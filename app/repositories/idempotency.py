"""Repository for idempotency key operations.

Provides atomic claim semantics for idempotency keys to prevent duplicate operations.
"""

import asyncio
import json
import time
from typing import Optional
from uuid import UUID

import structlog

from app.services.idempotency import IdempotencyRecord

logger = structlog.get_logger(__name__)

# Stale pending threshold (10 minutes)
STALE_PENDING_MINUTES = 10


class IdempotencyRepository:
    """Repository for idempotency key operations."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def claim_or_get(
        self,
        workspace_id: UUID,
        idempotency_key: str,
        request_hash: str,
        endpoint: str,
        http_method: str = "POST",
        api_version: str | None = None,
    ) -> tuple[bool, IdempotencyRecord]:
        """Atomically claim an idempotency key or get existing.

        Uses INSERT ... ON CONFLICT DO NOTHING for atomic claim semantics.
        Includes endpoint in constraint so same key can be used across endpoints.

        Args:
            workspace_id: Workspace UUID
            idempotency_key: Client-provided key (max 200 chars)
            request_hash: SHA256 hash of canonical request JSON
            endpoint: API endpoint (e.g., 'backtests.tune')
            http_method: HTTP method (default 'POST')
            api_version: API version for behavior change detection

        Returns:
            Tuple of (is_new, record)
            - (True, record) = key claimed, proceed with creation
            - (False, record) = key exists, check hash and replay/wait
        """
        async with self.pool.acquire() as conn:
            # Try to insert new key atomically
            result = await conn.fetchrow(
                """
                INSERT INTO idempotency_keys
                    (workspace_id, idempotency_key, request_hash,
                     endpoint, http_method, api_version)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (workspace_id, endpoint, idempotency_key) DO NOTHING
                RETURNING id, workspace_id, idempotency_key, request_hash, endpoint,
                          http_method, api_version, resource_id, response_json,
                          error_code, error_json, status
                """,
                workspace_id,
                idempotency_key,
                request_hash,
                endpoint,
                http_method,
                api_version,
            )

            if result:
                # We claimed it
                logger.info(
                    "idempotency_key_claimed",
                    idempotency_key=idempotency_key,
                    endpoint=endpoint,
                    workspace_id=str(workspace_id),
                )
                return (True, IdempotencyRecord.from_row(dict(result)))

            # Conflict - read existing
            existing = await conn.fetchrow(
                """
                SELECT id, workspace_id, idempotency_key, request_hash, endpoint,
                       http_method, api_version, resource_id, response_json,
                       error_code, error_json, status, created_at, updated_at
                FROM idempotency_keys
                WHERE workspace_id = $1 AND endpoint = $2 AND idempotency_key = $3
                """,
                workspace_id,
                endpoint,
                idempotency_key,
            )

            if not existing:
                # Rare race: was deleted between conflict and select
                raise RuntimeError("Idempotency key disappeared during claim")

            logger.info(
                "idempotency_key_exists",
                idempotency_key=idempotency_key,
                endpoint=endpoint,
                status=existing["status"],
            )
            return (False, IdempotencyRecord.from_row(dict(existing)))

    async def wait_for_completion_or_timeout(
        self,
        workspace_id: UUID,
        endpoint: str,
        idempotency_key: str,
        max_wait_seconds: float = 5.0,
        poll_interval: float = 0.5,
    ) -> Optional[IdempotencyRecord]:
        """Poll for pending request to complete.

        Also detects stale pending keys (created > STALE_PENDING_MINUTES ago).

        Args:
            workspace_id: Workspace UUID
            endpoint: API endpoint
            idempotency_key: Client-provided key
            max_wait_seconds: Maximum time to wait (default 5s)
            poll_interval: Time between polls (default 0.5s)

        Returns:
            Completed/failed record or None if still pending after timeout
        """
        deadline = time.monotonic() + max_wait_seconds

        async with self.pool.acquire() as conn:
            while time.monotonic() < deadline:
                row = await conn.fetchrow(
                    """
                    SELECT id, workspace_id, idempotency_key, request_hash, endpoint,
                           http_method, api_version, resource_id, response_json,
                           error_code, error_json, status, created_at, updated_at
                    FROM idempotency_keys
                    WHERE workspace_id = $1 AND endpoint = $2 AND idempotency_key = $3
                    """,
                    workspace_id,
                    endpoint,
                    idempotency_key,
                )

                if not row:
                    return None

                if row["status"] != "pending":
                    return IdempotencyRecord.from_row(dict(row))

                # Check if stale (created > 10 minutes ago and still pending)
                created_at = row["created_at"]
                if created_at:
                    from datetime import datetime, timezone

                    now = datetime.now(timezone.utc)
                    age_minutes = (now - created_at).total_seconds() / 60
                    if age_minutes > STALE_PENDING_MINUTES:
                        logger.warning(
                            "idempotency_key_stale",
                            idempotency_key=idempotency_key,
                            endpoint=endpoint,
                            age_minutes=age_minutes,
                        )
                        # Mark as failed due to staleness
                        await self.mark_failed(
                            row["id"],
                            error_code="stale_pending",
                            error_json={
                                "reason": f"Pending for {age_minutes:.1f} minutes"
                            },
                        )
                        # Re-fetch the updated record
                        updated = await conn.fetchrow(
                            """
                            SELECT * FROM idempotency_keys WHERE id = $1
                            """,
                            row["id"],
                        )
                        return (
                            IdempotencyRecord.from_row(dict(updated))
                            if updated
                            else None
                        )

                await asyncio.sleep(poll_interval)

        logger.info(
            "idempotency_wait_timeout",
            idempotency_key=idempotency_key,
            endpoint=endpoint,
            max_wait_seconds=max_wait_seconds,
        )
        return None  # Still pending after timeout

    async def complete(
        self,
        idempotency_id: UUID,
        resource_id: UUID,
        response_json: dict,
    ) -> None:
        """Mark idempotency key as completed with response.

        Args:
            idempotency_id: Idempotency record ID
            resource_id: Created resource ID (e.g., tune_id)
            response_json: Exact response to replay
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE idempotency_keys
                SET status = 'completed',
                    resource_id = $2,
                    response_json = $3,
                    updated_at = NOW()
                WHERE id = $1
                """,
                idempotency_id,
                resource_id,
                json.dumps(response_json),
            )

        logger.info(
            "idempotency_completed",
            idempotency_id=str(idempotency_id),
            resource_id=str(resource_id),
        )

    async def mark_failed(
        self,
        idempotency_id: UUID,
        error_code: str,
        error_json: dict | None = None,
    ) -> None:
        """Mark idempotency key as failed.

        Failed keys are also replayed deterministically (same error on retry).

        Args:
            idempotency_id: Idempotency record ID
            error_code: Error classification (e.g., 'validation_error', 'internal_error')
            error_json: Error details for replay
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE idempotency_keys
                SET status = 'failed',
                    error_code = $2,
                    error_json = $3,
                    updated_at = NOW()
                WHERE id = $1
                """,
                idempotency_id,
                error_code,
                json.dumps(error_json) if error_json else None,
            )

        logger.info(
            "idempotency_failed",
            idempotency_id=str(idempotency_id),
            error_code=error_code,
        )

    async def get_by_resource(
        self,
        resource_id: UUID,
    ) -> Optional[IdempotencyRecord]:
        """Get idempotency record by resource ID.

        Useful for looking up the original request that created a resource.

        Args:
            resource_id: Created resource ID

        Returns:
            Idempotency record or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, workspace_id, idempotency_key, request_hash, endpoint,
                       http_method, api_version, resource_id, response_json,
                       error_code, error_json, status
                FROM idempotency_keys
                WHERE resource_id = $1
                """,
                resource_id,
            )

        return IdempotencyRecord.from_row(dict(row)) if row else None

    async def prune_expired(
        self,
        batch_size: int = 10000,
        dry_run: bool = True,
    ) -> int:
        """Delete expired idempotency keys.

        Args:
            batch_size: Rows per delete batch
            dry_run: If True, count only without deleting

        Returns:
            Number of rows deleted (or would be deleted if dry_run)
        """
        async with self.pool.acquire() as conn:
            if dry_run:
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM idempotency_keys
                    WHERE expires_at < NOW()
                    """
                )
                return count or 0

            total_deleted = 0
            while True:
                # Batch delete with ORDER BY for deterministic behavior
                result = await conn.fetchval(
                    """
                    WITH candidates AS (
                        SELECT id FROM idempotency_keys
                        WHERE expires_at < NOW()
                        ORDER BY expires_at ASC
                        LIMIT $1
                    ),
                    deleted AS (
                        DELETE FROM idempotency_keys
                        WHERE id IN (SELECT id FROM candidates)
                        RETURNING 1
                    )
                    SELECT COUNT(*) FROM deleted
                    """,
                    batch_size,
                )

                batch_deleted = result or 0
                total_deleted += batch_deleted

                if batch_deleted < batch_size:
                    break

                # Small sleep between batches
                await asyncio.sleep(0.1)

            logger.info(
                "idempotency_keys_pruned",
                total_deleted=total_deleted,
            )
            return total_deleted
