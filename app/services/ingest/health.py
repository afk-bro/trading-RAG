"""Health validation for ingested sources.

This module provides validation to ensure ingested documents are healthy:
- Chunks were created successfully (chunk_count > 0)
- Embeddings were created for all chunks (embedding_count == chunk_count)
- Status transitions are correct

Run validation after chunking + embedding + DB writes, before marking "active".
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Overall health status for a source."""

    OK = "ok"  # All checks pass
    DEGRADED = (
        "degraded"  # Some checks fail but source is usable (e.g., missing embeddings)
    )
    FAILED = "failed"  # Critical checks fail, source is not usable


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    passed: bool
    message: str
    expected: Optional[int] = None
    actual: Optional[int] = None


@dataclass
class HealthResult:
    """Complete health validation result for a source."""

    source_id: UUID
    workspace_id: UUID
    status: HealthStatus
    checks: list[HealthCheckResult] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        """Return True if status is OK."""
        return self.status == HealthStatus.OK

    @property
    def chunk_count_ok(self) -> bool:
        """Return True if chunk count check passed."""
        for check in self.checks:
            if check.name == "chunk_count":
                return check.passed
        return False

    @property
    def embeddings_ok(self) -> bool:
        """Return True if embeddings check passed."""
        for check in self.checks:
            if check.name == "embeddings_parity":
                return check.passed
        return False


async def validate_source_health(
    pool,
    workspace_id: UUID,
    source_id: UUID,
    *,
    expected_chunk_count: Optional[int] = None,
    expected_vector_count: Optional[int] = None,
    require_embeddings: bool = True,
) -> HealthResult:
    """
    Validate health of an ingested source.

    Args:
        pool: asyncpg connection pool
        workspace_id: Workspace ID
        source_id: Document/source ID to validate
        expected_chunk_count: Expected number of chunks (if known from ingest)
        expected_vector_count: Expected number of vectors (if known from ingest)
        require_embeddings: If True, missing embeddings causes FAILED status.
                           If False, missing embeddings causes DEGRADED status.

    Returns:
        HealthResult with status and individual check results.

    Usage:
        After ingest pipeline completes:

        >>> result = await validate_source_health(pool, workspace_id, doc_id)
        >>> if result.status == HealthStatus.FAILED:
        ...     # Mark document as failed, store error
        >>> elif result.status == HealthStatus.DEGRADED:
        ...     # Mark as active but log warning
        >>> else:
        ...     # All good, mark as active
    """
    checks: list[HealthCheckResult] = []
    log = logger.bind(workspace_id=str(workspace_id), source_id=str(source_id))

    async with pool.acquire() as conn:
        # Check 1: Document exists
        doc = await conn.fetchrow(
            """
            SELECT id, status, last_indexed_at
            FROM documents
            WHERE id = $1 AND workspace_id = $2
            """,
            source_id,
            workspace_id,
        )

        if not doc:
            return HealthResult(
                source_id=source_id,
                workspace_id=workspace_id,
                status=HealthStatus.FAILED,
                checks=[
                    HealthCheckResult(
                        name="document_exists",
                        passed=False,
                        message="Document not found",
                    )
                ],
                error_code="document_not_found",
                error_message=f"Document {source_id} not found in workspace {workspace_id}",
            )

        checks.append(
            HealthCheckResult(
                name="document_exists",
                passed=True,
                message="Document exists",
            )
        )

        # Check 2: Chunk count > 0
        chunk_count_row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM chunks WHERE doc_id = $1",
            source_id,
        )
        actual_chunk_count = chunk_count_row["count"] if chunk_count_row else 0

        chunk_check_passed = actual_chunk_count > 0
        chunk_check_message = (
            f"Found {actual_chunk_count} chunks"
            if chunk_check_passed
            else "No chunks found"
        )

        # Also validate against expected if provided
        if (
            expected_chunk_count is not None
            and actual_chunk_count != expected_chunk_count
        ):
            chunk_check_passed = False
            chunk_check_message = (
                f"Chunk count mismatch: expected {expected_chunk_count}, "
                f"got {actual_chunk_count}"
            )

        checks.append(
            HealthCheckResult(
                name="chunk_count",
                passed=chunk_check_passed,
                message=chunk_check_message,
                expected=expected_chunk_count,
                actual=actual_chunk_count,
            )
        )

        # Check 3: Embedding parity (chunk_vectors count == chunks count)
        vector_count_row = await conn.fetchrow(
            """
            SELECT COUNT(*) as count
            FROM chunk_vectors cv
            JOIN chunks c ON cv.chunk_id = c.id
            WHERE c.doc_id = $1
            """,
            source_id,
        )
        actual_vector_count = vector_count_row["count"] if vector_count_row else 0

        embedding_check_passed = actual_vector_count == actual_chunk_count
        if embedding_check_passed:
            embedding_check_message = f"All {actual_chunk_count} chunks have embeddings"
        else:
            embedding_check_message = (
                f"Embedding mismatch: {actual_vector_count} vectors "
                f"for {actual_chunk_count} chunks"
            )

        # Also validate against expected if provided
        if (
            expected_vector_count is not None
            and actual_vector_count != expected_vector_count
        ):
            embedding_check_passed = False
            embedding_check_message = (
                f"Vector count mismatch: expected {expected_vector_count}, "
                f"got {actual_vector_count}"
            )

        checks.append(
            HealthCheckResult(
                name="embeddings_parity",
                passed=embedding_check_passed,
                message=embedding_check_message,
                expected=expected_vector_count or actual_chunk_count,
                actual=actual_vector_count,
            )
        )

        # Check 4: last_indexed_at is set (indicates indexing completed)
        indexed_check_passed = doc["last_indexed_at"] is not None
        checks.append(
            HealthCheckResult(
                name="indexed_timestamp",
                passed=indexed_check_passed,
                message=(
                    f"Indexed at {doc['last_indexed_at']}"
                    if indexed_check_passed
                    else "last_indexed_at not set"
                ),
            )
        )

    # Determine overall status
    # Critical failures: no chunks, document not found
    critical_failures = [
        c
        for c in checks
        if not c.passed and c.name in ("document_exists", "chunk_count")
    ]

    # Non-critical failures: missing embeddings (if not required), missing timestamp
    non_critical_failures = [
        c
        for c in checks
        if not c.passed
        and c.name not in ("document_exists", "chunk_count")
        and (c.name != "embeddings_parity" or not require_embeddings)
    ]

    # Embedding failure when required
    embedding_failure = not embedding_check_passed and require_embeddings

    if critical_failures or embedding_failure:
        status = HealthStatus.FAILED
        # Build error message from failures
        failure_names = [c.name for c in checks if not c.passed]
        error_code = "health_check_failed"
        error_message = f"Health checks failed: {', '.join(failure_names)}"
    elif non_critical_failures:
        status = HealthStatus.DEGRADED
        error_code = None
        error_message = None
    else:
        status = HealthStatus.OK
        error_code = None
        error_message = None

    result = HealthResult(
        source_id=source_id,
        workspace_id=workspace_id,
        status=status,
        checks=checks,
        error_code=error_code,
        error_message=error_message,
    )

    log.info(
        "source_health_validated",
        status=status.value,
        chunk_count=actual_chunk_count,
        vector_count=actual_vector_count,
        checks_passed=len([c for c in checks if c.passed]),
        checks_failed=len([c for c in checks if not c.passed]),
    )

    return result


async def get_source_health_summary(
    pool,
    workspace_id: UUID,
    source_id: UUID,
) -> dict:
    """
    Get health summary for a source (lightweight version for API responses).

    Returns dict suitable for including in SourceDetailResponse.
    """
    result = await validate_source_health(
        pool,
        workspace_id,
        source_id,
        require_embeddings=False,  # Don't fail, just report
    )

    return {
        "status": result.status.value,
        "chunk_count_ok": result.chunk_count_ok,
        "embeddings_ok": result.embeddings_ok,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "message": c.message,
            }
            for c in result.checks
        ],
    }
