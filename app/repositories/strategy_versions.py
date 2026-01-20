"""Repository for strategy version lifecycle management.

Handles version creation, state transitions, and audit logging.
All state changes are recorded in strategy_version_transitions table.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


# Valid state transitions
VALID_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"active", "retired"},
    "active": {"paused", "retired"},
    "paused": {"active", "retired"},
    "retired": set(),  # Terminal state
}


@dataclass
class StrategyVersion:
    """Strategy version model."""

    id: UUID
    strategy_id: UUID
    strategy_entity_id: UUID
    version_number: int
    version_tag: Optional[str]
    config_snapshot: dict
    config_hash: str
    state: str
    regime_awareness: dict
    created_at: datetime
    created_by: Optional[str]
    activated_at: Optional[datetime]
    paused_at: Optional[datetime]
    retired_at: Optional[datetime]
    kb_strategy_spec_id: Optional[UUID]

    @classmethod
    def from_row(cls, row: dict) -> "StrategyVersion":
        """Create from database row."""
        config = row["config_snapshot"]
        if isinstance(config, str):
            config = json.loads(config)

        regime = row.get("regime_awareness", {})
        if isinstance(regime, str):
            regime = json.loads(regime)

        return cls(
            id=row["id"],
            strategy_id=row["strategy_id"],
            strategy_entity_id=row["strategy_entity_id"],
            version_number=row["version_number"],
            version_tag=row.get("version_tag"),
            config_snapshot=config,
            config_hash=row["config_hash"].strip() if row["config_hash"] else "",
            state=row["state"],
            regime_awareness=regime or {},
            created_at=row["created_at"],
            created_by=row.get("created_by"),
            activated_at=row.get("activated_at"),
            paused_at=row.get("paused_at"),
            retired_at=row.get("retired_at"),
            kb_strategy_spec_id=row.get("kb_strategy_spec_id"),
        )


@dataclass
class VersionTransition:
    """State transition audit record."""

    id: UUID
    version_id: UUID
    from_state: Optional[str]
    to_state: str
    triggered_by: str
    triggered_at: datetime
    reason: Optional[str]

    @classmethod
    def from_row(cls, row: dict) -> "VersionTransition":
        """Create from database row."""
        return cls(
            id=row["id"],
            version_id=row["version_id"],
            from_state=row.get("from_state"),
            to_state=row["to_state"],
            triggered_by=row["triggered_by"],
            triggered_at=row["triggered_at"],
            reason=row.get("reason"),
        )


def compute_config_hash(config: dict) -> str:
    """Compute SHA256 hash of config for deduplication.

    Returns 64-char hex string.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StrategyVersionsRepository:
    """Repository for strategy version CRUD and state transitions."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def create_version(
        self,
        strategy_id: UUID,
        config_snapshot: dict,
        created_by: str = "system",
        version_tag: Optional[str] = None,
        regime_awareness: Optional[dict] = None,
        kb_strategy_spec_id: Optional[UUID] = None,
    ) -> StrategyVersion:
        """
        Create a new strategy version in draft state.

        Auto-increments version_number, computes config_hash,
        copies strategy_entity_id from parent strategy.

        Args:
            strategy_id: Parent strategy UUID
            config_snapshot: Immutable config dict
            created_by: Actor creating version (e.g., "admin:token_name")
            version_tag: Optional tag (e.g., "v1.0-beta")
            regime_awareness: Optional regime config for v1.5
            kb_strategy_spec_id: Optional link to kb_strategy_specs

        Returns:
            Created StrategyVersion in draft state

        Raises:
            ValueError: If strategy doesn't exist or has no entity_id mapping
            ValueError: If config_hash already exists for this strategy
        """
        config_hash = compute_config_hash(config_snapshot)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get parent strategy's entity_id with lock
                parent = await conn.fetchrow(
                    """
                    SELECT strategy_entity_id FROM strategies
                    WHERE id = $1
                    FOR SHARE
                    """,
                    strategy_id,
                )

                if not parent:
                    raise ValueError(f"Strategy {strategy_id} not found")

                strategy_entity_id = parent["strategy_entity_id"]
                if not strategy_entity_id:
                    raise ValueError(
                        f"Strategy {strategy_id} has no strategy_entity_id mapping"
                    )

                # Get next version number
                max_version = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX(version_number), 0)
                    FROM strategy_versions
                    WHERE strategy_id = $1
                    """,
                    strategy_id,
                )
                version_number = max_version + 1

                # Insert version
                row = await conn.fetchrow(
                    """
                    INSERT INTO strategy_versions (
                        strategy_id, strategy_entity_id, version_number, version_tag,
                        config_snapshot, config_hash, state, regime_awareness,
                        created_by, kb_strategy_spec_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'draft', $7, $8, $9)
                    RETURNING *
                    """,
                    strategy_id,
                    strategy_entity_id,
                    version_number,
                    version_tag,
                    json.dumps(config_snapshot),
                    config_hash,
                    json.dumps(regime_awareness or {}),
                    created_by,
                    kb_strategy_spec_id,
                )

                # Record initial transition
                await conn.execute(
                    """
                    INSERT INTO strategy_version_transitions (
                        version_id, from_state, to_state, triggered_by, reason
                    ) VALUES ($1, NULL, 'draft', $2, 'Version created')
                    """,
                    row["id"],
                    created_by,
                )

        version = StrategyVersion.from_row(dict(row))

        logger.info(
            "strategy_version_created",
            version_id=str(version.id),
            strategy_id=str(strategy_id),
            version_number=version_number,
            created_by=created_by,
        )

        return version

    async def get_version(
        self,
        version_id: UUID,
        strategy_id: Optional[UUID] = None,
    ) -> Optional[StrategyVersion]:
        """
        Get version by ID.

        Args:
            version_id: Version UUID
            strategy_id: Optional strategy_id for scoping

        Returns:
            StrategyVersion or None if not found
        """
        if strategy_id:
            query = """
                SELECT * FROM strategy_versions
                WHERE id = $1 AND strategy_id = $2
            """
            params = [version_id, strategy_id]
        else:
            query = "SELECT * FROM strategy_versions WHERE id = $1"
            params = [version_id]

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        return StrategyVersion.from_row(dict(row)) if row else None

    async def list_versions(
        self,
        strategy_id: UUID,
        state: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StrategyVersion], int]:
        """
        List versions for a strategy.

        Args:
            strategy_id: Parent strategy UUID
            state: Optional filter by state
            limit: Max results
            offset: Pagination offset

        Returns:
            (list of versions, total count)
        """
        conditions = ["strategy_id = $1"]
        params: list[Any] = [strategy_id]
        param_idx = 2

        if state:
            conditions.append(f"state = ${param_idx}")
            params.append(state)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        count_query = f"""
            SELECT COUNT(*) FROM strategy_versions WHERE {where_clause}
        """
        data_query = f"""
            SELECT * FROM strategy_versions
            WHERE {where_clause}
            ORDER BY version_number DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(data_query, *params)

        versions = [StrategyVersion.from_row(dict(r)) for r in rows]
        return versions, total or 0

    async def get_active_version(
        self,
        strategy_id: UUID,
    ) -> Optional[StrategyVersion]:
        """Get the active version for a strategy (if any)."""
        query = """
            SELECT * FROM strategy_versions
            WHERE strategy_id = $1 AND state = 'active'
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, strategy_id)

        return StrategyVersion.from_row(dict(row)) if row else None

    async def activate(
        self,
        version_id: UUID,
        triggered_by: str,
        reason: Optional[str] = None,
    ) -> StrategyVersion:
        """
        Activate a version (draft or paused -> active).

        Transactional:
        1. Pause current active version (if any) with reason "Superseded by version X"
        2. Set this version to active
        3. Update strategies.active_version_id
        4. Record transitions

        Args:
            version_id: Version to activate
            triggered_by: Actor performing activation
            reason: Optional reason for activation

        Returns:
            Activated StrategyVersion

        Raises:
            ValueError: If version not found or invalid transition
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get version with lock
                version_row = await conn.fetchrow(
                    """
                    SELECT * FROM strategy_versions
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    version_id,
                )

                if not version_row:
                    raise ValueError(f"Version {version_id} not found")

                current_state = version_row["state"]
                if "active" not in VALID_TRANSITIONS.get(current_state, set()):
                    raise ValueError(f"Cannot activate from state '{current_state}'")

                strategy_id = version_row["strategy_id"]
                version_number = version_row["version_number"]

                # Pause current active version (if any)
                old_active = await conn.fetchrow(
                    """
                    SELECT id, version_number FROM strategy_versions
                    WHERE strategy_id = $1 AND state = 'active' AND id != $2
                    FOR UPDATE
                    """,
                    strategy_id,
                    version_id,
                )

                if old_active:
                    await conn.execute(
                        """
                        UPDATE strategy_versions
                        SET state = 'paused', paused_at = NOW()
                        WHERE id = $1
                        """,
                        old_active["id"],
                    )
                    await conn.execute(
                        """
                        INSERT INTO strategy_version_transitions (
                            version_id, from_state, to_state, triggered_by, reason
                        ) VALUES ($1, 'active', 'paused', $2, $3)
                        """,
                        old_active["id"],
                        triggered_by,
                        f"Superseded by version {version_number}",
                    )
                    logger.info(
                        "strategy_version_paused",
                        version_id=str(old_active["id"]),
                        reason=f"Superseded by version {version_number}",
                    )

                # Activate this version
                updated_row = await conn.fetchrow(
                    """
                    UPDATE strategy_versions
                    SET state = 'active', activated_at = NOW()
                    WHERE id = $1
                    RETURNING *
                    """,
                    version_id,
                )

                # Update strategy's active_version_id
                await conn.execute(
                    """
                    UPDATE strategies
                    SET active_version_id = $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    version_id,
                    strategy_id,
                )

                # Record transition
                await conn.execute(
                    """
                    INSERT INTO strategy_version_transitions (
                        version_id, from_state, to_state, triggered_by, reason
                    ) VALUES ($1, $2, 'active', $3, $4)
                    """,
                    version_id,
                    current_state,
                    triggered_by,
                    reason,
                )

        version = StrategyVersion.from_row(dict(updated_row))

        logger.info(
            "strategy_version_activated",
            version_id=str(version_id),
            strategy_id=str(strategy_id),
            version_number=version_number,
            triggered_by=triggered_by,
        )

        return version

    async def pause(
        self,
        version_id: UUID,
        triggered_by: str,
        reason: Optional[str] = None,
    ) -> StrategyVersion:
        """
        Pause an active version.

        Clears strategies.active_version_id if this was the active version.

        Args:
            version_id: Version to pause
            triggered_by: Actor performing pause
            reason: Optional reason for pausing

        Returns:
            Paused StrategyVersion

        Raises:
            ValueError: If version not found or invalid transition
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get version with lock
                version_row = await conn.fetchrow(
                    """
                    SELECT * FROM strategy_versions
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    version_id,
                )

                if not version_row:
                    raise ValueError(f"Version {version_id} not found")

                current_state = version_row["state"]
                if "paused" not in VALID_TRANSITIONS.get(current_state, set()):
                    raise ValueError(f"Cannot pause from state '{current_state}'")

                strategy_id = version_row["strategy_id"]

                # Pause this version
                updated_row = await conn.fetchrow(
                    """
                    UPDATE strategy_versions
                    SET state = 'paused', paused_at = NOW()
                    WHERE id = $1
                    RETURNING *
                    """,
                    version_id,
                )

                # Clear strategy's active_version_id if this was active
                if current_state == "active":
                    await conn.execute(
                        """
                        UPDATE strategies
                        SET active_version_id = NULL, updated_at = NOW()
                        WHERE id = $1 AND active_version_id = $2
                        """,
                        strategy_id,
                        version_id,
                    )

                # Record transition
                await conn.execute(
                    """
                    INSERT INTO strategy_version_transitions (
                        version_id, from_state, to_state, triggered_by, reason
                    ) VALUES ($1, $2, 'paused', $3, $4)
                    """,
                    version_id,
                    current_state,
                    triggered_by,
                    reason,
                )

        version = StrategyVersion.from_row(dict(updated_row))

        logger.info(
            "strategy_version_paused",
            version_id=str(version_id),
            strategy_id=str(strategy_id),
            triggered_by=triggered_by,
            reason=reason,
        )

        return version

    async def retire(
        self,
        version_id: UUID,
        triggered_by: str,
        reason: Optional[str] = None,
    ) -> StrategyVersion:
        """
        Retire a version (terminal state).

        Can retire from draft, active, or paused.
        Clears strategies.active_version_id if this was active.

        Args:
            version_id: Version to retire
            triggered_by: Actor performing retirement
            reason: Optional reason for retiring

        Returns:
            Retired StrategyVersion

        Raises:
            ValueError: If version not found or already retired
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get version with lock
                version_row = await conn.fetchrow(
                    """
                    SELECT * FROM strategy_versions
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    version_id,
                )

                if not version_row:
                    raise ValueError(f"Version {version_id} not found")

                current_state = version_row["state"]
                if "retired" not in VALID_TRANSITIONS.get(current_state, set()):
                    raise ValueError(f"Cannot retire from state '{current_state}'")

                strategy_id = version_row["strategy_id"]

                # Retire this version
                updated_row = await conn.fetchrow(
                    """
                    UPDATE strategy_versions
                    SET state = 'retired', retired_at = NOW()
                    WHERE id = $1
                    RETURNING *
                    """,
                    version_id,
                )

                # Clear strategy's active_version_id if this was active
                if current_state == "active":
                    await conn.execute(
                        """
                        UPDATE strategies
                        SET active_version_id = NULL, updated_at = NOW()
                        WHERE id = $1 AND active_version_id = $2
                        """,
                        strategy_id,
                        version_id,
                    )

                # Record transition
                await conn.execute(
                    """
                    INSERT INTO strategy_version_transitions (
                        version_id, from_state, to_state, triggered_by, reason
                    ) VALUES ($1, $2, 'retired', $3, $4)
                    """,
                    version_id,
                    current_state,
                    triggered_by,
                    reason,
                )

        version = StrategyVersion.from_row(dict(updated_row))

        logger.info(
            "strategy_version_retired",
            version_id=str(version_id),
            strategy_id=str(strategy_id),
            triggered_by=triggered_by,
            reason=reason,
        )

        return version

    async def get_transitions(
        self,
        version_id: UUID,
        limit: int = 50,
    ) -> list[VersionTransition]:
        """Get state transition history for a version."""
        query = """
            SELECT * FROM strategy_version_transitions
            WHERE version_id = $1
            ORDER BY triggered_at DESC
            LIMIT $2
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, version_id, limit)

        return [VersionTransition.from_row(dict(r)) for r in rows]

    async def get_version_by_hash(
        self,
        strategy_id: UUID,
        config_hash: str,
    ) -> Optional[StrategyVersion]:
        """Get version by config hash (for deduplication)."""
        query = """
            SELECT * FROM strategy_versions
            WHERE strategy_id = $1 AND config_hash = $2
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, strategy_id, config_hash)

        return StrategyVersion.from_row(dict(row)) if row else None
