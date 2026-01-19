"""Repository for operational alerts (health, coverage, drift, confidence)."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class UpsertResult:
    """Result from upserting an ops alert."""

    id: UUID
    is_new: bool
    previous_severity: Optional[str]
    current_severity: str
    escalated: bool


@dataclass
class OpsAlert:
    """Operational alert model."""

    id: UUID
    workspace_id: UUID
    rule_type: str
    severity: str
    status: str
    rule_version: str
    dedupe_key: str
    payload: dict
    source: str
    job_run_id: Optional[UUID]
    created_at: datetime
    last_seen_at: datetime
    resolved_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    occurrence_count: int = 1

    @classmethod
    def from_row(cls, row: dict) -> "OpsAlert":
        """Create from database row."""
        return cls(
            id=row["id"],
            workspace_id=row["workspace_id"],
            rule_type=row["rule_type"],
            severity=row["severity"],
            status=row["status"],
            rule_version=row["rule_version"],
            dedupe_key=row["dedupe_key"],
            payload=(
                row["payload"]
                if isinstance(row["payload"], dict)
                else json.loads(row["payload"])
            ),
            source=row["source"],
            job_run_id=row.get("job_run_id"),
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            resolved_at=row.get("resolved_at"),
            acknowledged_at=row.get("acknowledged_at"),
            acknowledged_by=row.get("acknowledged_by"),
            occurrence_count=row.get("occurrence_count", 1),
        )


class OpsAlertsRepository:
    """Repository for operational alerts with deduplication."""

    # Severity ordering for escalation detection
    SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def upsert(
        self,
        workspace_id: UUID,
        rule_type: str,
        severity: str,
        dedupe_key: str,
        payload: dict,
        source: str = "alert_evaluator",
        job_run_id: Optional[UUID] = None,
        rule_version: str = "v1",
    ) -> UpsertResult:
        """
        Upsert an operational alert.

        Returns UpsertResult with:
        - is_new: True if this is a new alert
        - escalated: True if severity increased
        - previous_severity: The severity before update (for escalation detection)

        Only updates active alerts - resolved alerts are not touched.
        """
        query = """
            INSERT INTO ops_alerts (
                workspace_id, rule_type, severity, dedupe_key,
                payload, source, job_run_id, rule_version
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (workspace_id, dedupe_key) DO UPDATE SET
                last_seen_at = NOW(),
                payload = EXCLUDED.payload,
                severity = EXCLUDED.severity,
                job_run_id = EXCLUDED.job_run_id,
                source = EXCLUDED.source,
                occurrence_count = ops_alerts.occurrence_count + 1
            WHERE ops_alerts.status = 'active'
            RETURNING
                id,
                (xmax = 0) AS is_new,
                severity AS current_severity
        """

        # First get existing severity if any (for escalation detection)
        existing_query = """
            SELECT severity FROM ops_alerts
            WHERE workspace_id = $1 AND dedupe_key = $2 AND status = 'active'
        """

        async with self.pool.acquire() as conn:
            existing_row = await conn.fetchrow(existing_query, workspace_id, dedupe_key)
            previous_severity = existing_row["severity"] if existing_row else None

            row = await conn.fetchrow(
                query,
                workspace_id,
                rule_type,
                severity,
                dedupe_key,
                json.dumps(payload),
                source,
                job_run_id,
                rule_version,
            )

        if not row:
            # This happens if the alert exists but is resolved
            # The WHERE clause prevents updating resolved alerts
            logger.debug(
                "ops_alert_upsert_skipped_resolved",
                workspace_id=str(workspace_id),
                dedupe_key=dedupe_key,
            )
            # Return a result indicating no change
            return UpsertResult(
                id=UUID("00000000-0000-0000-0000-000000000000"),
                is_new=False,
                previous_severity=None,
                current_severity=severity,
                escalated=False,
            )

        is_new = row["is_new"]
        current_severity = row["current_severity"]

        # Detect escalation
        escalated = False
        if not is_new and previous_severity:
            prev_order = self.SEVERITY_ORDER.get(previous_severity, 0)
            curr_order = self.SEVERITY_ORDER.get(current_severity, 0)
            escalated = curr_order > prev_order

        logger.info(
            "ops_alert_upserted",
            alert_id=str(row["id"]),
            workspace_id=str(workspace_id),
            rule_type=rule_type,
            dedupe_key=dedupe_key,
            is_new=is_new,
            escalated=escalated,
            severity=current_severity,
        )

        return UpsertResult(
            id=row["id"],
            is_new=is_new,
            previous_severity=previous_severity,
            current_severity=current_severity,
            escalated=escalated,
        )

    async def resolve(self, alert_id: UUID) -> Optional[OpsAlert]:
        """
        Resolve an active alert by ID.

        Returns the resolved alert or None if not found/already resolved.
        Idempotent: returns alert even if already resolved.
        """
        query = """
            UPDATE ops_alerts
            SET status = 'resolved', resolved_at = NOW()
            WHERE id = $1 AND status = 'active'
            RETURNING *
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, alert_id)

        if row:
            logger.info("ops_alert_resolved", alert_id=str(alert_id))
            return OpsAlert.from_row(dict(row))

        # Check if already resolved (for idempotent return)
        get_query = "SELECT * FROM ops_alerts WHERE id = $1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(get_query, alert_id)

        if row:
            return OpsAlert.from_row(dict(row))
        return None

    async def resolve_by_dedupe_key(
        self, workspace_id: UUID, dedupe_key: str
    ) -> Optional[OpsAlert]:
        """
        Resolve an active alert by dedupe key.

        Returns the resolved alert or None if not found/already resolved.
        """
        query = """
            UPDATE ops_alerts
            SET status = 'resolved', resolved_at = NOW()
            WHERE workspace_id = $1 AND dedupe_key = $2 AND status = 'active'
            RETURNING *
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, dedupe_key)

        if row:
            logger.info(
                "ops_alert_resolved_by_key",
                workspace_id=str(workspace_id),
                dedupe_key=dedupe_key,
            )
            return OpsAlert.from_row(dict(row))
        return None

    async def acknowledge(
        self, alert_id: UUID, acknowledged_by: Optional[str] = None
    ) -> tuple[bool, bool]:
        """
        Acknowledge an alert.

        Returns (success, was_already_acknowledged).
        Idempotent: returns True even if already acknowledged.
        """
        query = """
            UPDATE ops_alerts
            SET acknowledged_at = NOW(), acknowledged_by = $2
            WHERE id = $1 AND acknowledged_at IS NULL
        """

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, alert_id, acknowledged_by)

        if result == "UPDATE 1":
            logger.info("ops_alert_acknowledged", alert_id=str(alert_id))
            return True, False

        # Check if already acknowledged
        check_query = "SELECT acknowledged_at FROM ops_alerts WHERE id = $1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(check_query, alert_id)

        if row and row["acknowledged_at"]:
            return True, True  # Already acknowledged
        return False, False  # Not found

    async def get(self, alert_id: UUID) -> Optional[OpsAlert]:
        """Get alert by ID."""
        query = "SELECT * FROM ops_alerts WHERE id = $1"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, alert_id)

        return OpsAlert.from_row(dict(row)) if row else None

    async def list_alerts(
        self,
        workspace_id: UUID,
        status: Optional[list[str]] = None,
        severity: Optional[list[str]] = None,
        rule_type: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[OpsAlert], int]:
        """
        List alerts with filters.

        Args:
            workspace_id: Required workspace filter
            status: Filter by status(es) - ['active'], ['resolved'], ['active', 'resolved']
            severity: Filter by severity(ies) - ['critical', 'high'], etc.
            rule_type: Filter by rule type(s)
            limit: Max results (default 50, max 100)
            offset: Pagination offset

        Returns:
            (alerts, total_count)

        Ordering:
            - If status=['active'] only: order by last_seen_at DESC (still happening first)
            - Otherwise: order by created_at DESC
        """
        limit = min(limit, 100)

        conditions = ["workspace_id = $1"]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if status:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(status)))
            conditions.append(f"status IN ({placeholders})")
            params.extend(status)
            param_idx += len(status)

        if severity:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(severity)))
            conditions.append(f"severity IN ({placeholders})")
            params.extend(severity)
            param_idx += len(severity)

        if rule_type:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(rule_type)))
            conditions.append(f"rule_type IN ({placeholders})")
            params.extend(rule_type)
            param_idx += len(rule_type)

        where_clause = " AND ".join(conditions)

        # Determine ordering
        if status == ["active"]:
            order_by = "last_seen_at DESC"
        else:
            order_by = "created_at DESC"

        query = f"""
            SELECT * FROM ops_alerts
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        count_query = f"SELECT COUNT(*) FROM ops_alerts WHERE {where_clause}"
        count_params = params[:-2]  # Exclude limit/offset

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            total = await conn.fetchval(count_query, *count_params)

        alerts = [OpsAlert.from_row(dict(r)) for r in rows]
        return alerts, total or 0

    async def get_active_alerts_by_rule(
        self, workspace_id: UUID, rule_type: str
    ) -> list[OpsAlert]:
        """Get all active alerts for a specific rule type."""
        query = """
            SELECT * FROM ops_alerts
            WHERE workspace_id = $1 AND rule_type = $2 AND status = 'active'
            ORDER BY created_at DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, rule_type)

        return [OpsAlert.from_row(dict(r)) for r in rows]

    async def get_active_dedupe_keys(
        self, workspace_id: UUID, rule_type_prefix: Optional[str] = None
    ) -> set[str]:
        """
        Get all active dedupe keys for a workspace.

        Used by resolution pass to find alerts that should be resolved.
        """
        if rule_type_prefix:
            query = """
                SELECT dedupe_key FROM ops_alerts
                WHERE workspace_id = $1 AND status = 'active' AND rule_type LIKE $2
            """
            params = [workspace_id, f"{rule_type_prefix}%"]
        else:
            query = """
                SELECT dedupe_key FROM ops_alerts
                WHERE workspace_id = $1 AND status = 'active'
            """
            params = [workspace_id]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return {r["dedupe_key"] for r in rows}
