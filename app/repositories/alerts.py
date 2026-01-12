"""Repository for alert rules and events."""

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.alerts.models import AlertStatus, RuleType, Severity

logger = structlog.get_logger(__name__)


class AlertsRepository:
    """Repository for alert rules and events queries."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    # =========================================================================
    # Alert Rules
    # =========================================================================

    async def list_rules(
        self,
        workspace_id: UUID,
        enabled_only: bool = False,
    ) -> list[dict]:
        """List alert rules for workspace."""
        query = """
            SELECT id, workspace_id, rule_type, strategy_entity_id, regime_key,
                   timeframe, enabled, config, cooldown_minutes, created_at, updated_at
            FROM alert_rules
            WHERE workspace_id = $1
        """
        params: list[Any] = [workspace_id]

        if enabled_only:
            query += " AND enabled = true"

        query += " ORDER BY created_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(r) for r in rows]

    async def get_rule(self, rule_id: UUID) -> Optional[dict]:
        """Get alert rule by ID."""
        query = """
            SELECT id, workspace_id, rule_type, strategy_entity_id, regime_key,
                   timeframe, enabled, config, cooldown_minutes, created_at, updated_at
            FROM alert_rules
            WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, rule_id)

        return dict(row) if row else None

    async def create_rule(
        self,
        workspace_id: UUID,
        rule_type: RuleType,
        config: dict,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
        timeframe: Optional[str] = None,
        cooldown_minutes: int = 60,
    ) -> dict:
        """Create new alert rule."""
        query = """
            INSERT INTO alert_rules (
                workspace_id, rule_type, strategy_entity_id, regime_key,
                timeframe, config, cooldown_minutes
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, workspace_id, rule_type, strategy_entity_id, regime_key,
                      timeframe, enabled, config, cooldown_minutes, created_at, updated_at
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                rule_type.value,
                strategy_entity_id,
                regime_key,
                timeframe,
                json.dumps(config),
                cooldown_minutes,
            )

        return dict(row)

    async def update_rule(
        self,
        rule_id: UUID,
        enabled: Optional[bool] = None,
        config: Optional[dict] = None,
        cooldown_minutes: Optional[int] = None,
    ) -> Optional[dict]:
        """Update alert rule."""
        updates = []
        params: list[Any] = []
        param_idx = 1

        if enabled is not None:
            updates.append(f"enabled = ${param_idx}")
            params.append(enabled)
            param_idx += 1

        if config is not None:
            updates.append(f"config = ${param_idx}")
            params.append(json.dumps(config))
            param_idx += 1

        if cooldown_minutes is not None:
            updates.append(f"cooldown_minutes = ${param_idx}")
            params.append(cooldown_minutes)
            param_idx += 1

        if not updates:
            return await self.get_rule(rule_id)

        params.append(rule_id)
        query = f"""
            UPDATE alert_rules
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, workspace_id, rule_type, strategy_entity_id, regime_key,
                      timeframe, enabled, config, cooldown_minutes, created_at, updated_at
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        return dict(row) if row else None

    async def delete_rule(self, rule_id: UUID) -> bool:
        """Delete alert rule."""
        query = "DELETE FROM alert_rules WHERE id = $1"
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, rule_id)
        return result == "DELETE 1"

    # =========================================================================
    # Alert Events
    # =========================================================================

    async def list_events(
        self,
        workspace_id: UUID,
        status: Optional[AlertStatus] = None,
        severity: Optional[Severity] = None,
        acknowledged: Optional[bool] = None,
        rule_type: Optional[RuleType] = None,
        strategy_entity_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List alert events with filters."""
        conditions = ["workspace_id = $1"]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1

        if severity:
            conditions.append(f"severity = ${param_idx}")
            params.append(severity.value)
            param_idx += 1

        if acknowledged is not None:
            conditions.append(f"acknowledged = ${param_idx}")
            params.append(acknowledged)
            param_idx += 1

        if rule_type:
            conditions.append(f"rule_type = ${param_idx}")
            params.append(rule_type.value)
            param_idx += 1

        if strategy_entity_id:
            conditions.append(f"strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT id, workspace_id, rule_id, strategy_entity_id, regime_key,
                   timeframe, rule_type, status, severity, acknowledged,
                   acknowledged_at, acknowledged_by, first_seen, activated_at,
                   last_seen, resolved_at, context_json, fingerprint,
                   created_at, updated_at
            FROM alert_events
            WHERE {where_clause}
            ORDER BY last_seen DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        count_query = f"SELECT COUNT(*) FROM alert_events WHERE {where_clause}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            total = await conn.fetchval(count_query, *params[:-2])

        return [dict(r) for r in rows], total or 0

    async def get_event(self, event_id: UUID) -> Optional[dict]:
        """Get alert event by ID."""
        query = """
            SELECT id, workspace_id, rule_id, strategy_entity_id, regime_key,
                   timeframe, rule_type, status, severity, acknowledged,
                   acknowledged_at, acknowledged_by, first_seen, activated_at,
                   last_seen, resolved_at, context_json, fingerprint,
                   created_at, updated_at
            FROM alert_events
            WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, event_id)

        return dict(row) if row else None

    async def get_existing_event(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
        rule_type: RuleType,
        fingerprint: str,
    ) -> Optional[dict]:
        """Get existing event by unique key."""
        query = """
            SELECT id, status, activated_at, last_seen
            FROM alert_events
            WHERE workspace_id = $1
              AND strategy_entity_id = $2
              AND regime_key = $3
              AND timeframe = $4
              AND rule_type = $5
              AND fingerprint = $6
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                strategy_entity_id,
                regime_key,
                timeframe,
                rule_type.value,
                fingerprint,
            )

        return dict(row) if row else None

    async def upsert_activate(
        self,
        workspace_id: UUID,
        rule_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
        rule_type: RuleType,
        severity: Severity,
        context_json: dict,
        fingerprint: str,
    ) -> dict:
        """Upsert alert event as active."""
        now = datetime.now(timezone.utc)
        query = """
            INSERT INTO alert_events (
                workspace_id, rule_id, strategy_entity_id, regime_key, timeframe,
                rule_type, status, severity, context_json, fingerprint,
                first_seen, activated_at, last_seen, acknowledged
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'active', $7, $8, $9, $10, $10, $10, false)
            ON CONFLICT (workspace_id, strategy_entity_id, regime_key, timeframe,
                         rule_type, fingerprint)
            DO UPDATE SET
                status = 'active',
                severity = $7,
                context_json = $8,
                activated_at = CASE
                    WHEN alert_events.status = 'resolved' THEN $10
                    ELSE alert_events.activated_at
                END,
                last_seen = $10,
                resolved_at = NULL,
                acknowledged = CASE
                    WHEN alert_events.status = 'resolved' THEN false
                    ELSE alert_events.acknowledged
                END,
                acknowledged_at = CASE
                    WHEN alert_events.status = 'resolved' THEN NULL
                    ELSE alert_events.acknowledged_at
                END,
                acknowledged_by = CASE
                    WHEN alert_events.status = 'resolved' THEN NULL
                    ELSE alert_events.acknowledged_by
                END
            RETURNING id, workspace_id, status, activated_at, last_seen
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                rule_id,
                strategy_entity_id,
                regime_key,
                timeframe,
                rule_type.value,
                severity.value,
                json.dumps(context_json),
                fingerprint,
                now,
            )

        return dict(row)

    async def update_last_seen(self, event_id: UUID) -> bool:
        """Update last_seen timestamp for active event."""
        query = """
            UPDATE alert_events
            SET last_seen = NOW()
            WHERE id = $1 AND status = 'active'
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id)
        return result == "UPDATE 1"

    async def resolve(self, event_id: UUID) -> bool:
        """Resolve active alert event."""
        query = """
            UPDATE alert_events
            SET status = 'resolved', resolved_at = NOW()
            WHERE id = $1 AND status = 'active'
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id)
        return result == "UPDATE 1"

    async def acknowledge(
        self, event_id: UUID, acknowledged_by: Optional[str] = None
    ) -> bool:
        """Acknowledge alert event."""
        query = """
            UPDATE alert_events
            SET acknowledged = true, acknowledged_at = NOW(), acknowledged_by = $2
            WHERE id = $1 AND acknowledged = false
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id, acknowledged_by)
        return result == "UPDATE 1"

    async def unacknowledge(self, event_id: UUID) -> bool:
        """Unacknowledge alert event."""
        query = """
            UPDATE alert_events
            SET acknowledged = false, acknowledged_at = NULL, acknowledged_by = NULL
            WHERE id = $1 AND acknowledged = true
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id)
        return result == "UPDATE 1"
