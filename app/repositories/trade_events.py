"""Trade Events Repository: Append-only event journal for trading decisions."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import structlog

from app.schemas import TradeEvent, TradeEventType


logger = structlog.get_logger(__name__)


@dataclass
class EventFilters:
    """Filters for querying trade events."""

    workspace_id: UUID
    event_types: Optional[list[TradeEventType]] = None
    strategy_entity_id: Optional[UUID] = None
    symbol: Optional[str] = None
    correlation_id: Optional[str] = None
    intent_id: Optional[UUID] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None


class TradeEventsRepository:
    """
    Repository for trade event journal.

    This is append-only by design - events are never updated or deleted.
    Provides methods for writing events and querying the journal.
    """

    def __init__(self, pool):
        """Initialize repository with asyncpg pool."""
        self.pool = pool

    async def insert(self, event: TradeEvent) -> UUID:
        """
        Insert a single event into the journal.

        Returns the event ID.
        """
        query = """
            INSERT INTO trade_events (
                id, correlation_id, workspace_id,
                event_type, created_at,
                strategy_entity_id, symbol, timeframe,
                intent_id, order_id, position_id,
                payload, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            )
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                event.id,
                event.correlation_id,
                event.workspace_id,
                event.event_type.value,
                event.created_at,
                event.strategy_entity_id,
                event.symbol,
                event.timeframe,
                event.intent_id,
                event.order_id,
                event.position_id,
                json.dumps(event.payload),
                json.dumps(event.metadata),
            )
            logger.debug(
                "Event inserted",
                event_id=str(row["id"]),
                event_type=event.event_type.value,
                correlation_id=event.correlation_id,
            )
            return row["id"]

    async def insert_many(self, events: list[TradeEvent]) -> int:
        """
        Insert multiple events in a single transaction.

        Returns count of events inserted.
        """
        if not events:
            return 0

        query = """
            INSERT INTO trade_events (
                id, correlation_id, workspace_id,
                event_type, created_at,
                strategy_entity_id, symbol, timeframe,
                intent_id, order_id, position_id,
                payload, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            )
        """

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for event in events:
                    await conn.execute(
                        query,
                        event.id,
                        event.correlation_id,
                        event.workspace_id,
                        event.event_type.value,
                        event.created_at,
                        event.strategy_entity_id,
                        event.symbol,
                        event.timeframe,
                        event.intent_id,
                        event.order_id,
                        event.position_id,
                        json.dumps(event.payload),
                        json.dumps(event.metadata),
                    )

        logger.info("Events batch inserted", count=len(events))
        return len(events)

    async def get_by_id(self, event_id: UUID) -> Optional[TradeEvent]:
        """Get a single event by ID."""
        query = """
            SELECT
                id, correlation_id, workspace_id,
                event_type, created_at,
                strategy_entity_id, symbol, timeframe,
                intent_id, order_id, position_id,
                payload, metadata
            FROM trade_events
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, event_id)

        if not row:
            return None

        return self._row_to_event(row)

    async def list_events(
        self,
        filters: EventFilters,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[TradeEvent], int]:
        """
        List events with filters and pagination.

        Returns (events, total_count).
        """
        # Build WHERE clause
        conditions = ["workspace_id = $1"]
        params: list = [filters.workspace_id]
        param_idx = 2

        if filters.event_types:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(filters.event_types)))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(et.value for et in filters.event_types)
            param_idx += len(filters.event_types)

        if filters.strategy_entity_id:
            conditions.append(f"strategy_entity_id = ${param_idx}")
            params.append(filters.strategy_entity_id)
            param_idx += 1

        if filters.symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(filters.symbol)
            param_idx += 1

        if filters.correlation_id:
            conditions.append(f"correlation_id = ${param_idx}")
            params.append(filters.correlation_id)
            param_idx += 1

        if filters.intent_id:
            conditions.append(f"intent_id = ${param_idx}")
            params.append(filters.intent_id)
            param_idx += 1

        if filters.since:
            conditions.append(f"created_at >= ${param_idx}")
            params.append(filters.since)
            param_idx += 1

        if filters.until:
            conditions.append(f"created_at <= ${param_idx}")
            params.append(filters.until)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
            SELECT COUNT(*) as total
            FROM trade_events
            WHERE {where_clause}
        """

        # Data query
        data_query = f"""
            SELECT
                id, correlation_id, workspace_id,
                event_type, created_at,
                strategy_entity_id, symbol, timeframe,
                intent_id, order_id, position_id,
                payload, metadata
            FROM trade_events
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            count_row = await conn.fetchrow(count_query, *params[:-2])
            total = count_row["total"]

            rows = await conn.fetch(data_query, *params)

        events = [self._row_to_event(row) for row in rows]
        return events, total

    async def get_by_correlation_id(
        self,
        correlation_id: str,
    ) -> list[TradeEvent]:
        """Get all events for a correlation ID, ordered by time."""
        query = """
            SELECT
                id, correlation_id, workspace_id,
                event_type, created_at,
                strategy_entity_id, symbol, timeframe,
                intent_id, order_id, position_id,
                payload, metadata
            FROM trade_events
            WHERE correlation_id = $1
            ORDER BY created_at ASC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, correlation_id)

        return [self._row_to_event(row) for row in rows]

    async def get_recent_by_strategy(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        hours: int = 24,
        limit: int = 100,
    ) -> list[TradeEvent]:
        """Get recent events for a specific strategy."""
        since = datetime.utcnow() - timedelta(hours=hours)

        query = """
            SELECT
                id, correlation_id, workspace_id,
                event_type, created_at,
                strategy_entity_id, symbol, timeframe,
                intent_id, order_id, position_id,
                payload, metadata
            FROM trade_events
            WHERE workspace_id = $1
              AND strategy_entity_id = $2
              AND created_at >= $3
            ORDER BY created_at DESC
            LIMIT $4
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, strategy_entity_id, since, limit)

        return [self._row_to_event(row) for row in rows]

    async def count_by_type(
        self,
        workspace_id: UUID,
        since_hours: int = 24,
    ) -> dict[str, int]:
        """Get event counts grouped by type."""
        since = datetime.utcnow() - timedelta(hours=since_hours)

        query = """
            SELECT event_type, COUNT(*) as count
            FROM trade_events
            WHERE workspace_id = $1 AND created_at >= $2
            GROUP BY event_type
            ORDER BY count DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, since)

        return {row["event_type"]: row["count"] for row in rows}

    def _row_to_event(self, row) -> TradeEvent:
        """Convert database row to TradeEvent."""
        return TradeEvent(
            id=row["id"],
            correlation_id=row["correlation_id"],
            workspace_id=row["workspace_id"],
            event_type=TradeEventType(row["event_type"]),
            created_at=row["created_at"],
            strategy_entity_id=row["strategy_entity_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            intent_id=row["intent_id"],
            order_id=row["order_id"],
            position_id=row["position_id"],
            payload=row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"] or "{}"),
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
        )
