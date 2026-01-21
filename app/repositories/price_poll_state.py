"""Repository for price poll state management (LP3 DB-backed scheduling)."""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PollState:
    """State for a single poll key (exchange, symbol, timeframe)."""

    exchange_id: str
    symbol: str
    timeframe: str
    next_poll_at: Optional[datetime] = None
    failure_count: int = 0
    last_success_at: Optional[datetime] = None
    last_candle_ts: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class PollKey:
    """Unique identifier for a poll target."""

    exchange_id: str
    symbol: str
    timeframe: str


class PricePollStateRepository:
    """Repository for price poll state operations."""

    def __init__(
        self,
        pool,
        interval_seconds: int = 60,
        jitter_seconds: int = 5,
        backoff_max_seconds: int = 900,
    ):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
            interval_seconds: Base poll interval for scheduling
            jitter_seconds: Max random jitter to prevent thundering herd
            backoff_max_seconds: Max backoff cap for exponential backoff
        """
        self._pool = pool
        self._interval_seconds = interval_seconds
        self._jitter_seconds = jitter_seconds
        self._backoff_max_seconds = backoff_max_seconds

    def _row_to_state(self, row) -> PollState:
        """Convert database row to PollState."""
        return PollState(
            exchange_id=row["exchange_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            next_poll_at=row["next_poll_at"],
            failure_count=row["failure_count"],
            last_success_at=row["last_success_at"],
            last_candle_ts=row["last_candle_ts"],
            last_error=row["last_error"],
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    async def get_state(self, key: PollKey) -> Optional[PollState]:
        """Get state for a single poll key."""
        query = """
            SELECT exchange_id, symbol, timeframe, next_poll_at, failure_count,
                   last_success_at, last_candle_ts, last_error, created_at, updated_at
            FROM price_poll_state
            WHERE exchange_id = $1 AND symbol = $2 AND timeframe = $3
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, key.exchange_id, key.symbol, key.timeframe)
        if not row:
            return None
        return self._row_to_state(row)

    async def list_due_pairs(self, limit: int = 50) -> list[PollState]:
        """
        Get pairs due for polling (next_poll_at <= NOW or NULL).

        Returns pairs ordered by:
        1. NULL next_poll_at first (new pairs, never polled)
        2. Oldest next_poll_at (most overdue)
        """
        query = """
            SELECT exchange_id, symbol, timeframe, next_poll_at, failure_count,
                   last_success_at, last_candle_ts, last_error, created_at, updated_at
            FROM price_poll_state
            WHERE next_poll_at IS NULL OR next_poll_at <= NOW()
            ORDER BY next_poll_at NULLS FIRST, last_success_at NULLS FIRST
            LIMIT $1
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, limit)
        return [self._row_to_state(row) for row in rows]

    async def upsert_state_if_missing(self, key: PollKey) -> bool:
        """
        Seed state row for a new pair (next_poll_at=NULL = due immediately).

        Returns True if inserted, False if already existed.
        """
        query = """
            INSERT INTO price_poll_state (exchange_id, symbol, timeframe)
            VALUES ($1, $2, $3)
            ON CONFLICT (exchange_id, symbol, timeframe) DO NOTHING
            RETURNING exchange_id
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, key.exchange_id, key.symbol, key.timeframe)
        return row is not None

    async def mark_success(
        self,
        key: PollKey,
        last_candle_ts: datetime,
    ) -> None:
        """
        Update state on successful fetch.

        Resets failure_count and schedules next poll at interval + jitter.
        """
        jitter = random.uniform(0, self._jitter_seconds)
        next_poll = datetime.now(timezone.utc) + timedelta(
            seconds=self._interval_seconds + jitter
        )

        query = """
            INSERT INTO price_poll_state
                (exchange_id, symbol, timeframe, next_poll_at, failure_count,
                 last_success_at, last_candle_ts, last_error)
            VALUES ($1, $2, $3, $4, 0, NOW(), $5, NULL)
            ON CONFLICT (exchange_id, symbol, timeframe) DO UPDATE SET
                next_poll_at = EXCLUDED.next_poll_at,
                failure_count = 0,
                last_success_at = NOW(),
                last_candle_ts = EXCLUDED.last_candle_ts,
                last_error = NULL,
                updated_at = NOW()
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                key.exchange_id,
                key.symbol,
                key.timeframe,
                next_poll,
                last_candle_ts,
            )

    async def mark_failure(self, key: PollKey, error: str) -> None:
        """
        Update state on failed fetch with exponential backoff.

        Backoff formula: min(interval * 2^(failure_count-1), backoff_max)
        1st failure: 1x interval
        2nd failure: 2x interval
        3rd failure: 4x interval
        ... capped at backoff_max_seconds
        """
        # Get current failure count
        state = await self.get_state(key)
        failure_count = (state.failure_count if state else 0) + 1

        # Exponential backoff with cap
        backoff = min(
            self._interval_seconds * (2 ** (failure_count - 1)),
            self._backoff_max_seconds,
        )
        next_poll = datetime.now(timezone.utc) + timedelta(seconds=backoff)

        query = """
            INSERT INTO price_poll_state
                (exchange_id, symbol, timeframe, next_poll_at, failure_count, last_error)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (exchange_id, symbol, timeframe) DO UPDATE SET
                next_poll_at = EXCLUDED.next_poll_at,
                failure_count = EXCLUDED.failure_count,
                last_error = EXCLUDED.last_error,
                updated_at = NOW()
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                key.exchange_id,
                key.symbol,
                key.timeframe,
                next_poll,
                failure_count,
                error[:500] if error else None,  # Truncate long errors
            )

    # =========================================================================
    # Health Helpers
    # =========================================================================

    async def count_due_pairs(self) -> int:
        """Count pairs where next_poll_at <= NOW() or IS NULL."""
        query = """
            SELECT COUNT(*) as cnt FROM price_poll_state
            WHERE next_poll_at IS NULL OR next_poll_at <= NOW()
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query)
        return row["cnt"] if row else 0

    async def count_never_polled(self) -> int:
        """Count pairs with last_candle_ts IS NULL (never successfully polled)."""
        query = """
            SELECT COUNT(*) as cnt FROM price_poll_state
            WHERE last_candle_ts IS NULL
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query)
        return row["cnt"] if row else 0

    async def get_worst_staleness(self) -> Optional[float]:
        """
        Get worst staleness in seconds (max age of last_candle_ts).

        Returns None if no pairs have been successfully polled.
        """
        query = """
            SELECT EXTRACT(EPOCH FROM (NOW() - MIN(last_candle_ts))) as staleness
            FROM price_poll_state
            WHERE last_candle_ts IS NOT NULL
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query)
        if row and row["staleness"] is not None:
            return float(row["staleness"])
        return None

    async def count_total(self) -> int:
        """Count total state rows."""
        query = "SELECT COUNT(*) as cnt FROM price_poll_state"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query)
        return row["cnt"] if row else 0
