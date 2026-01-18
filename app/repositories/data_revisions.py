"""Repository for data revision tracking.

Data revisions provide drift detection by tracking checksums of fetched data
for specific exchange/symbol/timeframe/date-range combinations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DataRevision:
    """A data revision record tracking a fetched data range."""

    id: int
    exchange_id: str
    symbol: str
    timeframe: str
    start_ts: datetime
    end_ts: datetime
    row_count: int
    checksum: str
    computed_at: datetime
    job_id: Optional[str] = None


class DataRevisionRepository:
    """Repository for data revision operations."""

    def __init__(self, pool):
        self._pool = pool

    async def upsert(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
        row_count: int,
        checksum: str,
        job_id: Optional[str] = None,
    ) -> DataRevision:
        """Upsert a data revision record.

        If a revision already exists for the same (exchange_id, symbol, timeframe,
        start_ts, end_ts) tuple, updates the row_count, checksum, and computed_at.

        Args:
            exchange_id: Exchange identifier (e.g., 'kucoin')
            symbol: Canonical symbol (e.g., 'BTC-USDT')
            timeframe: Canonical timeframe (e.g., '1h')
            start_ts: Start timestamp of the data range
            end_ts: End timestamp of the data range
            row_count: Number of candles in the revision
            checksum: SHA256 checksum of the data (truncated)
            job_id: Optional job ID that triggered this revision

        Returns:
            The created or updated DataRevision
        """
        query = """
            INSERT INTO data_revisions
                (exchange_id, symbol, timeframe, start_ts, end_ts, row_count, checksum, job_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (exchange_id, symbol, timeframe, start_ts, end_ts) DO UPDATE SET
                row_count = EXCLUDED.row_count,
                checksum = EXCLUDED.checksum,
                computed_at = now(),
                job_id = EXCLUDED.job_id
            RETURNING *
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                exchange_id,
                symbol,
                timeframe,
                start_ts,
                end_ts,
                row_count,
                checksum,
                job_id,
            )

        logger.debug(
            "data_revision_upserted",
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            checksum=checksum,
            row_count=row_count,
        )

        return self._row_to_revision(row)

    async def get(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> Optional[DataRevision]:
        """Get a revision for a specific data range.

        Args:
            exchange_id: Exchange identifier
            symbol: Canonical symbol
            timeframe: Canonical timeframe
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            DataRevision if found, None otherwise
        """
        query = """
            SELECT * FROM data_revisions
            WHERE exchange_id = $1
              AND symbol = $2
              AND timeframe = $3
              AND start_ts = $4
              AND end_ts = $5
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query, exchange_id, symbol, timeframe, start_ts, end_ts
            )
        return self._row_to_revision(row) if row else None

    async def get_latest(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
    ) -> Optional[DataRevision]:
        """Get the most recent revision for a symbol/timeframe.

        Args:
            exchange_id: Exchange identifier
            symbol: Canonical symbol
            timeframe: Canonical timeframe

        Returns:
            Most recent DataRevision if any exists, None otherwise
        """
        query = """
            SELECT * FROM data_revisions
            WHERE exchange_id = $1 AND symbol = $2 AND timeframe = $3
            ORDER BY computed_at DESC
            LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, symbol, timeframe)
        return self._row_to_revision(row) if row else None

    async def has_changed(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
        checksum: str,
    ) -> bool:
        """Check if data has changed compared to stored revision.

        Args:
            exchange_id: Exchange identifier
            symbol: Canonical symbol
            timeframe: Canonical timeframe
            start_ts: Start timestamp
            end_ts: End timestamp
            checksum: Current checksum to compare

        Returns:
            True if revision doesn't exist or checksum differs, False if unchanged
        """
        existing = await self.get(exchange_id, symbol, timeframe, start_ts, end_ts)
        if existing is None:
            return True  # No prior revision = new data
        return existing.checksum != checksum

    async def list_for_symbol(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[DataRevision]:
        """List revisions for a symbol/timeframe, ordered by computed_at desc.

        Args:
            exchange_id: Exchange identifier
            symbol: Canonical symbol
            timeframe: Canonical timeframe
            limit: Max records to return

        Returns:
            List of DataRevision records
        """
        query = """
            SELECT * FROM data_revisions
            WHERE exchange_id = $1 AND symbol = $2 AND timeframe = $3
            ORDER BY computed_at DESC
            LIMIT $4
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, exchange_id, symbol, timeframe, limit)
        return [self._row_to_revision(row) for row in rows]

    def _row_to_revision(self, row) -> DataRevision:
        """Convert a database row to a DataRevision model."""
        return DataRevision(
            id=row["id"],
            exchange_id=row["exchange_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            start_ts=row["start_ts"],
            end_ts=row["end_ts"],
            row_count=row["row_count"],
            checksum=row["checksum"],
            computed_at=row["computed_at"],
            job_id=row["job_id"],
        )
