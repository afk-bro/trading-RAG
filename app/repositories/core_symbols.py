"""Repository for core symbols management."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CoreSymbol:
    """A symbol in the core universe."""

    exchange_id: str
    canonical_symbol: str
    raw_symbol: str
    timeframes: list[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"]
    )
    is_enabled: bool = True
    added_at: Optional[datetime] = None
    added_by: Optional[str] = None


class CoreSymbolsRepository:
    """Repository for core symbols operations."""

    def __init__(self, pool):
        self._pool = pool

    async def add_symbol(self, symbol: CoreSymbol) -> bool:
        """Add a symbol to the core universe. Returns True if inserted."""
        query = """
            INSERT INTO core_symbols
                (exchange_id, canonical_symbol, raw_symbol, timeframes, is_enabled, added_by)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (exchange_id, canonical_symbol) DO NOTHING
            RETURNING canonical_symbol
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                symbol.exchange_id,
                symbol.canonical_symbol,
                symbol.raw_symbol,
                symbol.timeframes,
                symbol.is_enabled,
                symbol.added_by,
            )
        return row is not None

    async def remove_symbol(self, exchange_id: str, canonical_symbol: str) -> bool:
        """Remove a symbol from core universe. Returns True if deleted."""
        query = """
            DELETE FROM core_symbols
            WHERE exchange_id = $1 AND canonical_symbol = $2
            RETURNING canonical_symbol
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, canonical_symbol)
        return row is not None

    async def set_enabled(
        self, exchange_id: str, canonical_symbol: str, enabled: bool
    ) -> bool:
        """Enable or disable a symbol. Returns True if updated."""
        query = """
            UPDATE core_symbols SET is_enabled = $3
            WHERE exchange_id = $1 AND canonical_symbol = $2
            RETURNING canonical_symbol
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, canonical_symbol, enabled)
        return row is not None

    async def list_symbols(
        self, exchange_id: Optional[str] = None, enabled_only: bool = True
    ) -> list[CoreSymbol]:
        """List core symbols, optionally filtered by exchange."""
        conditions = []
        params = []
        if exchange_id:
            params.append(exchange_id)
            conditions.append(f"exchange_id = ${len(params)}")
        if enabled_only:
            conditions.append("is_enabled = true")
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT exchange_id, canonical_symbol, raw_symbol, timeframes,
                   is_enabled, added_at, added_by
            FROM core_symbols
            {where_clause}
            ORDER BY exchange_id, canonical_symbol
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [
            CoreSymbol(
                exchange_id=row["exchange_id"],
                canonical_symbol=row["canonical_symbol"],
                raw_symbol=row["raw_symbol"],
                timeframes=row["timeframes"],
                is_enabled=row["is_enabled"],
                added_at=row["added_at"],
                added_by=row["added_by"],
            )
            for row in rows
        ]

    async def get_symbol(
        self, exchange_id: str, canonical_symbol: str
    ) -> Optional[CoreSymbol]:
        """Get a single core symbol."""
        query = """
            SELECT exchange_id, canonical_symbol, raw_symbol, timeframes,
                   is_enabled, added_at, added_by
            FROM core_symbols
            WHERE exchange_id = $1 AND canonical_symbol = $2
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, canonical_symbol)
        if not row:
            return None
        return CoreSymbol(
            exchange_id=row["exchange_id"],
            canonical_symbol=row["canonical_symbol"],
            raw_symbol=row["raw_symbol"],
            timeframes=row["timeframes"],
            is_enabled=row["is_enabled"],
            added_at=row["added_at"],
            added_by=row["added_by"],
        )
