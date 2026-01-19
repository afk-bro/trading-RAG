"""Admin endpoints for data management."""

from typing import Literal, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.repositories.core_symbols import CoreSymbol, CoreSymbolsRepository

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/data", tags=["admin-data"])

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_core_symbols_repo() -> CoreSymbolsRepository:
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return CoreSymbolsRepository(_db_pool)


# Request/Response Models
class CoreSymbolResponse(BaseModel):
    """Response model for a core symbol."""

    exchange_id: str
    canonical_symbol: str
    raw_symbol: str
    timeframes: list[str]
    is_enabled: bool
    added_at: Optional[str] = None
    added_by: Optional[str] = None


class AddCoreSymbolsRequest(BaseModel):
    """Request to add symbols to core universe."""

    exchange_id: str = Field(..., description="Exchange ID (e.g., 'kucoin')")
    symbols: list[str] = Field(
        ..., min_length=1, max_length=50, description="Canonical symbols to add"
    )
    timeframes: Optional[list[str]] = Field(
        default=None, description="Timeframes to sync (defaults to all)"
    )
    added_by: Optional[str] = Field(default=None, description="Who added these symbols")


class ModifyCoreSymbolsRequest(BaseModel):
    """Request to modify core symbols."""

    action: Literal["remove", "enable", "disable"]
    exchange_id: str
    symbols: list[str] = Field(..., min_length=1, max_length=50)


class CoreSymbolsActionResponse(BaseModel):
    """Response from core symbols actions."""

    action: str
    exchange_id: str
    affected: int
    symbols: list[str]


# Endpoints
@router.get(
    "/core-symbols",
    response_model=list[CoreSymbolResponse],
    dependencies=[Depends(require_admin_token)],
)
async def list_core_symbols(
    exchange_id: Optional[str] = Query(None, description="Filter by exchange"),
    include_disabled: bool = Query(False, description="Include disabled symbols"),
):
    """List all core symbols."""
    repo = _get_core_symbols_repo()
    symbols = await repo.list_symbols(
        exchange_id=exchange_id, enabled_only=not include_disabled
    )
    return [
        CoreSymbolResponse(
            exchange_id=s.exchange_id,
            canonical_symbol=s.canonical_symbol,
            raw_symbol=s.raw_symbol,
            timeframes=s.timeframes,
            is_enabled=s.is_enabled,
            added_at=s.added_at.isoformat() if s.added_at else None,
            added_by=s.added_by,
        )
        for s in symbols
    ]


@router.post(
    "/core-symbols",
    response_model=CoreSymbolsActionResponse,
    dependencies=[Depends(require_admin_token)],
)
async def add_core_symbols(request: AddCoreSymbolsRequest):
    """Add symbols to the core universe."""
    repo = _get_core_symbols_repo()
    added = []
    for sym in request.symbols:
        cs = CoreSymbol(
            exchange_id=request.exchange_id,
            canonical_symbol=sym,
            raw_symbol=sym,
            timeframes=request.timeframes or ["1m", "5m", "15m", "1h", "1d"],
            is_enabled=True,
            added_by=request.added_by,
        )
        if await repo.add_symbol(cs):
            added.append(sym)
    logger.info(
        "core_symbols_added",
        exchange_id=request.exchange_id,
        added_count=len(added),
        requested_count=len(request.symbols),
    )
    return CoreSymbolsActionResponse(
        action="add",
        exchange_id=request.exchange_id,
        affected=len(added),
        symbols=added,
    )


@router.patch(
    "/core-symbols",
    response_model=CoreSymbolsActionResponse,
    dependencies=[Depends(require_admin_token)],
)
async def modify_core_symbols(request: ModifyCoreSymbolsRequest):
    """Modify (remove/enable/disable) core symbols."""
    repo = _get_core_symbols_repo()
    affected = []
    for sym in request.symbols:
        if request.action == "remove":
            if await repo.remove_symbol(request.exchange_id, sym):
                affected.append(sym)
        elif request.action == "enable":
            if await repo.set_enabled(request.exchange_id, sym, True):
                affected.append(sym)
        elif request.action == "disable":
            if await repo.set_enabled(request.exchange_id, sym, False):
                affected.append(sym)
    logger.info(
        "core_symbols_modified",
        action=request.action,
        exchange_id=request.exchange_id,
        affected_count=len(affected),
    )
    return CoreSymbolsActionResponse(
        action=request.action,
        exchange_id=request.exchange_id,
        affected=len(affected),
        symbols=affected,
    )
