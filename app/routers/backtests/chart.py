"""Backtest chart data endpoints."""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["backtests"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


# =============================================================================
# Pydantic Models
# =============================================================================


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    t: str = Field(..., description="ISO timestamp with Z suffix")
    equity: float = Field(..., description="Equity value")


class TradeRecord(BaseModel):
    """Single trade record for chart display."""

    t_entry: str = Field(..., description="Entry timestamp")
    t_exit: str = Field(..., description="Exit timestamp")
    side: str = Field(..., description="Trade direction: long or short")
    size: Optional[float] = Field(None, description="Position size")
    entry_price: Optional[float] = Field(None, description="Entry price")
    exit_price: Optional[float] = Field(None, description="Exit price")
    pnl: float = Field(..., description="Profit/loss in currency")
    return_pct: float = Field(..., description="Return percentage")


class ChartSummary(BaseModel):
    """Summary metrics for chart display."""

    return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    sharpe: Optional[float] = None
    trades: Optional[int] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_pct: Optional[float] = None
    buy_hold_return_pct: Optional[float] = None


class DatasetMeta(BaseModel):
    """Dataset metadata for chart header."""

    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    row_count: Optional[int] = None


class TradesPagination(BaseModel):
    """Pagination info for trades."""

    page: int
    page_size: int
    total: int


class ExportLinks(BaseModel):
    """Export endpoint URLs."""

    trades_csv: Optional[str] = None
    json_snapshot: str


class RegimeInfo(BaseModel):
    """Regime information for chart overlay."""

    trend_tag: Optional[str] = Field(None, description="uptrend, downtrend, flat, etc.")
    vol_tag: Optional[str] = Field(None, description="high_vol, low_vol")
    efficiency_tag: Optional[str] = Field(None, description="efficient, noisy")
    ts_start: Optional[str] = Field(None, description="Window start timestamp")
    ts_end: Optional[str] = Field(None, description="Window end timestamp")


class BacktestChartData(BaseModel):
    """Full chart data response - powers the entire visualization page."""

    run_id: str
    status: str
    dataset_meta: Optional[DatasetMeta] = None
    params: dict[str, Any]
    summary: ChartSummary
    equity: list[EquityPoint]
    equity_source: Literal["jsonb", "csv", "missing"]
    trades_page: list[TradeRecord]
    trades_pagination: TradesPagination
    exports: ExportLinks
    notes: list[str] = Field(default_factory=list)
    # Regime data (for overlay)
    regime_is: Optional[RegimeInfo] = Field(None, description="In-sample regime")
    regime_oos: Optional[RegimeInfo] = Field(None, description="Out-of-sample regime")


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_timestamp(ts: Any) -> str:
    """Normalize timestamp to ISO format with Z suffix."""
    if ts is None:
        return ""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat().replace("+00:00", "Z")
    # String timestamp
    s = str(ts)
    if not s.endswith("Z") and "+" not in s:
        s = s + "Z"
    return s


EquitySource = Literal["jsonb", "csv", "missing"]


def _parse_equity_curve(raw: Any) -> tuple[list[EquityPoint], EquitySource]:
    """Parse equity curve from JSONB, return (points, source)."""
    if not raw:
        return [], "missing"

    # Handle string JSON (JSONB sometimes returned as string)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [], "missing"

    if isinstance(raw, list):
        points = []
        for p in raw:
            if isinstance(p, dict):
                t = p.get("t") or p.get("timestamp") or p.get("time")
                equity = p.get("equity") or p.get("value") or p.get("Equity")
                if t is not None and equity is not None:
                    points.append(
                        EquityPoint(t=_normalize_timestamp(t), equity=float(equity))
                    )
        return points, "jsonb" if points else "missing"

    return [], "missing"


def _parse_trades(raw: Any, page: int, page_size: int) -> tuple[list[TradeRecord], int]:
    """Parse trades from JSONB with pagination, return (trades, total)."""
    if not raw:
        return [], 0

    # Handle string JSON (JSONB sometimes returned as string)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [], 0

    if not isinstance(raw, list):
        return [], 0

    total = len(raw)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = raw[start:end]

    trades = []
    for t in page_items:
        if isinstance(t, dict):
            # Handle various field name conventions
            t_entry = (
                t.get("t_entry")
                or t.get("entry_time")
                or t.get("EntryTime")
                or t.get("entry_bar")
            )
            t_exit = (
                t.get("t_exit")
                or t.get("exit_time")
                or t.get("ExitTime")
                or t.get("exit_bar")
            )
            side = t.get("side") or t.get("Size", "long")
            if isinstance(side, (int, float)):
                side = "long" if side > 0 else "short"

            trades.append(
                TradeRecord(
                    t_entry=_normalize_timestamp(t_entry),
                    t_exit=_normalize_timestamp(t_exit),
                    side=str(side).lower(),
                    size=t.get("size") or t.get("Size"),
                    entry_price=t.get("entry_price") or t.get("EntryPrice"),
                    exit_price=t.get("exit_price") or t.get("ExitPrice"),
                    pnl=float(t.get("pnl") or t.get("PnL") or 0),
                    return_pct=float(t.get("return_pct") or t.get("ReturnPct") or 0),
                )
            )

    return trades, total


def _parse_summary(raw: Any) -> ChartSummary:
    """Parse summary from JSONB."""
    # Handle string JSON
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ChartSummary()

    if not raw or not isinstance(raw, dict):
        return ChartSummary()

    return ChartSummary(
        return_pct=raw.get("return_pct"),
        max_drawdown_pct=raw.get("max_drawdown_pct"),
        sharpe=raw.get("sharpe"),
        trades=raw.get("trades"),
        win_rate=raw.get("win_rate"),
        profit_factor=raw.get("profit_factor"),
        avg_trade_pct=raw.get("avg_trade_pct"),
        buy_hold_return_pct=raw.get("buy_hold_return_pct"),
    )


def _parse_dataset_meta(raw: Any) -> Optional[DatasetMeta]:
    """Parse dataset metadata from JSONB."""
    # Handle string JSON
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None

    if not raw or not isinstance(raw, dict):
        return None

    return DatasetMeta(
        symbol=raw.get("symbol"),
        timeframe=raw.get("timeframe"),
        date_min=_normalize_timestamp(raw.get("date_min")),
        date_max=_normalize_timestamp(raw.get("date_max")),
        row_count=raw.get("row_count"),
    )


def _parse_jsonb_field(raw: Any) -> dict:
    """Parse a JSONB field that might be stored as string or dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_regime_info(raw: Any) -> Optional[RegimeInfo]:
    """Parse regime snapshot from JSONB into RegimeInfo."""
    data = _parse_jsonb_field(raw)
    if not data:
        return None

    # Extract tags from regime snapshot
    tags = data.get("regime_tags") or data.get("tags") or []
    trend_tag = None
    vol_tag = None
    efficiency_tag = None

    for tag in tags:
        tag_lower = tag.lower() if isinstance(tag, str) else ""
        if tag_lower in ("uptrend", "downtrend", "flat", "ranging", "trending"):
            trend_tag = tag_lower
        elif tag_lower in ("high_vol", "low_vol"):
            vol_tag = tag_lower
        elif tag_lower in ("efficient", "noisy"):
            efficiency_tag = tag_lower

    # Also check denormalized fields
    if not trend_tag:
        trend_tag = data.get("trend_tag")
    if not vol_tag:
        vol_tag = data.get("vol_tag")
    if not efficiency_tag:
        efficiency_tag = data.get("efficiency_tag")

    # If no tags found, try to infer from numeric values
    td = data.get("trend_dir")
    if not trend_tag and td is not None:
        if td > 0:
            trend_tag = "uptrend"
        elif td < 0:
            trend_tag = "downtrend"
        else:
            trend_tag = "flat"

    if not any([trend_tag, vol_tag, efficiency_tag]):
        return None

    return RegimeInfo(
        trend_tag=trend_tag,
        vol_tag=vol_tag,
        efficiency_tag=efficiency_tag,
        ts_start=_normalize_timestamp(data.get("ts_start")),
        ts_end=_normalize_timestamp(data.get("ts_end")),
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/runs/{run_id}/chart-data",
    response_model=BacktestChartData,
    responses={
        200: {"description": "Chart data retrieved"},
        404: {"description": "Run not found"},
    },
    summary="Get backtest chart data",
    description="Returns chart-ready data for visualization: equity curve, trades, summary.",
)
async def get_chart_data(
    run_id: UUID,
    page: int = Query(1, ge=1, description="Trades page number"),
    page_size: int = Query(50, ge=10, le=100, description="Trades per page"),
):
    """Get chart data for a backtest run."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, status, params, summary, dataset_meta,
                equity_curve, trades, run_kind,
                regime_is, regime_oos
            FROM backtest_runs
            WHERE id = $1
            """,
            run_id,
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    notes: list[str] = []
    run_kind = row.get("run_kind")

    # Parse equity curve
    equity, equity_source = _parse_equity_curve(row.get("equity_curve"))

    # Check for tune variant with no equity
    if not equity and run_kind == "tune_variant":
        notes.append("Equity curve not stored for tune variants")
        equity_source = "missing"
    elif not equity:
        notes.append("Equity curve not available for this run")

    # Log warning for huge series
    if len(equity) > 10000:
        logger.warning(
            "Large equity curve returned",
            run_id=str(run_id),
            point_count=len(equity),
        )

    # Parse trades with pagination
    trades_page, trades_total = _parse_trades(
        row.get("trades"), page=page, page_size=page_size
    )

    if trades_total == 0 and row.get("trades") is None:
        notes.append("Trades not stored for this run")

    # Build export links
    base_url = f"/backtests/runs/{run_id}/export"
    exports = ExportLinks(
        trades_csv=f"{base_url}/trades.csv" if trades_total > 0 else None,
        json_snapshot=f"{base_url}/snapshot.json",
    )

    # Parse regime data
    regime_is = _parse_regime_info(row.get("regime_is"))
    regime_oos = _parse_regime_info(row.get("regime_oos"))

    return BacktestChartData(
        run_id=str(run_id),
        status=row["status"],
        dataset_meta=_parse_dataset_meta(row.get("dataset_meta")),
        params=_parse_jsonb_field(row.get("params")),
        summary=_parse_summary(row.get("summary")),
        equity=equity,
        equity_source=equity_source,
        trades_page=trades_page,
        trades_pagination=TradesPagination(
            page=page,
            page_size=page_size,
            total=trades_total,
        ),
        exports=exports,
        notes=notes,
        regime_is=regime_is,
        regime_oos=regime_oos,
    )


@router.get(
    "/runs/{run_id}/export/trades.csv",
    responses={
        200: {"description": "CSV file", "content": {"text/csv": {}}},
        404: {"description": "Run not found or no trades"},
    },
    summary="Export trades as CSV",
)
async def export_trades_csv(run_id: UUID):
    """Export all trades as CSV file."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT trades FROM backtest_runs WHERE id = $1",
            run_id,
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    trades_raw = row.get("trades")

    # Handle string JSON (JSONB sometimes returned as string)
    if trades_raw and isinstance(trades_raw, str):
        try:
            trades_raw = json.loads(trades_raw)
        except json.JSONDecodeError:
            trades_raw = None

    if not trades_raw or not isinstance(trades_raw, list) or len(trades_raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trades stored for this run",
        )

    # Build CSV
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "entry_time",
        "exit_time",
        "side",
        "size",
        "entry_price",
        "exit_price",
        "pnl",
        "return_pct",
    ]
    writer.writerow(headers)

    for t in trades_raw:
        if isinstance(t, dict):
            t_entry = (
                t.get("t_entry")
                or t.get("entry_time")
                or t.get("EntryTime")
                or t.get("entry_bar")
                or ""
            )
            t_exit = (
                t.get("t_exit")
                or t.get("exit_time")
                or t.get("ExitTime")
                or t.get("exit_bar")
                or ""
            )
            side = t.get("side") or t.get("Size", "long")
            if isinstance(side, (int, float)):
                side = "long" if side > 0 else "short"

            writer.writerow(
                [
                    _normalize_timestamp(t_entry),
                    _normalize_timestamp(t_exit),
                    str(side).lower(),
                    t.get("size") or t.get("Size") or "",
                    t.get("entry_price") or t.get("EntryPrice") or "",
                    t.get("exit_price") or t.get("ExitPrice") or "",
                    t.get("pnl") or t.get("PnL") or 0,
                    t.get("return_pct") or t.get("ReturnPct") or 0,
                ]
            )

    output.seek(0)
    filename = f"trades_{str(run_id)[:8]}_{datetime.now().strftime('%Y%m%d')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/runs/{run_id}/export/snapshot.json",
    responses={
        200: {"description": "JSON snapshot"},
        404: {"description": "Run not found"},
    },
    summary="Export full JSON snapshot",
)
async def export_json_snapshot(run_id: UUID):
    """Export full run data as JSON snapshot (all trades, no pagination)."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, status, params, summary, dataset_meta,
                equity_curve, trades, run_kind, created_at
            FROM backtest_runs
            WHERE id = $1
            """,
            run_id,
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    # Parse equity
    equity, equity_source = _parse_equity_curve(row.get("equity_curve"))

    # Parse all trades (no pagination for export)
    all_trades = []
    trades_raw = row.get("trades")

    # Handle string JSON (JSONB sometimes returned as string)
    if trades_raw and isinstance(trades_raw, str):
        try:
            trades_raw = json.loads(trades_raw)
        except json.JSONDecodeError:
            trades_raw = None

    if trades_raw and isinstance(trades_raw, list):
        for t in trades_raw:
            if isinstance(t, dict):
                t_entry = (
                    t.get("t_entry")
                    or t.get("entry_time")
                    or t.get("EntryTime")
                    or t.get("entry_bar")
                )
                t_exit = (
                    t.get("t_exit")
                    or t.get("exit_time")
                    or t.get("ExitTime")
                    or t.get("exit_bar")
                )
                side = t.get("side") or t.get("Size", "long")
                if isinstance(side, (int, float)):
                    side = "long" if side > 0 else "short"

                all_trades.append(
                    {
                        "t_entry": _normalize_timestamp(t_entry),
                        "t_exit": _normalize_timestamp(t_exit),
                        "side": str(side).lower(),
                        "size": t.get("size") or t.get("Size"),
                        "entry_price": t.get("entry_price") or t.get("EntryPrice"),
                        "exit_price": t.get("exit_price") or t.get("ExitPrice"),
                        "pnl": t.get("pnl") or t.get("PnL") or 0,
                        "return_pct": t.get("return_pct") or t.get("ReturnPct") or 0,
                    }
                )

    snapshot = {
        "run_id": str(run_id),
        "status": row["status"],
        "created_at": _normalize_timestamp(row.get("created_at")),
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_meta": _parse_jsonb_field(row.get("dataset_meta")),
        "params": _parse_jsonb_field(row.get("params")),
        "summary": _parse_jsonb_field(row.get("summary")),
        "equity_source": equity_source,
        "equity": [{"t": p.t, "equity": p.equity} for p in equity],
        "trades": all_trades,
    }

    return snapshot


# =============================================================================
# Sparkline Endpoint
# =============================================================================

SPARKLINE_MAX_POINTS = 96


class SparklineData(BaseModel):
    """Sparkline data for table display."""

    y: list[float] = Field(..., description="Equity values (downsampled)")
    status: str = Field("ok", description="ok, empty, or error")


def _downsample_equity(points: list[EquityPoint], max_points: int) -> list[float]:
    """
    Downsample equity curve to max_points using even spacing.

    Args:
        points: Full equity curve
        max_points: Maximum number of points to return

    Returns:
        List of equity values (y-axis only)
    """
    n = len(points)
    if n == 0:
        return []
    if n <= max_points:
        return [p.equity for p in points]

    # Even spacing: pick indices at regular intervals
    indices = [int(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    return [points[i].equity for i in indices]


@router.get(
    "/runs/{run_id}/sparkline",
    response_model=SparklineData,
    responses={
        200: {"description": "Sparkline data"},
        404: {"description": "Run not found"},
    },
    summary="Get sparkline data for table display",
    description=(
        f"Returns downsampled equity curve (max {SPARKLINE_MAX_POINTS} points) "
        "for sparkline rendering."
    ),
)
async def get_sparkline(
    run_id: UUID,
    max_points: int = Query(
        SPARKLINE_MAX_POINTS, ge=10, le=200, description="Maximum points to return"
    ),
):
    """Get sparkline data for a backtest run."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT equity_curve, status
            FROM backtest_runs
            WHERE id = $1
            """,
            run_id,
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    # Don't return sparkline for non-completed runs
    if row["status"] != "completed":
        return SparklineData(y=[], status="pending")

    # Parse and downsample
    equity, source = _parse_equity_curve(row.get("equity_curve"))

    if not equity:
        return SparklineData(y=[], status="empty")

    downsampled = _downsample_equity(equity, max_points)

    return SparklineData(y=downsampled, status="ok")
