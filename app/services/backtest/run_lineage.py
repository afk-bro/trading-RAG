"""Run lineage detection and delta computation for backtest coaching.

Provides functions to find previous runs for the same strategy,
compute KPI deltas, parameter diffs, and trajectory data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGHER_IS_BETTER: dict[str, bool] = {
    "return_pct": True,
    "max_drawdown_pct": False,
    "sharpe": True,
    "win_rate": True,
    "profit_factor": True,
    "trades": False,
    "avg_trade_pct": True,
    "buy_hold_return_pct": True,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class KpiDelta:
    metric: str
    current: Optional[float]
    previous: Optional[float]
    delta: Optional[float]
    improved: Optional[bool]
    higher_is_better: bool


@dataclass
class LineagePreviousRun:
    run_id: str
    completed_at: str
    summary: Optional[dict[str, Any]] = None


@dataclass
class LineageCandidate:
    run_id: str
    completed_at: str
    sharpe: Optional[float] = None
    return_pct: Optional[float] = None
    is_auto_baseline: bool = False


@dataclass
class TrajectoryRun:
    run_id: str
    completed_at: str
    sharpe: Optional[float] = None
    return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate: Optional[float] = None
    trades: Optional[int] = None


@dataclass
class LineageResult:
    previous_run_id: Optional[str]
    previous_completed_at: Optional[str]
    deltas: list[KpiDelta] = field(default_factory=list)
    params_changed: bool = False
    param_diffs: dict[str, list[Any]] = field(default_factory=dict)
    comparison_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compute_deltas(
    current_summary: dict[str, Any],
    previous_summary: dict[str, Any],
) -> list[KpiDelta]:
    """Compute KPI deltas between current and previous run summaries."""
    metrics = list(HIGHER_IS_BETTER.keys())
    deltas: list[KpiDelta] = []

    for metric in metrics:
        curr = _safe_float(current_summary.get(metric))
        prev = _safe_float(previous_summary.get(metric))
        higher = HIGHER_IS_BETTER[metric]

        delta_val: Optional[float] = None
        improved: Optional[bool] = None

        if curr is not None and prev is not None:
            delta_val = round(curr - prev, 6)
            if delta_val == 0:
                improved = None  # unchanged
            elif higher:
                improved = delta_val > 0
            else:
                improved = delta_val < 0

        deltas.append(
            KpiDelta(
                metric=metric,
                current=curr,
                previous=prev,
                delta=delta_val,
                improved=improved,
                higher_is_better=higher,
            )
        )

    return deltas


def compute_param_diffs(
    current_params: dict[str, Any],
    previous_params: dict[str, Any],
) -> dict[str, list[Any]]:
    """Compute changed keys between current and previous params.

    Returns dict of {key: [old_json, new_json]} for changed keys.
    Values are JSON-serialized for complex objects.
    """
    diffs: dict[str, list[Any]] = {}
    all_keys = set(current_params.keys()) | set(previous_params.keys())

    for key in sorted(all_keys):
        curr = current_params.get(key)
        prev = previous_params.get(key)

        if curr != prev:
            diffs[key] = [
                _json_serialize(prev),
                _json_serialize(curr),
            ]

    return diffs


def _json_serialize(val: Any) -> str:
    """JSON-serialize a value for display."""
    if val is None:
        return "null"
    if isinstance(val, (dict, list)):
        return json.dumps(val, separators=(",", ":"))
    return str(val)


def compute_comparison_warnings(
    current_run: dict[str, Any],
    previous_run: dict[str, Any],
) -> list[str]:
    """Compute list of mismatch warnings between runs."""
    warnings: list[str] = []

    # Symbol check
    curr_dataset = current_run.get("dataset") or current_run.get("dataset_meta") or {}
    prev_dataset = previous_run.get("dataset") or previous_run.get("dataset_meta") or {}

    curr_symbol = curr_dataset.get("symbol", "")
    prev_symbol = prev_dataset.get("symbol", "")
    if curr_symbol and prev_symbol and curr_symbol != prev_symbol:
        warnings.append(f"Different symbol ({curr_symbol} vs {prev_symbol})")

    # Timeframe check
    curr_tf = curr_dataset.get("timeframe", "")
    prev_tf = prev_dataset.get("timeframe", "")
    if curr_tf and prev_tf and curr_tf != prev_tf:
        warnings.append(f"Different timeframe ({curr_tf} vs {prev_tf})")

    # Date range overlap check
    curr_min = str(curr_dataset.get("date_min", ""))[:10]
    curr_max = str(curr_dataset.get("date_max", ""))[:10]
    prev_min = str(prev_dataset.get("date_min", ""))[:10]
    prev_max = str(prev_dataset.get("date_max", ""))[:10]

    if curr_min and curr_max and prev_min and prev_max:
        if curr_max < prev_min or prev_max < curr_min:
            warnings.append("Non-overlapping date ranges")

    # Run kind check
    curr_kind = current_run.get("run_kind", "")
    prev_kind = previous_run.get("run_kind", "")
    if curr_kind and prev_kind and curr_kind != prev_kind:
        warnings.append(f"Different run kind ({curr_kind} vs {prev_kind})")

    return warnings


# ---------------------------------------------------------------------------
# Async DB functions
# ---------------------------------------------------------------------------


async def find_previous_run(
    pool: Any,
    workspace_id: UUID,
    strategy_entity_id: UUID,
    current_run_id: UUID,
    current_completed_at: Any,
) -> Optional[LineagePreviousRun]:
    """Find the most recent completed run before the current one."""
    query = """
        SELECT id, completed_at, summary
        FROM backtest_runs
        WHERE workspace_id = $1
          AND strategy_entity_id = $2
          AND id != $3
          AND status = 'completed'
          AND completed_at < $4
        ORDER BY completed_at DESC
        LIMIT 1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query,
            workspace_id,
            strategy_entity_id,
            current_run_id,
            current_completed_at,
        )

    if not row:
        return None

    summary = row["summary"]
    if isinstance(summary, str):
        import json as _json

        summary = _json.loads(summary)

    return LineagePreviousRun(
        run_id=str(row["id"]),
        completed_at=row["completed_at"].isoformat() if row["completed_at"] else "",
        summary=summary,
    )


async def find_lineage_candidates(
    pool: Any,
    workspace_id: UUID,
    strategy_entity_id: UUID,
    current_run_id: UUID,
    limit: int = 10,
) -> list[LineageCandidate]:
    """Find recent completed runs for the same strategy."""
    query = """
        SELECT id, completed_at, summary
        FROM backtest_runs
        WHERE workspace_id = $1
          AND strategy_entity_id = $2
          AND id != $3
          AND status = 'completed'
        ORDER BY completed_at DESC
        LIMIT $4
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            query,
            workspace_id,
            strategy_entity_id,
            current_run_id,
            limit,
        )

    candidates: list[LineageCandidate] = []
    for i, row in enumerate(rows):
        summary = row["summary"]
        if isinstance(summary, str):
            import json as _json

            summary = _json.loads(summary)

        summary = summary or {}
        candidates.append(
            LineageCandidate(
                run_id=str(row["id"]),
                completed_at=(
                    row["completed_at"].isoformat() if row["completed_at"] else ""
                ),
                sharpe=_safe_float(summary.get("sharpe")),
                return_pct=_safe_float(summary.get("return_pct")),
                is_auto_baseline=(i == 0),  # first = most recent = auto
            )
        )

    return candidates


async def build_trajectory(
    pool: Any,
    workspace_id: UUID,
    strategy_entity_id: UUID,
    limit: int = 10,
) -> list[TrajectoryRun]:
    """Fetch last N completed run summaries for sparklines."""
    query = """
        SELECT id, completed_at, summary
        FROM backtest_runs
        WHERE workspace_id = $1
          AND strategy_entity_id = $2
          AND status = 'completed'
        ORDER BY completed_at DESC
        LIMIT $1
    """
    # Fix: use positional params properly
    query = """
        SELECT id, completed_at, summary
        FROM backtest_runs
        WHERE workspace_id = $1
          AND strategy_entity_id = $2
          AND status = 'completed'
        ORDER BY completed_at DESC
        LIMIT $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            query,
            workspace_id,
            strategy_entity_id,
            limit,
        )

    runs: list[TrajectoryRun] = []
    for row in reversed(rows):  # chronological order (oldest first)
        summary = row["summary"]
        if isinstance(summary, str):
            import json as _json

            summary = _json.loads(summary)

        summary = summary or {}
        runs.append(
            TrajectoryRun(
                run_id=str(row["id"]),
                completed_at=(
                    row["completed_at"].isoformat() if row["completed_at"] else ""
                ),
                sharpe=_safe_float(summary.get("sharpe")),
                return_pct=_safe_float(summary.get("return_pct")),
                max_drawdown_pct=_safe_float(summary.get("max_drawdown_pct")),
                win_rate=_safe_float(summary.get("win_rate")),
                trades=(
                    int(summary["trades"])
                    if summary.get("trades") is not None
                    else None
                ),
            )
        )

    return runs


async def find_run_by_id(
    pool: Any,
    run_id: UUID,
    workspace_id: UUID,
) -> Optional[dict[str, Any]]:
    """Fetch a single run's summary, params, dataset_meta, and run_kind."""
    query = """
        SELECT id, completed_at, summary, params, dataset_meta, run_kind
        FROM backtest_runs
        WHERE id = $1 AND workspace_id = $2
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, run_id, workspace_id)

    if not row:
        return None

    def _parse(raw: Any) -> Any:
        if isinstance(raw, str):
            import json as _json

            return _json.loads(raw)
        return raw

    return {
        "run_id": str(row["id"]),
        "completed_at": (
            row["completed_at"].isoformat() if row["completed_at"] else None
        ),
        "summary": _parse(row["summary"]) or {},
        "params": _parse(row["params"]) or {},
        "dataset": _parse(row["dataset_meta"]) or {},
        "run_kind": row["run_kind"],
    }
