"""Loss attribution analysis for backtest coaching.

Pure-function module analyzing losing trades to provide actionable
insights and policy counterfactuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPUTE_CEILING = 50_000

SIZE_BUCKET_LABELS = ["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LossCluster:
    label: str
    trade_count: int
    total_loss: float
    pct_of_total_losses: float


@dataclass
class RegimeSummary:
    regime_tags: list[str]
    loss_count: int
    context: str


@dataclass
class Counterfactual:
    description: str
    metric_name: str
    actual: float
    hypothetical: float
    delta: float


@dataclass
class LossAttributionResult:
    time_clusters: list[LossCluster] = field(default_factory=list)
    size_clusters: list[LossCluster] = field(default_factory=list)
    regime_summary: Optional[RegimeSummary] = None
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    total_losses: int = 0
    total_loss_amount: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
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


def _parse_hour(t_entry: Any) -> Optional[int]:
    """Extract hour-of-day from a timestamp string."""
    if not t_entry:
        return None
    s = str(t_entry)
    # Try common formats: "2024-01-15T09:30:00", "2024-01-15 09:30:00"
    for sep in ("T", " "):
        if sep in s:
            time_part = s.split(sep)[1] if len(s.split(sep)) > 1 else None
            if time_part and ":" in time_part:
                try:
                    return int(time_part.split(":")[0])
                except (ValueError, IndexError):
                    pass
    return None


def _compute_total_return(trades: list[dict[str, Any]]) -> float:
    """Compute compounded return from trade return_pct values."""
    equity = 1.0
    for t in trades:
        ret = _safe_float(t.get("return_pct"))
        if ret is not None:
            equity *= 1.0 + ret
    return (equity - 1.0) * 100.0  # as percentage


def _compute_max_drawdown(trades: list[dict[str, Any]]) -> float:
    """Compute max drawdown percentage from trade sequence."""
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        ret = _safe_float(t.get("return_pct"))
        if ret is not None:
            equity *= 1.0 + ret
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd * 100.0  # as percentage


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------


def _compute_time_clusters(
    losing_trades: list[dict[str, Any]],
    total_loss_amount: float,
) -> list[LossCluster]:
    """Bucket losing trades by hour-of-day."""
    hour_buckets: dict[int, list[float]] = {}

    for t in losing_trades:
        hour = _parse_hour(t.get("t_entry"))
        if hour is None:
            continue
        pnl = _safe_float(t.get("pnl")) or 0.0
        hour_buckets.setdefault(hour, []).append(pnl)

    if not hour_buckets:
        return []

    abs_total = abs(total_loss_amount) if total_loss_amount != 0 else 1.0

    clusters = []
    for hour in sorted(hour_buckets):
        losses = hour_buckets[hour]
        total_loss = sum(losses)
        clusters.append(
            LossCluster(
                label=f"{hour:02d}:00",
                trade_count=len(losses),
                total_loss=round(total_loss, 2),
                pct_of_total_losses=round(abs(total_loss) / abs_total * 100, 1),
            )
        )

    return clusters


def _compute_size_clusters(
    losing_trades: list[dict[str, Any]],
    total_loss_amount: float,
) -> list[LossCluster]:
    """Bucket trades by |pnl| into small/medium/large."""
    if not losing_trades:
        return []

    abs_losses = []
    for t in losing_trades:
        pnl = _safe_float(t.get("pnl"))
        if pnl is not None:
            abs_losses.append(abs(pnl))

    if not abs_losses:
        return []

    abs_losses.sort()
    n = len(abs_losses)

    # Tercile boundaries
    t1 = abs_losses[n // 3] if n >= 3 else abs_losses[0]
    t2 = abs_losses[2 * n // 3] if n >= 3 else abs_losses[-1]

    buckets: dict[str, list[float]] = {"small": [], "medium": [], "large": []}

    for t in losing_trades:
        pnl = _safe_float(t.get("pnl"))
        if pnl is None:
            continue
        abs_pnl = abs(pnl)
        if abs_pnl <= t1:
            buckets["small"].append(pnl)
        elif abs_pnl <= t2:
            buckets["medium"].append(pnl)
        else:
            buckets["large"].append(pnl)

    abs_total = abs(total_loss_amount) if total_loss_amount != 0 else 1.0

    clusters = []
    for label in SIZE_BUCKET_LABELS:
        losses = buckets[label]
        if not losses:
            continue
        total_loss = sum(losses)
        clusters.append(
            LossCluster(
                label=label,
                trade_count=len(losses),
                total_loss=round(total_loss, 2),
                pct_of_total_losses=round(abs(total_loss) / abs_total * 100, 1),
            )
        )

    return clusters


def _compute_regime_summary(
    losing_trades: list[dict[str, Any]],
    regime_is: Optional[dict[str, Any]],
    regime_oos: Optional[dict[str, Any]],
) -> Optional[RegimeSummary]:
    """Associate regime tags with aggregate loss stats."""
    tags: list[str] = []
    for regime in (regime_is, regime_oos):
        if regime and isinstance(regime, dict):
            for key in ("trend_tag", "vol_tag", "efficiency_tag"):
                val = regime.get(key)
                if isinstance(val, str) and val:
                    tags.append(val)

    if not tags:
        return None

    tag_str = ", ".join(tags)
    return RegimeSummary(
        regime_tags=tags,
        loss_count=len(losing_trades),
        context=f"All trades ran during [{tag_str}] regime conditions",
    )


# ---------------------------------------------------------------------------
# Counterfactual computations
# ---------------------------------------------------------------------------


def _counterfactual_skip_hour(
    all_trades: list[dict[str, Any]],
    losing_trades: list[dict[str, Any]],
    actual_return: float,
) -> list[Counterfactual]:
    """If you skipped trades during hour X, return would be Y%."""
    # Find the worst hour
    hour_losses: dict[int, float] = {}
    for t in losing_trades:
        hour = _parse_hour(t.get("t_entry"))
        if hour is None:
            continue
        pnl = _safe_float(t.get("pnl")) or 0.0
        hour_losses[hour] = hour_losses.get(hour, 0.0) + pnl

    if not hour_losses:
        return []

    worst_hour = min(hour_losses, key=lambda h: hour_losses[h])
    worst_loss = hour_losses[worst_hour]

    if worst_loss >= 0:
        return []

    # Recompute equity without trades during that hour
    filtered = [t for t in all_trades if _parse_hour(t.get("t_entry")) != worst_hour]
    hyp_return = _compute_total_return(filtered)

    return [
        Counterfactual(
            description=(
                f"If you skipped trades during hour {worst_hour:02d}:00, "
                f"return would be {hyp_return:.2f}%"
            ),
            metric_name="return_pct",
            actual=round(actual_return, 4),
            hypothetical=round(hyp_return, 4),
            delta=round(hyp_return - actual_return, 4),
        )
    ]


def _counterfactual_regime_filter(
    all_trades: list[dict[str, Any]],
    regime_is: Optional[dict[str, Any]],
    regime_oos: Optional[dict[str, Any]],
    actual_dd: float,
) -> list[Counterfactual]:
    """If you enforced regime filter [tag], drawdown would be X%."""
    # Find regime tags
    tags: list[str] = []
    for regime in (regime_is, regime_oos):
        if regime and isinstance(regime, dict):
            trend = regime.get("trend_tag")
            if isinstance(trend, str) and trend:
                tags.append(trend.lower())

    if not tags:
        return []

    preferred_side = None
    for tag in tags:
        from app.services.backtest.process_score import REGIME_PREFERRED_SIDE

        if tag in REGIME_PREFERRED_SIDE:
            preferred_side = REGIME_PREFERRED_SIDE[tag]
            break

    if preferred_side is None:
        return []

    filtered = [
        t for t in all_trades if str(t.get("side", "")).lower() == preferred_side
    ]

    if not filtered or len(filtered) == len(all_trades):
        return []

    hyp_dd = _compute_max_drawdown(filtered)

    tag_str = ", ".join(tags)
    return [
        Counterfactual(
            description=(
                f"If you enforced regime filter [{tag_str}], "
                f"drawdown would be {hyp_dd:.2f}%"
            ),
            metric_name="max_drawdown_pct",
            actual=round(actual_dd, 4),
            hypothetical=round(hyp_dd, 4),
            delta=round(hyp_dd - actual_dd, 4),
        )
    ]


def _counterfactual_max_trades_per_day(
    all_trades: list[dict[str, Any]],
    actual_return: float,
) -> list[Counterfactual]:
    """If you capped max trades/day at K, return would be Y%."""
    # Group trades by date
    day_buckets: dict[str, list[dict[str, Any]]] = {}
    for t in all_trades:
        t_entry = str(t.get("t_entry", ""))
        date = t_entry[:10] if len(t_entry) >= 10 else ""
        if date:
            day_buckets.setdefault(date, []).append(t)

    if not day_buckets:
        return []

    max_per_day = max(len(trades) for trades in day_buckets.values())
    if max_per_day <= 1:
        return []

    # Try capping at median trades/day
    counts = sorted(len(v) for v in day_buckets.values())
    n = len(counts)
    median_count = (
        counts[n // 2] if n % 2 == 1 else (counts[n // 2 - 1] + counts[n // 2]) // 2
    )
    cap = max(1, median_count)

    if cap >= max_per_day:
        return []

    # Keep only first `cap` trades per day
    filtered: list[dict[str, Any]] = []
    for date in sorted(day_buckets):
        filtered.extend(day_buckets[date][:cap])

    hyp_return = _compute_total_return(filtered)

    if abs(hyp_return - actual_return) < 0.01:
        return []

    return [
        Counterfactual(
            description=(
                f"If you capped max trades/day at {cap}, "
                f"return would be {hyp_return:.2f}%"
            ),
            metric_name="return_pct",
            actual=round(actual_return, 4),
            hypothetical=round(hyp_return, 4),
            delta=round(hyp_return - actual_return, 4),
        )
    ]


def _counterfactual_skip_orb_minutes(
    all_trades: list[dict[str, Any]],
    events: Optional[list[dict[str, Any]]],
    actual_return: float,
) -> list[Counterfactual]:
    """If you skipped the first N minutes after ORB lock, return would be Y%."""
    if not events:
        return []

    # Find ORB lock events to get lock times
    orb_locks: dict[str, str] = {}  # date -> lock timestamp
    for e in events:
        if e.get("type") == "orb_range_locked":
            ts = str(e.get("ts", ""))
            date = ts[:10] if len(ts) >= 10 else ""
            if date:
                orb_locks[date] = ts

    if not orb_locks:
        return []

    # Filter out trades that entered within 15 min of ORB lock
    skip_minutes = 15
    filtered: list[dict[str, Any]] = []
    skipped = 0

    for t in all_trades:
        t_entry = str(t.get("t_entry", ""))
        date = t_entry[:10] if len(t_entry) >= 10 else ""
        lock_ts = orb_locks.get(date)

        if lock_ts and t_entry > lock_ts:
            # Check if within skip window
            # Simple comparison: if entry time is within first N minutes
            try:
                lock_time = (
                    lock_ts.split("T")[1] if "T" in lock_ts else lock_ts.split(" ")[1]
                )
                entry_time = (
                    t_entry.split("T")[1] if "T" in t_entry else t_entry.split(" ")[1]
                )
                lock_parts = lock_time.split(":")
                entry_parts = entry_time.split(":")
                lock_min = int(lock_parts[0]) * 60 + int(lock_parts[1])
                entry_min = int(entry_parts[0]) * 60 + int(entry_parts[1])
                if entry_min - lock_min < skip_minutes:
                    skipped += 1
                    continue
            except (ValueError, IndexError):
                pass

        filtered.append(t)

    if skipped == 0:
        return []

    hyp_return = _compute_total_return(filtered)

    if abs(hyp_return - actual_return) < 0.01:
        return []

    return [
        Counterfactual(
            description=(
                f"If you skipped the first {skip_minutes} minutes after "
                f"ORB lock, return would be {hyp_return:.2f}%"
            ),
            metric_name="return_pct",
            actual=round(actual_return, 4),
            hypothetical=round(hyp_return, 4),
            delta=round(hyp_return - actual_return, 4),
        )
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_loss_attribution(
    trades: list[dict[str, Any]],
    regime_is: Optional[dict[str, Any]] = None,
    regime_oos: Optional[dict[str, Any]] = None,
    events: Optional[list[dict[str, Any]]] = None,
) -> LossAttributionResult:
    """Analyze losing trades and produce attribution insights.

    Returns counts-only (no counterfactuals) if trade count exceeds ceiling.
    """
    losing_trades = [t for t in trades if (_safe_float(t.get("pnl")) or 0.0) < 0]

    total_loss_amount = sum(_safe_float(t.get("pnl")) or 0.0 for t in losing_trades)

    if not losing_trades:
        return LossAttributionResult(
            total_losses=0,
            total_loss_amount=0.0,
        )

    time_clusters = _compute_time_clusters(losing_trades, total_loss_amount)
    size_clusters = _compute_size_clusters(losing_trades, total_loss_amount)
    regime_summary = _compute_regime_summary(losing_trades, regime_is, regime_oos)

    # Compute ceiling: skip counterfactuals for large datasets
    counterfactuals: list[Counterfactual] = []
    if len(trades) <= COMPUTE_CEILING:
        actual_return = _compute_total_return(trades)
        actual_dd = _compute_max_drawdown(trades)

        counterfactuals.extend(
            _counterfactual_skip_hour(trades, losing_trades, actual_return)
        )
        counterfactuals.extend(
            _counterfactual_regime_filter(trades, regime_is, regime_oos, actual_dd)
        )
        counterfactuals.extend(
            _counterfactual_max_trades_per_day(trades, actual_return)
        )
        counterfactuals.extend(
            _counterfactual_skip_orb_minutes(trades, events, actual_return)
        )

    return LossAttributionResult(
        time_clusters=time_clusters,
        size_clusters=size_clusters,
        regime_summary=regime_summary,
        counterfactuals=counterfactuals,
        total_losses=len(losing_trades),
        total_loss_amount=round(total_loss_amount, 2),
    )
