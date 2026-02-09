"""Process score computation for backtest coaching.

Pure-function module computing a 0-100 composite process score from
existing run data. Evaluates trading process quality independent of
P&L outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPUTE_CEILING = 50_000

REGIME_PREFERRED_SIDE: dict[str, str] = {
    "uptrend": "long",
    "downtrend": "short",
    "trending_up": "long",
    "trending_down": "short",
}

# Max bars between setup_valid and entry_signal for rule adherence
MAX_SETUP_WINDOW_BARS = 5

GRADE_MAP = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
    (0, "F"),
]

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

# Component weights (must sum to 1.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "rule_adherence": 0.25,
    "regime_alignment": 0.20,
    "risk_discipline": 0.25,
    "exit_quality": 0.15,
    "consistency": 0.15,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ScoreComponent:
    name: str
    score: float
    weight: float
    detail: str
    available: bool


@dataclass
class ProcessScoreResult:
    total: Optional[float]
    grade: str
    components: list[ScoreComponent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float, returning None for NaN/Inf/None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _grade(score: Optional[float]) -> str:
    if score is None:
        return "unavailable"
    for threshold, letter in GRADE_MAP:
        if score >= threshold:
            return letter
    return "F"


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def _score_rule_adherence(
    events: Optional[list[dict[str, Any]]],
) -> Optional[float]:
    """% of entry_signal events preceded by setup_valid within window."""
    if not events:
        return None

    entry_signals = [e for e in events if e.get("type") == "entry_signal"]
    setup_valids = [e for e in events if e.get("type") == "setup_valid"]

    if not entry_signals:
        return None

    if not setup_valids:
        return 0.0

    # Sort setup_valids by bar_index for efficient lookup
    setup_bars = sorted(e.get("bar_index", 0) for e in setup_valids)

    matched = 0
    for entry in entry_signals:
        entry_bar = entry.get("bar_index", 0)
        # Find most recent setup_valid where 0 <= entry_bar - setup_bar <= MAX_SETUP_WINDOW_BARS
        for sb in reversed(setup_bars):
            diff = entry_bar - sb
            if 0 <= diff <= MAX_SETUP_WINDOW_BARS:
                matched += 1
                break
            if diff > MAX_SETUP_WINDOW_BARS:
                break

    return (matched / len(entry_signals)) * 100.0


def _score_regime_alignment(
    trades: list[dict[str, Any]],
    regime_is: Optional[dict[str, Any]],
    regime_oos: Optional[dict[str, Any]],
) -> Optional[float]:
    """% of trades aligned with preferred direction for the regime."""
    if not trades:
        return None

    # Collect regime tags from IS and OOS
    regime_tags: list[str] = []
    for regime in (regime_is, regime_oos):
        if regime and isinstance(regime, dict):
            for key in ("trend_tag", "tags"):
                val = regime.get(key)
                if isinstance(val, str) and val:
                    regime_tags.append(val.lower())
                elif isinstance(val, list):
                    regime_tags.extend(t.lower() for t in val if isinstance(t, str))

    if not regime_tags:
        return None

    # Find preferred side from regime tags
    preferred_side: Optional[str] = None
    for tag in regime_tags:
        if tag in REGIME_PREFERRED_SIDE:
            preferred_side = REGIME_PREFERRED_SIDE[tag]
            break

    # Neutral regimes (flat, ranging, etc.) - all sides score 100%
    if preferred_side is None:
        return 100.0

    aligned = sum(1 for t in trades if str(t.get("side", "")).lower() == preferred_side)
    return (aligned / len(trades)) * 100.0


def _score_risk_discipline(trades: list[dict[str, Any]]) -> Optional[float]:
    """Score based on worst_trade_loss / median_abs_trade_loss ratio."""
    if not trades:
        return None

    losses = []
    for t in trades:
        pnl = _safe_float(t.get("pnl"))
        if pnl is not None and pnl < 0:
            losses.append(abs(pnl))

    if not losses:
        # No losing trades = perfect risk discipline
        return 100.0

    losses.sort()
    n = len(losses)
    if n % 2 == 0:
        median_loss = (losses[n // 2 - 1] + losses[n // 2]) / 2
    else:
        median_loss = losses[n // 2]

    if median_loss == 0:
        return 100.0

    worst_ratio = max(losses) / median_loss

    # Score: ratio of 1 = perfect (100), ratio of 5+ = poor (0)
    # Linear scale: score = max(0, 100 - (ratio - 1) * 25)
    score = max(0.0, min(100.0, 100.0 - (worst_ratio - 1.0) * 25.0))

    # Position sizing CV sub-component (if size field exists)
    sizes = []
    for t in trades:
        size = _safe_float(t.get("size"))
        if size is not None and size > 0:
            sizes.append(size)

    if len(sizes) >= 2:
        mean_size = sum(sizes) / len(sizes)
        if mean_size > 0:
            variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            cv = math.sqrt(variance) / mean_size
            # Low CV is good. CV of 0 = 100, CV of 1+ = 0
            sizing_score = max(0.0, min(100.0, 100.0 - cv * 100.0))
            # Average the two sub-components
            score = (score + sizing_score) / 2.0

    return score


def _score_exit_quality(trades: list[dict[str, Any]]) -> Optional[float]:
    """Avg winner return / avg loser return ratio."""
    if not trades:
        return None

    winner_returns: list[float] = []
    loser_returns: list[float] = []

    for t in trades:
        ret = _safe_float(t.get("return_pct"))
        if ret is None:
            continue
        if ret > 0:
            winner_returns.append(ret)
        elif ret < 0:
            loser_returns.append(abs(ret))

    if not winner_returns or not loser_returns:
        return None

    avg_win = sum(winner_returns) / len(winner_returns)
    avg_loss = sum(loser_returns) / len(loser_returns)

    if avg_loss == 0:
        return 100.0

    ratio = avg_win / avg_loss

    # Score: ratio of 2+ = 100, ratio of 0.5 = 0
    # Linear scale between 0.5 and 2
    score = max(0.0, min(100.0, (ratio - 0.5) / 1.5 * 100.0))
    return score


def _score_consistency(trades: list[dict[str, Any]]) -> Optional[float]:
    """CV of return_pct. Low CV = consistent = good."""
    if len(trades) < 5:
        return None

    returns: list[float] = []
    for t in trades:
        ret = _safe_float(t.get("return_pct"))
        if ret is not None:
            returns.append(ret)

    if len(returns) < 5:
        return None

    mean_ret = sum(returns) / len(returns)

    # Zero-mean edge case: CV is undefined
    if mean_ret == 0:
        return None

    variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)
    cv = abs(std / mean_ret)

    # Score: CV of 0 = 100, CV of 3+ = 0
    score = max(0.0, min(100.0, 100.0 - (cv / 3.0) * 100.0))
    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_process_score(
    trades: list[dict[str, Any]],
    events: Optional[list[dict[str, Any]]],
    regime_is: Optional[dict[str, Any]],
    regime_oos: Optional[dict[str, Any]],
    summary: Optional[dict[str, Any]] = None,
) -> ProcessScoreResult:
    """Compute a 0-100 composite process score.

    Returns partial result with total=None if compute ceiling is exceeded.
    """
    # Compute ceiling check
    if len(trades) > COMPUTE_CEILING or (events and len(events) > COMPUTE_CEILING):
        return ProcessScoreResult(
            total=None,
            grade="unavailable",
            components=[],
        )

    # Compute each component
    raw_scores: dict[str, Optional[float]] = {
        "rule_adherence": _score_rule_adherence(events),
        "regime_alignment": _score_regime_alignment(trades, regime_is, regime_oos),
        "risk_discipline": _score_risk_discipline(trades),
        "exit_quality": _score_exit_quality(trades),
        "consistency": _score_consistency(trades),
    }

    components: list[ScoreComponent] = []
    available_weight = 0.0
    weighted_sum = 0.0

    # First pass: find total available weight
    for name, score in raw_scores.items():
        if score is not None:
            available_weight += DEFAULT_WEIGHTS[name]

    # Second pass: build components with redistributed weights
    for name, score in raw_scores.items():
        base_weight = DEFAULT_WEIGHTS[name]
        available = score is not None

        if available and available_weight > 0:
            # Redistribute proportionally
            effective_weight = base_weight / available_weight
            weighted_sum += score * effective_weight  # type: ignore[operator]
        else:
            effective_weight = 0.0

        detail = _component_detail(name, score)

        components.append(
            ScoreComponent(
                name=name,
                score=round(score, 1) if score is not None else 0.0,
                weight=round(effective_weight, 3),
                detail=detail,
                available=available,
            )
        )

    if available_weight == 0:
        return ProcessScoreResult(
            total=None,
            grade="unavailable",
            components=components,
        )

    total = round(weighted_sum, 1)
    return ProcessScoreResult(
        total=total,
        grade=_grade(total),
        components=components,
    )


def _component_detail(name: str, score: Optional[float]) -> str:
    """Generate a human-readable detail string for a component."""
    if score is None:
        return "Insufficient data"

    details: dict[str, str] = {
        "rule_adherence": (f"{score:.0f}% of entries followed a valid setup signal"),
        "regime_alignment": (f"{score:.0f}% of trades aligned with regime direction"),
        "risk_discipline": (f"Risk control score: {score:.0f}/100"),
        "exit_quality": (f"Winner/loser return ratio score: {score:.0f}/100"),
        "consistency": (f"Return consistency score: {score:.0f}/100"),
    }
    return details.get(name, f"Score: {score:.0f}")
