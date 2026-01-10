"""Scoring functions for backtest optimization.

Pure functions for computing objective scores from backtest summaries.
These are used by the tuner to rank parameter combinations.
"""

import math
from typing import Any, Optional


def compute_score(
    summary: dict[str, Any],
    objective: str = "sharpe",
    min_trades: int = 5,
) -> Optional[float]:
    """
    Compute optimization score from backtest summary.

    Args:
        summary: Backtest summary dict with metrics
        objective: Objective function - "sharpe", "return", or "calmar"
        min_trades: Minimum trades required (fewer = skipped)

    Returns:
        Score value, or None if trial should be skipped.
        Higher is better for all objectives.
        None indicates insufficient trades or missing metrics.
    """
    trades = summary.get("trades", 0)

    # Insufficient trades - skip this trial
    if trades < min_trades:
        return None

    if objective == "sharpe":
        return _compute_sharpe_score(summary)
    elif objective == "return":
        return _compute_return_score(summary)
    elif objective == "calmar":
        return _compute_calmar_score(summary)
    else:
        # Unknown objective - skip
        return None


def _compute_sharpe_score(summary: dict[str, Any]) -> Optional[float]:
    """Compute Sharpe ratio score."""
    sharpe = summary.get("sharpe")

    if sharpe is None:
        return None

    # Handle NaN/Inf
    if isinstance(sharpe, float) and (math.isnan(sharpe) or math.isinf(sharpe)):
        return None

    return float(sharpe)


def _compute_return_score(summary: dict[str, Any]) -> Optional[float]:
    """Compute total return score."""
    return_pct = summary.get("return_pct")

    if return_pct is None:
        return None

    if isinstance(return_pct, float) and (
        math.isnan(return_pct) or math.isinf(return_pct)
    ):
        return None

    return float(return_pct)


def _compute_calmar_score(summary: dict[str, Any]) -> Optional[float]:
    """
    Compute Calmar ratio score (return / max drawdown).

    Higher is better - good returns with low drawdown.
    """
    return_pct = summary.get("return_pct")
    max_dd = summary.get("max_drawdown_pct")

    if return_pct is None or max_dd is None:
        return None

    # Handle NaN/Inf
    if isinstance(return_pct, float) and (
        math.isnan(return_pct) or math.isinf(return_pct)
    ):
        return None
    if isinstance(max_dd, float) and (math.isnan(max_dd) or math.isinf(max_dd)):
        return None

    # Max drawdown is typically negative, take absolute value
    dd_abs = abs(float(max_dd))

    if dd_abs == 0:
        # No drawdown - if positive return, excellent; if negative, terrible
        if return_pct > 0:
            return float(return_pct) * 10  # Bonus for no drawdown
        else:
            return float(return_pct)

    return float(return_pct) / dd_abs


def rank_trials(
    trials: list[dict[str, Any]],
    objective: str = "sharpe",
    min_trades: int = 5,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    Rank trials by score and return top N.

    Args:
        trials: List of trial dicts with "summary" and "params" keys
        objective: Objective function
        min_trades: Minimum trades for valid trial
        top_n: Number of top results to return

    Returns:
        Top N trials sorted by score descending, with "rank" and "score" added.
    """
    scored = []

    for trial in trials:
        summary = trial.get("summary", {})
        score = compute_score(summary, objective, min_trades)

        if score is not None:
            scored.append(
                {
                    **trial,
                    "score": score,
                }
            )

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Add ranks and return top N
    result = []
    for i, trial in enumerate(scored[:top_n]):
        result.append(
            {
                **trial,
                "rank": i + 1,
            }
        )

    return result
