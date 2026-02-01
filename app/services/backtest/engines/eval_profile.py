"""Eval account profile for R-native risk budgeting.

Prop-firm style trailing drawdown model. Per-trade risk (R_day) is derived
from remaining drawdown room rather than a fixed dollar amount.

    R_day = clamp(room * risk_fraction, r_min, r_max)
    room  = max_drawdown_dollars - (peak_equity - current_equity)

When room <= 0 the eval is blown and R_day returns 0.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalAccountProfile:
    """Immutable prop-firm account profile.

    All dollar values are absolute (not percentages).
    """

    account_size: float
    max_drawdown_dollars: float
    max_daily_loss_dollars: float
    risk_fraction: float = 0.15
    r_min_dollars: float = 100.0
    r_max_dollars: float = 300.0


def compute_r_day(
    profile: EvalAccountProfile,
    current_equity: float,
    peak_equity: float,
) -> float:
    """Compute the per-trade risk budget for the current day.

    Pure function: no state, no I/O.

    Returns:
        R_day in dollars.  0.0 when the eval is blown (room exhausted).
    """
    trailing_dd = peak_equity - current_equity
    room = profile.max_drawdown_dollars - trailing_dd
    if room <= 0:
        return 0.0
    raw = room * profile.risk_fraction
    return max(profile.r_min_dollars, min(profile.r_max_dollars, raw))
