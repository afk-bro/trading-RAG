"""Daily risk governor for eval/prop-firm backtesting."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DailyGovernorStats:
    """Aggregated stats for governor behavior across the backtest."""

    signals_skipped: int = 0  # setups that passed criteria but governor blocked
    days_halted: int = 0  # calendar days where governor halted early
    half_size_trades: int = 0  # trades taken at reduced size
    total_days_traded: int = 0  # days with at least one trade


@dataclass
class DailyGovernor:
    """Per-day risk state tracker with half-size stepdown.

    Policy:
        - Full loss (day_loss <= -max_daily_loss) -> halt for day
        - Half loss (day_loss <= -half_threshold) -> risk_multiplier drops to half_size_multiplier
        - Trade count cap -> halt for day
    """

    # --- Config (set once) ---
    max_daily_loss_dollars: float = 300.0
    max_trades_per_day: int = 2
    half_size_multiplier: float = 0.5
    max_daily_loss_r: Optional[float] = None  # R-based daily loss cap (e.g. 1.0 = 1R)

    # Adaptive confidence tiering config (None = disabled)
    adaptive_tier_streak: Optional[int] = None  # activate after N consecutive losses
    adaptive_tier_dd_pct: Optional[float] = (
        None  # activate when trailing DD > X% of max DD
    )
    adaptive_tier_a: float = 0.80  # tier A threshold when adaptive is active
    adaptive_tier_b: float = 0.70  # tier B threshold when adaptive is active

    # --- Per-day mutable state ---
    day_loss_dollars: float = 0.0
    day_trade_count: int = 0
    risk_multiplier: float = 1.0
    halted_for_day: bool = False
    halt_reason: str = ""  # "loss_limit" or "trade_limit" when halted
    current_date: object = None  # date object, set on first bar of day

    # Adaptive state (reset daily)
    consecutive_losses: int = 0
    confidence_tier_active: bool = False

    @property
    def half_loss_threshold(self) -> float:
        """Loss level that triggers half-size stepdown."""
        return self.max_daily_loss_dollars * self.half_size_multiplier

    def update_r_day(self, r_day: float) -> None:
        """Recalculate max_daily_loss_dollars from R-based cap when configured.

        Called at day boundary after R_day is recomputed from eval profile.
        If max_daily_loss_r is None, this is a no-op.
        """
        if self.max_daily_loss_r is not None and r_day > 0:
            self.max_daily_loss_dollars = self.max_daily_loss_r * r_day

    def allows_entry(self) -> bool:
        """Check if governor permits a new trade entry."""
        if self.halted_for_day:
            return False
        if self.day_trade_count >= self.max_trades_per_day:
            self.halted_for_day = True
            self.halt_reason = "trade_limit"
            return False
        if self.day_loss_dollars <= -self.max_daily_loss_dollars:
            self.halted_for_day = True
            self.halt_reason = "loss_limit"
            return False
        return True

    def record_trade_close(
        self, pnl_dollars: float, is_partial_leg: bool = False
    ) -> None:
        """Update state after a trade closes.

        Args:
            pnl_dollars: Realized P&L for this close.
            is_partial_leg: When True (scale-out leg), P&L accumulates
                but the close does not count toward the daily trade cap.
        """
        if not is_partial_leg:
            self.day_trade_count += 1
        if pnl_dollars < 0:
            self.day_loss_dollars += pnl_dollars
            if not is_partial_leg:
                self.consecutive_losses += 1
        elif not is_partial_leg:
            self.consecutive_losses = 0

        # Adaptive confidence tiering: streak trigger
        if (
            self.adaptive_tier_streak is not None
            and self.consecutive_losses >= self.adaptive_tier_streak
        ):
            self.confidence_tier_active = True

        # Half-size stepdown: if cumulative loss hits half threshold,
        # next trade uses reduced size
        if self.day_loss_dollars <= -self.half_loss_threshold:
            self.risk_multiplier = self.half_size_multiplier

        # Full halt
        if self.day_loss_dollars <= -self.max_daily_loss_dollars:
            self.halted_for_day = True
            self.halt_reason = "loss_limit"

    def maybe_reset(self, bar_date) -> str:
        """Reset state if calendar day changed.

        Returns:
            "" if no reset or day wasn't halted,
            halt_reason ("loss_limit" or "trade_limit") if day was halted.
        """
        if self.current_date != bar_date:
            prev_halt_reason = self.halt_reason if self.halted_for_day else ""
            self.reset_day()
            self.current_date = bar_date
            return prev_halt_reason
        return ""

    def reset_day(self) -> None:
        """Reset all per-day state."""
        self.day_loss_dollars = 0.0
        self.day_trade_count = 0
        self.risk_multiplier = 1.0
        self.halted_for_day = False
        self.halt_reason = ""
        self.consecutive_losses = 0
        self.confidence_tier_active = False

    def check_adaptive_dd(self, trailing_dd_pct: float) -> None:
        """Activate confidence tiering if trailing DD exceeds threshold."""
        if (
            self.adaptive_tier_dd_pct is not None
            and trailing_dd_pct >= self.adaptive_tier_dd_pct
        ):
            self.confidence_tier_active = True
