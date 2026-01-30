"""
Unicorn Model Backtest Runner.

Dedicated backtester for ICT Unicorn Model strategy with detailed analytics:
- Per-trade breakdown of which criteria fired
- Bias confidence at entry
- MFE (Maximum Favorable Excursion) / MAE (Maximum Adverse Excursion)
- Time-of-day statistics (NY AM vs PM vs London vs Asia)
- Criteria bottleneck analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional
import random
import statistics

from app.utils.time import to_eastern_time

from app.services.strategy.models import OHLCVBar
from app.services.strategy.indicators.ict_patterns import (
    detect_fvgs,
    detect_breaker_blocks,
    detect_mitigation_blocks,
    detect_liquidity_sweeps,
    detect_mss,
    FVGType,
    BlockType,
)
from app.services.strategy.indicators.tf_bias import (
    BiasDirection,
    TimeframeBias,
    compute_tf_bias,
)
from app.services.strategy.strategies.unicorn_model import (
    UnicornSetup,
    UnicornConfig,
    SessionProfile,
    CriteriaScore as StrategyCriteriaScore,
    analyze_unicorn_setup,
    is_in_macro_window,
    get_max_stop_points,
    get_point_value,
    find_entry_zone,
    calculate_stop_and_target,
    MACRO_WINDOWS,
    DEFAULT_CONFIG,
    _calculate_atr,
)


class TradingSession(str, Enum):
    """Trading session classification."""
    NY_AM = "ny_am"       # 9:30-11:00 ET
    NY_PM = "ny_pm"       # 13:30-15:00 ET
    LONDON = "london"     # 3:00-4:00 ET
    ASIA = "asia"         # 19:00-20:00 ET
    OFF_HOURS = "off_hours"


class TradeOutcome(str, Enum):
    """Trade result classification."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"


class IntrabarPolicy(str, Enum):
    """
    Policy for resolving stop/target ambiguity when both are hit in same bar.

    This is critical for realistic backtesting - assuming "TP wins" leads to
    artificially inflated win rates. Default to WORST for conservative estimates.
    """
    WORST = "worst"       # Assume stop hit first (conservative)
    BEST = "best"         # Assume target hit first (optimistic, unrealistic)
    RANDOM = "random"     # 50/50 coin flip
    OHLC_PATH = "ohlc_path"  # Deterministic: O→H→L→C (bullish) or O→L→H→C (bearish)


@dataclass(frozen=True)
class ExitResult:
    """Result of intrabar exit resolution.

    Returned by resolve_bar_exit when a trade exits on a given bar.
    """
    exit_price: float
    exit_reason: str   # "stop_loss", "target", "time_stop", "eod"
    pnl_points: float


def resolve_bar_exit(
    trade: "TradeRecord",
    bar: OHLCVBar,
    intrabar_policy: IntrabarPolicy,
    slippage_points: float,
    eod_exit: bool = False,
    eod_time: Optional[time] = None,
    time_stop_minutes: Optional[int] = None,
    time_stop_r_threshold: float = 0.25,
) -> Optional[ExitResult]:
    """
    Determine if and how a trade exits on a given bar.

    Evaluation order (explicit priority):
        1. Stop/target hit detection (touching counts: <= / >=)
        2. Same-bar ambiguity resolved via IntrabarPolicy
        3. Gap-through stop pricing: fill at bar.open when open is
           worse than the stop level
        4. Time-stop: exit at bar.close if unrealized R below threshold
        5. EOD: exit at bar.close

    Fill assumptions
    ----------------
    * Stops are stop-market orders.  Gap fills at bar.open (worst-case).
      No worse-than-open scenario is modeled.
    * Targets are limit orders.  Fill at limit price with adverse
      slippage applied.  No price improvement is modeled.
    * ``pnl_points = exit_price - entry_price`` for longs,
      ``entry_price - exit_price`` for shorts.
      Slippage is already baked into exit_price.

    OHLC_PATH logic (deterministic, no simulation):
        * Bullish bar (close >= open): assumed path O -> H -> L -> C
          - Long:  target first (O->H leg), stop second (H->L leg)
          - Short: stop first  (O->H leg), target second (H->L leg)
        * Bearish bar (close < open): assumed path O -> L -> H -> C
          - Long:  stop first  (O->L leg), target second (L->H leg)
          - Short: target first (O->L leg), stop second (L->H leg)

    Returns:
        ExitResult if the trade exits on this bar, None otherwise.
    """
    is_long = trade.direction == BiasDirection.BULLISH
    ts = bar.ts

    # --- 1. Detect stop / target hits ---
    if is_long:
        stop_hit = bar.low <= trade.stop_price
        target_hit = bar.high >= trade.target_price
    else:
        stop_hit = bar.high >= trade.stop_price
        target_hit = bar.low <= trade.target_price

    # --- 2 & 3. Resolve exit when at least one level is hit ---
    if stop_hit or target_hit:
        # Decide which fires first when both are hit
        if stop_hit and target_hit:
            if intrabar_policy == IntrabarPolicy.WORST:
                take_stop = True
            elif intrabar_policy == IntrabarPolicy.BEST:
                take_stop = False
            elif intrabar_policy == IntrabarPolicy.OHLC_PATH:
                bullish_bar = bar.close >= bar.open
                if is_long:
                    # Bullish bar O→H→L→C: target on O→H, stop on H→L => target first
                    # Bearish bar O→L→H→C: stop on O→L, target on L→H => stop first
                    take_stop = not bullish_bar
                else:
                    # Bullish bar O→H→L→C: stop on O→H, target on H→L => stop first
                    # Bearish bar O→L→H→C: target on O→L, stop on L→H => target first
                    take_stop = bullish_bar
            else:  # RANDOM
                take_stop = random.random() < 0.5
        else:
            take_stop = stop_hit  # Only one was hit

        if take_stop:
            # Gap-through: fill at bar.open when open is already past the stop
            if is_long:
                fill_price = min(trade.stop_price, bar.open)
                exit_price = fill_price - slippage_points
                pnl = exit_price - trade.entry_price
            else:
                fill_price = max(trade.stop_price, bar.open)
                exit_price = fill_price + slippage_points
                pnl = trade.entry_price - exit_price
            return ExitResult(exit_price=exit_price, exit_reason="stop_loss", pnl_points=pnl)
        else:
            # Target is a limit order — fills at limit price (no gap improvement)
            if is_long:
                exit_price = trade.target_price - slippage_points
                pnl = exit_price - trade.entry_price
            else:
                exit_price = trade.target_price + slippage_points
                pnl = trade.entry_price - exit_price
            return ExitResult(exit_price=exit_price, exit_reason="target", pnl_points=pnl)

    # --- 4. Time-stop ---
    if (
        time_stop_minutes is not None
        and (ts - trade.entry_time).total_seconds() / 60 >= time_stop_minutes
    ):
        if trade.risk_points > 0:
            if is_long:
                unrealized_r = (bar.close - trade.entry_price) / trade.risk_points
            else:
                unrealized_r = (trade.entry_price - bar.close) / trade.risk_points
        else:
            unrealized_r = 0.0

        if unrealized_r < time_stop_r_threshold:
            if is_long:
                exit_price = bar.close - slippage_points
                pnl = exit_price - trade.entry_price
            else:
                exit_price = bar.close + slippage_points
                pnl = trade.entry_price - exit_price
            return ExitResult(exit_price=exit_price, exit_reason="time_stop", pnl_points=pnl)

    # --- 5. EOD exit ---
    if eod_exit and eod_time is not None and to_eastern_time(ts) >= eod_time:
        if is_long:
            exit_price = bar.close - slippage_points
            pnl = exit_price - trade.entry_price
        else:
            exit_price = bar.close + slippage_points
            pnl = trade.entry_price - exit_price
        return ExitResult(exit_price=exit_price, exit_reason="eod", pnl_points=pnl)

    return None


def compute_adverse_wick_ratio(bar: OHLCVBar, direction: BiasDirection) -> float:
    """
    Adverse wick ratio: fraction of bar range that is the wick opposing our direction.

    Long entry: upper wick is adverse (rejection after buying).
        ratio = (high - max(open, close)) / (high - low)
    Short entry: lower wick is adverse (rejection after selling).
        ratio = (min(open, close) - low) / (high - low)

    Returns 0.0 if bar range is zero.
    """
    bar_range = bar.high - bar.low
    if bar_range <= 0:
        return 0.0
    body_top = max(bar.open, bar.close)
    body_bot = min(bar.open, bar.close)
    if direction == BiasDirection.BULLISH:
        return (bar.high - body_top) / bar_range
    else:
        return (body_bot - bar.low) / bar_range


def compute_range_atr_mult(bar: OHLCVBar, atr: float) -> float:
    """Bar range as a multiple of ATR. Returns 0.0 if ATR is zero."""
    if atr <= 0:
        return 0.0
    return (bar.high - bar.low) / atr


# Mandatory criteria that MUST pass before soft scoring applies.
# These protect core risk logic and cannot be bypassed by scoring.
MANDATORY_CRITERIA = frozenset({"htf_bias", "stop_valid", "macro_window"})

# Scored criteria - soft scoring threshold applies to these
SCORED_CRITERIA = frozenset({
    "liquidity_sweep", "htf_fvg", "breaker_block", "ltf_fvg", "mss"
})


@dataclass
class CriteriaCheck:
    """Results of checking each of the 8 Unicorn criteria."""
    htf_bias_aligned: bool = False
    liquidity_sweep_found: bool = False
    htf_fvg_found: bool = False
    breaker_block_found: bool = False
    ltf_fvg_found: bool = False
    mss_found: bool = False
    stop_valid: bool = False
    in_macro_window: bool = False

    # Details
    htf_bias_direction: Optional[BiasDirection] = None
    htf_bias_confidence: float = 0.0
    sweep_type: Optional[str] = None
    fvg_size: float = 0.0
    stop_points: float = 0.0
    session: TradingSession = TradingSession.OFF_HOURS

    @property
    def criteria_met_count(self) -> int:
        """Count how many criteria are satisfied."""
        return sum([
            self.htf_bias_aligned,
            self.liquidity_sweep_found,
            self.htf_fvg_found,
            self.breaker_block_found,
            self.ltf_fvg_found,
            self.mss_found,
            self.stop_valid,
            self.in_macro_window,
        ])

    @property
    def all_criteria_met(self) -> bool:
        """Check if all 8 criteria are satisfied."""
        return self.criteria_met_count == 8

    def missing_criteria(self) -> list[str]:
        """Return list of criteria that failed."""
        missing = []
        if not self.htf_bias_aligned:
            missing.append("htf_bias")
        if not self.liquidity_sweep_found:
            missing.append("liquidity_sweep")
        if not self.htf_fvg_found:
            missing.append("htf_fvg")
        if not self.breaker_block_found:
            missing.append("breaker_block")
        if not self.ltf_fvg_found:
            missing.append("ltf_fvg")
        if not self.mss_found:
            missing.append("mss")
        if not self.stop_valid:
            missing.append("stop_valid")
        if not self.in_macro_window:
            missing.append("macro_window")
        return missing

    @property
    def mandatory_criteria_met(self) -> bool:
        """
        Check if all MANDATORY criteria are satisfied.

        Mandatory criteria (htf_bias, stop_valid, macro_window) MUST pass
        before soft scoring can be applied. These protect core risk logic.
        """
        return (
            self.htf_bias_aligned and
            self.stop_valid and
            self.in_macro_window
        )

    @property
    def scored_criteria_count(self) -> int:
        """
        Count how many SCORED criteria are satisfied.

        Scored criteria: liquidity_sweep, htf_fvg, breaker_block, ltf_fvg, mss
        Soft scoring threshold applies only to these.
        """
        return sum([
            self.liquidity_sweep_found,
            self.htf_fvg_found,
            self.breaker_block_found,
            self.ltf_fvg_found,
            self.mss_found,
        ])

    def meets_entry_requirements(self, min_scored: int = 3) -> bool:
        """
        Check if criteria meet entry requirements with guardrails.

        Args:
            min_scored: Minimum scored criteria required (out of 5)

        Returns:
            True if all mandatory criteria pass AND scored count >= threshold
        """
        return self.mandatory_criteria_met and self.scored_criteria_count >= min_scored


@dataclass
class TradeRecord:
    """Complete record of a single trade."""
    # Entry info
    entry_time: datetime
    entry_price: float
    direction: BiasDirection
    quantity: float
    session: TradingSession

    # Criteria at entry
    criteria: CriteriaCheck

    # Risk management
    stop_price: float
    target_price: float
    risk_points: float

    # MFE/MAE tracking
    mfe: float = 0.0  # Maximum Favorable Excursion (best unrealized profit)
    mae: float = 0.0  # Maximum Adverse Excursion (worst unrealized loss)
    mfe_time: Optional[datetime] = None
    mae_time: Optional[datetime] = None

    # Exit info
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # Result
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    outcome: TradeOutcome = TradeOutcome.OPEN
    r_multiple: float = 0.0  # PnL as multiple of risk

    # Timing
    duration_minutes: int = 0
    bars_held: int = 0


@dataclass
class SetupOccurrence:
    """Record of a potential setup (even if not taken)."""
    timestamp: datetime
    direction: BiasDirection
    criteria: CriteriaCheck
    taken: bool = False
    reason_not_taken: Optional[str] = None

    # Parity diagnostics: record the gating inputs so live vs backtest
    # decisions can be compared on the same snapshot set.
    mandatory_met: bool = False
    scored_count: int = 0
    min_scored_required: int = 0
    decide_entry_result: bool = False
    scored_missing: list[str] = field(default_factory=list)

    # Bar-quality guard diagnostics
    signal_wick_ratio: float = 0.0       # adverse wick / range of signal bar
    signal_range_atr_mult: float = 0.0   # signal bar range / ATR
    wick_guard_rejected: bool = False
    range_guard_rejected: bool = False
    guard_reason_code: Optional[str] = None  # stable key: "wick_guard" | "range_guard"


@dataclass
class SessionStats:
    """Statistics for a trading session."""
    session: TradingSession
    total_setups: int = 0
    valid_setups: int = 0  # All 8 criteria met
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_points: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    avg_r_multiple: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return self.wins / self.trades_taken

    @property
    def setup_conversion_rate(self) -> float:
        """How often valid setups become trades."""
        if self.valid_setups == 0:
            return 0.0
        return self.trades_taken / self.valid_setups


@dataclass
class CriteriaBottleneck:
    """Analysis of which criteria fail most often."""
    criterion: str
    fail_count: int = 0
    fail_rate: float = 0.0  # Percentage of setups where this was the blocker

    # When this criterion passes, what's the outcome?
    pass_count: int = 0
    pass_win_rate: float = 0.0


@dataclass
class ConfidenceBucket:
    """Statistics grouped by confidence level."""
    min_confidence: float
    max_confidence: float
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    avg_r_multiple: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count


@dataclass
class UnicornBacktestResult:
    """Complete backtest results with analytics."""
    # Summary
    symbol: str
    start_date: datetime
    end_date: datetime
    total_bars: int

    # Setup analysis
    total_setups_scanned: int = 0
    partial_setups: int = 0  # Some criteria met but not all
    valid_setups: int = 0    # All 8 criteria met
    trades_taken: int = 0

    # Trade results
    trades: list[TradeRecord] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    breakevens: int = 0

    # PnL
    total_pnl_points: float = 0.0
    total_pnl_dollars: float = 0.0
    largest_win_points: float = 0.0
    largest_loss_points: float = 0.0

    # MFE/MAE analysis
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_capture_rate: float = 0.0  # How much of MFE was captured

    # R-Multiple analysis
    avg_r_multiple: float = 0.0
    best_r_multiple: float = 0.0
    worst_r_multiple: float = 0.0

    # Session breakdown
    session_stats: dict[TradingSession, SessionStats] = field(default_factory=dict)

    # Criteria bottleneck analysis
    criteria_bottlenecks: list[CriteriaBottleneck] = field(default_factory=list)

    # Confidence correlation
    confidence_buckets: list[ConfidenceBucket] = field(default_factory=list)
    confidence_win_correlation: float = 0.0  # Pearson correlation

    # Setup occurrences (for debugging)
    all_setups: list[SetupOccurrence] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return self.wins / self.trades_taken

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_points for t in self.trades if t.pnl_points > 0)
        gross_loss = abs(sum(t.pnl_points for t in self.trades if t.pnl_points < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def expectancy_points(self) -> float:
        """Expected value per trade in points."""
        if self.trades_taken == 0:
            return 0.0
        return self.total_pnl_points / self.trades_taken

    @property
    def setup_to_trade_ratio(self) -> float:
        """How often valid setups become trades."""
        if self.valid_setups == 0:
            return 0.0
        return self.trades_taken / self.valid_setups


def classify_session(ts: datetime) -> TradingSession:
    """Classify timestamp into trading session.

    Args:
        ts: Timezone-aware datetime (any timezone).

    Raises:
        ValueError: If ts is a naive datetime.
    """
    t = to_eastern_time(ts)

    # NY AM: 9:30-11:00
    if time(9, 30) <= t <= time(11, 0):
        return TradingSession.NY_AM
    # NY PM: 13:30-15:00
    elif time(13, 30) <= t <= time(15, 0):
        return TradingSession.NY_PM
    # London: 3:00-4:00
    elif time(3, 0) <= t <= time(4, 0):
        return TradingSession.LONDON
    # Asia: 19:00-20:00
    elif time(19, 0) <= t <= time(20, 0):
        return TradingSession.ASIA
    else:
        return TradingSession.OFF_HOURS


def check_criteria(
    bars: list[OHLCVBar],
    htf_bars: list[OHLCVBar],
    ltf_bars: list[OHLCVBar],
    symbol: str,
    ts: datetime,
    direction_filter: Optional[BiasDirection] = None,
    config: Optional[UnicornConfig] = None,
) -> CriteriaCheck:
    """
    Check all 8 Unicorn criteria at a specific point in time.

    Args:
        bars: Primary timeframe bars
        htf_bars: Higher timeframe bars for bias
        ltf_bars: Lower timeframe bars for confirmation
        symbol: Trading symbol
        ts: Current timestamp
        direction_filter: Only check for this direction (optional)
        config: Strategy configuration (ATR thresholds, session profile)

    Returns:
        CriteriaCheck with all criteria results
    """
    if config is None:
        config = DEFAULT_CONFIG

    check = CriteriaCheck()
    check.session = classify_session(ts)

    # Compute ATR for volatility-normalized thresholds
    atr_values = _calculate_atr(htf_bars, config.atr_period)
    current_atr = atr_values[-1] if atr_values else 0.0

    # 1. HTF Bias
    htf_bias = compute_tf_bias(
        m15_bars=htf_bars,
        m5_bars=ltf_bars,
        timestamp=ts,
    )

    check.htf_bias_direction = htf_bias.final_direction
    check.htf_bias_confidence = htf_bias.final_confidence
    check.htf_bias_aligned = htf_bias.is_tradeable

    if direction_filter and htf_bias.final_direction != direction_filter:
        check.htf_bias_aligned = False

    direction = htf_bias.final_direction
    if direction == BiasDirection.NEUTRAL:
        return check  # Can't proceed without direction

    current_price = bars[-1].close if bars else 0

    # 2. Liquidity Sweep
    sweeps = detect_liquidity_sweeps(htf_bars, lookback=50)
    for sweep in sweeps:
        if direction == BiasDirection.BULLISH and sweep.sweep_type == "low":
            check.liquidity_sweep_found = True
            check.sweep_type = "low"
            break
        elif direction == BiasDirection.BEARISH and sweep.sweep_type == "high":
            check.liquidity_sweep_found = True
            check.sweep_type = "high"
            break

    # 3. HTF FVG (ATR-normalized threshold)
    htf_fvgs = detect_fvgs(
        htf_bars,
        lookback=50,
        min_gap_atr_mult=config.fvg_min_atr_mult,
        atr_period=config.atr_period,
    )
    for fvg in htf_fvgs:
        if not fvg.invalidated(config.max_fvg_fill_pct):
            if (direction == BiasDirection.BULLISH and fvg.fvg_type == FVGType.BULLISH):
                check.htf_fvg_found = True
                check.fvg_size = fvg.gap_size
                break
            elif (direction == BiasDirection.BEARISH and fvg.fvg_type == FVGType.BEARISH):
                check.htf_fvg_found = True
                check.fvg_size = fvg.gap_size
                break

    # 4. Breaker/Mitigation Block (must overlap with HTF FVG, matching strategy logic)
    breakers = detect_breaker_blocks(htf_bars, lookback=50)
    mitigations = detect_mitigation_blocks(htf_bars, lookback=50)

    _, entry_block_check, _ = find_entry_zone(
        htf_fvgs, breakers, mitigations, direction, current_price,
        max_fvg_fill_pct=config.max_fvg_fill_pct,
    )
    check.breaker_block_found = entry_block_check is not None

    # 5. LTF FVG (ATR-normalized, looser threshold for LTF)
    ltf_fvgs = detect_fvgs(
        ltf_bars,
        lookback=30,
        min_gap_atr_mult=config.fvg_min_atr_mult * 0.5,  # 50% of HTF threshold
        atr_period=config.atr_period,
    )
    for fvg in ltf_fvgs:
        if not fvg.invalidated(config.max_fvg_fill_pct):
            if (direction == BiasDirection.BULLISH and fvg.fvg_type == FVGType.BULLISH):
                check.ltf_fvg_found = True
                break
            elif (direction == BiasDirection.BEARISH and fvg.fvg_type == FVGType.BEARISH):
                check.ltf_fvg_found = True
                break

    # 6. MSS
    mss_list = detect_mss(htf_bars, lookback=50)
    for mss in mss_list:
        if (direction == BiasDirection.BULLISH and mss.shift_type == "bullish"):
            check.mss_found = True
            break
        elif (direction == BiasDirection.BEARISH and mss.shift_type == "bearish"):
            check.mss_found = True
            break

    # 7. Stop validation (ATR-based max stop)
    max_points = get_max_stop_points(symbol, atr=current_atr, config=config)
    entry_fvg, entry_block, entry_price = find_entry_zone(
        htf_fvgs, breakers, mitigations, direction, current_price,
        max_fvg_fill_pct=config.max_fvg_fill_pct,
    )

    if entry_fvg:
        if direction == BiasDirection.BULLISH:
            stop_distance = current_price - entry_fvg.gap_low
        else:
            stop_distance = entry_fvg.gap_high - current_price

        check.stop_points = abs(stop_distance)
        check.stop_valid = check.stop_points <= max_points
    else:
        check.stop_points = max_points + 1  # Invalid
        check.stop_valid = False

    # 8. Macro window
    check.in_macro_window = is_in_macro_window(ts, profile=config.session_profile)

    return check


def run_unicorn_backtest(
    symbol: str,
    htf_bars: list[OHLCVBar],  # 15m bars
    ltf_bars: list[OHLCVBar],  # 5m bars
    dollars_per_trade: float = 1000.0,
    scan_interval: int = 1,  # Check every N bars
    max_concurrent_trades: int = 1,
    eod_exit: bool = True,
    eod_time: time = time(15, 45),
    min_criteria_score: int = 3,  # Soft scoring: enter at >= N of 5 scored criteria
    config: Optional[UnicornConfig] = None,
    # Friction parameters (critical for realistic backtesting)
    slippage_ticks: float = 1.0,  # Slippage per side in ticks
    commission_per_contract: float = 2.50,  # Round-trip commission per contract
    intrabar_policy: IntrabarPolicy = IntrabarPolicy.WORST,  # Stop/target ambiguity
    # Direction filter (diagnostics showed longs profitable, shorts not)
    direction_filter: Optional[BiasDirection] = None,  # None=both, BULLISH=long-only
    # Time-stop: exit if not profitable within N minutes
    time_stop_minutes: Optional[int] = None,  # None=disabled, e.g., 30=exit if not +0.25R in 30 min
    time_stop_r_threshold: float = 0.25,  # R-multiple to reach within time_stop_minutes
) -> UnicornBacktestResult:
    """
    Run Unicorn Model backtest with detailed analytics.

    Args:
        symbol: Trading symbol (e.g., "NQ", "ES")
        htf_bars: Higher timeframe bars (15m recommended)
        ltf_bars: Lower timeframe bars (5m recommended)
        dollars_per_trade: Risk amount per trade
        scan_interval: Check for setups every N bars
        max_concurrent_trades: Maximum simultaneous positions
        eod_exit: Exit all positions at EOD
        eod_time: Time for EOD exit
        min_criteria_score: Minimum SCORED criteria to enter (out of 5, not 8)
        config: Strategy configuration (ATR thresholds, session profile)
        slippage_ticks: Slippage per side in ticks (default 1 tick each way)
        commission_per_contract: Round-trip commission per contract (default $2.50)
        intrabar_policy: How to resolve stop/target ambiguity when both hit in
                        same bar. WORST=stop first (conservative), BEST=target
                        first (optimistic), RANDOM=50/50. Default WORST.
        direction_filter: Only take trades in this direction. None=both directions,
                         BULLISH=long-only (diagnostics showed longs profitable).
        time_stop_minutes: Exit if not at +time_stop_r_threshold R within N minutes.
                          None=disabled. Helps cut grindy losers early.
        time_stop_r_threshold: R-multiple to reach within time_stop_minutes (default 0.25R).

    Returns:
        UnicornBacktestResult with complete analytics

    Note on soft scoring:
        - 3 MANDATORY criteria must always pass: htf_bias, stop_valid, macro_window
        - 5 SCORED criteria: liquidity_sweep, htf_fvg, breaker_block, ltf_fvg, mss
        - min_criteria_score applies only to the scored set (default 3/5)
        - This prevents bypassing core risk management via scoring
    """
    if config is None:
        config = UnicornConfig(min_scored_criteria=min_criteria_score)
    else:
        config.min_scored_criteria = min_criteria_score

    if len(htf_bars) < 50 or len(ltf_bars) < 30:
        raise ValueError("Insufficient bars for backtest (need 50 HTF, 30 LTF minimum)")

    result = UnicornBacktestResult(
        symbol=symbol,
        start_date=htf_bars[0].ts,
        end_date=htf_bars[-1].ts,
        total_bars=len(htf_bars),
    )

    # Initialize session stats
    for session in TradingSession:
        result.session_stats[session] = SessionStats(session=session)

    # Criteria failure tracking
    criteria_fails: dict[str, int] = {
        "htf_bias": 0,
        "liquidity_sweep": 0,
        "htf_fvg": 0,
        "breaker_block": 0,
        "ltf_fvg": 0,
        "mss": 0,
        "stop_valid": 0,
        "macro_window": 0,
    }
    criteria_passes: dict[str, list[TradeRecord]] = {k: [] for k in criteria_fails}

    # Trade tracking
    open_trades: list[TradeRecord] = []
    point_value = get_point_value(symbol)

    # Tick size for slippage calculation (NQ/ES = 0.25)
    tick_size = 0.25
    slippage_points = slippage_ticks * tick_size  # Per side

    # ATR for volatility-normalized stops (computed per-bar in loop)
    htf_atr_values = _calculate_atr(htf_bars, config.atr_period)

    # Build LTF index for quick lookup
    ltf_by_time = {b.ts: b for b in ltf_bars}

    # Main backtest loop
    # Warmup needs extra bar because we use i-1 for signals
    warmup = 51  # Need history for indicators + 1 for causal signal

    for i in range(warmup, len(htf_bars), scan_interval):
        bar = htf_bars[i]
        ts = bar.ts

        # CRITICAL: Use bar OPEN for entry price (we know open at start of bar)
        # We do NOT use bar.close - that's future information
        entry_price_raw = bar.open

        # CRITICAL: Signal window uses PREVIOUS closed bars only (excludes current bar)
        # Pattern detection at time t uses data from bars [0..t-1]
        htf_window = htf_bars[max(0, i-100):i]  # Excludes bar[i]

        if len(htf_window) < 3:
            continue

        # CRITICAL: ATR for sizing uses previous bar's ATR (not current)
        # htf_atr_values[i] includes bar[i]'s TR, so use i-1
        signal_atr = htf_atr_values[i-1] if i > 0 else htf_atr_values[0]

        # Find corresponding LTF bars - must be BEFORE current HTF bar timestamp
        ltf_end_idx = None
        for j, lb in enumerate(ltf_bars):
            if lb.ts >= ts:
                ltf_end_idx = j
                break

        if ltf_end_idx is None:
            ltf_end_idx = len(ltf_bars)

        # LTF window: bars strictly before current timestamp
        ltf_window = ltf_bars[max(0, ltf_end_idx-60):ltf_end_idx]

        if not ltf_window:
            continue

        # For stop/target checks, we use the CURRENT bar's OHLC (this is correct -
        # we're checking if price hit levels during bar[i], not predicting)

        # Update open trades (MFE/MAE tracking + exit checks)
        # For exit checking, we use the current bar's OHLC (this is correct -
        # checking if stop/target were hit during bar execution is not look-ahead)
        for trade in open_trades:
            # MFE/MAE tracking (stays inline — not exit logic)
            if trade.direction == BiasDirection.BULLISH:
                unrealized = bar.close - trade.entry_price
            else:
                unrealized = trade.entry_price - bar.close
            if unrealized > trade.mfe:
                trade.mfe = unrealized
                trade.mfe_time = ts
            if unrealized < -trade.mae:
                trade.mae = abs(unrealized)
                trade.mae_time = ts

            # Exit resolution via extracted function
            exit_result = resolve_bar_exit(
                trade, bar, intrabar_policy, slippage_points,
                eod_exit=eod_exit, eod_time=eod_time,
                time_stop_minutes=time_stop_minutes,
                time_stop_r_threshold=time_stop_r_threshold,
            )
            if exit_result:
                trade.exit_price = exit_result.exit_price
                trade.exit_time = ts
                trade.exit_reason = exit_result.exit_reason
                trade.pnl_points = exit_result.pnl_points

        # Finalize closed trades (apply commission)
        closed = [t for t in open_trades if t.exit_time is not None]
        open_trades = [t for t in open_trades if t.exit_time is None]

        for trade in closed:
            # Calculate PnL including slippage (already in pnl_points) and commission
            commission_total = commission_per_contract * trade.quantity
            trade.pnl_dollars = (trade.pnl_points * point_value * trade.quantity) - commission_total
            trade.duration_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

            if trade.risk_points > 0:
                trade.r_multiple = trade.pnl_points / trade.risk_points

            if trade.pnl_dollars > 0:  # Use dollars (includes commission) for outcome
                trade.outcome = TradeOutcome.WIN
                result.wins += 1
            elif trade.pnl_dollars < 0:
                trade.outcome = TradeOutcome.LOSS
                result.losses += 1
            else:
                trade.outcome = TradeOutcome.BREAKEVEN
                result.breakevens += 1

            result.trades.append(trade)
            result.total_pnl_points += trade.pnl_points
            result.total_pnl_dollars += trade.pnl_dollars

            # Update session stats
            session_stat = result.session_stats[trade.session]
            session_stat.trades_taken += 1
            session_stat.total_pnl_points += trade.pnl_points
            if trade.outcome == TradeOutcome.WIN:
                session_stat.wins += 1
            elif trade.outcome == TradeOutcome.LOSS:
                session_stat.losses += 1

        # Check for new setups (only if we have capacity)
        if len(open_trades) >= max_concurrent_trades:
            continue

        result.total_setups_scanned += 1

        # Check all criteria
        criteria = check_criteria(
            bars=htf_window,
            htf_bars=htf_window,
            ltf_bars=ltf_window,
            symbol=symbol,
            ts=ts,
        )

        session = criteria.session
        result.session_stats[session].total_setups += 1

        # Track setup occurrence with parity diagnostics
        entry_decision = criteria.meets_entry_requirements(min_scored=min_criteria_score)
        setup_record = SetupOccurrence(
            timestamp=ts,
            direction=criteria.htf_bias_direction or BiasDirection.NEUTRAL,
            criteria=criteria,
            mandatory_met=criteria.mandatory_criteria_met,
            scored_count=criteria.scored_criteria_count,
            min_scored_required=min_criteria_score,
            decide_entry_result=entry_decision,
            scored_missing=[
                c for c in criteria.missing_criteria() if c in SCORED_CRITERIA
            ],
        )

        if criteria.criteria_met_count > 0:
            result.partial_setups += 1

        # Track which criteria fail
        for criterion in criteria.missing_criteria():
            criteria_fails[criterion] += 1

        # Track valid setups (all 8/8)
        if criteria.all_criteria_met:
            result.valid_setups += 1
            result.session_stats[session].valid_setups += 1

        # SOFT SCORING with mandatory/scored separation:
        # - 3 MANDATORY must all pass: htf_bias, stop_valid, macro_window
        # - 5 SCORED: min_criteria_score of these must pass
        if entry_decision:
            direction = criteria.htf_bias_direction

            # Direction filter: skip if direction doesn't match filter
            # Diagnostics showed longs are profitable, shorts bleed
            if direction_filter is not None and direction != direction_filter:
                setup_record.taken = False
                setup_record.reason_not_taken = f"direction_filter: {direction.value} != {direction_filter.value}"
                result.all_setups.append(setup_record)
                continue

            # --- Bar-quality guards ---
            # signal_bar is last closed HTF bar (not current forming bar)
            signal_bar = htf_window[-1]
            wick_ratio = compute_adverse_wick_ratio(signal_bar, direction)
            range_atr_mult = compute_range_atr_mult(signal_bar, signal_atr)

            # Always record diagnostics (even when guards disabled)
            setup_record.signal_wick_ratio = wick_ratio
            setup_record.signal_range_atr_mult = range_atr_mult

            # Wick guard
            if config.max_wick_ratio is not None and wick_ratio > config.max_wick_ratio:
                setup_record.taken = False
                setup_record.wick_guard_rejected = True
                setup_record.guard_reason_code = "wick_guard"
                setup_record.reason_not_taken = (
                    f"wick_guard: {wick_ratio:.2f} > {config.max_wick_ratio}"
                )
                result.all_setups.append(setup_record)
                continue

            # Range guard
            if config.max_range_atr_mult is not None and range_atr_mult > config.max_range_atr_mult:
                setup_record.taken = False
                setup_record.range_guard_rejected = True
                setup_record.guard_reason_code = "range_guard"
                setup_record.reason_not_taken = (
                    f"range_guard: {range_atr_mult:.1f}x ATR > {config.max_range_atr_mult}"
                )
                result.all_setups.append(setup_record)
                continue

            # CRITICAL: Use signal_atr (from previous bar) for sizing, not current bar
            # signal_atr was computed above from htf_atr_values[i-1]

            # Find FVG/blocks for stop placement (htf_window excludes current bar, causal)
            htf_fvgs = detect_fvgs(
                htf_window,
                lookback=50,
                min_gap_atr_mult=config.fvg_min_atr_mult,
                atr_period=config.atr_period,
            )
            breakers = detect_breaker_blocks(htf_window, lookback=50)
            mitigations = detect_mitigation_blocks(htf_window, lookback=50)

            # Get FVG info for stop/target calculation (but NOT for entry price)
            entry_fvg, entry_block, _ = find_entry_zone(
                htf_fvgs, breakers, mitigations, direction, entry_price_raw,
                max_fvg_fill_pct=config.max_fvg_fill_pct,
            )

            # CRITICAL: Entry is at MARKET (bar.open), not at theoretical FVG level
            # We can only enter at prices that actually traded
            # Apply slippage: worse price for our direction
            if direction == BiasDirection.BULLISH:
                entry_price = bar.open + slippage_points  # Buy at open + slip
            else:
                entry_price = bar.open - slippage_points  # Sell at open - slip

            # Calculate stop and target (ATR-based validation)
            # Uses signal_atr from previous closed bar
            sweeps = detect_liquidity_sweeps(htf_window, lookback=50)
            relevant_sweep = None
            for sweep in sweeps:
                if direction == BiasDirection.BULLISH and sweep.sweep_type == "low":
                    relevant_sweep = sweep
                    break
                elif direction == BiasDirection.BEARISH and sweep.sweep_type == "high":
                    relevant_sweep = sweep
                    break

            stop_price, target_price, risk_points, _ = calculate_stop_and_target(
                entry_price, direction, entry_fvg, relevant_sweep, symbol,
                atr=signal_atr, config=config,  # Use causal ATR
            )

            # Position sizing (account for slippage in risk)
            risk_dollars = (risk_points + slippage_points) * point_value
            if risk_dollars > 0:
                quantity = max(1, int(dollars_per_trade / risk_dollars))
            else:
                quantity = 1

            # Create trade
            trade = TradeRecord(
                entry_time=ts,
                entry_price=entry_price,
                direction=direction,
                quantity=quantity,
                session=session,
                criteria=criteria,
                stop_price=stop_price,
                target_price=target_price,
                risk_points=risk_points,
            )

            open_trades.append(trade)
            result.trades_taken += 1

            # Entry-bar exit check: stop/target may be pierced on the
            # same bar the trade was opened.  The general open-trades loop
            # above ran BEFORE this trade existed, so no double-process risk.
            entry_bar_exit = resolve_bar_exit(
                trade, bar, intrabar_policy, slippage_points,
                eod_exit=eod_exit, eod_time=eod_time,
                time_stop_minutes=time_stop_minutes,
                time_stop_r_threshold=time_stop_r_threshold,
            )
            if entry_bar_exit:
                trade.exit_price = entry_bar_exit.exit_price
                trade.exit_time = ts
                trade.exit_reason = entry_bar_exit.exit_reason
                trade.pnl_points = entry_bar_exit.pnl_points

            setup_record.taken = True
        else:
            setup_record.taken = False
            # Build detailed reason
            if not criteria.mandatory_criteria_met:
                missing_mandatory = []
                if not criteria.htf_bias_aligned:
                    missing_mandatory.append("htf_bias")
                if not criteria.stop_valid:
                    missing_mandatory.append("stop_valid")
                if not criteria.in_macro_window:
                    missing_mandatory.append("macro_window")
                setup_record.reason_not_taken = f"mandatory failed: {', '.join(missing_mandatory)}"
            else:
                setup_record.reason_not_taken = f"scored {criteria.scored_criteria_count}/5 < {min_criteria_score}"

        result.all_setups.append(setup_record)

    # Close any remaining open trades at last bar (with slippage + commission)
    for trade in open_trades:
        last_close = htf_bars[-1].close
        trade.exit_time = htf_bars[-1].ts
        trade.exit_reason = "backtest_end"

        # Apply slippage to exit (worse for trade direction)
        if trade.direction == BiasDirection.BULLISH:
            trade.exit_price = last_close - slippage_points
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.exit_price = last_close + slippage_points
            trade.pnl_points = trade.entry_price - trade.exit_price

        # Apply commission
        commission_total = commission_per_contract * trade.quantity
        trade.pnl_dollars = (trade.pnl_points * point_value * trade.quantity) - commission_total
        trade.duration_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

        if trade.risk_points > 0:
            trade.r_multiple = trade.pnl_points / trade.risk_points

        if trade.pnl_dollars > 0:  # Use dollars (includes commission) for outcome
            trade.outcome = TradeOutcome.WIN
            result.wins += 1
        elif trade.pnl_dollars < 0:
            trade.outcome = TradeOutcome.LOSS
            result.losses += 1
        else:
            trade.outcome = TradeOutcome.BREAKEVEN
            result.breakevens += 1

        result.trades.append(trade)
        result.total_pnl_points += trade.pnl_points
        result.total_pnl_dollars += trade.pnl_dollars

    # Calculate aggregate stats
    if result.trades:
        result.avg_mfe = statistics.mean(t.mfe for t in result.trades)
        result.avg_mae = statistics.mean(t.mae for t in result.trades)
        result.avg_r_multiple = statistics.mean(t.r_multiple for t in result.trades)
        result.best_r_multiple = max(t.r_multiple for t in result.trades)
        result.worst_r_multiple = min(t.r_multiple for t in result.trades)
        result.largest_win_points = max((t.pnl_points for t in result.trades), default=0)
        result.largest_loss_points = min((t.pnl_points for t in result.trades), default=0)

        # MFE capture rate
        mfe_captures = []
        for t in result.trades:
            if t.mfe > 0:
                mfe_captures.append(t.pnl_points / t.mfe if t.pnl_points > 0 else 0)
        result.mfe_capture_rate = statistics.mean(mfe_captures) if mfe_captures else 0

        # Session averages
        for session, stats in result.session_stats.items():
            session_trades = [t for t in result.trades if t.session == session]
            if session_trades:
                stats.avg_mfe = statistics.mean(t.mfe for t in session_trades)
                stats.avg_mae = statistics.mean(t.mae for t in session_trades)
                stats.avg_r_multiple = statistics.mean(t.r_multiple for t in session_trades)

    # Criteria bottleneck analysis
    total_checks = result.total_setups_scanned
    for criterion, fail_count in criteria_fails.items():
        bottleneck = CriteriaBottleneck(
            criterion=criterion,
            fail_count=fail_count,
            fail_rate=fail_count / total_checks if total_checks > 0 else 0,
        )
        result.criteria_bottlenecks.append(bottleneck)

    # Sort by fail rate (biggest bottleneck first)
    result.criteria_bottlenecks.sort(key=lambda x: x.fail_rate, reverse=True)

    # Confidence bucket analysis
    buckets = [
        (0.0, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    ]

    for min_conf, max_conf in buckets:
        bucket_trades = [
            t for t in result.trades
            if min_conf <= t.criteria.htf_bias_confidence < max_conf
        ]

        bucket = ConfidenceBucket(
            min_confidence=min_conf,
            max_confidence=max_conf,
            trade_count=len(bucket_trades),
            win_count=sum(1 for t in bucket_trades if t.outcome == TradeOutcome.WIN),
            total_pnl=sum(t.pnl_points for t in bucket_trades),
        )

        if bucket_trades:
            bucket.avg_r_multiple = statistics.mean(t.r_multiple for t in bucket_trades)

        result.confidence_buckets.append(bucket)

    # Confidence-win correlation (simple Pearson)
    if len(result.trades) >= 5:
        confidences = [t.criteria.htf_bias_confidence for t in result.trades]
        wins = [1.0 if t.outcome == TradeOutcome.WIN else 0.0 for t in result.trades]

        if len(set(confidences)) > 1 and len(set(wins)) > 1:
            mean_conf = statistics.mean(confidences)
            mean_win = statistics.mean(wins)

            numerator = sum((c - mean_conf) * (w - mean_win) for c, w in zip(confidences, wins))
            denom_conf = sum((c - mean_conf) ** 2 for c in confidences) ** 0.5
            denom_win = sum((w - mean_win) ** 2 for w in wins) ** 0.5

            if denom_conf > 0 and denom_win > 0:
                result.confidence_win_correlation = numerator / (denom_conf * denom_win)

    return result


def format_backtest_report(result: UnicornBacktestResult) -> str:
    """Format backtest results as a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"UNICORN MODEL BACKTEST REPORT - {result.symbol}")
    lines.append("=" * 70)
    lines.append(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    lines.append(f"Total bars analyzed: {result.total_bars}")
    lines.append("")

    # Setup Analysis
    lines.append("-" * 40)
    lines.append("SETUP ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Total setups scanned:  {result.total_setups_scanned}")
    lines.append(f"Partial setups:        {result.partial_setups} ({result.partial_setups/max(1,result.total_setups_scanned)*100:.1f}%)")
    lines.append(f"Valid setups (8/8):    {result.valid_setups} ({result.valid_setups/max(1,result.total_setups_scanned)*100:.1f}%)")
    lines.append(f"Trades taken:          {result.trades_taken}")
    lines.append(f"Setup→Trade ratio:     {result.setup_to_trade_ratio*100:.1f}%")
    lines.append("")

    # Trade Results
    lines.append("-" * 40)
    lines.append("TRADE RESULTS")
    lines.append("-" * 40)
    lines.append(f"Wins / Losses / BE:    {result.wins} / {result.losses} / {result.breakevens}")
    lines.append(f"Win rate:              {result.win_rate*100:.1f}%")
    lines.append(f"Profit factor:         {result.profit_factor:.2f}")
    lines.append(f"Total PnL (points):    {result.total_pnl_points:+.2f}")
    lines.append(f"Total PnL (dollars):   ${result.total_pnl_dollars:+,.2f}")
    lines.append(f"Expectancy/trade:      {result.expectancy_points:+.2f} points")
    lines.append(f"Largest win:           {result.largest_win_points:+.2f} points")
    lines.append(f"Largest loss:          {result.largest_loss_points:+.2f} points")
    lines.append("")

    # R-Multiple Analysis
    lines.append("-" * 40)
    lines.append("R-MULTIPLE ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Average R:             {result.avg_r_multiple:+.2f}R")
    lines.append(f"Best R:                {result.best_r_multiple:+.2f}R")
    lines.append(f"Worst R:               {result.worst_r_multiple:+.2f}R")
    lines.append("")

    # MFE/MAE Analysis
    lines.append("-" * 40)
    lines.append("MFE / MAE ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Avg MFE:               {result.avg_mfe:.2f} points")
    lines.append(f"Avg MAE:               {result.avg_mae:.2f} points")
    lines.append(f"MFE capture rate:      {result.mfe_capture_rate*100:.1f}%")
    lines.append("")

    # Session Breakdown
    lines.append("-" * 40)
    lines.append("SESSION BREAKDOWN")
    lines.append("-" * 40)
    lines.append(f"{'Session':<12} {'Setups':>8} {'Valid':>8} {'Trades':>8} {'WinRate':>8} {'PnL':>10}")
    lines.append("-" * 56)

    for session in [TradingSession.NY_AM, TradingSession.NY_PM, TradingSession.LONDON, TradingSession.ASIA]:
        stats = result.session_stats[session]
        lines.append(
            f"{session.value:<12} {stats.total_setups:>8} {stats.valid_setups:>8} "
            f"{stats.trades_taken:>8} {stats.win_rate*100:>7.1f}% {stats.total_pnl_points:>+10.2f}"
        )
    lines.append("")

    # Criteria Bottleneck
    lines.append("-" * 40)
    lines.append("CRITERIA BOTTLENECK (sorted by fail rate)")
    lines.append("-" * 40)
    for bottleneck in result.criteria_bottlenecks:
        bar = "█" * int(bottleneck.fail_rate * 20)
        lines.append(f"{bottleneck.criterion:<18} {bottleneck.fail_rate*100:>5.1f}% {bar}")
    lines.append("")

    # Confidence Correlation
    lines.append("-" * 40)
    lines.append("CONFIDENCE vs WIN RATE")
    lines.append("-" * 40)
    lines.append(f"{'Confidence':<15} {'Trades':>8} {'WinRate':>10} {'AvgR':>8} {'TotalPnL':>10}")
    lines.append("-" * 56)

    for bucket in result.confidence_buckets:
        conf_range = f"{bucket.min_confidence:.1f}-{bucket.max_confidence:.1f}"
        lines.append(
            f"{conf_range:<15} {bucket.trade_count:>8} {bucket.win_rate*100:>9.1f}% "
            f"{bucket.avg_r_multiple:>+7.2f}R {bucket.total_pnl:>+10.2f}"
        )

    lines.append("")
    lines.append(f"Confidence-Win Correlation: {result.confidence_win_correlation:+.3f}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
