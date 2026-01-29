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
import statistics

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
    get_max_stop_handles,
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
    stop_handles: float = 0.0
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
    risk_handles: float

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
    pnl_handles: float = 0.0
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


@dataclass
class SessionStats:
    """Statistics for a trading session."""
    session: TradingSession
    total_setups: int = 0
    valid_setups: int = 0  # All 8 criteria met
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_handles: float = 0.0
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
    total_pnl_handles: float = 0.0
    total_pnl_dollars: float = 0.0
    largest_win_handles: float = 0.0
    largest_loss_handles: float = 0.0

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
        gross_profit = sum(t.pnl_handles for t in self.trades if t.pnl_handles > 0)
        gross_loss = abs(sum(t.pnl_handles for t in self.trades if t.pnl_handles < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def expectancy_handles(self) -> float:
        """Expected value per trade in handles."""
        if self.trades_taken == 0:
            return 0.0
        return self.total_pnl_handles / self.trades_taken

    @property
    def setup_to_trade_ratio(self) -> float:
        """How often valid setups become trades."""
        if self.valid_setups == 0:
            return 0.0
        return self.trades_taken / self.valid_setups


def classify_session(ts: datetime) -> TradingSession:
    """Classify timestamp into trading session."""
    t = ts.time()

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
        if not fvg.filled:
            if (direction == BiasDirection.BULLISH and fvg.fvg_type == FVGType.BULLISH):
                check.htf_fvg_found = True
                check.fvg_size = fvg.gap_size
                break
            elif (direction == BiasDirection.BEARISH and fvg.fvg_type == FVGType.BEARISH):
                check.htf_fvg_found = True
                check.fvg_size = fvg.gap_size
                break

    # 4. Breaker/Mitigation Block
    breakers = detect_breaker_blocks(htf_bars, lookback=50)
    mitigations = detect_mitigation_blocks(htf_bars, lookback=50)

    for breaker in breakers:
        if (direction == BiasDirection.BULLISH and breaker.block_type == BlockType.BULLISH):
            check.breaker_block_found = True
            break
        elif (direction == BiasDirection.BEARISH and breaker.block_type == BlockType.BEARISH):
            check.breaker_block_found = True
            break

    if not check.breaker_block_found:
        for mit in mitigations:
            if (direction == BiasDirection.BULLISH and mit.block_type == BlockType.BULLISH):
                check.breaker_block_found = True
                break
            elif (direction == BiasDirection.BEARISH and mit.block_type == BlockType.BEARISH):
                check.breaker_block_found = True
                break

    # 5. LTF FVG (ATR-normalized, looser threshold for LTF)
    ltf_fvgs = detect_fvgs(
        ltf_bars,
        lookback=30,
        min_gap_atr_mult=config.fvg_min_atr_mult * 0.5,  # 50% of HTF threshold
        atr_period=config.atr_period,
    )
    for fvg in ltf_fvgs:
        if not fvg.filled:
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
    max_handles = get_max_stop_handles(symbol, atr=current_atr, config=config)
    entry_fvg, entry_block, entry_price = find_entry_zone(
        htf_fvgs, breakers, mitigations, direction, current_price
    )

    if entry_fvg:
        if direction == BiasDirection.BULLISH:
            stop_distance = current_price - entry_fvg.gap_low
        else:
            stop_distance = entry_fvg.gap_high - current_price

        check.stop_handles = abs(stop_distance)
        check.stop_valid = check.stop_handles <= max_handles
    else:
        check.stop_handles = max_handles + 1  # Invalid
        check.stop_valid = False

    # 8. Macro window
    check.in_macro_window = is_in_macro_window(ts)

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
    min_criteria_score: int = 6,  # Soft scoring: enter at >= N/8
    config: Optional[UnicornConfig] = None,
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
        min_criteria_score: Minimum criteria to enter (default 6/8)
        config: Strategy configuration (ATR thresholds, session profile)

    Returns:
        UnicornBacktestResult with complete analytics
    """
    if config is None:
        config = UnicornConfig(min_criteria_score=min_criteria_score)
    else:
        config.min_criteria_score = min_criteria_score

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

    # ATR for volatility-normalized stops (computed per-bar in loop)
    htf_atr_values = _calculate_atr(htf_bars, config.atr_period)

    # Build LTF index for quick lookup
    ltf_by_time = {b.ts: b for b in ltf_bars}

    # Main backtest loop
    warmup = 50  # Need history for indicators

    for i in range(warmup, len(htf_bars), scan_interval):
        bar = htf_bars[i]
        ts = bar.ts
        current_price = bar.close

        # Get historical windows
        htf_window = htf_bars[max(0, i-100):i+1]

        # Find corresponding LTF bars
        ltf_end_idx = None
        for j, lb in enumerate(ltf_bars):
            if lb.ts >= ts:
                ltf_end_idx = j
                break

        if ltf_end_idx is None:
            ltf_end_idx = len(ltf_bars)

        ltf_window = ltf_bars[max(0, ltf_end_idx-60):ltf_end_idx]

        if not ltf_window:
            continue

        # Update open trades (MFE/MAE tracking)
        for trade in open_trades:
            if trade.direction == BiasDirection.BULLISH:
                # Long trade
                unrealized = current_price - trade.entry_price
                if unrealized > trade.mfe:
                    trade.mfe = unrealized
                    trade.mfe_time = ts
                if unrealized < -trade.mae:
                    trade.mae = abs(unrealized)
                    trade.mae_time = ts

                # Check stop hit
                if bar.low <= trade.stop_price:
                    trade.exit_price = trade.stop_price
                    trade.exit_time = ts
                    trade.exit_reason = "stop_loss"
                    trade.pnl_handles = trade.stop_price - trade.entry_price

                # Check target hit
                elif bar.high >= trade.target_price:
                    trade.exit_price = trade.target_price
                    trade.exit_time = ts
                    trade.exit_reason = "target"
                    trade.pnl_handles = trade.target_price - trade.entry_price

            else:  # Short trade
                unrealized = trade.entry_price - current_price
                if unrealized > trade.mfe:
                    trade.mfe = unrealized
                    trade.mfe_time = ts
                if unrealized < -trade.mae:
                    trade.mae = abs(unrealized)
                    trade.mae_time = ts

                # Check stop hit
                if bar.high >= trade.stop_price:
                    trade.exit_price = trade.stop_price
                    trade.exit_time = ts
                    trade.exit_reason = "stop_loss"
                    trade.pnl_handles = trade.entry_price - trade.stop_price

                # Check target hit
                elif bar.low <= trade.target_price:
                    trade.exit_price = trade.target_price
                    trade.exit_time = ts
                    trade.exit_reason = "target"
                    trade.pnl_handles = trade.entry_price - trade.target_price

            # EOD exit
            if eod_exit and ts.time() >= eod_time and trade.exit_time is None:
                trade.exit_price = current_price
                trade.exit_time = ts
                trade.exit_reason = "eod"
                if trade.direction == BiasDirection.BULLISH:
                    trade.pnl_handles = current_price - trade.entry_price
                else:
                    trade.pnl_handles = trade.entry_price - current_price

        # Finalize closed trades
        closed = [t for t in open_trades if t.exit_time is not None]
        open_trades = [t for t in open_trades if t.exit_time is None]

        for trade in closed:
            trade.pnl_dollars = trade.pnl_handles * point_value * trade.quantity
            trade.duration_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

            if trade.risk_handles > 0:
                trade.r_multiple = trade.pnl_handles / trade.risk_handles

            if trade.pnl_handles > 0:
                trade.outcome = TradeOutcome.WIN
                result.wins += 1
            elif trade.pnl_handles < 0:
                trade.outcome = TradeOutcome.LOSS
                result.losses += 1
            else:
                trade.outcome = TradeOutcome.BREAKEVEN
                result.breakevens += 1

            result.trades.append(trade)
            result.total_pnl_handles += trade.pnl_handles
            result.total_pnl_dollars += trade.pnl_dollars

            # Update session stats
            session_stat = result.session_stats[trade.session]
            session_stat.trades_taken += 1
            session_stat.total_pnl_handles += trade.pnl_handles
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

        # Track setup occurrence
        setup_record = SetupOccurrence(
            timestamp=ts,
            direction=criteria.htf_bias_direction or BiasDirection.NEUTRAL,
            criteria=criteria,
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

        # SOFT SCORING: Enter if score >= threshold (configurable, default 6/8)
        if criteria.criteria_met_count >= min_criteria_score:
            direction = criteria.htf_bias_direction

            # Get current ATR for volatility-normalized calculations
            current_atr = htf_atr_values[i] if i < len(htf_atr_values) else htf_atr_values[-1]

            # Find entry zone (with ATR-normalized FVG detection)
            htf_fvgs = detect_fvgs(
                htf_window,
                lookback=50,
                min_gap_atr_mult=config.fvg_min_atr_mult,
                atr_period=config.atr_period,
            )
            breakers = detect_breaker_blocks(htf_window, lookback=50)
            mitigations = detect_mitigation_blocks(htf_window, lookback=50)

            entry_fvg, entry_block, entry_price = find_entry_zone(
                htf_fvgs, breakers, mitigations, direction, current_price
            )

            # Calculate stop and target (ATR-based validation)
            sweeps = detect_liquidity_sweeps(htf_window, lookback=50)
            relevant_sweep = None
            for sweep in sweeps:
                if direction == BiasDirection.BULLISH and sweep.sweep_type == "low":
                    relevant_sweep = sweep
                    break
                elif direction == BiasDirection.BEARISH and sweep.sweep_type == "high":
                    relevant_sweep = sweep
                    break

            stop_price, target_price, risk_handles, _ = calculate_stop_and_target(
                entry_price, direction, entry_fvg, relevant_sweep, symbol,
                atr=current_atr, config=config,
            )

            # Position sizing
            risk_dollars = risk_handles * point_value
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
                risk_handles=risk_handles,
            )

            open_trades.append(trade)
            result.trades_taken += 1

            setup_record.taken = True
        else:
            setup_record.taken = False
            setup_record.reason_not_taken = f"score {criteria.criteria_met_count}/8 < {min_criteria_score}"

        result.all_setups.append(setup_record)

    # Close any remaining open trades at last bar
    for trade in open_trades:
        trade.exit_price = htf_bars[-1].close
        trade.exit_time = htf_bars[-1].ts
        trade.exit_reason = "backtest_end"
        if trade.direction == BiasDirection.BULLISH:
            trade.pnl_handles = trade.exit_price - trade.entry_price
        else:
            trade.pnl_handles = trade.entry_price - trade.exit_price

        trade.pnl_dollars = trade.pnl_handles * point_value * trade.quantity
        trade.duration_minutes = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

        if trade.risk_handles > 0:
            trade.r_multiple = trade.pnl_handles / trade.risk_handles

        if trade.pnl_handles > 0:
            trade.outcome = TradeOutcome.WIN
            result.wins += 1
        elif trade.pnl_handles < 0:
            trade.outcome = TradeOutcome.LOSS
            result.losses += 1
        else:
            trade.outcome = TradeOutcome.BREAKEVEN
            result.breakevens += 1

        result.trades.append(trade)
        result.total_pnl_handles += trade.pnl_handles
        result.total_pnl_dollars += trade.pnl_dollars

    # Calculate aggregate stats
    if result.trades:
        result.avg_mfe = statistics.mean(t.mfe for t in result.trades)
        result.avg_mae = statistics.mean(t.mae for t in result.trades)
        result.avg_r_multiple = statistics.mean(t.r_multiple for t in result.trades)
        result.best_r_multiple = max(t.r_multiple for t in result.trades)
        result.worst_r_multiple = min(t.r_multiple for t in result.trades)
        result.largest_win_handles = max((t.pnl_handles for t in result.trades), default=0)
        result.largest_loss_handles = min((t.pnl_handles for t in result.trades), default=0)

        # MFE capture rate
        mfe_captures = []
        for t in result.trades:
            if t.mfe > 0:
                mfe_captures.append(t.pnl_handles / t.mfe if t.pnl_handles > 0 else 0)
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
            total_pnl=sum(t.pnl_handles for t in bucket_trades),
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
    lines.append(f"Total PnL (handles):   {result.total_pnl_handles:+.2f}")
    lines.append(f"Total PnL (dollars):   ${result.total_pnl_dollars:+,.2f}")
    lines.append(f"Expectancy/trade:      {result.expectancy_handles:+.2f} handles")
    lines.append(f"Largest win:           {result.largest_win_handles:+.2f} handles")
    lines.append(f"Largest loss:          {result.largest_loss_handles:+.2f} handles")
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
    lines.append(f"Avg MFE:               {result.avg_mfe:.2f} handles")
    lines.append(f"Avg MAE:               {result.avg_mae:.2f} handles")
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
            f"{stats.trades_taken:>8} {stats.win_rate*100:>7.1f}% {stats.total_pnl_handles:>+10.2f}"
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
