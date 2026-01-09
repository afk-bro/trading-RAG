"""
Duration stats backfill job.

Runs hysteresis FSM over historical OHLCV to segment regimes and compute
duration distributions by (symbol, timeframe, regime_key).

Used for v1.5 Live Intelligence regime persistence estimation:
- regime_half_life_bars (median historical duration)
- expected_remaining_bars (median - current_age)
- duration_iqr_bars ([p25, p75] historical duration)
"""

from dataclasses import dataclass, field
from statistics import median, quantiles
from typing import Optional

import structlog

from app.repositories.duration_stats import DurationStats, DurationStatsRepository
from app.services.kb.regime_fsm import FSMConfig, RegimeFSM
from app.services.kb.regime_key import RegimeDims, canonicalize_regime_key

logger = structlog.get_logger(__name__)

# Default values
DEFAULT_LOOKBACK = 20
DEFAULT_MIN_SEGMENT_BARS = 5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BackfillResult:
    """Result of duration stats backfill job."""

    symbol: str = ""
    timeframe: str = ""
    bars_processed: int = 0
    segments_found: int = 0
    segments_filtered: int = 0  # Filtered due to min_segment_bars
    stats_written: int = 0
    stats_would_write: int = 0  # For dry run
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class RegimeSegment:
    """A continuous segment of a single regime."""

    regime_key: str
    start_bar: int
    end_bar: int

    @property
    def duration_bars(self) -> int:
        """Duration in bars (inclusive of both endpoints)."""
        return self.end_bar - self.start_bar + 1


# =============================================================================
# OHLCV Analysis Helpers
# =============================================================================


def compute_atr_pct(bars: list[dict]) -> float:
    """
    Compute Average True Range as percentage of price.

    Args:
        bars: List of OHLCV dicts with 'high', 'low', 'close' keys

    Returns:
        ATR as percentage (e.g., 0.02 = 2%)
    """
    if not bars:
        return 0.0

    if len(bars) == 1:
        bar = bars[0]
        hl_range = bar["high"] - bar["low"]
        avg_price = (bar["high"] + bar["low"]) / 2
        return hl_range / avg_price if avg_price > 0 else 0.0

    # Compute true ranges
    true_ranges = []
    for i in range(1, len(bars)):
        current = bars[i]
        prev_close = bars[i - 1]["close"]

        hl = current["high"] - current["low"]
        hc = abs(current["high"] - prev_close)
        lc = abs(current["low"] - prev_close)

        true_ranges.append(max(hl, hc, lc))

    if not true_ranges:
        return 0.0

    atr = sum(true_ranges) / len(true_ranges)

    # Get average price for percentage
    avg_price = sum((b["high"] + b["low"]) / 2 for b in bars) / len(bars)

    return atr / avg_price if avg_price > 0 else 0.0


def classify_regime_from_ohlcv(
    ohlcv: list[dict],
    index: int,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[Optional[str], float]:
    """
    Classify regime at given index using lookback window.

    Uses simple feature-based classification:
    - trend: based on average returns
    - vol: based on ATR percentile

    Args:
        ohlcv: Full OHLCV bar list
        index: Current bar index to classify
        lookback: Number of bars to use for classification

    Returns:
        (regime_key, confidence) or (None, 0.0) if insufficient data
    """
    if index < lookback:
        return None, 0.0

    window = ohlcv[index - lookback : index + 1]

    if len(window) < lookback:
        return None, 0.0

    # Trend classification based on average returns
    returns = []
    for i in range(1, len(window)):
        if window[i - 1]["close"] != 0:
            ret = (window[i]["close"] - window[i - 1]["close"]) / window[i - 1]["close"]
            returns.append(ret)

    if not returns:
        return None, 0.0

    avg_return = sum(returns) / len(returns)

    # Trend thresholds
    if avg_return > 0.001:  # >0.1% average return
        trend = "uptrend"
    elif avg_return < -0.001:  # <-0.1% average return
        trend = "downtrend"
    else:
        trend = "flat"

    # Vol classification based on ATR
    atr_pct = compute_atr_pct(window)

    if atr_pct < 0.01:  # <1%
        vol = "low_vol"
    elif atr_pct < 0.025:  # <2.5%
        vol = "mid_vol"
    else:  # >=2.5%
        vol = "high_vol"

    # Compute confidence based on signal strength
    # Stronger trend signal = higher confidence
    trend_strength = min(abs(avg_return) / 0.003, 1.0)  # Saturates at 0.3%
    vol_strength = min(atr_pct / 0.04, 1.0)  # Saturates at 4%
    confidence = 0.5 + 0.5 * ((trend_strength + vol_strength) / 2)

    # Canonical regime key
    dims = RegimeDims(trend=trend, vol=vol)
    regime_key = canonicalize_regime_key(dims)

    return regime_key, confidence


# =============================================================================
# Segment Extraction
# =============================================================================


def extract_regime_segments(
    ohlcv: list[dict],
    fsm_config: FSMConfig,
    lookback: int = DEFAULT_LOOKBACK,
) -> list[RegimeSegment]:
    """
    Extract regime segments from OHLCV data using hysteresis FSM.

    Args:
        ohlcv: List of OHLCV dicts
        fsm_config: FSM configuration
        lookback: Bars needed for classification

    Returns:
        List of RegimeSegment instances
    """
    if not ohlcv:
        return []

    if len(ohlcv) <= lookback:
        return []

    segments: list[RegimeSegment] = []
    fsm = RegimeFSM(fsm_config)

    current_segment_start: Optional[int] = None
    current_regime_key: Optional[str] = None

    for i in range(lookback, len(ohlcv)):
        regime_key, confidence = classify_regime_from_ohlcv(ohlcv, i, lookback)

        if regime_key is None:
            continue

        # Update FSM
        transition = fsm.update(regime_key, confidence)

        state = fsm.get_state()
        stable_key = state.stable_regime_key

        # First stable regime
        if current_regime_key is None and stable_key is not None:
            current_regime_key = stable_key
            current_segment_start = i
            continue

        # Regime transition confirmed
        if transition is not None:
            # Close previous segment
            if current_regime_key is not None and current_segment_start is not None:
                segments.append(
                    RegimeSegment(
                        regime_key=current_regime_key,
                        start_bar=current_segment_start,
                        end_bar=i - 1,  # End at bar before transition
                    )
                )

            # Start new segment
            current_regime_key = transition.to_key
            current_segment_start = i

    # Close final segment
    if current_regime_key is not None and current_segment_start is not None:
        segments.append(
            RegimeSegment(
                regime_key=current_regime_key,
                start_bar=current_segment_start,
                end_bar=len(ohlcv) - 1,
            )
        )

    return segments


# =============================================================================
# Duration Aggregation
# =============================================================================


def aggregate_segment_durations(
    segments: list[RegimeSegment],
    min_segment_bars: int = DEFAULT_MIN_SEGMENT_BARS,
) -> dict[str, dict]:
    """
    Aggregate segment durations by regime key.

    Args:
        segments: List of regime segments
        min_segment_bars: Minimum segment duration to include

    Returns:
        Dict mapping regime_key -> {n_segments, median, p25, p75}
    """
    if not segments:
        return {}

    # Group durations by regime key
    durations_by_key: dict[str, list[int]] = {}

    for segment in segments:
        duration = segment.duration_bars
        if duration < min_segment_bars:
            continue

        if segment.regime_key not in durations_by_key:
            durations_by_key[segment.regime_key] = []

        durations_by_key[segment.regime_key].append(duration)

    # Compute percentiles
    result = {}
    for regime_key, durations in durations_by_key.items():
        if not durations:
            continue

        n = len(durations)
        sorted_durations = sorted(durations)

        # Compute percentiles
        if n == 1:
            med = sorted_durations[0]
            p25 = sorted_durations[0]
            p75 = sorted_durations[0]
        elif n == 2:
            med = (sorted_durations[0] + sorted_durations[1]) // 2
            p25 = sorted_durations[0]
            p75 = sorted_durations[1]
        else:
            # Use Python's quantiles for n >= 3
            try:
                quartiles = quantiles(sorted_durations, n=4)
                p25 = int(quartiles[0])
                med = int(quartiles[1])
                p75 = int(quartiles[2])
            except Exception:
                # Fallback for edge cases
                med = int(median(sorted_durations))
                p25 = sorted_durations[n // 4] if n >= 4 else sorted_durations[0]
                p75 = sorted_durations[(3 * n) // 4] if n >= 4 else sorted_durations[-1]

        result[regime_key] = {
            "n_segments": n,
            "median_duration_bars": med,
            "p25_duration_bars": p25,
            "p75_duration_bars": p75,
        }

    return result


# =============================================================================
# Main Backfill Job
# =============================================================================


async def run_duration_stats_backfill(
    db_pool,
    symbol: str,
    timeframe: str,
    ohlcv_data: list[dict],
    fsm_config: Optional[FSMConfig] = None,
    dry_run: bool = False,
    min_segment_bars: int = DEFAULT_MIN_SEGMENT_BARS,
    lookback: int = DEFAULT_LOOKBACK,
) -> BackfillResult:
    """
    Backfill duration stats from OHLCV data.

    Runs hysteresis FSM over OHLCV to segment regimes, then aggregates
    duration distributions by regime_key.

    Args:
        db_pool: asyncpg connection pool
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframe: Timeframe string (e.g., "5m", "1h")
        ohlcv_data: List of OHLCV dicts
        fsm_config: FSM configuration (uses defaults if None)
        dry_run: If True, don't write to DB
        min_segment_bars: Minimum segment duration to include
        lookback: Bars needed for classification

    Returns:
        BackfillResult with counts and errors
    """
    result = BackfillResult(
        symbol=symbol,
        timeframe=timeframe,
        dry_run=dry_run,
    )

    if fsm_config is None:
        fsm_config = FSMConfig()

    logger.info(
        "duration_stats_backfill_started",
        symbol=symbol,
        timeframe=timeframe,
        n_bars=len(ohlcv_data),
        dry_run=dry_run,
        min_segment_bars=min_segment_bars,
        lookback=lookback,
    )

    result.bars_processed = len(ohlcv_data)

    # Extract segments
    try:
        segments = extract_regime_segments(ohlcv_data, fsm_config, lookback)
        result.segments_found = len(segments)

        logger.debug(
            "duration_stats_segments_extracted",
            n_segments=len(segments),
        )
    except Exception as e:
        error_msg = f"Failed to extract segments: {str(e)}"
        logger.error("duration_stats_segment_error", error=str(e))
        result.errors.append(error_msg)
        return result

    # Count filtered segments
    original_count = len(segments)
    segments_after_filter = [s for s in segments if s.duration_bars >= min_segment_bars]
    result.segments_filtered = original_count - len(segments_after_filter)

    # Aggregate durations
    aggregated = aggregate_segment_durations(segments, min_segment_bars)

    logger.info(
        "duration_stats_aggregated",
        n_regime_keys=len(aggregated),
        segments_filtered=result.segments_filtered,
    )

    # Write or count stats
    if dry_run:
        result.stats_would_write = len(aggregated)
        logger.info(
            "duration_stats_backfill_dry_run_complete",
            would_write=result.stats_would_write,
        )
    else:
        repo = DurationStatsRepository(db_pool)

        for regime_key, stats in aggregated.items():
            try:
                duration_stats = DurationStats(
                    symbol=symbol,
                    timeframe=timeframe,
                    regime_key=regime_key,
                    n_segments=stats["n_segments"],
                    median_duration_bars=stats["median_duration_bars"],
                    p25_duration_bars=stats["p25_duration_bars"],
                    p75_duration_bars=stats["p75_duration_bars"],
                )
                await repo.upsert_stats(duration_stats)
                result.stats_written += 1
            except Exception as e:
                error_msg = f"Failed to upsert stats for {regime_key}: {str(e)}"
                logger.error(
                    "duration_stats_upsert_error",
                    regime_key=regime_key,
                    error=str(e),
                )
                result.errors.append(error_msg)

        logger.info(
            "duration_stats_backfill_complete",
            symbol=symbol,
            timeframe=timeframe,
            stats_written=result.stats_written,
            errors=len(result.errors),
        )

    return result
