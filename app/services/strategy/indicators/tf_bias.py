"""
Multi-Timeframe Bias System.

Implements a hierarchical bias determination across timeframes:
- Daily: Macro gate (close vs EMA(200) or VWAP)
- 4H: Primary bias (EMA slope, Efficiency Ratio)
- 1H: Confirmation (EMA(20) vs EMA(50))
- 15m: Momentum health (RSI, VWAP slope)
- 5m: Micro-structure (HH/HL patterns)

Each timeframe contributes to a final bias with confidence scoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from app.services.strategy.models import OHLCVBar


class BiasDirection(str, Enum):
    """Bias direction for trading."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class BiasStrength(str, Enum):
    """Strength of bias signal."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class TimeframeBiasComponent:
    """Bias component from a single timeframe."""

    timeframe: str
    direction: BiasDirection
    strength: BiasStrength
    confidence: float  # 0.0 to 1.0
    factors: dict = field(default_factory=dict)  # Debug info


@dataclass
class TimeframeBias:
    """
    Complete multi-timeframe bias result.

    Combines signals from all timeframes into a final bias
    with confidence scoring.
    """

    final_direction: BiasDirection
    final_confidence: float  # 0.0 to 1.0
    final_strength: BiasStrength
    timestamp: datetime

    # Component biases
    daily_bias: Optional[TimeframeBiasComponent] = None
    h4_bias: Optional[TimeframeBiasComponent] = None
    h1_bias: Optional[TimeframeBiasComponent] = None
    m15_bias: Optional[TimeframeBiasComponent] = None
    m5_bias: Optional[TimeframeBiasComponent] = None

    # Alignment
    alignment_score: float = 0.0  # How aligned are all timeframes
    conflicting_timeframes: list[str] = field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        """Check if bias is clear enough to trade."""
        return (
            self.final_direction != BiasDirection.NEUTRAL
            and self.final_confidence >= 0.6
            and self.alignment_score >= 0.6
        )


def compute_ema(prices: list[float], period: int) -> list[float]:
    """
    Compute Exponential Moving Average.

    Args:
        prices: Price series
        period: EMA period

    Returns:
        List of EMA values
    """
    if not prices:
        return []

    ema: list[float] = []
    multiplier = 2 / (period + 1)

    for i, price in enumerate(prices):
        if i == 0:
            ema.append(price)
        elif i < period:
            # Use SMA for initial warmup
            ema.append(sum(prices[: i + 1]) / (i + 1))
        else:
            ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema


def compute_sma(prices: list[float], period: int) -> list[float]:
    """
    Compute Simple Moving Average.

    Args:
        prices: Price series
        period: SMA period

    Returns:
        List of SMA values
    """
    if len(prices) < period:
        return [sum(prices) / len(prices)] * len(prices) if prices else []

    sma: list[float] = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(sum(prices[: i + 1]) / (i + 1))
        else:
            sma.append(sum(prices[i - period + 1 : i + 1]) / period)

    return sma


def compute_efficiency_ratio(
    prices: list[float],
    period: int = 10,
) -> list[float]:
    """
    Compute Kaufman Efficiency Ratio (ER).

    ER = Direction / Volatility
    Where:
    - Direction = abs(close - close[period])
    - Volatility = sum(abs(close - close[1])) over period

    ER ranges from 0 (choppy/ranging) to 1 (trending)

    Args:
        prices: Price series
        period: Lookback period

    Returns:
        List of ER values (0-1)
    """
    if len(prices) < period + 1:
        return [0.0] * len(prices)

    er_values: list[float] = [0.0] * period

    for i in range(period, len(prices)):
        # Direction: absolute price change over period
        direction = abs(prices[i] - prices[i - period])

        # Volatility: sum of absolute bar-to-bar changes
        volatility = sum(
            abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1)
        )

        if volatility > 0:
            er = direction / volatility
        else:
            er = 0.0

        er_values.append(min(1.0, max(0.0, er)))

    return er_values


def compute_rsi(prices: list[float], period: int = 14) -> list[float]:
    """
    Compute Relative Strength Index.

    Args:
        prices: Price series
        period: RSI period

    Returns:
        List of RSI values (0-100)
    """
    if len(prices) < 2:
        return [50.0] * len(prices)

    gains: list[float] = []
    losses: list[float] = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(0, change))
        losses.append(max(0, -change))

    # Calculate average gains and losses using EMA
    rsi_values: list[float] = [50.0]  # First value
    avg_gain = sum(gains[:period]) / period if len(gains) >= period else 0
    avg_loss = sum(losses[:period]) / period if len(losses) >= period else 0

    for i in range(period - 1):
        rsi_values.append(50.0)  # Warmup period

    multiplier = 1 / period

    for i in range(period - 1, len(gains)):
        if i >= period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi_values.append(rsi)

    return rsi_values


def compute_vwap(bars: list[OHLCVBar]) -> list[float]:
    """
    Compute Volume Weighted Average Price (session-based).

    Args:
        bars: OHLCV bars

    Returns:
        List of VWAP values
    """
    if not bars:
        return []

    vwap_values: list[float] = []
    cumulative_tp_vol = 0.0
    cumulative_vol = 0.0

    for bar in bars:
        typical_price = (bar.high + bar.low + bar.close) / 3
        cumulative_tp_vol += typical_price * bar.volume
        cumulative_vol += bar.volume

        if cumulative_vol > 0:
            vwap_values.append(cumulative_tp_vol / cumulative_vol)
        else:
            vwap_values.append(typical_price)

    return vwap_values


def detect_hh_hl_pattern(
    bars: list[OHLCVBar],
    lookback: int = 10,
) -> tuple[BiasDirection, float]:
    """
    Detect Higher Highs / Higher Lows (bullish) or Lower Highs / Lower Lows (bearish).

    Args:
        bars: OHLCV bars
        lookback: Number of bars to analyze

    Returns:
        Tuple of (direction, confidence)
    """
    if len(bars) < lookback:
        return BiasDirection.NEUTRAL, 0.0

    recent = bars[-lookback:]

    # Find swing points (simplified - use local extremes)
    highs: list[float] = []
    lows: list[float] = []

    for i in range(1, len(recent) - 1):
        if recent[i].high > recent[i - 1].high and recent[i].high > recent[i + 1].high:
            highs.append(recent[i].high)
        if recent[i].low < recent[i - 1].low and recent[i].low < recent[i + 1].low:
            lows.append(recent[i].low)

    if len(highs) < 2 or len(lows) < 2:
        return BiasDirection.NEUTRAL, 0.0

    # Check for HH/HL (bullish) or LH/LL (bearish)
    hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
    lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
    ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])

    bullish_score = (hh_count + hl_count) / (len(highs) + len(lows) - 2)
    bearish_score = (lh_count + ll_count) / (len(highs) + len(lows) - 2)

    if bullish_score > 0.6:
        return BiasDirection.BULLISH, bullish_score
    elif bearish_score > 0.6:
        return BiasDirection.BEARISH, bearish_score
    else:
        return BiasDirection.NEUTRAL, max(bullish_score, bearish_score)


def _compute_ema_slope(ema_values: list[float], period: int = 5) -> float:
    """Compute the slope of EMA over recent bars."""
    if len(ema_values) < period:
        return 0.0

    recent = ema_values[-period:]
    if recent[0] == 0:
        return 0.0

    # Normalized slope (percent change per bar)
    return (recent[-1] - recent[0]) / (recent[0] * period)


def compute_daily_bias(
    bars: list[OHLCVBar],
    ema_period: int = 200,
) -> TimeframeBiasComponent:
    """
    Compute Daily timeframe bias (Macro Gate).

    Criteria:
    - Close vs EMA(200)
    - Provides the overarching market direction

    Args:
        bars: Daily OHLCV bars
        ema_period: EMA period (default 200)

    Returns:
        TimeframeBiasComponent for daily
    """
    if len(bars) < ema_period:
        return TimeframeBiasComponent(
            timeframe="daily",
            direction=BiasDirection.NEUTRAL,
            strength=BiasStrength.WEAK,
            confidence=0.0,
            factors={"error": "insufficient_data"},
        )

    closes = [b.close for b in bars]
    ema200 = compute_ema(closes, ema_period)

    current_close = closes[-1]
    current_ema = ema200[-1]

    # Calculate distance from EMA as percentage
    distance_pct = (current_close - current_ema) / current_ema * 100

    # Determine direction
    if distance_pct > 0.5:
        direction = BiasDirection.BULLISH
    elif distance_pct < -0.5:
        direction = BiasDirection.BEARISH
    else:
        direction = BiasDirection.NEUTRAL

    # Determine strength based on distance
    abs_distance = abs(distance_pct)
    if abs_distance > 3.0:
        strength = BiasStrength.STRONG
        confidence = min(1.0, 0.7 + abs_distance / 30)
    elif abs_distance > 1.0:
        strength = BiasStrength.MODERATE
        confidence = 0.5 + abs_distance / 10
    else:
        strength = BiasStrength.WEAK
        confidence = 0.3 + abs_distance / 5

    return TimeframeBiasComponent(
        timeframe="daily",
        direction=direction,
        strength=strength,
        confidence=min(1.0, confidence),
        factors={
            "close": current_close,
            "ema200": current_ema,
            "distance_pct": distance_pct,
        },
    )


def compute_h4_bias(
    bars: list[OHLCVBar],
    ema_fast: int = 50,
    ema_slow: int = 200,
    er_period: int = 10,
    er_threshold: float = 0.22,
    hysteresis: float = 0.03,
) -> TimeframeBiasComponent:
    """
    Compute 4H timeframe bias (Primary Bias).

    Criteria:
    - EMA(50) vs EMA(200) slope comparison
    - Efficiency Ratio threshold (~0.20-0.25)
    - Hysteresis for flip prevention

    Args:
        bars: 4H OHLCV bars
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period
        er_period: Efficiency Ratio period
        er_threshold: ER threshold for trend qualification
        hysteresis: Buffer to prevent whipsaws

    Returns:
        TimeframeBiasComponent for 4H
    """
    min_bars = max(ema_slow, er_period) + 10
    if len(bars) < min_bars:
        return TimeframeBiasComponent(
            timeframe="4h",
            direction=BiasDirection.NEUTRAL,
            strength=BiasStrength.WEAK,
            confidence=0.0,
            factors={"error": "insufficient_data"},
        )

    closes = [b.close for b in bars]
    ema50 = compute_ema(closes, ema_fast)
    ema200 = compute_ema(closes, ema_slow)
    er_values = compute_efficiency_ratio(closes, er_period)

    current_er = er_values[-1]
    ema50_slope = _compute_ema_slope(ema50, 5)
    ema200_slope = _compute_ema_slope(ema200, 5)

    # EMA crossover position
    ema_diff = (ema50[-1] - ema200[-1]) / ema200[-1] * 100

    # Determine direction
    if ema_diff > hysteresis and ema50_slope > 0:
        base_direction = BiasDirection.BULLISH
    elif ema_diff < -hysteresis and ema50_slope < 0:
        base_direction = BiasDirection.BEARISH
    else:
        base_direction = BiasDirection.NEUTRAL

    # ER must confirm trend
    if current_er < er_threshold:
        # Low ER = choppy market, reduce confidence
        confidence = current_er / er_threshold * 0.5
        strength = BiasStrength.WEAK
        if confidence < 0.3:
            base_direction = BiasDirection.NEUTRAL
    else:
        # Good ER = trending
        confidence = 0.5 + (current_er - er_threshold) / (1 - er_threshold) * 0.5
        if current_er > 0.5:
            strength = BiasStrength.STRONG
        elif current_er > 0.35:
            strength = BiasStrength.MODERATE
        else:
            strength = BiasStrength.WEAK

    return TimeframeBiasComponent(
        timeframe="4h",
        direction=base_direction,
        strength=strength,
        confidence=min(1.0, confidence),
        factors={
            "ema50": ema50[-1],
            "ema200": ema200[-1],
            "ema_diff_pct": ema_diff,
            "ema50_slope": ema50_slope,
            "efficiency_ratio": current_er,
        },
    )


def compute_h1_bias(
    bars: list[OHLCVBar],
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> TimeframeBiasComponent:
    """
    Compute 1H timeframe bias (Confirmation).

    Criteria:
    - EMA(20) vs EMA(50) relationship

    Args:
        bars: 1H OHLCV bars
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period

    Returns:
        TimeframeBiasComponent for 1H
    """
    min_bars = ema_slow + 5
    if len(bars) < min_bars:
        return TimeframeBiasComponent(
            timeframe="1h",
            direction=BiasDirection.NEUTRAL,
            strength=BiasStrength.WEAK,
            confidence=0.0,
            factors={"error": "insufficient_data"},
        )

    closes = [b.close for b in bars]
    ema20 = compute_ema(closes, ema_fast)
    ema50 = compute_ema(closes, ema_slow)

    current_close = closes[-1]
    ema_diff = (ema20[-1] - ema50[-1]) / ema50[-1] * 100
    price_vs_ema20 = (current_close - ema20[-1]) / ema20[-1] * 100

    # Direction based on EMA alignment and price position
    if ema20[-1] > ema50[-1] and current_close > ema20[-1]:
        direction = BiasDirection.BULLISH
        strength = BiasStrength.STRONG if ema_diff > 0.5 else BiasStrength.MODERATE
    elif ema20[-1] < ema50[-1] and current_close < ema20[-1]:
        direction = BiasDirection.BEARISH
        strength = BiasStrength.STRONG if ema_diff < -0.5 else BiasStrength.MODERATE
    elif ema20[-1] > ema50[-1]:
        direction = BiasDirection.BULLISH
        strength = BiasStrength.WEAK
    elif ema20[-1] < ema50[-1]:
        direction = BiasDirection.BEARISH
        strength = BiasStrength.WEAK
    else:
        direction = BiasDirection.NEUTRAL
        strength = BiasStrength.WEAK

    confidence = 0.3 + min(0.7, abs(ema_diff) / 2)

    return TimeframeBiasComponent(
        timeframe="1h",
        direction=direction,
        strength=strength,
        confidence=min(1.0, confidence),
        factors={
            "ema20": ema20[-1],
            "ema50": ema50[-1],
            "ema_diff_pct": ema_diff,
            "price_vs_ema20_pct": price_vs_ema20,
        },
    )


def compute_m15_bias(
    bars: list[OHLCVBar],
    rsi_period: int = 14,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
) -> TimeframeBiasComponent:
    """
    Compute 15m timeframe bias (Momentum Health).

    Criteria:
    - RSI(14) for momentum
    - VWAP slope for session bias

    Args:
        bars: 15m OHLCV bars
        rsi_period: RSI period
        rsi_oversold: Oversold threshold
        rsi_overbought: Overbought threshold

    Returns:
        TimeframeBiasComponent for 15m
    """
    min_bars = rsi_period + 5
    if len(bars) < min_bars:
        return TimeframeBiasComponent(
            timeframe="15m",
            direction=BiasDirection.NEUTRAL,
            strength=BiasStrength.WEAK,
            confidence=0.0,
            factors={"error": "insufficient_data"},
        )

    closes = [b.close for b in bars]
    rsi = compute_rsi(closes, rsi_period)
    vwap = compute_vwap(bars)

    current_rsi = rsi[-1]
    current_close = closes[-1]
    current_vwap = vwap[-1]

    # VWAP slope (recent 5 bars)
    vwap_slope = 0.0
    if len(vwap) >= 5:
        vwap_slope = (vwap[-1] - vwap[-5]) / vwap[-5] * 100 if vwap[-5] > 0 else 0

    # Determine direction
    # RSI midpoint divergence
    rsi_bias = current_rsi - 50

    if current_close > current_vwap and rsi_bias > 5:
        direction = BiasDirection.BULLISH
    elif current_close < current_vwap and rsi_bias < -5:
        direction = BiasDirection.BEARISH
    else:
        direction = BiasDirection.NEUTRAL

    # Strength based on RSI extremes and VWAP position
    if current_rsi > rsi_overbought or current_rsi < rsi_oversold:
        strength = BiasStrength.STRONG
        confidence = 0.8
    elif abs(rsi_bias) > 15:
        strength = BiasStrength.MODERATE
        confidence = 0.6
    else:
        strength = BiasStrength.WEAK
        confidence = 0.4

    return TimeframeBiasComponent(
        timeframe="15m",
        direction=direction,
        strength=strength,
        confidence=confidence,
        factors={
            "rsi": current_rsi,
            "vwap": current_vwap,
            "close": current_close,
            "vwap_slope_pct": vwap_slope,
            "price_vs_vwap": (current_close - current_vwap) / current_vwap * 100,
        },
    )


def compute_m5_bias(
    bars: list[OHLCVBar],
    lookback: int = 12,
) -> TimeframeBiasComponent:
    """
    Compute 5m timeframe bias (Micro-Structure Sanity).

    Criteria:
    - HH/HL patterns for bullish
    - LH/LL patterns for bearish

    Args:
        bars: 5m OHLCV bars
        lookback: Bars to analyze for structure

    Returns:
        TimeframeBiasComponent for 5m
    """
    if len(bars) < lookback:
        return TimeframeBiasComponent(
            timeframe="5m",
            direction=BiasDirection.NEUTRAL,
            strength=BiasStrength.WEAK,
            confidence=0.0,
            factors={"error": "insufficient_data"},
        )

    direction, confidence = detect_hh_hl_pattern(bars, lookback)

    if confidence > 0.8:
        strength = BiasStrength.STRONG
    elif confidence > 0.6:
        strength = BiasStrength.MODERATE
    else:
        strength = BiasStrength.WEAK

    return TimeframeBiasComponent(
        timeframe="5m",
        direction=direction,
        strength=strength,
        confidence=confidence,
        factors={
            "pattern_confidence": confidence,
        },
    )


def compute_tf_bias(
    daily_bars: Optional[list[OHLCVBar]] = None,
    h4_bars: Optional[list[OHLCVBar]] = None,
    h1_bars: Optional[list[OHLCVBar]] = None,
    m15_bars: Optional[list[OHLCVBar]] = None,
    m5_bars: Optional[list[OHLCVBar]] = None,
    timestamp: Optional[datetime] = None,
) -> TimeframeBias:
    """
    Compute complete multi-timeframe bias.

    Combines signals from all available timeframes into a final
    bias with confidence scoring.

    Weights:
    - Daily: 25% (macro gate)
    - 4H: 30% (primary bias)
    - 1H: 20% (confirmation)
    - 15m: 15% (momentum)
    - 5m: 10% (micro-structure)

    Args:
        daily_bars: Daily OHLCV bars (optional)
        h4_bars: 4H OHLCV bars (optional)
        h1_bars: 1H OHLCV bars (optional)
        m15_bars: 15m OHLCV bars (optional)
        m5_bars: 5m OHLCV bars (optional)
        timestamp: Current timestamp

    Returns:
        TimeframeBias with all components and final direction
    """
    if timestamp is None:
        timestamp = datetime.utcnow()

    # Compute individual biases
    daily_bias = compute_daily_bias(daily_bars) if daily_bars else None
    h4_bias = compute_h4_bias(h4_bars) if h4_bars else None
    h1_bias = compute_h1_bias(h1_bars) if h1_bars else None
    m15_bias = compute_m15_bias(m15_bars) if m15_bars else None
    m5_bias = compute_m5_bias(m5_bars) if m5_bars else None

    # Weights for each timeframe
    weights = {
        "daily": 0.25,
        "4h": 0.30,
        "1h": 0.20,
        "15m": 0.15,
        "5m": 0.10,
    }

    # Calculate weighted bias
    components = [
        (daily_bias, weights["daily"]),
        (h4_bias, weights["4h"]),
        (h1_bias, weights["1h"]),
        (m15_bias, weights["15m"]),
        (m5_bias, weights["5m"]),
    ]

    bullish_score = 0.0
    bearish_score = 0.0
    total_weight = 0.0
    weighted_confidence = 0.0
    conflicting: list[str] = []

    for component, weight in components:
        if component is None:
            continue

        total_weight += weight
        weighted_confidence += component.confidence * weight

        if component.direction == BiasDirection.BULLISH:
            bullish_score += weight * component.confidence
        elif component.direction == BiasDirection.BEARISH:
            bearish_score += weight * component.confidence

    # Normalize scores
    if total_weight > 0:
        bullish_score /= total_weight
        bearish_score /= total_weight
        weighted_confidence /= total_weight

    # Determine final direction
    if bullish_score > bearish_score + 0.1:
        final_direction = BiasDirection.BULLISH
        direction_confidence = bullish_score
    elif bearish_score > bullish_score + 0.1:
        final_direction = BiasDirection.BEARISH
        direction_confidence = bearish_score
    else:
        final_direction = BiasDirection.NEUTRAL
        direction_confidence = max(bullish_score, bearish_score)

    # Check for conflicts
    primary_direction = final_direction
    for component, _ in components:
        if component is None:
            continue
        if (
            component.direction != BiasDirection.NEUTRAL
            and component.direction != primary_direction
            and component.confidence > 0.5
        ):
            conflicting.append(component.timeframe)

    # Alignment score (how many TFs agree)
    agreeing_count = 0
    total_count = 0
    for component, _ in components:
        if component is None:
            continue
        total_count += 1
        if (
            component.direction == final_direction
            or component.direction == BiasDirection.NEUTRAL
        ):
            agreeing_count += 1

    alignment_score = agreeing_count / total_count if total_count > 0 else 0.0

    # Final confidence combines direction confidence, weighted confidence, and alignment
    final_confidence = (
        direction_confidence * 0.4
        + weighted_confidence * 0.4
        + alignment_score * 0.2
    )

    # Determine strength
    if final_confidence > 0.75:
        final_strength = BiasStrength.STRONG
    elif final_confidence > 0.5:
        final_strength = BiasStrength.MODERATE
    else:
        final_strength = BiasStrength.WEAK

    return TimeframeBias(
        final_direction=final_direction,
        final_confidence=final_confidence,
        final_strength=final_strength,
        timestamp=timestamp,
        daily_bias=daily_bias,
        h4_bias=h4_bias,
        h1_bias=h1_bias,
        m15_bias=m15_bias,
        m5_bias=m5_bias,
        alignment_score=alignment_score,
        conflicting_timeframes=conflicting,
    )
