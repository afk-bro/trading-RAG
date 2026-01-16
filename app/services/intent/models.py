"""Intent models and canonical tag mappings for trading content extraction."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Canonical tag mappings: pattern -> tag
# All tags are lowercase for consistency

STRATEGY_ARCHETYPE_PATTERNS: dict[str, list[str]] = {
    "trend_following": [
        "trend",
        "ride the trend",
        "trend continuation",
        "higher highs",
        "higher lows",
    ],
    "mean_reversion": [
        "mean revert",
        "reversal",
        "snap back",
        "oversold bounce",
        "overbought",
    ],
    "breakout": [
        "breakout",
        "range break",
        "ath breakout",
        "support break",
        "resistance break",
        "52 week high",
    ],
    "momentum": ["momentum", "impulse", "strength", "relative strength"],
    "range_bound": ["range", "chop", "sideways", "box", "consolidation"],
    "volatility": ["volatility expansion", "squeeze", "atr", "iv expansion"],
}

INDICATOR_PATTERNS: dict[str, list[str]] = {
    "rsi": ["rsi", "relative strength index"],
    "macd": ["macd"],
    "bollinger": ["bollinger", "bbands", "b bands"],
    "moving_average": ["sma", "ema", "wma", "moving average", "ma crossover"],
    "vwap": ["vwap"],
    "atr": ["atr", "average true range"],
    "volume": ["volume", "obv", "volume profile"],
    "stochastic": ["stoch", "stochastic"],
    "adx": ["adx"],
    "ichimoku": ["ichimoku", "cloud"],
    "pivot": ["pivot"],
    "support_resistance": ["support", "resistance", "s/r"],
}

# Timeframe bucket patterns
TIMEFRAME_BUCKET_PATTERNS: dict[str, list[str]] = {
    "scalp": ["scalp", "scalping"],
    "intraday": ["day trade", "daytrade", "intraday"],
    "swing": ["swing", "swing trade"],
    "position": ["position", "long-term", "long term"],
}

# Explicit timeframes (normalized format)
TIMEFRAME_EXPLICIT_PATTERNS: dict[str, list[str]] = {
    "1m": ["1m", "1 min", "1 minute", "one minute"],
    "5m": ["5m", "5 min", "5 minute", "five minute"],
    "15m": ["15m", "15 min", "15 minute", "fifteen minute"],
    "30m": ["30m", "30 min", "30 minute"],
    "1h": ["1h", "1 hour", "one hour", "hourly"],
    "4h": ["4h", "4 hour", "four hour"],
    "1d": ["1d", "daily", "1 day"],
    "1w": ["1w", "weekly", "1 week"],
}

RISK_TERM_PATTERNS: dict[str, list[str]] = {
    "stop_loss": ["stop loss", "stop-loss", "stoploss", " sl "],
    "take_profit": ["take profit", "take-profit", "takeprofit", " tp ", "target"],
    "trailing_stop": ["trailing stop", "trailing"],
    "position_sizing": [
        "position size",
        "position sizing",
        "risk per trade",
        "r-multiple",
        "risk reward",
    ],
    "dca": ["dca", "dollar cost average", "pyramiding", "scaling in", "scaling out"],
}

# Script type inference cues
STRATEGY_CUES = [
    "strategy",
    "strategy.entry",
    "strategy.exit",
    "backtest",
    "backtesting",
    "drawdown",
    "win rate",
    "entry",
    "exit",
    "pnl",
    "profit",
    "trade",
]
INDICATOR_CUES = [
    "indicator",
    "plot",
    "alert",
    "overlay",
    "oscillator",
    "histogram",
    "signal line",
    "divergence",
    "tradingview indicator",
]

# Valid canonical tags for validation
VALID_ARCHETYPES = set(STRATEGY_ARCHETYPE_PATTERNS.keys())
VALID_INDICATORS = set(INDICATOR_PATTERNS.keys())
VALID_TIMEFRAME_BUCKETS = set(TIMEFRAME_BUCKET_PATTERNS.keys())
VALID_TIMEFRAME_EXPLICIT = set(TIMEFRAME_EXPLICIT_PATTERNS.keys())
VALID_RISK_TERMS = set(RISK_TERM_PATTERNS.keys())


def _filter_and_dedupe(tags: list[str], valid_set: set[str]) -> list[str]:
    """Filter to valid tags and dedupe while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag in valid_set and tag not in seen:
            result.append(tag)
            seen.add(tag)
    return result


class MatchIntent(BaseModel):
    """Extracted trading intent from content."""

    # From existing MetadataExtractor
    symbols: list[str] = Field(
        default_factory=list, description="Ticker symbols (uppercase)"
    )
    topics: list[str] = Field(
        default_factory=list, description="Topic tags (lowercase)"
    )
    entities: list[str] = Field(default_factory=list, description="Named entities")

    # Trading-specific (lowercase canonical tags, deduped, order-preserved)
    strategy_archetypes: list[str] = Field(
        default_factory=list, description="Strategy types"
    )
    indicators: list[str] = Field(
        default_factory=list, description="Technical indicators"
    )
    timeframe_buckets: list[str] = Field(
        default_factory=list, description="Timeframe categories"
    )
    timeframe_explicit: list[str] = Field(
        default_factory=list, description="Explicit timeframes"
    )
    risk_terms: list[str] = Field(
        default_factory=list, description="Risk management terms"
    )

    # Script type inference
    inferred_script_type: Optional[Literal["strategy", "indicator"]] = Field(
        default=None, description="Inferred Pine script type"
    )
    script_type_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in script type (Laplace smoothed)"
    )
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )

    @field_validator("strategy_archetypes")
    @classmethod
    def validate_archetypes(cls, v: list[str]) -> list[str]:
        """Filter to valid archetype tags and dedupe preserving order."""
        return _filter_and_dedupe(v, VALID_ARCHETYPES)

    @field_validator("indicators")
    @classmethod
    def validate_indicators(cls, v: list[str]) -> list[str]:
        """Filter to valid indicator tags and dedupe preserving order."""
        return _filter_and_dedupe(v, VALID_INDICATORS)

    @field_validator("timeframe_buckets")
    @classmethod
    def validate_timeframe_buckets(cls, v: list[str]) -> list[str]:
        """Filter to valid timeframe bucket tags and dedupe preserving order."""
        return _filter_and_dedupe(v, VALID_TIMEFRAME_BUCKETS)

    @field_validator("timeframe_explicit")
    @classmethod
    def validate_timeframe_explicit(cls, v: list[str]) -> list[str]:
        """Filter to valid explicit timeframe tags and dedupe preserving order."""
        return _filter_and_dedupe(v, VALID_TIMEFRAME_EXPLICIT)

    @field_validator("risk_terms")
    @classmethod
    def validate_risk_terms(cls, v: list[str]) -> list[str]:
        """Filter to valid risk term tags and dedupe preserving order."""
        return _filter_and_dedupe(v, VALID_RISK_TERMS)
