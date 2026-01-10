"""
Strategy runner package.

Provides models and runtime for executing trading strategies.

Models:
- OHLCVBar: Single candlestick data
- MarketSnapshot: Point-in-time market state
- EntryConfig, ExitConfig, RiskConfig: Configuration models
- ExecutionSpec: Strategy instance definition
- StrategyEvaluation: Runner output

Runner:
- StrategyRunner: Main evaluation engine
"""

from app.services.strategy.models import (
    OHLCVBar,
    MarketSnapshot,
    EntryConfig,
    ExitConfig,
    RiskConfig,
    ExecutionSpec,
    StrategyEvaluation,
)
from app.services.strategy.runner import StrategyRunner

__all__ = [
    "OHLCVBar",
    "MarketSnapshot",
    "EntryConfig",
    "ExitConfig",
    "RiskConfig",
    "ExecutionSpec",
    "StrategyEvaluation",
    "StrategyRunner",
]
