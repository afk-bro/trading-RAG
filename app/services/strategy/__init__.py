"""
Strategy runner package.

Provides models and runtime for executing trading strategies,
plus the strategy registry for multi-engine strategy catalog.

Models:
- OHLCVBar: Single candlestick data
- MarketSnapshot: Point-in-time market state
- EntryConfig, ExitConfig, RiskConfig: Configuration models
- ExecutionSpec: Strategy instance definition
- StrategyEvaluation: Runner output

Runner:
- StrategyRunner: Main evaluation engine

Registry:
- StrategyRepository: Multi-engine strategy catalog
- slugify: URL-safe slug generation
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
from app.services.strategy.repository import StrategyRepository, slugify

__all__ = [
    # Models
    "OHLCVBar",
    "MarketSnapshot",
    "EntryConfig",
    "ExitConfig",
    "RiskConfig",
    "ExecutionSpec",
    "StrategyEvaluation",
    # Runner
    "StrategyRunner",
    # Registry
    "StrategyRepository",
    "slugify",
]
