"""
Strategy specifications and registry.

Provides:
- ParamSpec: Parameter definition with type, bounds, and validation
- StrategySpec: Full strategy specification
- StrategyRegistry: Central registry for strategies
- Built-in strategies: mean_reversion, trend_following, breakout, rsi_strategy
"""

from app.services.strategies.params import (
    ParamType,
    ParamSpec,
    ValidationResult,
    validate_params,
    repair_params,
    validate_and_repair_params,
)

from app.services.strategies.registry import (
    ObjectiveType,
    OBJECTIVE_DESCRIPTIONS,
    StrategySpec,
    StrategyRegistry,
    create_default_registry,
    get_default_registry,
    register_strategy,
    get_strategy,
    validate_strategy,
    # Built-in spec creators
    create_mean_reversion_spec,
    create_trend_following_spec,
    create_breakout_spec,
    create_rsi_strategy_spec,
)

__all__ = [
    # Params
    "ParamType",
    "ParamSpec",
    "ValidationResult",
    "validate_params",
    "repair_params",
    "validate_and_repair_params",
    # Registry
    "ObjectiveType",
    "OBJECTIVE_DESCRIPTIONS",
    "StrategySpec",
    "StrategyRegistry",
    "create_default_registry",
    "get_default_registry",
    "register_strategy",
    "get_strategy",
    "validate_strategy",
    # Built-in specs
    "create_mean_reversion_spec",
    "create_trend_following_spec",
    "create_breakout_spec",
    "create_rsi_strategy_spec",
]
