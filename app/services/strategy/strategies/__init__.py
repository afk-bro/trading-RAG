"""
Strategy implementations for the execution engine.

Each strategy is a pure function that evaluates market conditions
and emits TradeIntents.
"""

from app.services.strategy.strategies.breakout_52w_high import (
    evaluate_breakout_52w_high,
    round_quantity,
    QUANTITY_DECIMALS,
)

from app.services.strategy.strategies.unicorn_model import (
    evaluate_unicorn_model,
    UnicornSetup,
    UnicornConfig,
    CriteriaScore,
    SessionProfile,
    analyze_unicorn_setup,
    is_in_macro_window,
    get_max_stop_points,
    get_point_value,
    MACRO_WINDOWS,
    SESSION_WINDOWS,
    MAX_STOP_POINTS_NQ,
    MAX_STOP_POINTS_ES,
    DEFAULT_CONFIG,
)

__all__ = [
    # Breakout 52W High
    "evaluate_breakout_52w_high",
    "round_quantity",
    "QUANTITY_DECIMALS",
    # ICT Unicorn Model
    "evaluate_unicorn_model",
    "UnicornSetup",
    "UnicornConfig",
    "CriteriaScore",
    "SessionProfile",
    "analyze_unicorn_setup",
    "is_in_macro_window",
    "get_max_stop_points",
    "get_point_value",
    "MACRO_WINDOWS",
    "SESSION_WINDOWS",
    "MAX_STOP_POINTS_NQ",
    "MAX_STOP_POINTS_ES",
    "DEFAULT_CONFIG",
]
