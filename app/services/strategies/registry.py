"""
Strategy registry for the Trading Knowledge Base.

Provides:
- StrategySpec: Full strategy specification including params and constraints
- StrategyRegistry: Central registry of known strategies
- Auto-discovery from backtest_tune_runs table
- Objective function registry
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from app.services.strategies.params import (
    ParamSpec,
    ParamType,
    ValidationResult,
    validate_and_repair_params,
)


# =============================================================================
# Objective Functions
# =============================================================================


class ObjectiveType(str, Enum):
    """Supported objective function types."""

    SHARPE = "sharpe"
    SHARPE_DD_PENALTY = "sharpe_dd_penalty"
    RETURN = "return"
    RETURN_DD_PENALTY = "return_dd_penalty"
    CALMAR = "calmar"


OBJECTIVE_DESCRIPTIONS = {
    ObjectiveType.SHARPE: "Sharpe ratio (risk-adjusted return)",
    ObjectiveType.SHARPE_DD_PENALTY: "Sharpe - lambda * max_drawdown_pct",
    ObjectiveType.RETURN: "Total return percentage",
    ObjectiveType.RETURN_DD_PENALTY: "Return - lambda * max_drawdown_pct",
    ObjectiveType.CALMAR: "Return / abs(max_drawdown_pct) (Calmar ratio)",
}


# =============================================================================
# Strategy Specification
# =============================================================================


@dataclass
class StrategySpec:
    """
    Complete specification for a trading strategy.

    Includes parameter definitions, constraints, and metadata.
    """

    # Identity
    name: str
    display_name: str = ""
    description: str = ""
    version: str = "1.0"

    # Parameters
    params: dict[str, ParamSpec] = field(default_factory=dict)

    # Constraints between parameters
    constraints: list[dict] = field(default_factory=list)
    # Example: {"type": "less_than", "a": "fast_period", "b": "slow_period"}

    # Supported objectives
    supported_objectives: list[ObjectiveType] = field(
        default_factory=lambda: list(ObjectiveType)
    )
    default_objective: ObjectiveType = ObjectiveType.SHARPE

    # Recommended tuning
    tuning_config: dict = field(default_factory=dict)
    # Example: {"method": "grid", "max_trials": 1000}

    # Status
    status: str = "active"  # active, deprecated, experimental

    # Derived from (for KB-compiled strategies)
    derived_from_claims: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate parameters for this strategy."""
        result = validate_and_repair_params(params, self.params)

        # Check constraints
        if result.is_valid and result.repaired_params:
            constraint_errors = self._check_constraints(result.repaired_params)
            if constraint_errors:
                return ValidationResult(
                    is_valid=False,
                    errors=constraint_errors,
                    warnings=result.warnings,
                    repaired_params=None,
                )

        return result

    def _check_constraints(self, params: dict[str, Any]) -> list[str]:
        """Check parameter constraints."""
        errors = []
        for constraint in self.constraints:
            ctype = constraint.get("type")

            if ctype == "less_than":
                a, b = constraint.get("a"), constraint.get("b")
                if a in params and b in params:
                    if params[a] >= params[b]:
                        errors.append(f"Constraint violated: {a} must be < {b}")

            elif ctype == "less_than_equal":
                a, b = constraint.get("a"), constraint.get("b")
                if a in params and b in params:
                    if params[a] > params[b]:
                        errors.append(f"Constraint violated: {a} must be <= {b}")

            elif ctype == "sum_max":
                fields = constraint.get("fields", [])
                max_val = constraint.get("max")
                total = sum(params.get(f, 0) for f in fields)
                if total > max_val:
                    errors.append(
                        f"Constraint violated: sum of {fields} must be <= {max_val}"
                    )

        return errors

    def get_default_params(self) -> dict[str, Any]:
        """Get default parameter values."""
        return {name: spec.default for name, spec in self.params.items()}

    def get_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = {}
        for name, spec in self.params.items():
            if spec.type in (ParamType.INT, ParamType.FLOAT):
                if spec.min_value is not None and spec.max_value is not None:
                    bounds[name] = (spec.min_value, spec.max_value)
        return bounds

    def to_public_schema(self) -> dict:
        """
        Get UI-ready JSON schema for parameters.

        Returns a JSON-ready dict with param definitions suitable for
        frontend form generation, including min/max/step/type/choices.
        """
        params_schema = {}
        for name, spec in self.params.items():
            param_def = {
                "name": name,
                "type": spec.type.value,
                "default": spec.default,
                "required": spec.required,
                "description": spec.description or "",
            }

            # Add bounds for numeric types
            if spec.type in (ParamType.INT, ParamType.FLOAT):
                if spec.min_value is not None:
                    param_def["min"] = spec.min_value
                if spec.max_value is not None:
                    param_def["max"] = spec.max_value
                if spec.step is not None:
                    param_def["step"] = spec.step
                if spec.unit:
                    param_def["unit"] = spec.unit

            # Add choices for enum
            if spec.type == ParamType.ENUM and spec.choices:
                param_def["choices"] = spec.choices

            params_schema[name] = param_def

        return {
            "strategy_name": self.name,
            "strategy_version": self.version,
            "display_name": self.display_name,
            "description": self.description,
            "status": self.status,
            "params": params_schema,
            "constraints": self.constraints,
            "supported_objectives": [o.value for o in self.supported_objectives],
            "default_objective": self.default_objective.value,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "params": {name: spec.to_dict() for name, spec in self.params.items()},
            "constraints": self.constraints,
            "supported_objectives": [o.value for o in self.supported_objectives],
            "default_objective": self.default_objective.value,
            "tuning_config": self.tuning_config,
            "status": self.status,
            "derived_from_claims": self.derived_from_claims,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategySpec":
        """Create from dictionary."""
        params = {}
        for name, spec_data in data.get("params", {}).items():
            params[name] = ParamSpec.from_dict(spec_data)

        supported_objectives = [
            ObjectiveType(o) for o in data.get("supported_objectives", [])
        ]
        if not supported_objectives:
            supported_objectives = list(ObjectiveType)

        return cls(
            name=data["name"],
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            params=params,
            constraints=data.get("constraints", []),
            supported_objectives=supported_objectives,
            default_objective=ObjectiveType(
                data.get("default_objective", "sharpe")
            ),
            tuning_config=data.get("tuning_config", {}),
            status=data.get("status", "active"),
            derived_from_claims=data.get("derived_from_claims", []),
        )


# =============================================================================
# Strategy Registry
# =============================================================================


class StrategyRegistry:
    """
    Central registry for trading strategies.

    Provides:
    - Registration of known strategies with full specs
    - Auto-discovery from backtest_tune_runs
    - Validation of strategy/param combinations
    """

    def __init__(self):
        self._strategies: dict[str, StrategySpec] = {}
        self._discovered: set[str] = set()  # Strategies found in data

    def register(self, spec: StrategySpec) -> None:
        """Register a strategy specification."""
        self._strategies[spec.name] = spec

    def get(self, name: str) -> Optional[StrategySpec]:
        """Get a strategy specification by name."""
        return self._strategies.get(name)

    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def list_active_strategies(self) -> list[str]:
        """List strategies with active status."""
        return [
            name
            for name, spec in self._strategies.items()
            if spec.status == "active"
        ]

    def is_known(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies

    def is_discovered(self, name: str) -> bool:
        """Check if a strategy was discovered from data."""
        return name in self._discovered

    def mark_discovered(self, name: str) -> None:
        """Mark a strategy as discovered from data."""
        self._discovered.add(name)

    def validate_strategy_params(
        self,
        strategy_name: str,
        params: dict[str, Any],
        allow_unknown_strategy: bool = False,
    ) -> ValidationResult:
        """
        Validate parameters for a strategy.

        Args:
            strategy_name: Strategy name
            params: Parameter values
            allow_unknown_strategy: If True, skip validation for unknown strategies

        Returns:
            ValidationResult
        """
        spec = self.get(strategy_name)

        if spec is None:
            if allow_unknown_strategy:
                return ValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=[f"Unknown strategy '{strategy_name}', skipping validation"],
                    repaired_params=params,
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Unknown strategy: {strategy_name}"],
                    warnings=[],
                    repaired_params=None,
                )

        return spec.validate_params(params)

    def get_objectives(self) -> list[dict]:
        """Get list of supported objective types."""
        return [
            {
                "type": obj.value,
                "description": OBJECTIVE_DESCRIPTIONS.get(obj, ""),
            }
            for obj in ObjectiveType
        ]


# =============================================================================
# Built-in Strategy Definitions
# =============================================================================


def create_mean_reversion_spec() -> StrategySpec:
    """Create specification for mean reversion strategy."""
    return StrategySpec(
        name="mean_reversion",
        display_name="Mean Reversion",
        description="Trade reversals when price deviates from moving average",
        params={
            "period": ParamSpec(
                name="period",
                type=ParamType.INT,
                default=20,
                min_value=5,
                max_value=200,
                step=5,
                description="Moving average period",
                unit="bars",
            ),
            "threshold": ParamSpec(
                name="threshold",
                type=ParamType.FLOAT,
                default=2.0,
                min_value=0.5,
                max_value=4.0,
                step=0.25,
                description="Standard deviation threshold for entry",
            ),
            "stop_loss": ParamSpec(
                name="stop_loss",
                type=ParamType.FLOAT,
                default=0.02,
                min_value=0.005,
                max_value=0.10,
                step=0.005,
                description="Stop loss percentage",
                unit="fraction",
            ),
            "take_profit": ParamSpec(
                name="take_profit",
                type=ParamType.FLOAT,
                default=0.04,
                min_value=0.01,
                max_value=0.20,
                step=0.01,
                description="Take profit percentage",
                unit="fraction",
                required=False,
            ),
        },
        constraints=[
            {"type": "less_than", "a": "stop_loss", "b": "take_profit"},
        ],
    )


def create_trend_following_spec() -> StrategySpec:
    """Create specification for trend following strategy."""
    return StrategySpec(
        name="trend_following",
        display_name="Trend Following",
        description="Follow price momentum using dual moving averages",
        params={
            "fast_period": ParamSpec(
                name="fast_period",
                type=ParamType.INT,
                default=10,
                min_value=2,
                max_value=50,
                step=2,
                description="Fast moving average period",
                unit="bars",
            ),
            "slow_period": ParamSpec(
                name="slow_period",
                type=ParamType.INT,
                default=50,
                min_value=10,
                max_value=200,
                step=5,
                description="Slow moving average period",
                unit="bars",
            ),
            "atr_multiplier": ParamSpec(
                name="atr_multiplier",
                type=ParamType.FLOAT,
                default=2.0,
                min_value=0.5,
                max_value=5.0,
                step=0.5,
                description="ATR multiplier for stop loss",
            ),
            "trailing_stop": ParamSpec(
                name="trailing_stop",
                type=ParamType.BOOL,
                default=True,
                description="Use trailing stop loss",
            ),
        },
        constraints=[
            {"type": "less_than", "a": "fast_period", "b": "slow_period"},
        ],
    )


def create_breakout_spec() -> StrategySpec:
    """Create specification for breakout strategy."""
    return StrategySpec(
        name="breakout",
        display_name="Breakout",
        description="Trade breakouts from price channels",
        params={
            "lookback": ParamSpec(
                name="lookback",
                type=ParamType.INT,
                default=20,
                min_value=5,
                max_value=100,
                step=5,
                description="Lookback period for highs/lows",
                unit="bars",
            ),
            "volume_threshold": ParamSpec(
                name="volume_threshold",
                type=ParamType.FLOAT,
                default=1.5,
                min_value=1.0,
                max_value=3.0,
                step=0.25,
                description="Volume multiplier threshold",
            ),
            "confirmation_bars": ParamSpec(
                name="confirmation_bars",
                type=ParamType.INT,
                default=1,
                min_value=0,
                max_value=5,
                step=1,
                description="Bars to wait for confirmation",
                unit="bars",
            ),
        },
    )


def create_rsi_strategy_spec() -> StrategySpec:
    """Create specification for RSI strategy."""
    return StrategySpec(
        name="rsi_strategy",
        display_name="RSI Strategy",
        description="Trade RSI overbought/oversold signals",
        params={
            "rsi_period": ParamSpec(
                name="rsi_period",
                type=ParamType.INT,
                default=14,
                min_value=5,
                max_value=50,
                step=1,
                description="RSI calculation period",
                unit="bars",
            ),
            "oversold": ParamSpec(
                name="oversold",
                type=ParamType.INT,
                default=30,
                min_value=10,
                max_value=40,
                step=5,
                description="Oversold threshold",
            ),
            "overbought": ParamSpec(
                name="overbought",
                type=ParamType.INT,
                default=70,
                min_value=60,
                max_value=90,
                step=5,
                description="Overbought threshold",
            ),
            "exit_middle": ParamSpec(
                name="exit_middle",
                type=ParamType.BOOL,
                default=True,
                description="Exit when RSI crosses 50",
            ),
        },
        constraints=[
            {"type": "less_than", "a": "oversold", "b": "overbought"},
        ],
    )


# =============================================================================
# Default Registry
# =============================================================================


def create_default_registry() -> StrategyRegistry:
    """Create a registry with built-in strategies."""
    registry = StrategyRegistry()

    # Register built-in strategies
    registry.register(create_mean_reversion_spec())
    registry.register(create_trend_following_spec())
    registry.register(create_breakout_spec())
    registry.register(create_rsi_strategy_spec())

    return registry


# Global default registry instance
_default_registry: Optional[StrategyRegistry] = None


def get_default_registry() -> StrategyRegistry:
    """Get the default strategy registry (lazy-initialized singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry


def register_strategy(spec: StrategySpec) -> None:
    """Register a strategy in the default registry."""
    get_default_registry().register(spec)


def get_strategy(name: str) -> Optional[StrategySpec]:
    """Get a strategy from the default registry."""
    return get_default_registry().get(name)


def validate_strategy(
    strategy_name: str,
    params: dict[str, Any],
    allow_unknown: bool = False,
) -> ValidationResult:
    """Validate strategy params using the default registry."""
    return get_default_registry().validate_strategy_params(
        strategy_name, params, allow_unknown
    )
