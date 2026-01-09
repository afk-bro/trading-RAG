"""Unit tests for strategy registry and parameter validation."""

import pytest

from app.services.strategies import (
    ParamType,
    ParamSpec,
    ValidationResult,
    validate_params,
    repair_params,
    validate_and_repair_params,
    ObjectiveType,
    StrategySpec,
    StrategyRegistry,
    create_default_registry,
    get_default_registry,
    get_strategy,
    validate_strategy,
    create_mean_reversion_spec,
    create_trend_following_spec,
)


# =============================================================================
# ParamSpec Tests
# =============================================================================


class TestParamSpec:
    """Tests for ParamSpec validation and repair."""

    def test_int_param_valid(self):
        """Valid int should pass validation."""
        spec = ParamSpec(
            name="period",
            type=ParamType.INT,
            default=20,
            min_value=5,
            max_value=100,
        )
        is_valid, error = spec.validate(20)
        assert is_valid
        assert error is None

    def test_int_param_out_of_bounds(self):
        """Out of bounds int should fail validation."""
        spec = ParamSpec(
            name="period",
            type=ParamType.INT,
            default=20,
            min_value=5,
            max_value=100,
        )
        is_valid, error = spec.validate(200)
        assert not is_valid
        assert "must be <=" in error

    def test_int_param_repair(self):
        """Out of bounds int should be clamped."""
        spec = ParamSpec(
            name="period",
            type=ParamType.INT,
            default=20,
            min_value=5,
            max_value=100,
        )
        repaired, warning = spec.repair(200)
        assert repaired == 100
        assert "clamped" in warning

    def test_float_param_valid(self):
        """Valid float should pass validation."""
        spec = ParamSpec(
            name="threshold",
            type=ParamType.FLOAT,
            default=2.0,
            min_value=0.5,
            max_value=4.0,
        )
        is_valid, error = spec.validate(2.5)
        assert is_valid
        assert error is None

    def test_bool_param_valid(self):
        """Valid bool should pass validation."""
        spec = ParamSpec(
            name="trailing_stop",
            type=ParamType.BOOL,
            default=True,
        )
        is_valid, error = spec.validate(False)
        assert is_valid
        assert error is None

    def test_bool_param_coerce(self):
        """String should be coerced to bool."""
        spec = ParamSpec(
            name="trailing_stop",
            type=ParamType.BOOL,
            default=True,
        )
        repaired, warning = spec.repair("true")
        assert repaired is True
        assert warning is None

    def test_enum_param_valid(self):
        """Valid enum value should pass."""
        spec = ParamSpec(
            name="direction",
            type=ParamType.ENUM,
            default="long",
            choices=["long", "short", "both"],
        )
        is_valid, error = spec.validate("short")
        assert is_valid
        assert error is None

    def test_enum_param_invalid(self):
        """Invalid enum value should fail."""
        spec = ParamSpec(
            name="direction",
            type=ParamType.ENUM,
            default="long",
            choices=["long", "short", "both"],
        )
        is_valid, error = spec.validate("invalid")
        assert not is_valid
        assert "must be one of" in error

    def test_nullable_param(self):
        """Nullable param should accept None."""
        spec = ParamSpec(
            name="optional",
            type=ParamType.FLOAT,
            default=None,
            nullable=True,
            required=False,
        )
        is_valid, error = spec.validate(None)
        assert is_valid
        assert error is None

    def test_required_param_missing(self):
        """Required param should fail on None."""
        spec = ParamSpec(
            name="required_param",
            type=ParamType.INT,
            default=10,
            required=True,
        )
        is_valid, error = spec.validate(None)
        assert not is_valid
        assert "required" in error

    def test_to_dict_from_dict(self):
        """Spec should round-trip through dict."""
        spec = ParamSpec(
            name="period",
            type=ParamType.INT,
            default=20,
            min_value=5,
            max_value=100,
            description="Test param",
        )
        d = spec.to_dict()
        restored = ParamSpec.from_dict(d)

        assert restored.name == spec.name
        assert restored.type == spec.type
        assert restored.default == spec.default
        assert restored.min_value == spec.min_value
        assert restored.max_value == spec.max_value


# =============================================================================
# Validation Functions Tests
# =============================================================================


class TestValidation:
    """Tests for validation functions."""

    @pytest.fixture
    def sample_specs(self):
        """Sample parameter specs for testing."""
        return {
            "period": ParamSpec(
                name="period",
                type=ParamType.INT,
                default=20,
                min_value=5,
                max_value=100,
            ),
            "threshold": ParamSpec(
                name="threshold",
                type=ParamType.FLOAT,
                default=2.0,
                min_value=0.5,
                max_value=4.0,
            ),
        }

    def test_validate_params_valid(self, sample_specs):
        """Valid params should pass validation."""
        params = {"period": 30, "threshold": 2.5}
        result = validate_params(params, sample_specs)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_params_missing(self, sample_specs):
        """Missing required param should fail."""
        params = {"period": 30}
        result = validate_params(params, sample_specs)

        assert not result.is_valid
        assert any("Missing required" in e for e in result.errors)

    def test_validate_params_unknown_strict(self, sample_specs):
        """Unknown param should fail in strict mode."""
        params = {"period": 30, "threshold": 2.0, "unknown": 123}
        result = validate_params(params, sample_specs, strict=True)

        assert not result.is_valid
        assert any("Unknown" in e for e in result.errors)

    def test_validate_params_unknown_lenient(self, sample_specs):
        """Unknown param should warn in lenient mode."""
        params = {"period": 30, "threshold": 2.0, "unknown": 123}
        result = validate_params(params, sample_specs, strict=False)

        assert result.is_valid
        assert any("Ignoring unknown" in w for w in result.warnings)

    def test_repair_params(self, sample_specs):
        """Repair should fix missing and out-of-bounds params."""
        params = {"period": 200}  # Missing threshold, out of bounds period
        result = repair_params(params, sample_specs)

        assert result.repaired_params is not None
        assert result.repaired_params["period"] == 100  # Clamped
        assert result.repaired_params["threshold"] == 2.0  # Default

    def test_validate_and_repair(self, sample_specs):
        """Validate and repair should fix issues and return valid params."""
        params = {"period": 200, "threshold": 10.0}  # Both out of bounds
        result = validate_and_repair_params(params, sample_specs)

        assert result.is_valid
        assert result.repaired_params["period"] == 100
        assert result.repaired_params["threshold"] == 4.0


# =============================================================================
# StrategySpec Tests
# =============================================================================


class TestStrategySpec:
    """Tests for StrategySpec."""

    def test_create_spec(self):
        """Should create spec with parameters."""
        spec = create_mean_reversion_spec()

        assert spec.name == "mean_reversion"
        assert "period" in spec.params
        assert "threshold" in spec.params
        assert "stop_loss" in spec.params

    def test_validate_params_valid(self):
        """Valid params should pass."""
        spec = create_mean_reversion_spec()
        params = {
            "period": 20,
            "threshold": 2.0,
            "stop_loss": 0.02,
            "take_profit": 0.04,
        }
        result = spec.validate_params(params)

        assert result.is_valid

    def test_validate_params_constraint_violation(self):
        """Constraint violation should fail."""
        spec = create_mean_reversion_spec()
        params = {
            "period": 20,
            "threshold": 2.0,
            "stop_loss": 0.05,  # Greater than take_profit
            "take_profit": 0.04,
        }
        result = spec.validate_params(params)

        assert not result.is_valid
        assert any("Constraint violated" in e for e in result.errors)

    def test_get_default_params(self):
        """Should return default params."""
        spec = create_mean_reversion_spec()
        defaults = spec.get_default_params()

        assert defaults["period"] == 20
        assert defaults["threshold"] == 2.0
        assert defaults["stop_loss"] == 0.02

    def test_get_param_bounds(self):
        """Should return param bounds for numeric params."""
        spec = create_mean_reversion_spec()
        bounds = spec.get_param_bounds()

        assert "period" in bounds
        assert bounds["period"] == (5, 200)
        assert "threshold" in bounds
        assert bounds["threshold"] == (0.5, 4.0)

    def test_to_dict_from_dict(self):
        """Spec should round-trip through dict."""
        spec = create_mean_reversion_spec()
        d = spec.to_dict()
        restored = StrategySpec.from_dict(d)

        assert restored.name == spec.name
        assert len(restored.params) == len(spec.params)
        assert restored.constraints == spec.constraints


# =============================================================================
# StrategyRegistry Tests
# =============================================================================


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_create_default_registry(self):
        """Default registry should have built-in strategies."""
        registry = create_default_registry()

        assert registry.is_known("mean_reversion")
        assert registry.is_known("trend_following")
        assert registry.is_known("breakout")
        assert registry.is_known("rsi_strategy")

    def test_register_strategy(self):
        """Should register new strategy."""
        registry = StrategyRegistry()
        spec = create_mean_reversion_spec()
        registry.register(spec)

        assert registry.is_known("mean_reversion")
        assert registry.get("mean_reversion") == spec

    def test_get_unknown_strategy(self):
        """Unknown strategy should return None."""
        registry = StrategyRegistry()

        assert registry.get("unknown") is None

    def test_list_strategies(self):
        """Should list all strategies."""
        registry = create_default_registry()
        strategies = registry.list_strategies()

        assert "mean_reversion" in strategies
        assert len(strategies) >= 4

    def test_validate_known_strategy(self):
        """Should validate known strategy params."""
        registry = create_default_registry()
        params = {"period": 20, "threshold": 2.0, "stop_loss": 0.02}
        result = registry.validate_strategy_params("mean_reversion", params)

        assert result.is_valid

    def test_validate_unknown_strategy_strict(self):
        """Unknown strategy should fail in strict mode."""
        registry = create_default_registry()
        params = {"foo": 123}
        result = registry.validate_strategy_params(
            "unknown_strategy", params, allow_unknown_strategy=False
        )

        assert not result.is_valid
        assert any("Unknown strategy" in e for e in result.errors)

    def test_validate_unknown_strategy_lenient(self):
        """Unknown strategy should pass through in lenient mode."""
        registry = create_default_registry()
        params = {"foo": 123}
        result = registry.validate_strategy_params(
            "unknown_strategy", params, allow_unknown_strategy=True
        )

        assert result.is_valid
        assert result.repaired_params == params

    def test_mark_discovered(self):
        """Should track discovered strategies."""
        registry = StrategyRegistry()
        registry.mark_discovered("new_strategy")

        assert registry.is_discovered("new_strategy")
        assert not registry.is_discovered("unknown")


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_default_registry(self):
        """Should return singleton registry."""
        reg1 = get_default_registry()
        reg2 = get_default_registry()

        assert reg1 is reg2

    def test_get_strategy(self):
        """Should get strategy from default registry."""
        spec = get_strategy("mean_reversion")

        assert spec is not None
        assert spec.name == "mean_reversion"

    def test_validate_strategy(self):
        """Should validate using default registry."""
        params = {"period": 20, "threshold": 2.0, "stop_loss": 0.02}
        result = validate_strategy("mean_reversion", params)

        assert result.is_valid


# =============================================================================
# Public Schema Tests
# =============================================================================


class TestPublicSchema:
    """Tests for to_public_schema() method."""

    def test_public_schema_basic_fields(self):
        """Should include basic strategy info."""
        spec = create_mean_reversion_spec()
        schema = spec.to_public_schema()

        assert schema["strategy_name"] == "mean_reversion"
        assert schema["strategy_version"] == "1.0"
        assert schema["display_name"] == "Mean Reversion"
        assert schema["status"] == "active"

    def test_public_schema_params(self):
        """Should include param definitions."""
        spec = create_mean_reversion_spec()
        schema = spec.to_public_schema()

        assert "params" in schema
        assert "period" in schema["params"]

        period = schema["params"]["period"]
        assert period["type"] == "int"
        assert period["default"] == 20
        assert period["min"] == 5
        assert period["max"] == 200
        assert period["step"] == 5
        assert period["unit"] == "bars"

    def test_public_schema_constraints(self):
        """Should include constraints."""
        spec = create_mean_reversion_spec()
        schema = spec.to_public_schema()

        assert "constraints" in schema
        assert len(schema["constraints"]) > 0

    def test_public_schema_objectives(self):
        """Should include objective info."""
        spec = create_mean_reversion_spec()
        schema = spec.to_public_schema()

        assert "supported_objectives" in schema
        assert "default_objective" in schema
        assert "sharpe" in schema["supported_objectives"]

    def test_public_schema_enum_choices(self):
        """Should include choices for enum params."""
        spec = StrategySpec(
            name="test_strategy",
            params={
                "direction": ParamSpec(
                    name="direction",
                    type=ParamType.ENUM,
                    default="long",
                    choices=["long", "short", "both"],
                ),
            },
        )
        schema = spec.to_public_schema()

        direction = schema["params"]["direction"]
        assert direction["type"] == "enum"
        assert direction["choices"] == ["long", "short", "both"]
