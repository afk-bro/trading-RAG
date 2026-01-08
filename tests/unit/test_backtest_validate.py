"""Unit tests for backtest parameter validation."""

import pytest

from app.services.backtest.validate import validate_params, ParamValidationError


class TestValidateParams:
    """Tests for validate_params function."""

    def test_valid_params_pass(self):
        """Valid parameters matching schema should pass."""
        params = {"period": 14, "threshold": 2.0}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer", "minimum": 1, "maximum": 100},
                "threshold": {"type": "number", "minimum": 0.5, "maximum": 5.0},
            },
        }

        result = validate_params(params, schema)

        assert result["period"] == 14
        assert result["threshold"] == 2.0

    def test_defaults_are_applied(self):
        """Missing params with defaults should have defaults applied."""
        params = {"period": 20}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer", "default": 14},
                "threshold": {"type": "number", "default": 2.0},
            },
        }

        result = validate_params(params, schema)

        assert result["period"] == 20  # User-provided
        assert result["threshold"] == 2.0  # Default applied

    def test_required_param_missing_raises_error(self):
        """Missing required parameter should raise error."""
        params = {"threshold": 2.0}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer"},
                "threshold": {"type": "number"},
            },
            "required": ["period"],
        }

        with pytest.raises(ParamValidationError) as exc_info:
            validate_params(params, schema)

        assert any(e["param"] == "period" for e in exc_info.value.errors)

    def test_type_coercion_string_to_int(self):
        """String that can be converted to int should be coerced."""
        params = {"period": "14"}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer"},
            },
        }

        result = validate_params(params, schema)

        assert result["period"] == 14
        assert isinstance(result["period"], int)

    def test_type_coercion_string_to_float(self):
        """String that can be converted to float should be coerced."""
        params = {"threshold": "2.5"}
        schema = {
            "type": "object",
            "properties": {
                "threshold": {"type": "number"},
            },
        }

        result = validate_params(params, schema)

        assert result["threshold"] == 2.5
        assert isinstance(result["threshold"], float)

    def test_invalid_type_raises_error(self):
        """Non-coercible value should raise error."""
        params = {"period": "not_a_number"}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer"},
            },
        }

        with pytest.raises(ParamValidationError) as exc_info:
            validate_params(params, schema)

        assert any(e["param"] == "period" for e in exc_info.value.errors)

    def test_value_below_minimum_raises_error(self):
        """Value below minimum should raise error."""
        params = {"period": 0}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer", "minimum": 1},
            },
        }

        with pytest.raises(ParamValidationError) as exc_info:
            validate_params(params, schema)

        error = next(e for e in exc_info.value.errors if e["param"] == "period")
        assert "minimum" in error["error"].lower()

    def test_value_above_maximum_raises_error(self):
        """Value above maximum should raise error."""
        params = {"period": 500}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer", "maximum": 100},
            },
        }

        with pytest.raises(ParamValidationError) as exc_info:
            validate_params(params, schema)

        error = next(e for e in exc_info.value.errors if e["param"] == "period")
        assert "maximum" in error["error"].lower()

    def test_enum_validation(self):
        """Value not in enum should raise error."""
        params = {"direction": "invalid"}
        schema = {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["long", "short", "both"]},
            },
        }

        with pytest.raises(ParamValidationError) as exc_info:
            validate_params(params, schema)

        error = next(e for e in exc_info.value.errors if e["param"] == "direction")
        assert "enum" in error["error"].lower() or "must be one of" in error["error"].lower()

    def test_extra_params_passed_through(self):
        """Parameters not in schema should be passed through."""
        params = {"period": 14, "custom_param": "custom_value"}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer"},
            },
        }

        result = validate_params(params, schema)

        assert result["period"] == 14
        assert result["custom_param"] == "custom_value"

    def test_empty_schema_passes_all(self):
        """Empty schema should pass all params through."""
        params = {"anything": "goes", "number": 42}

        result = validate_params(params, {})

        assert result == params

    def test_none_schema_passes_all(self):
        """None schema should pass all params through."""
        params = {"anything": "goes"}

        result = validate_params(params, None)

        assert result == params

    def test_multiple_validation_errors(self):
        """Multiple validation errors should all be reported."""
        params = {"period": 0, "threshold": 10.0}
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer", "minimum": 1},
                "threshold": {"type": "number", "maximum": 5.0},
            },
        }

        with pytest.raises(ParamValidationError) as exc_info:
            validate_params(params, schema)

        assert len(exc_info.value.errors) == 2
        param_names = [e["param"] for e in exc_info.value.errors]
        assert "period" in param_names
        assert "threshold" in param_names
