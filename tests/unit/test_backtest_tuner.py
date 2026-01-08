"""Unit tests for backtest tuning functions."""

import pytest

from app.services.backtest.tuner import derive_param_space, ParamTuner


class TestDeriveParamSpace:
    """Tests for derive_param_space function."""

    def test_enum_values_used_directly(self):
        """Enum values should be used as-is."""
        schema = {
            "properties": {
                "direction": {"type": "string", "enum": ["long", "short", "both"]}
            }
        }
        space = derive_param_space(schema)
        assert space["direction"] == ["long", "short", "both"]

    def test_integer_range_creates_discrete_points(self):
        """Integer with min/max should create discrete points around default."""
        schema = {
            "properties": {
                "period": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 50,
                    "default": 20,
                }
            }
        }
        space = derive_param_space(schema)

        # Should have multiple discrete integer values
        assert isinstance(space["period"], list)
        assert all(isinstance(v, int) for v in space["period"])
        assert all(5 <= v <= 50 for v in space["period"])
        # Should include or be near default
        assert any(abs(v - 20) <= 5 for v in space["period"])

    def test_float_range_grid_creates_discrete_points(self):
        """Float with min/max in grid mode should create discrete points."""
        schema = {
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "default": 2.0,
                }
            }
        }
        space = derive_param_space(schema, search_type="grid")

        assert isinstance(space["threshold"], list)
        assert all(1.0 <= v <= 5.0 for v in space["threshold"])

    def test_float_range_random_creates_continuous(self):
        """Float with min/max in random mode should create continuous range."""
        schema = {
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "default": 2.0,
                }
            }
        }
        space = derive_param_space(schema, search_type="random")

        assert isinstance(space["threshold"], dict)
        assert space["threshold"]["min"] == 1.0
        assert space["threshold"]["max"] == 5.0
        assert space["threshold"]["type"] == "float"

    def test_default_only_creates_single_value(self):
        """Param with only default should create single-value list."""
        schema = {
            "properties": {
                "fixed": {"type": "integer", "default": 42}
            }
        }
        space = derive_param_space(schema)

        assert space["fixed"] == [42]

    def test_empty_properties(self):
        """Empty properties should return empty space."""
        schema = {"properties": {}}
        space = derive_param_space(schema)
        assert space == {}

    def test_no_properties_key(self):
        """Schema without properties key should return empty space."""
        schema = {}
        space = derive_param_space(schema)
        assert space == {}

    def test_multiple_params(self):
        """Multiple parameters should all be processed."""
        schema = {
            "properties": {
                "period": {"type": "integer", "minimum": 5, "maximum": 30, "default": 14},
                "threshold": {"type": "number", "minimum": 1.0, "maximum": 3.0, "default": 2.0},
                "direction": {"type": "string", "enum": ["long", "short"]},
            }
        }
        space = derive_param_space(schema, search_type="grid")

        assert "period" in space
        assert "threshold" in space
        assert "direction" in space
        assert len(space) == 3


class TestParamTunerGeneration:
    """Tests for ParamTuner param generation methods."""

    def test_generate_grid_params_lists(self):
        """Grid search with list values should generate all combinations."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": [1, 2],
            "b": [10, 20],
        }
        params = tuner._generate_grid_params(space, max_trials=100)

        assert len(params) == 4
        assert {"a": 1, "b": 10} in params
        assert {"a": 1, "b": 20} in params
        assert {"a": 2, "b": 10} in params
        assert {"a": 2, "b": 20} in params

    def test_generate_grid_params_max_trials_limit(self):
        """Grid search should respect max_trials limit."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
        params = tuner._generate_grid_params(space, max_trials=10)

        assert len(params) == 10  # 5*5=25 but capped at 10

    def test_generate_grid_params_continuous_range(self):
        """Grid search with continuous range should discretize."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": {"min": 0.0, "max": 1.0},
        }
        params = tuner._generate_grid_params(space, max_trials=100)

        assert len(params) == 5  # Default 5 discretization points
        assert all(0.0 <= p["a"] <= 1.0 for p in params)

    def test_generate_random_params_count(self):
        """Random search should generate exactly n_trials params."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": [1, 2, 3, 4, 5],
            "b": {"min": 0.0, "max": 1.0},
        }
        params = tuner._generate_random_params(space, n_trials=20, seed=42)

        assert len(params) == 20

    def test_generate_random_params_seed_deterministic(self):
        """Same seed should produce same results."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": [1, 2, 3, 4, 5],
            "b": {"min": 0.0, "max": 10.0, "type": "float"},
        }

        params1 = tuner._generate_random_params(space, n_trials=10, seed=123)
        params2 = tuner._generate_random_params(space, n_trials=10, seed=123)

        assert params1 == params2

    def test_generate_random_params_different_seeds(self):
        """Different seeds should produce different results."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": {"min": 0.0, "max": 10.0, "type": "float"},
        }

        params1 = tuner._generate_random_params(space, n_trials=10, seed=123)
        params2 = tuner._generate_random_params(space, n_trials=10, seed=456)

        # Highly unlikely to be identical with different seeds
        assert params1 != params2

    def test_generate_random_params_integer_type(self):
        """Integer type should generate integers."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": {"min": 1, "max": 100, "type": "int"},
        }
        params = tuner._generate_random_params(space, n_trials=10, seed=42)

        assert all(isinstance(p["a"], int) for p in params)
        assert all(1 <= p["a"] <= 100 for p in params)

    def test_generate_random_params_float_type(self):
        """Float type should generate floats."""
        tuner = ParamTuner(None, None, None)
        space = {
            "a": {"min": 0.0, "max": 1.0, "type": "float"},
        }
        params = tuner._generate_random_params(space, n_trials=10, seed=42)

        assert all(isinstance(p["a"], float) for p in params)
        assert all(0.0 <= p["a"] <= 1.0 for p in params)

    def test_generate_random_params_list_choice(self):
        """List values should be sampled from."""
        tuner = ParamTuner(None, None, None)
        space = {
            "direction": ["long", "short"],
        }
        params = tuner._generate_random_params(space, n_trials=100, seed=42)

        values = {p["direction"] for p in params}
        # With 100 trials, should have seen both values
        assert values == {"long", "short"}

    def test_generate_random_params_single_value(self):
        """Single value should always be that value."""
        tuner = ParamTuner(None, None, None)
        space = {
            "fixed": 42,
        }
        params = tuner._generate_random_params(space, n_trials=10, seed=42)

        assert all(p["fixed"] == 42 for p in params)
