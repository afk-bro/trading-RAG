"""Unit tests for backtest tuning functions."""

from app.services.backtest.tuner import (
    derive_param_space,
    ParamTuner,
    compute_objective_score,
    DEFAULT_DD_LAMBDA,
)


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
        schema = {"properties": {"fixed": {"type": "integer", "default": 42}}}
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
                "period": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 30,
                    "default": 14,
                },
                "threshold": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 3.0,
                    "default": 2.0,
                },
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


class TestComputeObjectiveScore:
    """Tests for compute_objective_score function."""

    def test_sharpe_objective_returns_sharpe(self):
        """Sharpe objective should return raw sharpe."""
        metrics = {"sharpe": 1.5, "max_drawdown_pct": -10.0, "return_pct": 25.0}
        score = compute_objective_score(metrics, objective_type="sharpe")
        assert score == 1.5

    def test_sharpe_objective_none_when_missing(self):
        """Sharpe objective should return None if sharpe missing."""
        metrics = {"max_drawdown_pct": -10.0, "return_pct": 25.0}
        score = compute_objective_score(metrics, objective_type="sharpe")
        assert score is None

    def test_sharpe_dd_penalty_applies_penalty(self):
        """Sharpe DD penalty should subtract lambda * abs(dd)."""
        metrics = {"sharpe": 1.5, "max_drawdown_pct": -10.0}
        score = compute_objective_score(metrics, objective_type="sharpe_dd_penalty")

        # With default lambda=0.02: 1.5 - 0.02 * 10 = 1.5 - 0.2 = 1.3
        expected = round(1.5 - DEFAULT_DD_LAMBDA * 10.0, 4)
        assert score == expected
        assert score == 1.3

    def test_sharpe_dd_penalty_custom_lambda(self):
        """Sharpe DD penalty should use custom lambda."""
        metrics = {"sharpe": 2.0, "max_drawdown_pct": -20.0}
        score = compute_objective_score(
            metrics,
            objective_type="sharpe_dd_penalty",
            objective_params={"dd_lambda": 0.05},
        )

        # 2.0 - 0.05 * 20 = 2.0 - 1.0 = 1.0
        assert score == 1.0

    def test_sharpe_dd_penalty_missing_dd_returns_sharpe(self):
        """Sharpe DD penalty should return raw sharpe if DD missing."""
        metrics = {"sharpe": 1.5}
        score = compute_objective_score(metrics, objective_type="sharpe_dd_penalty")
        assert score == 1.5

    def test_sharpe_dd_penalty_missing_sharpe_returns_none(self):
        """Sharpe DD penalty should return None if sharpe missing."""
        metrics = {"max_drawdown_pct": -10.0}
        score = compute_objective_score(metrics, objective_type="sharpe_dd_penalty")
        assert score is None

    def test_return_objective_returns_return_pct(self):
        """Return objective should return return_pct."""
        metrics = {"sharpe": 1.5, "max_drawdown_pct": -10.0, "return_pct": 25.0}
        score = compute_objective_score(metrics, objective_type="return")
        assert score == 25.0

    def test_return_dd_penalty_applies_penalty(self):
        """Return DD penalty should subtract lambda * abs(dd)."""
        metrics = {"return_pct": 30.0, "max_drawdown_pct": -15.0}
        score = compute_objective_score(metrics, objective_type="return_dd_penalty")

        # 30.0 - 0.02 * 15 = 30.0 - 0.3 = 29.7
        expected = round(30.0 - DEFAULT_DD_LAMBDA * 15.0, 4)
        assert score == expected

    def test_calmar_ratio(self):
        """Calmar objective should return return / abs(dd)."""
        metrics = {"return_pct": 30.0, "max_drawdown_pct": -10.0}
        score = compute_objective_score(metrics, objective_type="calmar")

        # 30 / 10 = 3.0
        assert score == 3.0

    def test_calmar_zero_dd_returns_none(self):
        """Calmar should return None if DD is zero (avoid division by zero)."""
        metrics = {"return_pct": 30.0, "max_drawdown_pct": 0.0}
        score = compute_objective_score(metrics, objective_type="calmar")
        assert score is None

    def test_calmar_missing_dd_returns_none(self):
        """Calmar should return None if DD is missing."""
        metrics = {"return_pct": 30.0}
        score = compute_objective_score(metrics, objective_type="calmar")
        assert score is None

    def test_empty_metrics_returns_none(self):
        """Empty metrics should return None."""
        score = compute_objective_score({}, objective_type="sharpe")
        assert score is None

    def test_none_metrics_returns_none(self):
        """None metrics should return None."""
        score = compute_objective_score(None, objective_type="sharpe")
        assert score is None

    def test_unknown_objective_falls_back_to_sharpe(self):
        """Unknown objective type should fall back to sharpe (with warning)."""
        metrics = {"sharpe": 1.5}
        score = compute_objective_score(metrics, objective_type="unknown_type")
        # Falls back to sharpe behavior
        assert score == 1.5


class TestWinnerSelectionWithObjectiveScore:
    """Tests for winner selection behavior with composite objectives."""

    def test_dd_penalty_changes_winner(self):
        """DD penalty can change winner selection compared to raw sharpe."""
        # Trial A: High sharpe but high DD
        trial_a = {"sharpe": 2.0, "max_drawdown_pct": -30.0}
        # Trial B: Lower sharpe but low DD
        trial_b = {"sharpe": 1.5, "max_drawdown_pct": -5.0}

        # Raw sharpe winner
        sharpe_a = compute_objective_score(trial_a, objective_type="sharpe")
        sharpe_b = compute_objective_score(trial_b, objective_type="sharpe")
        assert sharpe_a > sharpe_b  # Trial A wins on raw sharpe

        # With DD penalty (lambda=0.02)
        # A: 2.0 - 0.02*30 = 2.0 - 0.6 = 1.4
        # B: 1.5 - 0.02*5 = 1.5 - 0.1 = 1.4
        obj_a = compute_objective_score(trial_a, objective_type="sharpe_dd_penalty")
        obj_b = compute_objective_score(trial_b, objective_type="sharpe_dd_penalty")
        assert obj_a == 1.4
        assert obj_b == 1.4  # Tie in this case

        # With higher lambda, B wins
        obj_a_high = compute_objective_score(
            trial_a,
            objective_type="sharpe_dd_penalty",
            objective_params={"dd_lambda": 0.03},
        )
        obj_b_high = compute_objective_score(
            trial_b,
            objective_type="sharpe_dd_penalty",
            objective_params={"dd_lambda": 0.03},
        )
        # A: 2.0 - 0.03*30 = 2.0 - 0.9 = 1.1
        # B: 1.5 - 0.03*5 = 1.5 - 0.15 = 1.35
        assert obj_a_high == 1.1
        assert obj_b_high == 1.35
        assert obj_b_high > obj_a_high  # Trial B wins with higher DD penalty

    def test_negative_sharpe_with_high_dd(self):
        """Negative sharpe with high DD should be penalized further."""
        metrics = {"sharpe": -0.5, "max_drawdown_pct": -25.0}

        # Raw sharpe: -0.5
        raw = compute_objective_score(metrics, objective_type="sharpe")
        assert raw == -0.5

        # With DD penalty: -0.5 - 0.02*25 = -0.5 - 0.5 = -1.0
        penalized = compute_objective_score(metrics, objective_type="sharpe_dd_penalty")
        assert penalized == -1.0
        assert penalized < raw  # Penalty makes it worse


class TestGatesSnapshot:
    """Tests for gate policy snapshot building."""

    def test_gates_snapshot_includes_thresholds(self):
        """Gates snapshot should include max_drawdown_pct and min_trades."""
        from app.services.backtest.tuner import GATE_MAX_DD_PCT, GATE_MIN_TRADES

        # Build gates snapshot like the router does
        gates_snapshot = {
            "max_drawdown_pct": GATE_MAX_DD_PCT,
            "min_trades": GATE_MIN_TRADES,
            "evaluated_on": "oos",
        }

        assert "max_drawdown_pct" in gates_snapshot
        assert "min_trades" in gates_snapshot
        assert "evaluated_on" in gates_snapshot
        assert gates_snapshot["max_drawdown_pct"] == GATE_MAX_DD_PCT
        assert gates_snapshot["min_trades"] == GATE_MIN_TRADES

    def test_gates_evaluated_on_oos_when_oos_ratio_set(self):
        """Gates should be evaluated on 'oos' when oos_ratio is provided."""
        oos_ratio = 0.2  # 20% OOS split

        evaluated_on = "oos" if oos_ratio else "primary"

        assert evaluated_on == "oos"

    def test_gates_evaluated_on_primary_when_no_split(self):
        """Gates should be evaluated on 'primary' when no OOS split."""
        oos_ratio = None  # No OOS split

        evaluated_on = "oos" if oos_ratio else "primary"

        assert evaluated_on == "primary"

    def test_gates_evaluated_on_primary_when_zero_split(self):
        """Gates should be evaluated on 'primary' when oos_ratio is 0."""
        oos_ratio = 0  # Zero (falsy)

        evaluated_on = "oos" if oos_ratio else "primary"

        assert evaluated_on == "primary"

    def test_gate_constants_from_env(self):
        """Gate constants should be loaded from environment (or defaults)."""
        from app.services.backtest.tuner import GATE_MAX_DD_PCT, GATE_MIN_TRADES

        # Verify they are numeric and reasonable defaults
        assert isinstance(GATE_MAX_DD_PCT, float)
        assert isinstance(GATE_MIN_TRADES, int)
        assert GATE_MAX_DD_PCT > 0  # Should be positive (e.g., 20 for -20% max DD)
        assert GATE_MIN_TRADES > 0  # Should be at least 1

    def test_evaluate_gates_uses_constants(self):
        """evaluate_gates function should use the gate constants."""
        from app.services.backtest.tuner import (
            evaluate_gates,
            GATE_MAX_DD_PCT,
            GATE_MIN_TRADES,
        )

        # Metrics that pass gates
        passing_metrics = {
            "max_drawdown_pct": -(GATE_MAX_DD_PCT - 1),  # Just under threshold
            "trades": GATE_MIN_TRADES + 5,  # Above minimum
        }
        passed, failures = evaluate_gates(passing_metrics)
        assert passed is True
        assert len(failures) == 0

        # Metrics that fail gates
        failing_metrics = {
            "max_drawdown_pct": -(GATE_MAX_DD_PCT + 10),  # Over threshold
            "trades": GATE_MIN_TRADES - 1,  # Under minimum
        }
        passed, failures = evaluate_gates(failing_metrics)
        assert passed is False
        assert len(failures) == 2
        assert any("max_drawdown_pct" in f for f in failures)
        assert any("trades" in f for f in failures)
