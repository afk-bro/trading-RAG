"""Unit tests for testing package models - TDD style."""

import pytest
from copy import deepcopy

from app.services.testing.models import (
    canonical_json,
    hash_variant,
    apply_overrides,
    RunPlanStatus,
    GeneratorConstraints,
    RunVariant,
    RunPlan,
    VariantMetrics,
    RunResult,
)


class TestCanonicalJson:
    """Tests for canonical_json function."""

    def test_same_dict_different_key_order_produces_same_output(self):
        """Same dict with different key order should produce identical output."""
        dict1 = {"b": 2, "a": 1, "c": 3}
        dict2 = {"a": 1, "c": 3, "b": 2}
        dict3 = {"c": 3, "a": 1, "b": 2}

        result1 = canonical_json(dict1)
        result2 = canonical_json(dict2)
        result3 = canonical_json(dict3)

        assert result1 == result2 == result3

    def test_no_whitespace_in_output(self):
        """Output should have no whitespace - no ' : ' and no newlines."""
        obj = {"key": "value", "nested": {"inner": 123}}
        result = canonical_json(obj)

        assert " : " not in result
        assert "\n" not in result
        assert " " not in result  # No spaces anywhere

    def test_nested_dict_sorting(self):
        """Nested dicts should also have sorted keys."""
        obj = {"z": {"b": 2, "a": 1}, "a": 1}
        result = canonical_json(obj)

        # Keys should be sorted at all levels
        assert result == '{"a":1,"z":{"a":1,"b":2}}'


class TestHashVariant:
    """Tests for hash_variant function."""

    def test_deterministic_same_inputs_shuffled_keys(self):
        """Same base+overrides with key-order shuffled should produce same 16-char ID."""
        base1 = {"strategy": "bb_reversal", "params": {"window": 20}}
        base2 = {"params": {"window": 20}, "strategy": "bb_reversal"}

        overrides1 = {"params.window": 30, "risk.stop_loss": 0.02}
        overrides2 = {"risk.stop_loss": 0.02, "params.window": 30}

        hash1 = hash_variant(base1, overrides1)
        hash2 = hash_variant(base2, overrides2)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_override_value_produces_different_id(self):
        """Different override values should produce different IDs."""
        base = {"strategy": "bb_reversal"}
        overrides1 = {"params.window": 20}
        overrides2 = {"params.window": 30}

        hash1 = hash_variant(base, overrides1)
        hash2 = hash_variant(base, overrides2)

        assert hash1 != hash2

    def test_hash_is_16_chars(self):
        """Hash should be exactly 16 characters."""
        base = {"x": 1}
        overrides = {"y.z": 2}

        result = hash_variant(base, overrides)
        assert len(result) == 16

    def test_hash_is_hex(self):
        """Hash should be hexadecimal."""
        base = {"x": 1}
        overrides = {"y.z": 2}

        result = hash_variant(base, overrides)
        # Should not raise - all chars are hex
        int(result, 16)


class TestApplyOverrides:
    """Tests for apply_overrides function."""

    def test_applies_dotted_path_into_nested_dict(self):
        """Dotted path should traverse and set nested values."""
        spec = {"params": {"window": 20, "std_dev": 2.0}, "risk": {"stop_loss": 0.01}}
        overrides = {"params.window": 30, "risk.stop_loss": 0.02}

        result = apply_overrides(spec, overrides)

        assert result["params"]["window"] == 30
        assert result["risk"]["stop_loss"] == 0.02
        # Other values unchanged
        assert result["params"]["std_dev"] == 2.0

    def test_does_not_mutate_original_dict(self):
        """Original dict should not be modified."""
        spec = {"params": {"window": 20}}
        original_spec = deepcopy(spec)
        overrides = {"params.window": 30}

        result = apply_overrides(spec, overrides)

        # Original unchanged
        assert spec == original_spec
        assert spec["params"]["window"] == 20
        # Result is different
        assert result["params"]["window"] == 30

    def test_raises_keyerror_on_missing_path(self):
        """Missing intermediate key should raise KeyError."""
        spec = {"params": {"window": 20}}
        overrides = {"nonexistent.key": 100}

        with pytest.raises(KeyError):
            apply_overrides(spec, overrides)

    def test_raises_keyerror_on_missing_leaf(self):
        """Missing leaf key should raise KeyError."""
        spec = {"params": {"window": 20}}
        overrides = {"params.nonexistent": 100}

        with pytest.raises(KeyError):
            apply_overrides(spec, overrides)

    def test_deep_nesting_works(self):
        """Multi-level nesting should work."""
        spec = {"a": {"b": {"c": {"d": 1}}}}
        overrides = {"a.b.c.d": 999}

        result = apply_overrides(spec, overrides)
        assert result["a"]["b"]["c"]["d"] == 999


class TestRunVariantCreate:
    """Tests for RunVariant.create() factory validation."""

    def test_invalid_path_no_dot_raises_valueerror(self):
        """Override key without dot should raise ValueError."""
        with pytest.raises(ValueError, match="must contain a dot"):
            RunVariant.create(
                base_spec={"params": {"x": 1}},
                overrides={"nodot": 100},
            )

    def test_empty_segments_raises_valueerror(self):
        """Empty path segments like 'risk..x' should raise ValueError."""
        with pytest.raises(ValueError, match="empty segment"):
            RunVariant.create(
                base_spec={"risk": {"x": 1}},
                overrides={"risk..x": 100},
            )

    def test_nested_dict_value_raises_valueerror(self):
        """Override value that is a dict should raise ValueError."""
        with pytest.raises(ValueError, match="scalar"):
            RunVariant.create(
                base_spec={"params": {"x": 1}},
                overrides={"params.x": {"nested": "dict"}},
            )

    def test_empty_overrides_allowed(self):
        """Empty overrides should be allowed (baseline variant)."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}

        variant = RunVariant.create(base_spec=base, overrides={})

        assert variant.overrides == {}
        assert len(variant.variant_id) == 16  # Still gets an ID

    def test_valid_overrides_produce_variant(self):
        """Valid overrides should create a proper variant."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}
        overrides = {"params.window": 30}

        variant = RunVariant.create(base_spec=base, overrides=overrides)

        assert variant.overrides == overrides
        assert len(variant.variant_id) == 16
        assert variant.tags == []  # Default empty list

    def test_variant_id_is_deterministic(self):
        """Same inputs should produce same variant ID."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}
        overrides = {"params.window": 30}

        variant1 = RunVariant.create(base_spec=base, overrides=overrides)
        variant2 = RunVariant.create(base_spec=base, overrides=overrides)

        assert variant1.variant_id == variant2.variant_id


class TestRunPlanNVariants:
    """Tests for RunPlan.n_variants computed property."""

    def test_n_variants_returns_len_of_variants(self):
        """n_variants should return the length of variants list."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}

        variants = [
            RunVariant.create(base_spec=base, overrides={"params.window": i})
            for i in [20, 30, 40]
        ]

        plan = RunPlan(
            plan_id="test-plan-001",
            base_spec=base,
            variants=variants,
            constraints=GeneratorConstraints(),
        )

        assert plan.n_variants == 3

    def test_n_variants_zero_for_empty_variants(self):
        """n_variants should be 0 for empty variants list."""
        base = {"strategy": "bb_reversal"}

        plan = RunPlan(
            plan_id="test-plan-002",
            base_spec=base,
            variants=[],
            constraints=GeneratorConstraints(),
        )

        assert plan.n_variants == 0


class TestGeneratorConstraints:
    """Tests for GeneratorConstraints model."""

    def test_default_values(self):
        """Default constraint values should match spec."""
        constraints = GeneratorConstraints()

        assert constraints.include_ablations is True
        assert constraints.max_variants == 25
        assert constraints.objective == "sharpe_dd_penalty"

    def test_custom_values(self):
        """Custom values should be accepted."""
        constraints = GeneratorConstraints(
            lookback_days_values=[30, 60, 90],
            dollars_per_trade_values=[100, 200],
            max_positions_values=[3, 5],
            include_ablations=False,
            max_variants=50,
            objective="sharpe",
        )

        assert constraints.lookback_days_values == [30, 60, 90]
        assert constraints.dollars_per_trade_values == [100, 200]
        assert constraints.max_positions_values == [3, 5]
        assert constraints.include_ablations is False
        assert constraints.max_variants == 50
        assert constraints.objective == "sharpe"


class TestRunPlanStatus:
    """Tests for RunPlanStatus enum."""

    def test_all_statuses_exist(self):
        """All expected statuses should exist."""
        assert RunPlanStatus.pending
        assert RunPlanStatus.running
        assert RunPlanStatus.completed
        assert RunPlanStatus.failed
        assert RunPlanStatus.canceled


class TestVariantMetrics:
    """Tests for VariantMetrics model."""

    def test_create_with_required_fields(self):
        """Should create metrics with required fields."""
        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=25.0,
            max_drawdown_pct=10.0,
            n_trades=50,
        )

        assert metrics.sharpe == 1.5
        assert metrics.return_pct == 25.0
        assert metrics.max_drawdown_pct == 10.0
        assert metrics.n_trades == 50

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=25.0,
            max_drawdown_pct=10.0,
            n_trades=50,
        )

        assert metrics.calmar is None
        assert metrics.win_rate is None
        assert metrics.profit_factor is None


class TestRunResult:
    """Tests for RunResult model."""

    def test_create_successful_result(self):
        """Should create a successful result."""
        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=25.0,
            max_drawdown_pct=10.0,
            n_trades=50,
        )

        result = RunResult(
            variant_id="abc123def456ghij",
            status="success",
            metrics=metrics,
        )

        assert result.variant_id == "abc123def456ghij"
        assert result.status == "success"
        assert result.metrics == metrics
        assert result.error_message is None

    def test_create_failed_result(self):
        """Should create a failed result with error message."""
        result = RunResult(
            variant_id="abc123def456ghij",
            status="failed",
            error_message="Backtest crashed: division by zero",
        )

        assert result.status == "failed"
        assert result.metrics is None
        assert result.error_message == "Backtest crashed: division by zero"
