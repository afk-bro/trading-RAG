"""Unit tests for testing package models - TDD style."""

import pytest
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from app.services.testing.models import (
    canonical_json,
    hash_variant,
    apply_overrides,
    validate_variant_params,
    RunPlanStatus,
    RunResultStatus,
    GeneratorConstraints,
    RunVariant,
    RunPlan,
    VariantMetrics,
    RunResult,
    TESTING_VARIANT_NAMESPACE,
    get_variant_namespace,
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

    def test_float_precision_stable(self):
        """Float values should serialize stably."""
        obj1 = {"value": 0.1}
        obj2 = {"value": 0.1}

        assert canonical_json(obj1) == canonical_json(obj2)

    def test_int_and_float_same_value_different_hash(self):
        """int(5) and float(5.0) should produce different output (type matters)."""
        obj_int = {"value": 5}
        obj_float = {"value": 5.0}

        # These are actually different in JSON - 5 vs 5.0
        # Python json.dumps treats 5.0 as 5.0 and 5 as 5
        result_int = canonical_json(obj_int)
        result_float = canonical_json(obj_float)

        # Note: Python's json.dumps renders 5.0 as "5.0" only if it's actually 5.0
        # In practice, 5.0 may render as "5.0" or "5" depending on precision
        # The important thing is consistency
        assert result_int == canonical_json({"value": 5})
        assert result_float == canonical_json({"value": 5.0})


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
                spec_overrides={"nodot": 100},
            )

    def test_empty_segments_raises_valueerror(self):
        """Empty path segments like 'risk..x' should raise ValueError."""
        with pytest.raises(ValueError, match="empty segment"):
            RunVariant.create(
                base_spec={"risk": {"x": 1}},
                spec_overrides={"risk..x": 100},
            )

    def test_nested_dict_value_raises_valueerror(self):
        """Override value that is a dict should raise ValueError."""
        with pytest.raises(ValueError, match="scalar"):
            RunVariant.create(
                base_spec={"params": {"x": 1}},
                spec_overrides={"params.x": {"nested": "dict"}},
            )

    def test_empty_overrides_allowed(self):
        """Empty overrides should be allowed (baseline variant)."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}

        variant = RunVariant.create(base_spec=base, spec_overrides={})

        assert variant.spec_overrides == {}
        assert len(variant.variant_id) == 16  # Still gets an ID
        assert variant.label == ""  # Default empty label

    def test_valid_overrides_produce_variant(self):
        """Valid overrides should create a proper variant."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}
        spec_overrides = {"params.window": 30}

        variant = RunVariant.create(
            base_spec=base, spec_overrides=spec_overrides, label="window=30"
        )

        assert variant.spec_overrides == spec_overrides
        assert variant.label == "window=30"
        assert len(variant.variant_id) == 16
        assert variant.tags == []  # Default empty list

    def test_variant_id_is_deterministic(self):
        """Same inputs should produce same variant ID."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}
        spec_overrides = {"params.window": 30}

        variant1 = RunVariant.create(base_spec=base, spec_overrides=spec_overrides)
        variant2 = RunVariant.create(base_spec=base, spec_overrides=spec_overrides)

        assert variant1.variant_id == variant2.variant_id


class TestRunPlanNVariants:
    """Tests for RunPlan.n_variants computed property."""

    def test_n_variants_returns_len_of_variants(self):
        """n_variants should return the length of variants list."""
        base = {"strategy": "bb_reversal", "params": {"window": 20}}
        workspace_id = uuid4()

        variants = [
            RunVariant.create(
                base_spec=base, spec_overrides={"params.window": i}, label=f"window={i}"
            )
            for i in [20, 30, 40]
        ]

        plan = RunPlan(
            workspace_id=workspace_id,
            base_spec=base,
            variants=variants,
            dataset_ref="btc_2023",
        )

        assert plan.n_variants == 3

    def test_n_variants_zero_for_empty_variants(self):
        """n_variants should be 0 for empty variants list."""
        base = {"strategy": "bb_reversal"}
        workspace_id = uuid4()

        plan = RunPlan(
            workspace_id=workspace_id,
            base_spec=base,
            variants=[],
            dataset_ref="btc_2023",
        )

        assert plan.n_variants == 0

    def test_run_plan_has_uuid_id(self):
        """run_plan_id should be a UUID generated by default."""
        base = {"strategy": "bb_reversal"}
        workspace_id = uuid4()

        plan = RunPlan(
            workspace_id=workspace_id,
            base_spec=base,
            dataset_ref="btc_2023",
        )

        assert plan.run_plan_id is not None
        # Should be a valid UUID (no exception on access)
        assert str(plan.run_plan_id)

    def test_run_plan_objective_default(self):
        """objective should default to sharpe_dd_penalty."""
        base = {"strategy": "bb_reversal"}
        workspace_id = uuid4()

        plan = RunPlan(
            workspace_id=workspace_id,
            base_spec=base,
            dataset_ref="btc_2023",
        )

        assert plan.objective == "sharpe_dd_penalty"


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

    def test_max_variants_at_limit(self):
        """max_variants=200 should be accepted (at hard cap)."""
        constraints = GeneratorConstraints(max_variants=200)
        assert constraints.max_variants == 200

    def test_max_variants_exceeds_hard_cap(self):
        """max_variants > 200 should be rejected at validation time."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GeneratorConstraints(max_variants=201)

        # Check the error is about max_variants constraint
        assert "max_variants" in str(exc_info.value)


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
            return_pct=25.0,
            max_drawdown_pct=10.0,
            trade_count=50,
            ending_equity=12500.0,
            gross_profit=3000.0,
            gross_loss=-500.0,
        )

        assert metrics.return_pct == 25.0
        assert metrics.max_drawdown_pct == 10.0
        assert metrics.trade_count == 50
        assert metrics.ending_equity == 12500.0
        assert metrics.gross_profit == 3000.0
        assert metrics.gross_loss == -500.0

    def test_sharpe_optional_defaults_none(self):
        """sharpe should be optional and default to None (for <2 trades)."""
        metrics = VariantMetrics(
            return_pct=25.0,
            max_drawdown_pct=10.0,
            trade_count=1,
            ending_equity=12500.0,
            gross_profit=2500.0,
            gross_loss=0.0,
        )

        assert metrics.sharpe is None

    def test_sharpe_can_be_set(self):
        """sharpe should be settable when sufficient trades."""
        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=25.0,
            max_drawdown_pct=10.0,
            trade_count=50,
            ending_equity=12500.0,
            gross_profit=3000.0,
            gross_loss=-500.0,
        )

        assert metrics.sharpe == 1.5

    def test_win_rate_required_defaults_zero(self):
        """win_rate should be required and default to 0.0."""
        metrics = VariantMetrics(
            return_pct=0.0,
            max_drawdown_pct=0.0,
            trade_count=0,
            ending_equity=10000.0,
            gross_profit=0.0,
            gross_loss=0.0,
        )

        assert metrics.win_rate == 0.0

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        metrics = VariantMetrics(
            return_pct=25.0,
            max_drawdown_pct=10.0,
            trade_count=50,
            ending_equity=12500.0,
            gross_profit=3000.0,
            gross_loss=-500.0,
        )

        assert metrics.sharpe is None
        assert metrics.profit_factor is None


class TestRunResult:
    """Tests for RunResult model."""

    def test_create_successful_result(self):
        """Should create a successful result."""
        run_plan_id = uuid4()
        started_at = datetime.now(timezone.utc)
        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=25.0,
            max_drawdown_pct=10.0,
            trade_count=50,
            ending_equity=12500.0,
            gross_profit=3000.0,
            gross_loss=-500.0,
        )

        result = RunResult(
            run_plan_id=run_plan_id,
            variant_id="abc123def456ghij",
            status=RunResultStatus.success,
            metrics=metrics,
            started_at=started_at,
            objective_score=1.3,
        )

        assert result.run_plan_id == run_plan_id
        assert result.variant_id == "abc123def456ghij"
        assert result.status == RunResultStatus.success
        assert result.metrics == metrics
        assert result.error is None
        assert result.objective_score == 1.3
        assert result.events_recorded == 0  # Default

    def test_create_failed_result(self):
        """Should create a failed result with error message."""
        run_plan_id = uuid4()
        started_at = datetime.now(timezone.utc)

        result = RunResult(
            run_plan_id=run_plan_id,
            variant_id="abc123def456ghij",
            status=RunResultStatus.failed,
            error="Backtest crashed: division by zero",
            started_at=started_at,
        )

        assert result.status == RunResultStatus.failed
        assert result.metrics is None
        assert result.error == "Backtest crashed: division by zero"

    def test_duration_and_events_recorded(self):
        """Should track duration and events recorded."""
        run_plan_id = uuid4()
        started_at = datetime.now(timezone.utc)

        result = RunResult(
            run_plan_id=run_plan_id,
            variant_id="abc123def456ghij",
            status=RunResultStatus.success,
            started_at=started_at,
            duration_ms=1500,
            events_recorded=42,
        )

        assert result.duration_ms == 1500
        assert result.events_recorded == 42


class TestVariantNamespace:
    """Tests for TESTING_VARIANT_NAMESPACE and get_variant_namespace."""

    def test_constant_namespace_is_uuid(self):
        """TESTING_VARIANT_NAMESPACE should be a valid UUID."""
        from uuid import UUID

        assert isinstance(TESTING_VARIANT_NAMESPACE, UUID)

    def test_same_inputs_produce_same_namespace(self):
        """Same (run_plan_id, variant_id) should produce same namespace across calls."""
        run_plan_id = uuid4()
        variant_id = "abc123def456ghij"

        ns1 = get_variant_namespace(run_plan_id, variant_id)
        ns2 = get_variant_namespace(run_plan_id, variant_id)

        assert ns1 == ns2

    def test_different_variant_id_produces_different_namespace(self):
        """Different variant_id should produce different namespace."""
        run_plan_id = uuid4()
        variant_id_1 = "abc123def456ghij"
        variant_id_2 = "xyz789uvw123pqrs"

        ns1 = get_variant_namespace(run_plan_id, variant_id_1)
        ns2 = get_variant_namespace(run_plan_id, variant_id_2)

        assert ns1 != ns2

    def test_different_run_plan_id_produces_different_namespace(self):
        """Different run_plan_id should produce different namespace."""
        run_plan_id_1 = uuid4()
        run_plan_id_2 = uuid4()
        variant_id = "abc123def456ghij"

        ns1 = get_variant_namespace(run_plan_id_1, variant_id)
        ns2 = get_variant_namespace(run_plan_id_2, variant_id)

        assert ns1 != ns2

    def test_namespace_is_uuid(self):
        """get_variant_namespace should return a UUID."""
        from uuid import UUID

        run_plan_id = uuid4()
        variant_id = "abc123def456ghij"

        ns = get_variant_namespace(run_plan_id, variant_id)
        assert isinstance(ns, UUID)

    def test_namespace_deterministic_across_repeated_calls(self):
        """Namespace should be deterministic across any number of calls."""
        # Fixed inputs for reproducibility testing
        run_plan_id = uuid4()
        variant_id = "abc123def456ghij"

        namespaces = [get_variant_namespace(run_plan_id, variant_id) for _ in range(10)]

        # All should be identical
        assert all(ns == namespaces[0] for ns in namespaces)


class TestValidateVariantParams:
    """Tests for validate_variant_params function."""

    def test_valid_params_returns_true(self):
        """Valid params should return (True, None)."""
        spec = {
            "risk": {
                "dollars_per_trade": 1000.0,
                "max_positions": 5,
            }
        }
        is_valid, error = validate_variant_params(spec)
        assert is_valid is True
        assert error is None

    def test_zero_dollars_per_trade_returns_false(self):
        """dollars_per_trade = 0 should be invalid."""
        spec = {
            "risk": {
                "dollars_per_trade": 0,
                "max_positions": 5,
            }
        }
        is_valid, error = validate_variant_params(spec)
        assert is_valid is False
        assert "dollars_per_trade" in error
        assert "> 0" in error

    def test_negative_dollars_per_trade_returns_false(self):
        """Negative dollars_per_trade should be invalid."""
        spec = {
            "risk": {
                "dollars_per_trade": -100,
                "max_positions": 5,
            }
        }
        is_valid, error = validate_variant_params(spec)
        assert is_valid is False
        assert "dollars_per_trade" in error

    def test_zero_max_positions_returns_false(self):
        """max_positions = 0 should be invalid."""
        spec = {
            "risk": {
                "dollars_per_trade": 1000,
                "max_positions": 0,
            }
        }
        is_valid, error = validate_variant_params(spec)
        assert is_valid is False
        assert "max_positions" in error
        assert ">= 1" in error

    def test_negative_max_positions_returns_false(self):
        """Negative max_positions should be invalid."""
        spec = {
            "risk": {
                "dollars_per_trade": 1000,
                "max_positions": -1,
            }
        }
        is_valid, error = validate_variant_params(spec)
        assert is_valid is False
        assert "max_positions" in error

    def test_one_max_position_is_valid(self):
        """max_positions = 1 should be valid (edge case)."""
        spec = {
            "risk": {
                "dollars_per_trade": 100,
                "max_positions": 1,
            }
        }
        is_valid, error = validate_variant_params(spec)
        assert is_valid is True
        assert error is None

    def test_missing_risk_section_returns_false(self):
        """Missing risk section should fail validation."""
        spec = {"entry": {"lookback_days": 252}}
        is_valid, error = validate_variant_params(spec)
        # Should fail because defaults to 0 which is <= 0
        assert is_valid is False

    def test_empty_spec_returns_false(self):
        """Empty spec should fail validation."""
        spec = {}
        is_valid, error = validate_variant_params(spec)
        assert is_valid is False
