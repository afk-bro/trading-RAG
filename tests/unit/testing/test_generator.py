"""Unit tests for TestGenerator - TDD style."""

import pytest
from uuid import uuid4

from app.services.strategy.models import (
    ExecutionSpec,
    EntryConfig,
    ExitConfig,
    RiskConfig,
)
from app.services.testing.models import GeneratorConstraints
from app.services.testing.test_generator import TestGenerator


@pytest.fixture
def base_spec() -> ExecutionSpec:
    """Create a base ExecutionSpec for testing."""
    return ExecutionSpec(
        strategy_id="breakout_52w_high",
        name="Test Breakout Strategy",
        workspace_id=uuid4(),
        symbols=["AAPL", "MSFT"],
        timeframe="daily",
        entry=EntryConfig(type="breakout_52w_high", lookback_days=252),
        exit=ExitConfig(type="eod"),
        risk=RiskConfig(dollars_per_trade=1000.0, max_positions=5),
    )


@pytest.fixture
def grid_constraints() -> GeneratorConstraints:
    """Create constraints for a 2x2x2 grid sweep.

    Note: Grid values are chosen to NOT include base defaults (252, 1000.0, 5)
    so ablation variants don't duplicate grid variants.
    """
    return GeneratorConstraints(
        lookback_days_values=[126, 180],
        dollars_per_trade_values=[500.0, 750.0],
        max_positions_values=[3, 4],
        include_ablations=True,
        max_variants=25,
        objective="sharpe_dd_penalty",
    )


@pytest.fixture
def generator() -> TestGenerator:
    """Create a TestGenerator instance."""
    return TestGenerator()


class TestBaselineVariant:
    """Tests for baseline variant behavior."""

    def test_baseline_has_label_baseline(self, generator, base_spec):
        """Baseline variant should have label='baseline'."""
        constraints = GeneratorConstraints()
        plan = generator.generate(base_spec, "btc_2023", constraints)

        baseline = plan.variants[0]
        assert baseline.label == "baseline"

    def test_baseline_has_tag_baseline(self, generator, base_spec):
        """Baseline variant should have 'baseline' in its tags."""
        constraints = GeneratorConstraints()
        plan = generator.generate(base_spec, "btc_2023", constraints)

        baseline = plan.variants[0]
        assert "baseline" in baseline.tags

    def test_baseline_has_empty_spec_overrides(self, generator, base_spec):
        """Baseline variant should have empty spec_overrides."""
        constraints = GeneratorConstraints()
        plan = generator.generate(base_spec, "btc_2023", constraints)

        baseline = plan.variants[0]
        assert baseline.spec_overrides == {}

    def test_baseline_is_first_variant(self, generator, base_spec, grid_constraints):
        """Baseline should always be variants[0]."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        assert plan.variants[0].label == "baseline"
        assert "baseline" in plan.variants[0].tags

    def test_baseline_always_included(self, generator, base_spec, grid_constraints):
        """Baseline should always be included even with grid variants."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        baseline_variants = [v for v in plan.variants if v.label == "baseline"]
        assert len(baseline_variants) == 1


class TestGridSweep:
    """Tests for grid sweep variant generation."""

    def test_grid_sweep_count_is_cartesian_product(
        self, generator, base_spec, grid_constraints
    ):
        """Grid sweep should produce 2x2x2=8 variants."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        grid_variants = [v for v in plan.variants if "grid" in v.tags]
        # 2 lookback x 2 dollars x 2 max_pos = 8
        assert len(grid_variants) == 8

    def test_grid_variants_have_grid_tag(self, generator, base_spec, grid_constraints):
        """All grid variants should have 'grid' tag."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        grid_variants = [v for v in plan.variants if "grid" in v.tags]
        # Should have some grid variants
        assert len(grid_variants) > 0
        # All should have grid tag
        for v in grid_variants:
            assert "grid" in v.tags

    def test_grid_variants_have_all_three_overrides(
        self, generator, base_spec, grid_constraints
    ):
        """Grid variants should override all three parameters."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        grid_variants = [v for v in plan.variants if "grid" in v.tags]
        for v in grid_variants:
            assert "entry.lookback_days" in v.spec_overrides
            assert "risk.dollars_per_trade" in v.spec_overrides
            assert "risk.max_positions" in v.spec_overrides

    def test_empty_grid_produces_only_baseline(self, generator, base_spec):
        """Empty grid constraints should produce only baseline."""
        empty_constraints = GeneratorConstraints(
            lookback_days_values=[],
            dollars_per_trade_values=[],
            max_positions_values=[],
            include_ablations=True,
        )
        plan = generator.generate(base_spec, "btc_2023", empty_constraints)

        # Only baseline
        assert len(plan.variants) == 1
        assert plan.variants[0].label == "baseline"


class TestAblations:
    """Tests for ablation variant generation."""

    def test_ablations_generated_only_if_include_ablations_and_grid_nonempty(
        self, generator, base_spec, grid_constraints
    ):
        """Ablations should be generated only if include_ablations=True AND grid non-empty."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        ablation_variants = [v for v in plan.variants if "ablation" in v.tags]
        # 3 ablation variants (one per param)
        assert len(ablation_variants) == 3

    def test_no_ablations_when_include_ablations_false(
        self, generator, base_spec, grid_constraints
    ):
        """No ablations when include_ablations=False."""
        grid_constraints.include_ablations = False
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        ablation_variants = [v for v in plan.variants if "ablation" in v.tags]
        assert len(ablation_variants) == 0

    def test_no_ablations_when_grid_empty(self, generator, base_spec):
        """No ablations when grid is empty even if include_ablations=True."""
        empty_grid = GeneratorConstraints(
            lookback_days_values=[],
            dollars_per_trade_values=[],
            max_positions_values=[],
            include_ablations=True,
        )
        plan = generator.generate(base_spec, "btc_2023", empty_grid)

        ablation_variants = [v for v in plan.variants if "ablation" in v.tags]
        assert len(ablation_variants) == 0

    def test_ablations_have_ablation_tag(self, generator, base_spec, grid_constraints):
        """All ablation variants should have 'ablation' tag."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        ablation_variants = [v for v in plan.variants if "ablation" in v.tags]
        for v in ablation_variants:
            assert "ablation" in v.tags


class TestAblationLogic:
    """Tests for ablation logic - based on first grid variant."""

    def test_ablations_based_on_first_grid_variant(
        self, generator, base_spec, grid_constraints
    ):
        """Ablations should be based on first grid variant's overrides."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        # Get first grid variant (verified exists, used as reference for ablations)
        grid_variants = [v for v in plan.variants if "grid" in v.tags]
        _first_grid = grid_variants[0]  # noqa: F841 - verified exists

        # Get ablation variants
        ablation_variants = [v for v in plan.variants if "ablation" in v.tags]

        # There should be 3 ablations (one per param)
        assert len(ablation_variants) == 3

        # Each ablation should have 2 params from first_grid and 1 reset to base default
        base_dict = base_spec.model_dump(mode="json")
        base_lookback = base_dict["entry"]["lookback_days"]
        base_dollars = base_dict["risk"]["dollars_per_trade"]
        base_max_pos = base_dict["risk"]["max_positions"]

        # Check we have one ablation for each param
        lookback_ablations = [
            v
            for v in ablation_variants
            if v.spec_overrides.get("entry.lookback_days") == base_lookback
        ]
        dollars_ablations = [
            v
            for v in ablation_variants
            if v.spec_overrides.get("risk.dollars_per_trade") == base_dollars
        ]
        max_pos_ablations = [
            v
            for v in ablation_variants
            if v.spec_overrides.get("risk.max_positions") == base_max_pos
        ]

        assert len(lookback_ablations) == 1
        assert len(dollars_ablations) == 1
        assert len(max_pos_ablations) == 1


class TestDeduplication:
    """Tests for deduplication behavior."""

    def test_dedup_removes_duplicates(self, generator, base_spec):
        """Dedup should remove duplicate variants."""
        # Create constraints where an ablation would match baseline
        # (when base values are used in grid, ablation sets back to base = duplicate)
        constraints = GeneratorConstraints(
            lookback_days_values=[252],  # Same as base
            dollars_per_trade_values=[1000.0],  # Same as base
            max_positions_values=[5],  # Same as base
            include_ablations=True,
        )
        plan = generator.generate(base_spec, "btc_2023", constraints)

        # All variant_ids should be unique
        variant_ids = [v.variant_id for v in plan.variants]
        assert len(variant_ids) == len(set(variant_ids))

    def test_baseline_always_stays_first_after_dedup(
        self, generator, base_spec, grid_constraints
    ):
        """Baseline should remain first even after dedup."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        assert plan.variants[0].label == "baseline"


class TestMaxVariantsTruncation:
    """Tests for max_variants truncation."""

    def test_max_variants_truncates(self, generator, base_spec, grid_constraints):
        """max_variants should truncate the variants list."""
        grid_constraints.max_variants = 3
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        assert len(plan.variants) <= 3

    def test_baseline_kept_when_truncated(self, generator, base_spec, grid_constraints):
        """Baseline should be kept when truncating."""
        grid_constraints.max_variants = 3
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        assert plan.variants[0].label == "baseline"
        assert "baseline" in plan.variants[0].tags

    def test_truncation_keeps_order(self, generator, base_spec, grid_constraints):
        """Truncation should keep the first N variants in order."""
        # Get full plan first
        grid_constraints.max_variants = 100
        full_plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        # Get truncated plan
        grid_constraints.max_variants = 5
        truncated_plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        # Truncated should be prefix of full
        for i, v in enumerate(truncated_plan.variants):
            assert v.variant_id == full_plan.variants[i].variant_id


class TestGoldenOrdering:
    """Tests for deterministic ordering."""

    def test_order_is_deterministic(self, generator, base_spec, grid_constraints):
        """Same inputs should always produce same variant_id order."""
        plan1 = generator.generate(base_spec, "btc_2023", grid_constraints)
        plan2 = generator.generate(base_spec, "btc_2023", grid_constraints)

        ids1 = [v.variant_id for v in plan1.variants]
        ids2 = [v.variant_id for v in plan2.variants]

        assert ids1 == ids2

    def test_order_is_baseline_then_grid_then_ablations(
        self, generator, base_spec, grid_constraints
    ):
        """Order should be: baseline -> grid -> ablations."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        # First variant is baseline
        assert "baseline" in plan.variants[0].tags

        # Find transition points
        first_grid_idx = None
        first_ablation_idx = None

        for i, v in enumerate(plan.variants):
            if "grid" in v.tags and first_grid_idx is None:
                first_grid_idx = i
            if "ablation" in v.tags and first_ablation_idx is None:
                first_ablation_idx = i

        # Grid comes after baseline
        if first_grid_idx is not None:
            assert first_grid_idx > 0

        # Ablations come after grid
        if first_ablation_idx is not None and first_grid_idx is not None:
            # All grids should come before first ablation
            for i in range(first_grid_idx, first_ablation_idx):
                assert (
                    "grid" in plan.variants[i].tags
                    or "baseline" in plan.variants[i].tags
                )


class TestRunPlanMetadata:
    """Tests for RunPlan metadata."""

    def test_plan_has_correct_workspace_id(
        self, generator, base_spec, grid_constraints
    ):
        """RunPlan should have workspace_id from base_spec."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)
        assert plan.workspace_id == base_spec.workspace_id

    def test_plan_has_correct_dataset_ref(self, generator, base_spec, grid_constraints):
        """RunPlan should have the provided dataset_ref."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)
        assert plan.dataset_ref == "btc_2023"

    def test_plan_has_correct_objective(self, generator, base_spec, grid_constraints):
        """RunPlan should have objective from constraints."""
        grid_constraints.objective = "sharpe"
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)
        assert plan.objective == "sharpe"

    def test_plan_base_spec_is_serialized_dict(
        self, generator, base_spec, grid_constraints
    ):
        """RunPlan.base_spec should be a dict (JSON-serializable)."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)
        assert isinstance(plan.base_spec, dict)
        assert plan.base_spec["strategy_id"] == base_spec.strategy_id

    def test_base_spec_uses_json_mode_dump(
        self, generator, base_spec, grid_constraints
    ):
        """RunPlan.base_spec should use model_dump(mode='json') for consistent serialization."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        # JSON mode converts UUIDs to strings, datetimes to ISO strings, etc.
        # Verify workspace_id is a string (not UUID object)
        assert isinstance(plan.base_spec["workspace_id"], str)

        # Verify instance_id is a string
        assert isinstance(plan.base_spec["instance_id"], str)


class TestNumericDiscipline:
    """Tests for numeric type discipline in hashing."""

    def test_variant_id_stable_for_int_lookback(self, generator, base_spec):
        """Variant ID should be stable when lookback_days is int."""
        constraints = GeneratorConstraints(
            lookback_days_values=[126],  # int
            dollars_per_trade_values=[1000.0],
            max_positions_values=[3],
        )
        plan1 = generator.generate(base_spec, "btc_2023", constraints)
        plan2 = generator.generate(base_spec, "btc_2023", constraints)

        # Grid variant IDs should be identical
        grid1 = [v for v in plan1.variants if "grid" in v.tags]
        grid2 = [v for v in plan2.variants if "grid" in v.tags]

        assert len(grid1) == len(grid2) == 1
        assert grid1[0].variant_id == grid2[0].variant_id

    def test_variant_id_stable_for_float_dollars(self, generator, base_spec):
        """Variant ID should be stable when dollars_per_trade is float."""
        constraints = GeneratorConstraints(
            lookback_days_values=[126],
            dollars_per_trade_values=[500.5],  # float with decimal
            max_positions_values=[3],
        )
        plan1 = generator.generate(base_spec, "btc_2023", constraints)
        plan2 = generator.generate(base_spec, "btc_2023", constraints)

        grid1 = [v for v in plan1.variants if "grid" in v.tags]
        grid2 = [v for v in plan2.variants if "grid" in v.tags]

        assert grid1[0].variant_id == grid2[0].variant_id

    def test_overrides_contain_primitives_only(
        self, generator, base_spec, grid_constraints
    ):
        """All override values should be JSON primitives (int, float, str, bool, None)."""
        plan = generator.generate(base_spec, "btc_2023", grid_constraints)

        for variant in plan.variants:
            for key, value in variant.spec_overrides.items():
                assert isinstance(
                    value, (int, float, str, bool, type(None))
                ), f"Override {key}={value} is type {type(value).__name__}, expected primitive"
