"""TestGenerator - Generates RunPlan variants from a base ExecutionSpec."""

import itertools

from app.services.strategy.models import ExecutionSpec
from app.services.testing.models import (
    GeneratorConstraints,
    RunPlan,
    RunVariant,
    hash_variant,
)


class TestGenerator:
    """Generates RunPlan with variants from a base ExecutionSpec.

    Produces:
    1. Baseline variant (always first, empty overrides)
    2. Grid variants (cartesian product of sweep values)
    3. Ablation variants (one param reset to base default per variant)

    Deduplication ensures no duplicate variant_ids.
    max_variants truncation keeps baseline first.
    """

    def generate(
        self,
        base_spec: ExecutionSpec,
        dataset_ref: str,
        constraints: GeneratorConstraints,
    ) -> RunPlan:
        """Generate a RunPlan with variants from constraints.

        Args:
            base_spec: The base strategy specification.
            dataset_ref: Reference to the dataset for backtesting.
            constraints: Generator constraints including sweep values.

        Returns:
            RunPlan with baseline + grid + ablation variants.
        """
        spec_dict = base_spec.model_dump(mode="json")

        # Generate variant categories
        baseline = self._baseline_variant(spec_dict)
        grid = self._grid_variants(spec_dict, constraints)
        ablations = self._ablation_variants(spec_dict, grid, constraints)

        # Combine, dedupe, and truncate
        all_variants = [baseline] + grid + ablations
        variants = self._dedupe_preserve_order(all_variants)
        variants = variants[: constraints.max_variants]

        return RunPlan(
            workspace_id=base_spec.workspace_id,
            base_spec=spec_dict,
            variants=variants,
            objective=constraints.objective,
            dataset_ref=dataset_ref,
        )

    def _baseline_variant(self, spec_dict: dict) -> RunVariant:
        """Create baseline variant with empty overrides.

        Baseline is special:
        - label="baseline"
        - tags=["baseline"]
        - spec_overrides={}

        Uses direct construction rather than RunVariant.create() since
        empty overrides don't require dotted path validation.
        """
        variant_id = hash_variant(spec_dict, {})
        return RunVariant(
            variant_id=variant_id,
            label="baseline",
            spec_overrides={},
            tags=["baseline"],
        )

    def _grid_variants(
        self, spec_dict: dict, constraints: GeneratorConstraints
    ) -> list[RunVariant]:
        """Generate grid variants from cartesian product of sweep values.

        Each grid variant overrides all three parameters:
        - entry.lookback_days
        - risk.dollars_per_trade
        - risk.max_positions
        """
        lookbacks = constraints.lookback_days_values
        dollars = constraints.dollars_per_trade_values
        max_positions = constraints.max_positions_values

        # If any dimension is empty, no grid variants
        if not lookbacks or not dollars or not max_positions:
            return []

        variants = []
        for lookback, dollar, max_pos in itertools.product(
            lookbacks, dollars, max_positions
        ):
            overrides = {
                "entry.lookback_days": lookback,
                "risk.dollars_per_trade": dollar,
                "risk.max_positions": max_pos,
            }
            label = f"lookback={lookback},dollars={dollar},max_pos={max_pos}"
            variant = RunVariant.create(
                base_spec=spec_dict,
                spec_overrides=overrides,
                label=label,
                tags=["grid"],
            )
            variants.append(variant)

        return variants

    def _ablation_variants(
        self,
        spec_dict: dict,
        grid: list[RunVariant],
        constraints: GeneratorConstraints,
    ) -> list[RunVariant]:
        """Generate ablation variants by resetting one param to base default.

        Only generated if:
        - include_ablations=True
        - grid is non-empty

        Takes first grid variant's overrides as reference.
        For each param, keeps other two from ref, sets ablated param to base default.
        """
        if not constraints.include_ablations or not grid:
            return []

        # Get first grid variant's overrides as reference
        first_grid = grid[0]
        ref_overrides = first_grid.spec_overrides.copy()

        # Base default values
        base_lookback = spec_dict["entry"]["lookback_days"]
        base_dollars = spec_dict["risk"]["dollars_per_trade"]
        base_max_pos = spec_dict["risk"]["max_positions"]

        variants = []

        # Ablation 1: Reset lookback_days to base
        ablation1_overrides = ref_overrides.copy()
        ablation1_overrides["entry.lookback_days"] = base_lookback
        variants.append(
            RunVariant.create(
                base_spec=spec_dict,
                spec_overrides=ablation1_overrides,
                label=f"ablation:lookback={base_lookback}",
                tags=["ablation"],
            )
        )

        # Ablation 2: Reset dollars_per_trade to base
        ablation2_overrides = ref_overrides.copy()
        ablation2_overrides["risk.dollars_per_trade"] = base_dollars
        variants.append(
            RunVariant.create(
                base_spec=spec_dict,
                spec_overrides=ablation2_overrides,
                label=f"ablation:dollars={base_dollars}",
                tags=["ablation"],
            )
        )

        # Ablation 3: Reset max_positions to base
        ablation3_overrides = ref_overrides.copy()
        ablation3_overrides["risk.max_positions"] = base_max_pos
        variants.append(
            RunVariant.create(
                base_spec=spec_dict,
                spec_overrides=ablation3_overrides,
                label=f"ablation:max_pos={base_max_pos}",
                tags=["ablation"],
            )
        )

        return variants

    def _dedupe_preserve_order(self, variants: list[RunVariant]) -> list[RunVariant]:
        """Remove duplicates while preserving order of first occurrence.

        Uses variant_id as the deduplication key.
        """
        seen: set[str] = set()
        result: list[RunVariant] = []

        for variant in variants:
            if variant.variant_id not in seen:
                seen.add(variant.variant_id)
                result.append(variant)

        return result
