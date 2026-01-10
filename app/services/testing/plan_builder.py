"""PlanBuilder: Constructs immutable plan JSONB for run_plans table."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class PlanBuilder:
    """
    Builder for run_plans.plan JSONB structure.

    Constructs the three-layer plan format:
    - inputs: what was requested
    - resolved: what variants were generated
    - provenance: how to interpret it later
    """

    def __init__(
        self,
        base_spec: dict[str, Any],
        objective: str,
        constraints: dict[str, Any],
        dataset_ref: str,
        generator_name: str,
        generator_version: str,
        generator_config: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize plan builder.

        Args:
            base_spec: Base strategy specification
            objective: Objective function name
            constraints: Generator constraints
            dataset_ref: Dataset reference string
            generator_name: Name of generator (e.g., "grid_search_v1")
            generator_version: Version of generator
            generator_config: Optional generator configuration
            seed: Optional random seed
        """
        self._base_spec = base_spec
        self._objective = objective
        self._constraints = constraints
        self._dataset_ref = dataset_ref
        self._generator_name = generator_name
        self._generator_version = generator_version
        self._generator_config = generator_config or {}
        self._seed = seed
        self._variants: list[dict[str, Any]] = []

    def add_variant(
        self,
        variant_index: int,
        params: dict[str, Any],
        param_source: str,
    ) -> "PlanBuilder":
        """
        Add a resolved variant to the plan.

        Args:
            variant_index: 0-based index
            params: Fully materialized params for this variant
            param_source: Source type ("baseline", "grid", "ablation", etc.)

        Returns:
            self for chaining
        """
        self._variants.append(
            {
                "variant_index": variant_index,
                "params": params,
                "param_source": param_source,
            }
        )
        return self

    def build(self) -> dict[str, Any]:
        """
        Build the final plan JSONB structure.

        Returns:
            Complete plan dict with inputs, resolved, provenance
        """
        created_at = datetime.now(timezone.utc).isoformat()

        inputs = {
            "base_spec": self._base_spec,
            "objective": {
                "name": self._objective,
                "direction": "maximize",
            },
            "constraints": self._constraints,
            "dataset_ref": self._dataset_ref,
            "generator_config": self._generator_config,
        }

        resolved = {
            "n_variants": len(self._variants),
            "variants": self._variants,
        }

        # Compute fingerprints (deterministic, excluding timestamp)
        plan_content = canonical_json(
            {
                "inputs": inputs,
                "resolved": resolved,
            }
        )
        plan_fingerprint = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

        provenance = {
            "generator": {
                "name": self._generator_name,
                "version": self._generator_version,
            },
            "created_at": created_at,
            "seed": self._seed,
            "fingerprints": {
                "plan": f"sha256:{plan_fingerprint}",
            },
        }

        return {
            "inputs": inputs,
            "resolved": resolved,
            "provenance": provenance,
        }
