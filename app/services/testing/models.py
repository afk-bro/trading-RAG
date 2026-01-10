"""Core models for testing package - Test generation and orchestration."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, Field, computed_field


# CRITICAL: Constant namespace for uuid5 - used to generate isolated namespaces per variant
# This MUST be a constant UUID, never run_plan_id (which varies per run)
TESTING_VARIANT_NAMESPACE = UUID("8f3c2a2e-9d9a-4b2e-9f6b-3c3a4c1e9a01")


def get_variant_namespace(run_plan_id: UUID, variant_id: str) -> UUID:
    """Generate a deterministic namespace for a variant.

    Same (run_plan_id, variant_id) → same namespace across runs.
    Different variant_id → different namespace.
    Used to isolate broker state between variants.

    Args:
        run_plan_id: The run plan's UUID
        variant_id: The variant's 16-char hex ID

    Returns:
        UUID derived from the constant namespace + plan+variant identifier
    """
    return uuid5(TESTING_VARIANT_NAMESPACE, f"{run_plan_id}:{variant_id}")


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, stable across runs.

    Produces deterministic JSON output regardless of key insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def hash_variant(base_spec: dict, overrides: dict) -> str:
    """Deterministic variant ID from canonical hash.

    Returns a 16-character hex string derived from SHA-256 hash of
    the canonical JSON representation of base_spec and overrides.
    """
    payload = {"base": base_spec, "overrides": overrides}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


def validate_variant_params(materialized_spec: dict) -> tuple[bool, str | None]:
    """Validate that materialized spec has valid parameters.

    Checks:
    - risk.dollars_per_trade > 0
    - risk.max_positions >= 1

    Args:
        materialized_spec: The spec dict after overrides are applied.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        dollars_per_trade = materialized_spec.get("risk", {}).get("dollars_per_trade", 0)
        max_positions = materialized_spec.get("risk", {}).get("max_positions", 0)

        if dollars_per_trade <= 0:
            return False, f"dollars_per_trade must be > 0, got {dollars_per_trade}"

        if max_positions < 1:
            return False, f"max_positions must be >= 1, got {max_positions}"

        return True, None
    except (KeyError, TypeError) as e:
        return False, f"Invalid spec structure: {e}"


def apply_overrides(spec_dict: dict, overrides: dict[str, Any]) -> dict:
    """Apply dotted-path overrides to spec dict.

    Args:
        spec_dict: The base specification dictionary to apply overrides to.
        overrides: Dict mapping dotted paths (e.g., "params.window") to values.

    Returns:
        A new dict with overrides applied (deep copy, does not mutate original).

    Raises:
        KeyError: If the path traverses missing keys (strict v0 behavior).
    """
    result = deepcopy(spec_dict)

    for path, value in overrides.items():
        keys = path.split(".")
        current = result

        # Traverse to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                raise KeyError(f"Missing key '{key}' in path '{path}'")
            current = current[key]

        # Set the final key
        final_key = keys[-1]
        if final_key not in current:
            raise KeyError(f"Missing leaf key '{final_key}' in path '{path}'")
        current[final_key] = value

    return result


class RunPlanStatus(str, Enum):
    """Status of a run plan."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


class GeneratorConstraints(BaseModel):
    """Constraints for test generation / parameter sweep."""

    lookback_days_values: list[int] = Field(default_factory=list)
    dollars_per_trade_values: list[float] = Field(default_factory=list)
    max_positions_values: list[int] = Field(default_factory=list)
    include_ablations: bool = True
    max_variants: int = Field(default=25, le=200, description="Max variants (hard cap at 200)")
    objective: str = "sharpe_dd_penalty"


class RunVariant(BaseModel):
    """Individual variant with parameter overrides.

    Represents a single configuration to be tested, consisting of
    a variant_id (deterministic hash), a human-readable label,
    and the specific overrides to apply to the base specification.
    """

    variant_id: str = Field(description="16-char hex ID derived from base+overrides hash")
    label: str = Field(description="Human-readable label for the variant")
    spec_overrides: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        base_spec: dict,
        spec_overrides: dict[str, Any],
        label: str = "",
        tags: list[str] | None = None,
    ) -> RunVariant:
        """Factory method that validates overrides and generates variant ID.

        Args:
            base_spec: The base specification dictionary.
            spec_overrides: Dict mapping dotted paths to scalar values.
            label: Human-readable label for the variant.
            tags: Optional list of tags for the variant.

        Returns:
            A validated RunVariant instance.

        Raises:
            ValueError: If override keys don't contain dots (unless empty),
                       if paths have empty segments, or if values are dicts.
        """
        # Validate override keys and values
        for key, value in spec_overrides.items():
            # Key must contain a dot
            if "." not in key:
                raise ValueError(
                    f"Override key '{key}' must contain a dot (dotted path required)"
                )

            # No empty segments in path
            segments = key.split(".")
            if any(not segment for segment in segments):
                raise ValueError(
                    f"Override key '{key}' has empty segment (e.g., 'risk..x' is invalid)"
                )

            # Value must be scalar (not a dict)
            if isinstance(value, dict):
                raise ValueError(
                    f"Override value for '{key}' must be a scalar, not a dict"
                )

        variant_id = hash_variant(base_spec, spec_overrides)

        return cls(
            variant_id=variant_id,
            label=label,
            spec_overrides=spec_overrides,
            tags=tags or [],
        )


class RunPlan(BaseModel):
    """Plan for executing a batch of backtest variants.

    Contains the base specification, list of variants to test,
    objective function, and dataset reference.
    """

    run_plan_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the run plan")
    workspace_id: UUID = Field(description="Workspace this plan belongs to")
    base_spec: dict = Field(description="Base strategy specification")
    variants: list[RunVariant] = Field(default_factory=list)
    objective: str = Field(default="sharpe_dd_penalty", description="Objective function for optimization")
    dataset_ref: str = Field(description="Reference to the dataset used for backtesting")
    status: RunPlanStatus = RunPlanStatus.pending
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def n_variants(self) -> int:
        """Number of variants in the plan."""
        return len(self.variants)


class VariantMetrics(BaseModel):
    """Metrics from a variant backtest run.

    Contains all the standard performance metrics captured
    from a backtest execution.
    """

    sharpe: Optional[float] = Field(default=None, description="Sharpe ratio (None if <2 trades)")
    return_pct: float
    max_drawdown_pct: float
    trade_count: int
    win_rate: float = Field(default=0.0, description="Win rate (0.0 if no trades)")
    ending_equity: float = Field(description="Final equity value")
    gross_profit: float = Field(description="Total profit from winning trades")
    gross_loss: float = Field(description="Total loss from losing trades (should be <= 0)")
    profit_factor: Optional[float] = None


class RunResult(BaseModel):
    """Results from executing a variant.

    Contains the outcome of running a single variant, including
    success/failure/skipped status, metrics (if successful), and any
    error information (if failed) or skip reason (if skipped).
    """

    run_plan_id: UUID = Field(description="ID of the run plan this result belongs to")
    variant_id: str = Field(description="ID of the variant that was run")
    status: str = Field(description="'success', 'failed', or 'skipped'")
    metrics: Optional[VariantMetrics] = None
    objective_score: Optional[float] = Field(default=None, description="Computed objective score")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    skip_reason: Optional[str] = Field(default=None, description="Reason if skipped (invalid params)")
    started_at: datetime = Field(description="When the run started")
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = Field(default=None, description="Duration in milliseconds")
    events_recorded: int = Field(default=0, description="Number of events recorded during run")
