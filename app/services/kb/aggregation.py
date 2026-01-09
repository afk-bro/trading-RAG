"""
KB aggregation module for combining trial parameters.

Handles:
- Computing weights per trial
- Weighted median for numeric params
- Weighted mode for categorical/bool params
- Constraint validation and repair
- Step snapping and clamping
"""

from dataclasses import dataclass, field
from typing import Optional
import math

import structlog

from app.services.kb.rerank import RerankedCandidate
from app.services.strategies.registry import StrategySpec, get_strategy
from app.services.strategies.params import ParamType

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Weight penalties
RELAXED_WEIGHT_MULTIPLIER = 0.5  # Penalty for relaxed filter candidates
METADATA_ONLY_WEIGHT_MULTIPLIER = 0.25  # Heavy penalty for no-embedding candidates

# Minimum weight to include in aggregation
MIN_AGGREGATION_WEIGHT = 0.01


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParamSpread:
    """Spread/confidence info for a single parameter."""

    name: str
    value: any  # Aggregated value
    count_used: int  # Number of trials contributing
    weight_sum: float  # Sum of weights contributing
    spread: Optional[float] = None  # IQR/stddev for numeric
    mode_fraction: Optional[float] = None  # Fraction voting for mode (categorical)


@dataclass
class AggregationResult:
    """Result of parameter aggregation."""

    params: dict  # Aggregated parameters
    spreads: dict[str, ParamSpread]  # Per-param spread info
    count_used: int  # Total trials used
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Weight Computation
# =============================================================================


def compute_weight(
    candidate: RerankedCandidate,
    max_rerank_score: float = 1.0,
) -> float:
    """
    Compute aggregation weight for a candidate.

    Args:
        candidate: Reranked candidate
        max_rerank_score: Maximum rerank score for normalization

    Returns:
        Weight (0-1), with penalties applied
    """
    # Base weight from rerank score
    if max_rerank_score > 0:
        base_weight = candidate.rerank_score / max_rerank_score
    else:
        base_weight = 0.5

    # Apply penalties
    if candidate._metadata_only:
        base_weight *= METADATA_ONLY_WEIGHT_MULTIPLIER
    elif candidate._relaxed:
        base_weight *= RELAXED_WEIGHT_MULTIPLIER

    return max(base_weight, MIN_AGGREGATION_WEIGHT)


# =============================================================================
# Weighted Statistics
# =============================================================================


def weighted_median(values: list[float], weights: list[float]) -> float:
    """
    Compute weighted median.

    Args:
        values: Numeric values
        weights: Corresponding weights

    Returns:
        Weighted median value
    """
    if not values or not weights:
        raise ValueError("Empty values or weights")

    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    # Sort by value
    sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
    sorted_values = [v for v, w in sorted_pairs]
    sorted_weights = [w for v, w in sorted_pairs]

    # Compute cumulative weights
    total_weight = sum(sorted_weights)
    if total_weight == 0:
        return sorted_values[len(sorted_values) // 2]

    cumsum = 0.0
    median_idx = 0

    for i, weight in enumerate(sorted_weights):
        cumsum += weight
        if cumsum >= total_weight / 2:
            median_idx = i
            break

    return sorted_values[median_idx]


def weighted_mode(values: list[any], weights: list[float]) -> tuple[any, float]:
    """
    Compute weighted mode (most common value weighted).

    Args:
        values: Any values (categorical, bool)
        weights: Corresponding weights

    Returns:
        (mode_value, fraction_of_total_weight)
    """
    if not values or not weights:
        raise ValueError("Empty values or weights")

    # Aggregate weights by value
    weight_by_value: dict[any, float] = {}
    for val, weight in zip(values, weights):
        # Handle hashability
        key = val if isinstance(val, (str, int, float, bool, type(None))) else str(val)
        weight_by_value[key] = weight_by_value.get(key, 0.0) + weight

    # Find max
    total_weight = sum(weights)
    if total_weight == 0:
        return values[0], 1.0

    mode_key = max(weight_by_value.keys(), key=lambda k: weight_by_value[k])
    mode_fraction = weight_by_value[mode_key] / total_weight

    # Convert back from key to original value type
    # Find first value matching key
    for val in values:
        key = val if isinstance(val, (str, int, float, bool, type(None))) else str(val)
        if key == mode_key:
            return val, mode_fraction

    return values[0], mode_fraction


def compute_iqr(values: list[float], weights: list[float]) -> float:
    """
    Compute weighted interquartile range.

    Args:
        values: Numeric values
        weights: Corresponding weights

    Returns:
        IQR (Q3 - Q1)
    """
    if len(values) < 2:
        return 0.0

    # Simple approach: use unweighted quartiles on weighted sample
    # For proper weighted quartiles, we'd need more complex logic
    sorted_values = sorted(values)
    n = len(sorted_values)

    q1_idx = n // 4
    q3_idx = (3 * n) // 4

    q1 = sorted_values[q1_idx]
    q3 = sorted_values[q3_idx]

    return q3 - q1


# =============================================================================
# Aggregation Logic
# =============================================================================


def aggregate_params(
    candidates: list[RerankedCandidate],
    strategy_spec: Optional[StrategySpec] = None,
) -> AggregationResult:
    """
    Aggregate parameters from top candidates.

    Args:
        candidates: Reranked candidates (assumed pre-sorted)
        strategy_spec: Optional strategy spec for type/constraint info

    Returns:
        AggregationResult with params, spreads, warnings
    """
    warnings: list[str] = []

    if not candidates:
        return AggregationResult(
            params={},
            spreads={},
            count_used=0,
            warnings=["no_candidates_for_aggregation"],
        )

    # Compute weights
    max_rerank = max(c.rerank_score for c in candidates) if candidates else 1.0
    weights = [compute_weight(c, max_rerank) for c in candidates]

    # Collect params from all candidates
    all_params: dict[str, list[tuple[any, float]]] = {}  # param_name -> [(value, weight)]

    for candidate, weight in zip(candidates, weights):
        params = candidate.payload.get("params", {})
        if not params:
            continue

        for param_name, value in params.items():
            if value is None:
                continue
            if param_name not in all_params:
                all_params[param_name] = []
            all_params[param_name].append((value, weight))

    # Aggregate each param
    aggregated: dict[str, any] = {}
    spreads: dict[str, ParamSpread] = {}

    for param_name, value_weight_pairs in all_params.items():
        values = [v for v, w in value_weight_pairs]
        param_weights = [w for v, w in value_weight_pairs]

        # Determine param type
        param_type = ParamType.FLOAT  # Default
        if strategy_spec and param_name in strategy_spec.params:
            param_type = strategy_spec.params[param_name].type

        # Aggregate based on type
        if param_type in (ParamType.INT, ParamType.FLOAT):
            # Numeric: weighted median
            try:
                numeric_values = [float(v) for v in values]
                agg_value = weighted_median(numeric_values, param_weights)

                # Preserve int type
                if param_type == ParamType.INT:
                    agg_value = int(round(agg_value))

                spread_value = compute_iqr(numeric_values, param_weights)

                spreads[param_name] = ParamSpread(
                    name=param_name,
                    value=agg_value,
                    count_used=len(values),
                    weight_sum=sum(param_weights),
                    spread=spread_value,
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to aggregate numeric param",
                    param=param_name,
                    error=str(e),
                )
                agg_value = values[0]  # Fallback to first
                spreads[param_name] = ParamSpread(
                    name=param_name,
                    value=agg_value,
                    count_used=len(values),
                    weight_sum=sum(param_weights),
                )

        else:
            # Categorical/bool: weighted mode
            mode_value, mode_frac = weighted_mode(values, param_weights)
            agg_value = mode_value

            spreads[param_name] = ParamSpread(
                name=param_name,
                value=agg_value,
                count_used=len(values),
                weight_sum=sum(param_weights),
                mode_fraction=mode_frac,
            )

        aggregated[param_name] = agg_value

    # Validate and repair if strategy spec available
    if strategy_spec:
        aggregated, repair_warnings = validate_and_repair_params(
            aggregated, strategy_spec
        )
        warnings.extend(repair_warnings)

    count_used = len(candidates)

    logger.info(
        "Parameter aggregation complete",
        param_count=len(aggregated),
        candidate_count=count_used,
    )

    return AggregationResult(
        params=aggregated,
        spreads=spreads,
        count_used=count_used,
        warnings=warnings,
    )


def validate_and_repair_params(
    params: dict,
    strategy_spec: StrategySpec,
) -> tuple[dict, list[str]]:
    """
    Validate and repair aggregated parameters.

    Applies:
    - Type coercion
    - Bound clamping
    - Step snapping (for int params)
    - Constraint checking

    Args:
        params: Aggregated parameters
        strategy_spec: Strategy specification

    Returns:
        (repaired_params, warnings)
    """
    warnings: list[str] = []
    repaired = params.copy()

    for param_name, spec in strategy_spec.params.items():
        if param_name not in repaired:
            continue

        value = repaired[param_name]
        original = value

        # Type coercion
        if spec.type == ParamType.INT:
            try:
                value = int(round(float(value)))
            except (ValueError, TypeError):
                warnings.append(f"param_{param_name}_type_coercion_failed")
                continue

        elif spec.type == ParamType.FLOAT:
            try:
                value = float(value)
            except (ValueError, TypeError):
                warnings.append(f"param_{param_name}_type_coercion_failed")
                continue

        elif spec.type == ParamType.BOOL:
            value = bool(value)

        elif spec.type == ParamType.ENUM:
            # Validate against choices
            if spec.choices and value not in spec.choices:
                # Find closest match or use default
                if spec.default is not None:
                    value = spec.default
                    warnings.append(f"param_{param_name}_invalid_enum_using_default")
                elif spec.choices:
                    value = spec.choices[0]
                    warnings.append(f"param_{param_name}_invalid_enum_using_first")

        # Bound clamping (numeric only)
        if spec.type in (ParamType.INT, ParamType.FLOAT):
            if spec.min_value is not None and value < spec.min_value:
                value = spec.min_value
                warnings.append(f"param_{param_name}_clamped_to_min")
            if spec.max_value is not None and value > spec.max_value:
                value = spec.max_value
                warnings.append(f"param_{param_name}_clamped_to_max")

            # Step snapping (int only)
            if spec.type == ParamType.INT and spec.step and spec.step > 1:
                # Snap to nearest step from min
                base = spec.min_value if spec.min_value is not None else 0
                offset = value - base
                steps = round(offset / spec.step)
                value = int(base + steps * spec.step)

                # Re-clamp after snapping
                if spec.min_value is not None and value < spec.min_value:
                    value = spec.min_value
                if spec.max_value is not None and value > spec.max_value:
                    value = spec.max_value

        if value != original:
            logger.debug(
                "Param repaired",
                param=param_name,
                original=original,
                repaired=value,
            )

        repaired[param_name] = value

    # Check inter-param constraints
    constraint_result = strategy_spec.validate_params(repaired)
    if not constraint_result.is_valid:
        for violation in constraint_result.constraint_violations:
            warnings.append(f"constraint_violation_{violation}")

    return repaired, warnings


# =============================================================================
# Confidence Computation
# =============================================================================


def compute_confidence(
    spreads: dict[str, ParamSpread],
    count_used: int,
    has_warnings: bool = False,
    used_relaxed: bool = False,
    used_metadata_fallback: bool = False,
    median_oos_trades: Optional[int] = None,
) -> float:
    """
    Compute confidence score for recommendation.

    Args:
        spreads: Parameter spreads
        count_used: Number of trials used
        has_warnings: Whether there are warnings
        used_relaxed: Whether relaxed filters were used
        used_metadata_fallback: Whether metadata-only fallback was used
        median_oos_trades: Median n_trades_oos across top trials

    Returns:
        Confidence score (0-1)
    """
    if count_used == 0:
        return 0.0

    # Base confidence from count
    # 10 trials = 0.5, 50 trials = 0.8, 100+ = 0.9
    count_conf = min(0.9, 0.3 + 0.006 * count_used)

    # Spread penalty (lower confidence if high variance)
    spread_penalties = []
    for spread in spreads.values():
        if spread.spread is not None and spread.value:
            # Relative spread
            try:
                rel_spread = abs(spread.spread / spread.value)
                penalty = min(rel_spread * 0.1, 0.2)
                spread_penalties.append(penalty)
            except (ZeroDivisionError, TypeError):
                pass
        if spread.mode_fraction is not None:
            # Mode fraction penalty: low agreement = low confidence
            if spread.mode_fraction < 0.5:
                spread_penalties.append(0.1)

    avg_spread_penalty = sum(spread_penalties) / len(spread_penalties) if spread_penalties else 0

    # Warning penalties
    warning_penalty = 0.1 if has_warnings else 0
    relaxed_penalty = 0.15 if used_relaxed else 0
    metadata_penalty = 0.3 if used_metadata_fallback else 0

    # Low trades penalty: cap confidence when statistical basis is weak
    # Trust threshold: 10 trades. Below this, cap confidence and apply penalty.
    low_trades_penalty = 0.0
    confidence_cap = 1.0
    if median_oos_trades is not None and median_oos_trades < 10:
        # With < 10 trades, Sharpe/drawdown are statistically noisy
        # Cap at "low" confidence (0.4) and add penalty
        confidence_cap = 0.4
        low_trades_penalty = 0.2

    confidence = count_conf - avg_spread_penalty - warning_penalty - relaxed_penalty - metadata_penalty - low_trades_penalty

    return max(0.0, min(confidence_cap, confidence))
