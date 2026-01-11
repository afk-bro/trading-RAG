# Step 2: Regime Explainability Payload

## Overview

Add structured evidence to `RegimeSnapshot` so both UI dashboards and LLM pipelines can understand *why* tags were assigned. This enables:
- Debugging ("why did this get tagged uptrend?")
- Near-miss surfacing ("almost overbought, RSI 68")
- LLM reasoning ("this regime is flat because trend_strength=0.22 is below 0.3 threshold")

## Consumers

- **UI**: Show badge explanations, highlight near-misses
- **LLM**: Include in context for parameter recommendation reasoning

## Data Structures

### Location

All new types go in `app/services/kb/types.py` alongside existing `RegimeSnapshot`.

### Type Definitions

```python
from dataclasses import dataclass, field
from typing import Literal

Op = Literal[">=", "<=", ">", "<", "=="]
Transform = Literal["abs"]

@dataclass
class TagRule:
    """Declarative rule for computing a single tag."""
    tag: str                           # e.g., "uptrend", "oversold"
    rule_id: str                       # e.g., "uptrend_strong", "oversold_zscore"
    metric: str                        # e.g., "trend_strength", "zscore"
    op: Op                             # Comparison operator
    threshold: float                   # Threshold value
    group: str = "default"             # Grouping key for OR-over-groups logic
    transform: Transform | None = None # Optional transform (e.g., "abs")
    units: str | None = None           # Display hint: "%", "σ", etc.
    is_headline: bool = False          # Surface in near-misses if failed


@dataclass
class TagEvidence:
    """Evidence of a single rule evaluation."""
    tag: str                           # Tag this rule contributes to
    rule_id: str                       # Which rule was evaluated
    passed: bool                       # Did this rule pass?
    metric: str                        # Which metric was evaluated
    value: float                       # Raw value from snapshot
    op: Op                             # Operator used
    threshold: float                   # Threshold compared against
    units: str | None = None           # Display hint
    margin: float | None = None        # Normalized: positive = passed by X
    transform: Transform | None = None # Transform applied (for UI: "|zscore|")
    computed_value: float | None = None # Value after transform (if different)
    confidence: float | None = None    # Reserved for v2
```

### Margin Normalization

Margin is always normalized so **positive = rule satisfied**:

| Op | Margin Formula | Interpretation |
|----|----------------|----------------|
| `>=` | `value - threshold` | +0.1 = "0.1 above threshold" |
| `>` | `value - threshold` | +0.1 = "0.1 above threshold" |
| `<=` | `threshold - value` | +0.1 = "0.1 below threshold" |
| `<` | `threshold - value` | +0.1 = "0.1 below threshold" |
| `==` | `-abs(value - threshold)` | 0 = exact match, negative = distance |

## Ruleset Design

### Evaluator Semantics

1. **Group rules by `(tag, group)`**
2. **AND within group**: All rules in a group must pass
3. **OR across groups**: Any group passing assigns the tag

This handles both:
- Compound requirements: `uptrend` needs `trend_strength >= 0.6 AND trend_dir > 0`
- Alternative triggers: `oversold` from `zscore < -1.5 OR rsi < 30`

### Exclusive Families

Some tags are mutually exclusive (at most one can be assigned):

```python
EXCLUSIVE_FAMILIES: dict[str, list[str]] = {
    "trend": ["uptrend", "downtrend", "flat"],
    # Note: "trending" (dir=0) is a fallback, handled by priority
}
```

**Legacy behavior to preserve:**
- If `trend_strength > 0.6`: assign uptrend/downtrend based on direction, or skip if dir=0
- If `trend_strength < 0.3`: assign flat
- Middle band (0.3-0.6): no trend tag

The evaluator applies priority order within families. The first passing tag in the family wins.

### Near-Miss Policy for OR Groups

When OR-over-groups is in play, near-misses can be noisy. Policy:

1. **If tag passes**: Skip near-miss emission entirely for that tag (no "almost" for something that happened)
2. **If tag fails**: Emit near-miss only for the **closest group** (highest group margin)

This prevents tooltips like "oversold: RSI almost triggered AND zscore almost triggered"

### Default Ruleset

```python
DEFAULT_RULESET: list[TagRule] = [
    # Uptrend: strong trend + positive direction (AND within group)
    TagRule("uptrend", "uptrend_strength", "trend_strength", ">=", TREND_STRONG_THRESHOLD,
            is_headline=True, group="default"),
    TagRule("uptrend", "uptrend_dir", "trend_dir", ">", 0, group="default"),

    # Downtrend: strong trend + negative direction (AND within group)
    TagRule("downtrend", "downtrend_strength", "trend_strength", ">=", TREND_STRONG_THRESHOLD,
            is_headline=True, group="default"),
    TagRule("downtrend", "downtrend_dir", "trend_dir", "<", 0, group="default"),

    # Flat: weak trend (single rule)
    TagRule("flat", "flat_weak_trend", "trend_strength", "<", TREND_WEAK_THRESHOLD,
            is_headline=True),

    # Volatility (mutually exclusive, no middle tag)
    TagRule("low_vol", "low_vol_atr", "atr_pct", "<", VOL_LOW_THRESHOLD,
            units="%", is_headline=True),
    TagRule("high_vol", "high_vol_atr", "atr_pct", ">", VOL_HIGH_THRESHOLD,
            units="%", is_headline=True),

    # Mean-reverting: requires flat + extreme zscore
    TagRule("mean_reverting", "mr_flat", "trend_strength", "<", TREND_WEAK_THRESHOLD,
            group="default"),
    TagRule("mean_reverting", "mr_zscore", "zscore", ">", ZSCORE_MEAN_REVERT_THRESHOLD,
            transform="abs", units="σ", is_headline=True, group="default"),

    # Choppy: requires flat + narrow bands
    TagRule("choppy", "choppy_flat", "trend_strength", "<", TREND_WEAK_THRESHOLD,
            group="default"),
    TagRule("choppy", "choppy_bb", "bb_width_pct", "<", BB_WIDTH_CHOPPY_THRESHOLD,
            units="%", is_headline=True, group="default"),

    # Efficiency (mutually exclusive, no middle tag)
    TagRule("noisy", "noisy_er", "efficiency_ratio", "<", ER_NOISY_THRESHOLD, is_headline=True),
    TagRule("efficient", "efficient_er", "efficiency_ratio", ">", ER_EFFICIENT_THRESHOLD, is_headline=True),

    # Oversold: zscore OR rsi (OR-over-groups)
    TagRule("oversold", "oversold_zscore", "zscore", "<", ZSCORE_OVERSOLD,
            units="σ", is_headline=True, group="zscore"),
    TagRule("oversold", "oversold_rsi", "rsi", "<", RSI_OVERSOLD,
            units="RSI", is_headline=True, group="rsi"),

    # Overbought: zscore OR rsi (OR-over-groups)
    TagRule("overbought", "overbought_zscore", "zscore", ">", ZSCORE_OVERBOUGHT,
            units="σ", is_headline=True, group="zscore"),
    TagRule("overbought", "overbought_rsi", "rsi", ">", RSI_OVERBOUGHT,
            units="RSI", is_headline=True, group="rsi"),
]
```

### Evaluator Logic

```python
def evaluate_rules(
    features: dict[str, float],
    ruleset: list[TagRule] = DEFAULT_RULESET,
) -> tuple[list[str], list[TagEvidence]]:
    """
    Evaluate ruleset against features.

    Returns:
        (assigned_tags, all_evidence)
    """
    evidence: list[TagEvidence] = []

    # Group rules by (tag, group)
    from collections import defaultdict
    grouped: dict[str, dict[str, list[TagRule]]] = defaultdict(lambda: defaultdict(list))
    for rule in ruleset:
        grouped[rule.tag][rule.group].append(rule)

    assigned_tags = []

    for tag, groups in grouped.items():
        tag_assigned = False

        for group_name, rules in groups.items():
            # AND within group: all rules must pass
            group_passed = True
            group_evidence = []

            for rule in rules:
                # Get raw value and apply transform
                raw_value = features.get(rule.metric, 0.0)
                if rule.transform == "abs":
                    computed = abs(raw_value)
                else:
                    computed = raw_value

                # Evaluate using computed value
                passed = _evaluate_op(computed, rule.op, rule.threshold)
                margin = _compute_margin(computed, rule.op, rule.threshold)

                ev = TagEvidence(
                    tag=rule.tag,
                    rule_id=rule.rule_id,
                    passed=passed,
                    metric=rule.metric,
                    value=raw_value,  # Always store raw
                    op=rule.op,
                    threshold=rule.threshold,
                    units=rule.units,
                    margin=margin,
                    transform=rule.transform,
                    computed_value=computed if rule.transform else None,
                )
                group_evidence.append(ev)

                if not passed:
                    group_passed = False

            evidence.extend(group_evidence)

            # OR across groups: any group passing assigns tag
            if group_passed:
                tag_assigned = True

        if tag_assigned:
            assigned_tags.append(tag)

    return sorted(assigned_tags), evidence


def _evaluate_op(value: float, op: Op, threshold: float) -> bool:
    """Evaluate comparison operation."""
    if op == ">=":
        return value >= threshold
    elif op == ">":
        return value > threshold
    elif op == "<=":
        return value <= threshold
    elif op == "<":
        return value < threshold
    elif op == "==":
        return value == threshold
    return False


def _compute_margin(value: float, op: Op, threshold: float) -> float:
    """Compute normalized margin (positive = passed)."""
    if op in (">=", ">"):
        return value - threshold
    elif op in ("<=", "<"):
        return threshold - value
    elif op == "==":
        return -abs(value - threshold)
    return 0.0
```

## RegimeSnapshot Changes

### New Fields

```python
@dataclass
class RegimeSnapshot:
    # ... existing fields ...

    # New explainability fields (v1.1)
    tag_evidence: list[TagEvidence] = field(default_factory=list)
    schema_version: str = "regime_v1_1"  # Bump from regime_v1
```

### Backward Compatibility

- Old snapshots with `schema_version="regime_v1"` will have empty `tag_evidence`
- Readers should check version and handle gracefully
- `regime_tags` remains the source of truth for filtering

## API Changes

### RegimeSnapshot Response

No API changes required. The new fields are automatically serialized.

### Near-Miss Extraction

```python
def extract_near_misses(
    evidence: list[TagEvidence],
    margin_threshold: float = 0.15,
) -> list[TagEvidence]:
    """
    Extract headline rules that nearly passed.

    A near-miss is a headline rule that:
    - Failed (passed=False)
    - Has small negative margin (within threshold of passing)
    """
    return [
        ev for ev in evidence
        if not ev.passed
        and ev.margin is not None
        and ev.margin > -margin_threshold
    ]
```

## Schema Version Bump

Update `app/services/kb/constants.py`:

```python
REGIME_SCHEMA_VERSION: Final[str] = "regime_v1_1"
```

## File Changes Summary

| File | Changes |
|------|---------|
| `app/services/kb/types.py` | Add `TagRule`, `TagEvidence`, `Op`, `Transform` types |
| `app/services/kb/types.py` | Add `tag_evidence` field to `RegimeSnapshot` |
| `app/services/kb/constants.py` | Bump `REGIME_SCHEMA_VERSION` to `regime_v1_1` |
| `app/services/kb/constants.py` | Add `DEFAULT_RULESET` |
| `app/services/kb/regime.py` | Add `evaluate_rules()`, refactor `compute_tags()` |
| `tests/unit/services/kb/test_regime.py` | Add tests for rule evaluation |

## Implementation Steps

### Task 1: Types + Constants
- Add `Op`, `Transform`, `TagRule`, `TagEvidence` to `types.py`
- Add `tag_evidence` field to `RegimeSnapshot`
- Add `EXCLUSIVE_FAMILIES` to `constants.py`
- Update `REGIME_SCHEMA_VERSION` to `"regime_v1_1"`

### Task 2: Evaluator (Pure Function)
- Add `DEFAULT_RULESET` to `constants.py`
- Add `evaluate_rules(features, ruleset)` to `regime.py`
- Add helper functions `_evaluate_op()`, `_compute_margin()`
- Keep it deterministic, no I/O

### Task 3: Wire into compute_tags()
- Replace existing tagging logic with `evaluate_rules()` call
- Set `snapshot.tag_evidence = evidence`
- Apply exclusive family logic
- **Critical**: Output tags must match legacy `compute_tags()` behavior

### Task 4: Tests (~12-18 tests)
Minimum test matrix:
- AND within group: uptrend requires both rules
- OR across groups: oversold fires on RSI alone
- `abs` transform: mean_reverting when zscore=-1.2 but abs>=1.0
- Margin sign correctness for each op family
- Near-miss emission only for headline rules
- Near-miss OR policy (closest group only)
- Exclusive family: at most one of uptrend/downtrend/flat
- Legacy behavior: middle band (0.3-0.6) yields no trend tag

### Task 5: Serialization / Persistence
- Ensure `tag_evidence` included in JSON serialization
- Verify old snapshots without `tag_evidence` default to `[]`
- KB ingestion/backfill runs without migrations (additive JSONB change)

## Test Cases

### Rule Evaluation

```python
def test_uptrend_requires_both_strength_and_direction():
    # Strength passes, direction fails → no uptrend
    features = {"trend_strength": 0.7, "trend_dir": 0}
    tags, _ = evaluate_rules(features)
    assert "uptrend" not in tags

def test_uptrend_assigned_when_both_pass():
    features = {"trend_strength": 0.7, "trend_dir": 1}
    tags, _ = evaluate_rules(features)
    assert "uptrend" in tags

def test_oversold_via_zscore_only():
    features = {"zscore": -1.6, "rsi": 50}  # zscore triggers, rsi doesn't
    tags, _ = evaluate_rules(features)
    assert "oversold" in tags

def test_oversold_via_rsi_only():
    features = {"zscore": 0, "rsi": 25}  # rsi triggers, zscore doesn't
    tags, _ = evaluate_rules(features)
    assert "oversold" in tags

def test_mean_reverting_uses_abs_zscore():
    features = {"trend_strength": 0.2, "zscore": -1.2}  # abs(-1.2) > 1.0
    tags, _ = evaluate_rules(features)
    assert "mean_reverting" in tags
```

### Margin Normalization

```python
def test_margin_positive_when_passed():
    # zscore=-1.6, threshold=-1.5, op="<"
    # passed (−1.6 < −1.5), margin = threshold - value = -1.5 - (-1.6) = +0.1
    ...

def test_margin_negative_when_failed():
    # zscore=-1.4, threshold=-1.5, op="<"
    # failed (−1.4 > −1.5), margin = threshold - value = -1.5 - (-1.4) = -0.1
    ...
```

## Definition of Done

Implementation is complete when:

1. **Tags match legacy**: For a fixed fixture snapshot set, tags match legacy `compute_tags()` output
2. **Evidence is stable**: Includes threshold/op/metric/value/margin for each evaluated rule
3. **UI-ready**: Dashboards can render tooltips without calling the LLM
4. **KB compatible**: Ingestion/backfill can run on existing DB without migrations (additive JSONB)
5. **Tests pass**: All ~12-18 tests covering the test matrix above

## Open Questions

None - design is complete based on feedback discussion.

## Revision History

- 2026-01-10: Initial design based on brainstorming session
- 2026-01-10: Added computed_value field, near-miss OR policy, exclusive families per feedback
