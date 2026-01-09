# Trading Knowledge Base Design

**Date:** 2026-01-08
**Status:** Approved
**Author:** Claude + Human collaboration

## Overview

A knowledge base that turns backtest results into searchable experience for parameter recommendations. Given a new dataset, the system finds historically similar market regimes and recommends strategy parameters based on what worked in those conditions.

**Core idea:** Case-based reasoning without ML. "Experience replay for strategies."

```
New Dataset → Compute Regime → Find Similar Trials → Aggregate Params → Recommend
                  ↓                    ↓                    ↓
            RegimeSnapshot      Qdrant Vector Search   Weighted Median
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         /kb/recommend                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. Parse OHLCV    │  2. Compute Regime  │  3. Retrieve Candidates  │
│     ↓              │        ↓            │         ↓                │
│  ParsedDataset     │  RegimeSnapshot     │  Qdrant + Filters        │
├─────────────────────────────────────────────────────────────────────┤
│  4. Rerank         │  5. Aggregate       │  6. Response             │
│     ↓              │        ↓            │         ↓                │
│  Jaccard + ties    │  Weighted Median    │  Params + Confidence     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Section 1: RegimeSnapshot Schema

Market regime features computed from an OHLCV window. Designed for embedding similarity and deterministic tagging.

### Dataclass

```python
@dataclass
class RegimeSnapshot:
    """Market regime features computed from OHLCV window (v1)."""

    # Versioning (reproducibility)
    schema_version: str = "regime_v1"
    feature_params: dict = field(default_factory=lambda: {
        "atr_period": 14,
        "rsi_period": 14,
        "bb_period": 20,
        "bb_k": 2.0,
        "z_period": 20,
        "trend_lookback": 50,
    })
    timeframe: str | None = None

    # Audit
    computed_at: str | None = None         # ISO UTC
    computation_source: str | None = None  # "live" | "backfill"

    # Window metadata
    n_bars: int = 0
    effective_n_bars: int = 0              # After NaN drop from rolling
    ts_start: str = ""                     # ISO UTC
    ts_end: str = ""

    # Volatility / dispersion
    atr_pct: float = 0.0                   # ATR / close (fraction)
    std_pct: float = 0.0                   # Rolling std / close
    bb_width_pct: float = 0.0              # (upper - lower) / middle
    range_pct: float = 0.0                 # (high - low) / close over window

    # Trend (magnitude + direction)
    trend_strength: float = 0.0            # 0-1 normalized
    trend_dir: int = 0                     # -1, 0, +1

    # Position in distribution
    zscore: float = 0.0                    # Current price vs rolling mean/std
    rsi: float = 50.0                      # 0-100

    # Return
    return_pct: float = 0.0                # Window return (fraction)
    drift_bps_per_bar: float = 0.0         # Avg return per bar in bps

    # Noise proxy
    efficiency_ratio: float = 0.5          # Kaufman ER: |net| / sum(|bars|), 0-1

    # Optional identifiers
    instrument: str | None = None
    source: str | None = None              # "is" | "oos" | "query"

    # Derived
    regime_tags: list[str] = field(default_factory=list)

    # Warnings from computation
    warnings: list[str] = field(default_factory=list)
```

### Tagging Rules

Deterministic, alphabetically sorted tags:

```python
def compute_tags(s: RegimeSnapshot) -> list[str]:
    tags = []

    # Trend
    if s.trend_strength > 0.6:
        if s.trend_dir > 0:
            tags.append("uptrend")
        elif s.trend_dir < 0:
            tags.append("downtrend")
        else:
            tags.append("trending")
    elif s.trend_strength < 0.3:
        tags.append("flat")

    # Volatility
    if s.atr_pct < 0.005:      # 0.5%
        tags.append("low_vol")
    elif s.atr_pct > 0.015:    # 1.5%
        tags.append("high_vol")

    # Regime type (mutually exclusive)
    if s.trend_strength < 0.3:
        if abs(s.zscore) > 1.0:
            tags.append("mean_reverting")
        elif s.bb_width_pct < 0.02:  # 2%
            tags.append("choppy")

    # Noise
    if s.efficiency_ratio < 0.3:
        tags.append("noisy")
    elif s.efficiency_ratio > 0.7:
        tags.append("efficient")

    # Oscillator extremes (independent)
    if s.zscore < -1.5 or s.rsi < 30:
        tags.append("oversold")
    if s.zscore > 1.5 or s.rsi > 70:
        tags.append("overbought")

    return sorted(tags)
```

### Text Template (for embedding)

```python
def regime_snapshot_to_text(s: RegimeSnapshot) -> str:
    tags_str = ", ".join(s.regime_tags) if s.regime_tags else "neutral"
    tf = f" ({s.timeframe})" if s.timeframe else ""

    return f"""Market regime{tf}: {tags_str}.
Volatility: ATR {s.atr_pct*100:.2f}%, BB width {s.bb_width_pct*100:.1f}%.
Trend: strength {s.trend_strength:.2f}, direction {s.trend_dir:+d}.
Position: z-score {s.zscore:.2f}, RSI {s.rsi:.0f}.
Efficiency: {s.efficiency_ratio:.2f}."""
```

---

## Section 2: Storage Design

### Authority

`backtest_tune_runs` is the authoritative source for regime data. Each trial stores its regime snapshot inline.

### JSON Shape

```json
{
  "metrics_is": {
    "sharpe": 1.23,
    "return_pct": 0.15,
    "max_drawdown_pct": 0.08,
    "n_trades": 45,
    "regime": {
      "schema_version": "regime_v1",
      "computed_at": "2026-01-08T14:30:00Z",
      "computation_source": "live",
      "n_bars": 200,
      "effective_n_bars": 186,
      "atr_pct": 0.0123,
      "bb_width_pct": 0.034,
      "trend_strength": 0.72,
      "trend_dir": 1,
      "zscore": 0.45,
      "rsi": 58,
      "efficiency_ratio": 0.65,
      "regime_tags": ["efficient", "uptrend"],
      "warnings": []
    }
  },
  "metrics_oos": {
    "sharpe": 0.98,
    "regime": { ... }
  }
}
```

### Null Policy

| Scenario | Value | Behavior |
|----------|-------|----------|
| OOS disabled | `metrics_oos: null` | No OOS regime |
| No OHLCV available | `regime: null` | Skip from KB ingestion |
| Partial computation | `regime: {..., warnings: ["incomplete_features"]}` | Include with warnings |

**Rule:** Explicit null for missing, never omit keys.

---

## Section 3: KB Ingestion

### TrialDoc Structure

```python
@dataclass
class TrialDoc:
    """KB document for a single backtest trial."""

    # Identity
    doc_type: Literal["trial"] = "trial"
    tune_run_id: UUID
    tune_id: UUID

    # Dataset identity
    workspace_id: UUID
    dataset_id: str | None
    instrument: str | None
    timeframe: str | None
    exchange: str | None = None

    # Strategy
    strategy_name: str
    params: dict[str, float | int | str | bool]

    # Performance (fractions 0-1, not percentages)
    sharpe_is: float | None
    sharpe_oos: float | None
    return_frac_is: float | None
    return_frac_oos: float | None
    max_dd_frac_is: float | None       # Positive value (0.10 = 10% drawdown)
    max_dd_frac_oos: float | None
    n_trades_is: int | None
    n_trades_oos: int | None

    # Overfit
    overfit_gap: float | None          # max(0, sharpe_is - sharpe_oos) or None

    # Regime (full snapshots preserved)
    regime_is: RegimeSnapshot | None
    regime_oos: RegimeSnapshot | None

    # Quality flags
    has_oos: bool
    is_valid: bool
    warnings: list[str]

    # Scoring
    objective_type: str
    objective_score: float | None

    created_at: str

    @property
    def regime_tags_is(self) -> list[str]:
        return self.regime_is.regime_tags if self.regime_is else []

    @property
    def regime_tags_oos(self) -> list[str]:
        return self.regime_oos.regime_tags if self.regime_oos else []

    @property
    def regime_primary(self) -> Literal["is", "oos"]:
        return "oos" if self.regime_oos else "is"

    @property
    def regime_tags(self) -> list[str]:
        return self.regime_tags_oos if self.regime_oos else self.regime_tags_is
```

### Text Template

```python
def trial_to_text(t: TrialDoc) -> str:
    # Provenance
    dataset_label = t.dataset_id or t.instrument or "unknown"
    provenance = f"Dataset: {dataset_label} {t.timeframe or ''}."
    if t.has_oos:
        provenance += " OOS enabled."

    # Regime
    regime_str = ", ".join(t.regime_tags) if t.regime_tags else "neutral"

    # Performance (prefer OOS)
    if t.has_oos and t.sharpe_oos is not None:
        perf = f"OOS Sharpe {t.sharpe_oos:.2f}"
        if t.return_frac_oos is not None:
            perf += f", return {t.return_frac_oos*100:.1f}%"
        if t.max_dd_frac_oos is not None:
            perf += f", max DD {t.max_dd_frac_oos*100:.1f}%"
    elif t.sharpe_is is not None:
        perf = f"IS Sharpe {t.sharpe_is:.2f}"
        if t.return_frac_is is not None:
            perf += f", return {t.return_frac_is*100:.1f}%"
        if t.max_dd_frac_is is not None:
            perf += f", max DD {t.max_dd_frac_is*100:.1f}%"
    else:
        perf = "metrics unavailable"

    # Objective
    obj_str = f"Objective: {t.objective_type}"
    if t.objective_score is not None:
        obj_str += f" (score {t.objective_score:.2f})"

    # Params
    params_str = ", ".join(f"{k}={v}" for k, v in sorted(t.params.items()))

    # Quality
    quality_notes = [w.replace("_", " ") for w in t.warnings]
    quality_str = f" ({', '.join(quality_notes)})" if quality_notes else ""

    return f"""{provenance}
Regime: {regime_str}.
Strategy: {t.strategy_name} with {params_str}.
Performance: {perf}. {obj_str}.{quality_str}"""
```

### Metadata for Qdrant

```python
def trial_to_metadata(t: TrialDoc) -> dict:
    return {
        # Identity
        "doc_type": "trial",
        "tune_run_id": str(t.tune_run_id),
        "tune_id": str(t.tune_id),
        "workspace_id": str(t.workspace_id),
        "dataset_id": t.dataset_id,
        "instrument": t.instrument,
        "timeframe": t.timeframe,

        # Strategy
        "strategy_name": t.strategy_name,
        "params": t.params,

        # Performance (numeric filters)
        "sharpe_oos": t.sharpe_oos,
        "return_frac_oos": t.return_frac_oos,
        "max_dd_frac_oos": t.max_dd_frac_oos,
        "n_trades_oos": t.n_trades_oos,
        "overfit_gap": t.overfit_gap,

        # Regime
        "regime_tags": t.regime_tags,
        "regime_tags_is": t.regime_tags_is,
        "regime_tags_oos": t.regime_tags_oos,
        "regime_tags_str": " ".join(t.regime_tags),
        "regime_snapshot_is": asdict(t.regime_is) if t.regime_is else None,
        "regime_snapshot_oos": asdict(t.regime_oos) if t.regime_oos else None,

        # Quality
        "has_oos": t.has_oos,
        "is_valid": t.is_valid,
        "warnings": t.warnings,

        # Scoring
        "objective_type": t.objective_type,
        "objective_score": t.objective_score,

        "created_at": t.created_at,
    }
```

### Warning Rules

```python
def compute_warnings(
    sharpe_is: float | None,
    sharpe_oos: float | None,
    overfit_gap: float | None,
    n_trades_oos: int | None,
    max_dd_frac_oos: float | None,
    is_valid: bool,
) -> list[str]:
    warnings = []

    if sharpe_is is None:
        warnings.append("missing_metrics")

    if not is_valid:
        warnings.append("failed_gates")

    if overfit_gap is not None:
        if overfit_gap > 0.5:
            warnings.append("high_overfit")
        elif overfit_gap > 0.3:
            warnings.append("moderate_overfit")

    if n_trades_oos is not None and n_trades_oos < 10:
        warnings.append("low_trades_oos")

    if max_dd_frac_oos is not None and max_dd_frac_oos > 0.15:
        warnings.append("high_drawdown")

    return sorted(warnings)
```

---

## Section 4: Query Flow

### Request Schema

```python
@dataclass
class RecommendParamsRequest:
    workspace_id: UUID
    strategy_name: str
    objective_type: str

    # Dataset
    ohlcv_file: UploadFile | None = None
    dataset_id: str | None = None

    # Metadata hints
    instrument: str | None = None
    timeframe: str | None = None

    # Performance floors
    min_oos_sharpe: float = 0.5
    min_trades: int = 20
    max_drawdown_frac: float = 0.20
    max_overfit_gap: float = 0.50

    # Candidate counts
    retrieve_k: int = 100
    rerank_keep: int = 30
    top_k: int = 15

    # Flags
    require_oos: bool = True
    run_backtest: bool = False
```

### Step 0: Parse Dataset

```python
@dataclass
class ParsedDataset:
    df: pd.DataFrame              # ts, open, high, low, close, volume
    n_bars: int
    ts_start: datetime
    ts_end: datetime
    instrument: str | None
    timeframe: str | None
    fingerprint: str              # SHA256 of canonical representation
    warnings: list[str]
```

### Step 1: Compute Query Regime

```python
REGIME_WINDOW_BARS = 200

def compute_query_regime(df: pd.DataFrame, hints: dict) -> RegimeSnapshot:
    window = df.tail(REGIME_WINDOW_BARS)
    return compute_regime_snapshot(
        window,
        source="query",
        instrument=hints.get("instrument"),
        timeframe=hints.get("timeframe"),
    )
```

### Step 2: Candidate Retrieval

**Hard filters:**
```python
filter = {
    "must": [
        {"key": "doc_type", "match": {"value": "trial"}},
        {"key": "workspace_id", "match": {"value": str(workspace_id)}},
        {"key": "strategy_name", "match": {"value": strategy_name}},
        {"key": "objective_type", "match": {"value": objective_type}},
        {"key": "is_valid", "match": {"value": True}},
    ]
}

if require_oos:
    filter["must"].append({"key": "has_oos", "match": {"value": True}})
```

**Performance floors:**
```python
filter["must"].extend([
    {"key": "sharpe_oos", "range": {"gte": min_oos_sharpe}},
    {"key": "n_trades_oos", "range": {"gte": min_trades}},
    {"key": "max_dd_frac_oos", "range": {"lte": max_drawdown_frac}},
    {"key": "overfit_gap", "range": {"lte": max_overfit_gap}},
])
```

**Two-phase fallback:**
```python
MIN_CANDIDATES_THRESHOLD = 10

# Phase 1: Strict (overfit_gap must exist and pass)
candidates = qdrant_search(filters=strict_filters, limit=retrieve_k)

if len(candidates) < MIN_CANDIDATES_THRESHOLD:
    # Phase 2: Relaxed (allow overfit_gap=None, tag as _relaxed)
    candidates = qdrant_search(filters=relaxed_filters, limit=retrieve_k)
    for c in candidates:
        if c.payload.get("overfit_gap") is None and c.payload.get("has_oos"):
            c.payload["_relaxed"] = True
```

### Step 3: Rerank

**Jaccard similarity:**
```python
def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
```

**Deterministic sort:**
```python
candidates = sorted(
    candidates,
    key=lambda c: (
        jaccard(c.payload.get("regime_tags", []), query_tags),
        c.payload.get("objective_score") or 0,
        c.payload.get("created_at") or "",
    ),
    reverse=True,
)[:rerank_keep]
```

### Step 4: Select Top Trials

```python
top_trials = sorted(
    candidates,
    key=lambda c: c.payload.get("objective_score") or 0,
    reverse=True
)[:top_k]
```

### Step 5: Weighted Median Aggregation

```python
def compute_weight(
    trial: dict,
    min_sharpe: float,
    sharpe_scale: float,
    jaccard: float,
    trades_ref: int = 50,
) -> float:
    sharpe = trial.get("sharpe_oos") or 0
    trades = trial.get("n_trades_oos") or 0

    base = max(0, sharpe - min_sharpe) / sharpe_scale
    regime_factor = 0.5 + 0.5 * jaccard
    trade_factor = math.log1p(trades) / math.log1p(trades_ref)

    weight = base * regime_factor * trade_factor

    if trial.get("_relaxed"):
        weight *= 0.25  # Relaxed penalty

    return min(weight, 1.0)


def weighted_median(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return float(np.median(values))  # Fallback

    sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
    cumsum = 0.0
    for val, w in sorted_pairs:
        cumsum += w
        if cumsum >= total / 2:
            return val
    return sorted_pairs[-1][0]


def weighted_mode(values: list, weights: list):
    """For categorical params."""
    weighted_counts = defaultdict(float)
    for v, w in zip(values, weights):
        weighted_counts[v] += w
    return max(weighted_counts, key=weighted_counts.get)
```

### Step 6: Response

```python
@dataclass
class RecommendParamsResponse:
    status: Literal["ok", "degraded", "none", "error"]
    message: str | None = None
    suggested_actions: list[str] | None = None

    # Query context
    query_regime: RegimeSnapshot
    query_tags: list[str]
    dataset_fingerprint: str

    # Retrieval stats
    candidates_requested: int
    candidates_returned: int
    candidates_after_rerank: int
    top_trials_count: int
    relaxed_filters_used: bool

    # Matching
    top_matching_tags: list[str]
    avg_jaccard: float

    # Top trials (transparency)
    top_trials: list[TrialSummary]

    # Recommendation
    recommended_params: dict[str, float | int]
    param_spreads: dict[str, ParamSpread]

    # Confidence
    confidence: Literal["high", "medium", "low", "none"]
    confidence_reasons: list[str]

    # Optional
    backtest_result: BacktestResult | None = None
```

### Confidence Rules

```python
def compute_confidence(
    top_trials_count: int,
    avg_jaccard: float,
    param_spreads: dict,
    warnings: list[str],
) -> tuple[str, list[str]]:
    reasons = []

    # Base confidence
    if top_trials_count >= 10 and avg_jaccard >= 0.5:
        confidence = "high"
        reasons.append("sufficient_trials")
        reasons.append("strong_regime_match")
    elif top_trials_count >= 5 and avg_jaccard >= 0.3:
        confidence = "medium"
    elif top_trials_count > 0:
        confidence = "low"
        if top_trials_count < 5:
            reasons.append("few_trials")
        if avg_jaccard < 0.3:
            reasons.append("weak_regime_match")
    else:
        return "none", ["no_candidates"]

    # Param stability
    high_dispersion = sum(1 for s in param_spreads.values() if s.iqr > s.median * 0.5)
    if high_dispersion > len(param_spreads) / 2:
        reasons.append("unstable_params")

    # Degradation from warnings
    degradation_warnings = {
        "relaxed_filters_missing_overfit_gap",
        "embedding_unavailable_fallback",
        "incomplete_query_regime",
        "timeframe_mismatch",
    }

    triggered = set(warnings) & degradation_warnings
    reasons.extend(sorted(triggered))

    # Downgrade per warning
    levels = ["high", "medium", "low"]
    idx = levels.index(confidence)
    idx = min(idx + len(triggered), len(levels) - 1)

    return levels[idx], reasons
```

---

## Section 5: Error Handling

### Response Status

| Status | Meaning |
|--------|---------|
| `ok` | Normal path, no fallbacks |
| `degraded` | Fallbacks used (relaxed filters, embed fallback, partial regime) |
| `none` | No recommendation possible (0 candidates) |
| `error` | Validation failure (4xx responses) |

### Dataset Parsing Errors

| Error | Detection | Handling |
|-------|-----------|----------|
| Invalid CSV | Parse exception | 400 |
| Missing columns | Column check | 400 |
| Column aliases | `timestamp`/`Close` variants | Alias map + warn |
| < 200 rows | Row count | 400 |
| Duplicate timestamps | `df.ts.duplicated()` | 400 |
| Timezone-naive | No tzinfo | Assume UTC + warn |
| Unix epochs | int ~1e9/1e12 | Convert + warn |
| Non-monotonic | ts diff | Sort + warn |
| Missing volume | Column check | Fill 0 + warn |
| Negative prices | Value check | 400 |

### Regime Computation Errors

| Error | Detection | Handling |
|-------|-----------|----------|
| Warmup reduces bars | Fewer valid rows | Track `effective_n_bars`, warn if < 50 |
| All NaN | Feature computation | Partial snapshot + warn |
| Zero variance | std = 0 | `bb_width = 0`, tag "flat" |
| Zero ATR | ATR = 0 | `atr_pct = 0` + warn |
| Outliers | abs(pct_change) > 0.5 | Warn `possible_bad_ticks` |

### Retrieval Errors

| Scenario | Handling |
|----------|----------|
| Qdrant unavailable | 503 |
| Embedding unavailable | Metadata-only fallback + warn |
| Zero candidates (strict) | Try Phase 2 |
| Zero candidates (relaxed) | status="none" + suggested_actions |
| Unknown strategy | 400 + list valid strategies |

### Aggregation Edge Cases

| Scenario | Handling |
|----------|----------|
| All weights = 0 | Unweighted median fallback |
| Single trial | Use params directly, confidence="low" |
| Categorical param | Weighted mode |
| Param out of bounds | Clamp + warn |
| Constraint violated | Repair + warn |

---

## Section 6: Testing Strategy

### Test Pyramid

| Layer | Speed | Purpose |
|-------|-------|---------|
| Unit | Fast | Parsing, regime, filters, rerank, aggregation |
| Contract | Medium | Adapter payloads, response schemas |
| Integration | Slow | End-to-end with seeded KB |
| Golden | Fast | Determinism checks |
| Property | Fast | Hypothesis fuzz tests |

### Key Test Cases

**Parsing:**
- Valid CSV, column aliases, epoch timestamps
- Missing volume, duplicate timestamps (400)
- Fingerprint stability

**Regime:**
- Flat price series → tags include "flat"
- Zero ATR → handled gracefully
- Deterministic tags and text

**Retrieval:**
- Strict vs relaxed filter building
- Phase 1 → Phase 2 fallback
- Embedding unavailable fallback

**Aggregation:**
- Weighted median with zero weights
- Categorical params (weighted mode)
- Constraint repair

**Golden Tests:**
- `regime_snapshot_to_text()` exact match
- `trial_to_text()` exact match
- `trial_to_metadata()` exact dict match

**Property-Based (Hypothesis):**
- `parse_ohlcv()` never crashes on random columns/NaNs
- `weighted_median()` always returns value in range

**Schema Drift:**
- `schema_version` always present in metadata
- Fail if version bumps without migration plan

### Folder Layout

```
tests/
├── unit/
│   ├── parsing/test_parse_ohlcv.py
│   ├── regime/test_regime_snapshot.py
│   ├── retrieval/test_filters.py
│   ├── retrieval/test_rerank.py
│   └── aggregation/test_weighted_median.py
├── contract/
│   ├── test_embed_adapter.py
│   └── test_qdrant_adapter.py
├── integration/recommend/test_recommend_flow.py
├── golden/test_text_determinism.py
├── slow/test_performance.py
└── fixtures/
    ├── ohlcv/
    └── golden/
```

---

## Section 7: Implementation Plan

### File Structure

```
app/
├── services/
│   ├── kb/
│   │   ├── constants.py          # Thresholds, timeouts, defaults
│   │   ├── types.py              # Shared dataclasses
│   │   ├── regime.py             # RegimeSnapshot, compute_regime_snapshot
│   │   ├── trial_doc.py          # TrialDoc, text/metadata conversion
│   │   ├── ingestion.py          # KB ingestion from tune_runs
│   │   ├── retrieval.py          # Filters, strict/relaxed, fallback
│   │   ├── rerank.py             # Jaccard, distance, tie-breakers
│   │   ├── aggregation.py        # Weighted median/mode
│   │   └── recommend.py          # Orchestration
│   ├── parsing/ohlcv.py          # CSV parsing, fingerprint
│   ├── strategies/
│   │   ├── registry.py           # Strategy + objective registries
│   │   └── params.py             # ParamSpec, validate_and_repair
│   └── backtest/tuner.py         # Add regime hook
├── repositories/kb.py            # Qdrant operations
├── routers/kb.py                 # API endpoints
├── schemas/kb.py                 # Pydantic models
└── adapters/
    ├── embed.py                  # Embedding with timeout
    └── qdrant.py                 # Qdrant client wrapper
```

### Phases

1. **Core Structures** — RegimeSnapshot, TrialDoc, parsing, constants
2. **Strategy Registry** — ParamSpec, validation, repair, registries
3. **Storage Integration** — Tuner hooks, regime persistence, migration
4. **Qdrant/Embed Adapters** — Collection setup, client wrappers, smoke tests
5. **KB Ingestion** — tune_runs → TrialDoc → Qdrant
6. **Retrieval Pipeline** — Filters, strict/relaxed, metadata-only fallback
7. **Rerank + Aggregation** — Jaccard, weights, median, confidence
8. **API Endpoints** — /kb/recommend, /kb/ingest, debug mode
9. **Polish** — Golden tests, property tests, performance baseline

### Migration

**Existing tune_runs:**
1. Add nullable `regime` to `metrics_is`, `metrics_oos`
2. Backfill: compute regime with `computation_source: "backfill"`
3. If no OHLCV: set `regime: null`

**KB Collection:**
1. Create `trading_kb_trials` collection
2. Batch ingest from tune_runs
3. Verify counts

### Rollout

| Stage | Actions |
|-------|---------|
| Dev | Full implementation, tests green |
| Staging | Seed with prod data, validate quality |
| Prod (soft) | Deploy with flag off, run backfill |
| Prod (enable) | Enable flag, monitor 24h |
| Prod (full) | Full recommend after clean metrics |

---

## Constants Reference

```python
# Regime
REGIME_WINDOW_BARS = 200
MIN_EFFECTIVE_BARS = 50

# Retrieval
MIN_CANDIDATES_THRESHOLD = 10
DEFAULT_RETRIEVE_K = 100
DEFAULT_RERANK_KEEP = 30
DEFAULT_TOP_K = 15

# Floors
DEFAULT_MIN_OOS_SHARPE = 0.5
DEFAULT_MIN_TRADES = 20
DEFAULT_MAX_DRAWDOWN_FRAC = 0.20
DEFAULT_MAX_OVERFIT_GAP = 0.50

# Timeouts
TIMEOUT_EMBED_S = 5.0
TIMEOUT_QDRANT_S = 10.0
TIMEOUT_BACKTEST_S = 60.0

# Weights
RELAXED_WEIGHT_PENALTY = 0.25
TRADES_REF = 50
```

---

## Appendix: Feature Scales

For regime distance normalization:

| Feature | Scale | Notes |
|---------|-------|-------|
| `atr_pct` | 2.0 | 0-2% typical range |
| `bb_width_pct` | 5.0 | 0-5% typical |
| `trend_strength` | 1.0 | Already 0-1 |
| `efficiency_ratio` | 1.0 | Already 0-1 |
| `zscore` | 3.0 | -3 to +3 typical |
