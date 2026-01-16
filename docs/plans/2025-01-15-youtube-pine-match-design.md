# YouTube to Pine Script Match Endpoint

**Date**: 2025-01-15
**Status**: Design Complete
**Author**: Claude + User collaboration

## Overview

New endpoint that accepts a YouTube URL, extracts trading intent from the transcript, and returns matching Pine Script strategies/indicators from the knowledge base.

This is the first "intelligent workflow" - proving the "paste a video, get strategies" story end-to-end.

## Architecture

### Hybrid Approach (Phase A)

```
POST /sources/youtube/match-pine
         │
         ▼
┌─────────────────────┐
│  Parse YouTube URL  │ → 422 if invalid
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Check KB: workspace_id + source_type='youtube' + video_id  │
│            + status='active'                                 │
└─────────┬───────────────────────────────────────────────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
  in_kb      not_in_kb (or force_transient=True)
    │           │
    ▼           ▼
┌───────────┐  ┌─────────────────┐
│ Load top-N│  │ Fetch transcript│ → 404 if unavailable
│ chunks    │  │ + metadata      │ → 502 if upstream fail
│ (bounded) │  │ (transient)     │
└─────┬─────┘  └────────┬────────┘
      │                 │
      └────────┬────────┘
               ▼
┌─────────────────────────────┐
│ Concatenate text            │ ← up to MAX_EXTRACTION_CHARS
│ (bounded input for extract) │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ IntentExtractor.extract()   │ ← RuleBasedIntentExtractor
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ QueryBuilder.build()        │ → query_string + filters
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ Pine match (internal call)  │ ← reuse existing match logic
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ Reranker.rerank()           │ → sorted RankedResults
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ Build response              │
└─────────────────────────────┘
```

### Phase B (Future)

- Add explicit `/sources/youtube/ingest` trigger from response hint
- Ingestion becomes queued job with notification
- Future calls automatically reuse stored chunks

## API Contract

### Request

```python
class YouTubeMatchPineRequest(BaseModel):
    workspace_id: UUID
    url: str
    symbols: Optional[list[str]] = None   # Override extracted symbols
    script_type: Optional[Literal["strategy", "indicator"]] = None
    lint_ok: bool = True                   # Filter to clean scripts
    top_k: conint(ge=1, le=50) = 10
    force_transient: bool = False          # Bypass KB, always fetch live
```

### Response

```python
class MatchFiltersApplied(BaseModel):
    script_type: Optional[Literal["strategy", "indicator"]]
    symbols: list[str]
    lint_ok: bool

class IngestRequestHint(BaseModel):
    workspace_id: UUID
    url: str

class PineMatchRankedResult(PineMatchResult):
    base_score: float
    boost: float
    final_score: float

class YouTubeMatchPineResponse(BaseModel):
    # Source metadata
    video_id: str
    title: Optional[str]
    channel: Optional[str]

    # KB status
    in_knowledge_base: bool
    transcript_source: Literal["kb", "transient"]
    transcript_chars_used: int

    # Extraction
    match_intent: MatchIntent
    extraction_method: Literal["rule_based", "llm"]

    # Match results
    results: list[PineMatchRankedResult]
    total_searched: int
    query_used: str
    filters_applied: MatchFiltersApplied

    # Next actions
    ingest_available: bool
    ingest_request_hint: Optional[IngestRequestHint]
```

### Authentication

Workspace-scoped via `X-Admin-Token` (consistent with existing `/sources/pine/*` endpoints). Validates workspace access.

### Errors

| Code | Condition |
|------|-----------|
| 422 | Invalid URL format, invalid script_type, top_k out of range |
| 404 | Video not found, transcript unavailable |
| 502 | Upstream transcript/YouTube API failure |

## MatchIntent Model

```python
class MatchIntent(BaseModel):
    # From existing MetadataExtractor
    symbols: list[str] = []               # ["BTC", "SPY"] - uppercase tickers
    topics: list[str] = []                # ["crypto", "macro"] - lowercase
    entities: list[str] = []              # ["Fed", "Powell"]

    # Trading-specific (lowercase canonical tags, deduped, order-preserved)
    strategy_archetypes: list[str] = []   # ["breakout", "momentum"]
    indicators: list[str] = []            # ["rsi", "macd", "bollinger"]
    timeframe_buckets: list[str] = []     # ["swing", "intraday"]
    timeframe_explicit: list[str] = []    # ["1h", "4h", "15m"]
    risk_terms: list[str] = []            # ["stop_loss", "take_profit"]

    # Script type inference
    inferred_script_type: Optional[Literal["strategy", "indicator"]] = None
    script_type_confidence: float = 0.0   # 0-1, Laplace smoothed
    overall_confidence: float = 0.0       # 0-1, weighted signal density
```

### Canonical Tag Mappings

#### Strategy Archetypes

| Tag | Patterns |
|-----|----------|
| `trend_following` | trend, ride the trend, trend continuation, higher highs |
| `mean_reversion` | mean revert, reversal, snap back, oversold bounce |
| `breakout` | breakout, range break, ATH breakout, support break, resistance break |
| `momentum` | momentum, impulse, strength, relative strength |
| `range_bound` | range, chop, sideways, box, consolidation |
| `volatility` | volatility expansion, squeeze, ATR, IV |

#### Indicators

| Tag | Patterns |
|-----|----------|
| `rsi` | rsi, relative strength index |
| `macd` | macd |
| `bollinger` | bollinger, bbands, b bands |
| `moving_average` | sma, ema, wma, moving average |
| `vwap` | vwap |
| `atr` | atr, average true range |
| `volume` | volume, obv, volume profile |
| `stochastic` | stoch, stochastic |
| `adx` | adx |
| `ichimoku` | ichimoku, cloud |
| `pivot` | pivot |
| `support_resistance` | support, resistance, s/r |

#### Timeframes

| Tag | Patterns | Type |
|-----|----------|------|
| `scalp` | scalp, 1m, 5m | bucket |
| `intraday` | day trade, intraday, 15m, 30m | bucket |
| `swing` | swing, 1h, 4h | bucket |
| `position` | position, long-term, daily, weekly | bucket |
| `1m`, `5m`, `15m`, `1h`, `4h`, `1d`, `1w` | explicit mentions | explicit |

#### Risk Terms

| Tag | Patterns |
|-----|----------|
| `stop_loss` | stop loss, stop-loss, SL |
| `take_profit` | take profit, TP, target |
| `trailing_stop` | trailing stop, trailing |
| `position_sizing` | position size, risk per trade, R-multiple |
| `dca` | DCA, dollar cost average, pyramiding, scaling |

### Script Type Inference

**Strategy cues**: strategy, strategy.entry, strategy.exit, backtest, drawdown, win rate, entry, exit, PnL, trade, any archetype present

**Indicator cues**: indicator, plot, alert, overlay, oscillator, histogram, signal line, divergence, "TradingView indicator"

**Confidence formula** (Laplace smoothed):
```
script_type_confidence = (strategy_cues + 1) / (strategy_cues + indicator_cues + 2)
```

### Overall Confidence

Weighted signal density:
```python
high_signal = [symbols, strategy_archetypes, indicators, timeframe_buckets, timeframe_explicit]
low_signal = [topics, entities, risk_terms]

weighted_sum = sum(1 for f in high_signal if f) + sum(0.5 for f in low_signal if f)
overall_confidence = min(1.0, weighted_sum / 6.0)
```

## Query Builder

```python
HIGH_SIGNAL_TOPICS = {"options", "crypto", "forex", "macro"}

def build_query_string(intent: MatchIntent, request: YouTubeMatchPineRequest) -> str:
    parts = []

    # 1. Archetypes (top 2)
    parts.extend(intent.strategy_archetypes[:2])

    # 2. Indicators (top 3)
    parts.extend(intent.indicators[:3])

    # 3. Timeframe: explicit beats bucket
    if intent.timeframe_explicit:
        parts.extend(intent.timeframe_explicit[:1])
    elif intent.timeframe_buckets:
        parts.extend(intent.timeframe_buckets[:1])

    # 4. One high-signal topic (always, if present)
    high_signal = [t for t in intent.topics if t in HIGH_SIGNAL_TOPICS]
    if high_signal:
        parts.append(high_signal[0])

    # 5. Risk terms (if confident)
    if intent.overall_confidence >= 0.5:
        parts.extend(intent.risk_terms[:1])

    # 6. Symbol (only if request override)
    if request.symbols:
        parts.append(request.symbols[0])

    # 7. Order-preserving dedupe
    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    # 8. Fallback if empty
    if not deduped:
        if intent.topics:
            deduped = [intent.topics[0]]
        else:
            deduped = ["trading"]

    return " ".join(deduped)
```

## Filter Builder

```python
def build_filters(intent: MatchIntent, request: YouTubeMatchPineRequest) -> MatchFiltersApplied:
    # Symbol filter: only if high confidence or explicit request
    # Avoids over-constraining when Pine scripts are generic
    if request.symbols:
        symbols = request.symbols
    elif len(intent.symbols) == 1 or (intent.overall_confidence >= 0.6 and intent.symbols):
        symbols = intent.symbols[:3]
    else:
        symbols = []  # No symbol filter - let query/rerank handle it

    # Script type: request override or inferred if confident
    script_type = request.script_type or (
        intent.inferred_script_type if intent.script_type_confidence >= 0.6 else None
    )

    return MatchFiltersApplied(
        symbols=symbols,
        script_type=script_type,
        lint_ok=request.lint_ok,
    )
```

**Note**: `symbols=[]` means "no symbol filtering", not "match scripts with no symbols".

## Reranker

```python
class RankedResult(BaseModel):
    result: PineMatchResult
    base_score: float
    boost: float
    final_score: float

def rerank(results: list[PineMatchResult], intent: MatchIntent) -> list[RankedResult]:
    ranked = []
    for r in results:
        boost = 0.0

        # Build haystack from available fields
        haystack = f"{r.title} {' '.join(r.inputs_preview)}".lower()

        # Indicator overlap: +0.15 * min(2, n)
        ind_matches = sum(1 for i in intent.indicators if i in haystack)
        boost += 0.15 * min(2, ind_matches)

        # Timeframe match: explicit +0.12, bucket +0.10
        for tf in intent.timeframe_explicit[:1]:
            if tf in haystack:
                boost += 0.12
                break
        else:
            for tf in intent.timeframe_buckets[:1]:
                if tf in haystack:
                    boost += 0.10
                    break

        # Archetype in title: +0.10 (no stack)
        if any(a in haystack for a in intent.strategy_archetypes):
            boost += 0.10

        # Risk mention: +0.05 (no stack)
        if any(rt in haystack for rt in intent.risk_terms):
            boost += 0.05

        # Cap total boost at 0.4
        boost = min(0.4, boost)
        final = min(1.0, r.score + boost)

        ranked.append(RankedResult(
            result=r,
            base_score=r.score,
            boost=boost,
            final_score=final,
        ))

    # Deterministic sort: final_score desc, then id asc
    return sorted(ranked, key=lambda x: (-x.final_score, str(x.result.id)))
```

## KB Chunk Loading

When loading from KB:
- Query: `workspace_id + source_type='youtube' + video_id + status='active'`
- Load top-N chunks (e.g., first 10 chunks by chunk_index)
- Concatenate up to `MAX_EXTRACTION_CHARS` (e.g., 50,000 chars)
- This bounds extraction input in both KB and transient modes

## force_transient Behavior

When `force_transient=True` and video IS in KB:
- `in_knowledge_base=True` (accurate - it IS in KB)
- `transcript_source="transient"` (accurate - we fetched live)

This distinction helps debugging and trust.

## File Structure

```
app/
├── routers/
│   └── youtube_pine.py              # New endpoint
├── services/
│   └── intent/
│       ├── __init__.py
│       ├── models.py                # MatchIntent, canonical tag mappings
│       ├── extractor.py             # IntentExtractor protocol + RuleBasedIntentExtractor
│       ├── query_builder.py         # build_query_string, build_filters
│       └── reranker.py              # rerank logic
└── tests/
    ├── unit/
    │   └── test_intent_extractor.py # High value, deterministic
    └── integration/
        └── test_youtube_pine_api.py # Future: mock transcript, verify wiring
```

## Implementation Order

1. `app/services/intent/models.py` - MatchIntent + canonical tag mappings with validators
2. `app/services/intent/extractor.py` - IntentExtractor protocol + RuleBasedIntentExtractor
3. `tests/unit/test_intent_extractor.py` - Test extraction immediately (reveals mapping gaps)
4. `app/services/intent/query_builder.py` - Query string + filter builders
5. `app/services/intent/reranker.py` - Rerank with score preservation
6. `app/routers/youtube_pine.py` - Endpoint wiring, KB check, response building
7. Manual testing against real YouTube URLs

## Future Enhancements (Phase B+)

1. **LLM-assisted extraction**: Add `LLMIntentExtractor` behind feature flag, same `MatchIntent` output
2. **Ingest on demand**: `/sources/youtube/ingest` triggered from `ingest_request_hint`
3. **Integration test**: Mock transcript fetch + verify API wiring
4. **asset_class inference**: Add to MatchIntent for routing (crypto/stocks/forex/options)
5. **Expanded haystack**: Include script header/function name in reranker if available

## Extensibility

The hybrid pattern generalizes to other sources:
- **PDF**: if ingested → use stored chunks; else transient parse → match → offer ingest
- **Web**: if in KB → use stored; else transient scrape → match → offer ingest

Single mental model: "Try now, save if useful."

## IntentExtractor Interface

```python
from abc import ABC, abstractmethod

class IntentExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, metadata: Optional[ExtractedMetadata] = None) -> MatchIntent:
        """Extract trading intent from text."""
        pass

class RuleBasedIntentExtractor(IntentExtractor):
    """Phase A: Regex + keyword matching."""
    pass

class LLMIntentExtractor(IntentExtractor):
    """Phase B: LLM-powered extraction (same output schema)."""
    pass
```

This allows swapping extraction strategy without pipeline changes.
