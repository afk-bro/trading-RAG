# Cross-Encoder Reranking

Optional two-stage retrieval that improves search relevance by reranking vector search results with a cross-encoder model.

## Overview

```
Query → Embed → Vector Search (candidates) → Cross-Encoder Rerank → Neighbor Expand → Results
                     ↓                              ↓
                 retrieve_k                      top_k
                 (default 50)                  (default 10)
```

**Default: Disabled.** Enable via `rerank: true` in request or workspace config.

## Quick Start

```bash
# Basic query with reranking
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "your-workspace-id",
    "question": "What causes inflation?",
    "rerank": true
  }'
```

## How It Works

### Stage 1: Vector Search
- Embeds query via Ollama (nomic-embed-text)
- Searches Qdrant for top `retrieve_k` candidates (default: 50)
- Fast but relies on embedding similarity alone

### Stage 2: Cross-Encoder Rerank
- Scores each (query, chunk) pair with BGE-reranker-v2-m3
- More accurate than embedding similarity (sees both texts together)
- Returns top `top_k` results (default: 10)

### Stage 3: Neighbor Expansion
- Fetches adjacent chunks for context continuity
- Configurable window size per source type
- Preserves document flow for LLM synthesis

## Configuration

### Request-Level (Highest Precedence)

```json
{
  "rerank": true,
  "retrieve_k": 50,
  "top_k": 10,
  "debug": true
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WARMUP_RERANKER` | `false` | Pre-load model at startup |
| `RERANK_TIMEOUT_S` | `10.0` | Timeout before fallback to vector order |

### Workspace Config (Future)

```json
{
  "rerank": {
    "enabled": true,
    "method": "cross_encoder",
    "candidates_k": 50,
    "final_k": 10
  },
  "neighbor": {
    "enabled": true,
    "window": 1,
    "pdf_window": 2
  }
}
```

## Response Fields

### QueryMeta

```json
{
  "meta": {
    "rerank_state": "ok",
    "rerank_enabled": true,
    "rerank_method": "cross_encoder",
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "rerank_ms": 127,
    "rerank_timeout": false,
    "rerank_fallback": false
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `rerank_state` | enum | `disabled`, `ok`, `timeout_fallback`, `error_fallback` |
| `rerank_enabled` | bool | Whether reranking was requested |
| `rerank_method` | string | `cross_encoder` or `llm` (future) |
| `rerank_model` | string | Model identifier |
| `rerank_ms` | int | Reranking latency in milliseconds |
| `rerank_timeout` | bool | True if timeout occurred |
| `rerank_fallback` | bool | True if fell back to vector order |

### Debug Mode

With `"debug": true`:

```json
{
  "results": [
    {
      "chunk_id": "...",
      "content": "...",
      "debug": {
        "vector_score": 0.85,
        "rerank_score": 0.92,
        "rerank_rank": 0,
        "is_neighbor": false
      }
    }
  ]
}
```

## Safety Features

### Timeout Fallback
- Default: 10 seconds
- On timeout: returns results in vector score order
- Logged as `rerank_state: timeout_fallback`

### Error Fallback
- On any reranker error: returns results in vector score order
- Logged as `rerank_state: error_fallback`

### Safety Caps (Hardcoded)

| Cap | Value | Enforced By |
|-----|-------|-------------|
| `retrieve_k` | ≤ 200 | Pydantic schema |
| `top_k` | ≤ 50 | Pydantic schema |
| `neighbor.max_total` | ≤ 50 | Runtime cap |

## Performance

| Metric | Typical Value |
|--------|---------------|
| Model load | 5-15s (first request or warmup) |
| Inference | 50-200ms for 50 candidates (GPU) |
| Memory | ~2GB VRAM |

### Optimization Tips

1. **Pre-warm model** - Set `WARMUP_RERANKER=true` to avoid cold start
2. **Reduce candidates** - Lower `retrieve_k` for faster reranking
3. **GPU recommended** - CPU inference is 5-10x slower

## Architecture

```
app/
├── services/
│   ├── reranker.py          # CrossEncoderReranker, singleton management
│   └── neighbor_expansion.py # Adjacent chunk fetching
├── routers/
│   └── query.py              # Pipeline orchestration, timeout handling
└── schemas.py                # RerankState enum, QueryMeta fields
```

### Key Design Decisions

1. **Disabled by default** - Opt-in to avoid latency surprise
2. **Fail-open** - Timeout/error falls back to vector order, never fails request
3. **Local inference** - No external API dependency
4. **Singleton model** - Loaded once, reused across requests
5. **Semaphore concurrency** - Prevents GPU contention

## Files

| File | Purpose |
|------|---------|
| [runbook.md](./runbook.md) | Operational guide, troubleshooting |

## Testing

```bash
# Unit tests (fast, mocked)
pytest tests/unit/test_reranker.py tests/unit/test_neighbor_expansion.py -v

# Integration tests (mocked services)
pytest tests/integration/test_rerank_pipeline.py -v

# Contract tests (schema stability)
pytest tests/unit/test_query_meta_contract.py -v

# Slow tests (real model, requires GPU or patience)
RUN_SLOW_TESTS=1 pytest tests/slow/test_real_reranker.py -v -m slow
```

## A/B Comparison Endpoint

`POST /query/compare` runs vector-only and reranked retrieval on the same candidate set for fair comparison.

### Usage

```bash
curl -X POST http://localhost:8000/query/compare \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "your-workspace-id",
    "question": "What causes inflation?",
    "retrieve_k": 50,
    "top_k": 10
  }'
```

### Response

```json
{
  "vector_only": { "results": [...], "meta": {...} },
  "reranked": { "results": [...], "meta": {...} },
  "metrics": {
    "jaccard": 0.67,
    "spearman": 0.42,
    "rank_delta_mean": 1.8,
    "rank_delta_max": 4,
    "overlap_count": 8,
    "union_count": 12
  }
}
```

### Structured Eval Logging

Each `/query/compare` call emits a `query_compare` log event with:

| Field | Description |
|-------|-------------|
| `workspace_id` | Target workspace |
| `candidates_k`, `top_k` | Retrieval parameters |
| `jaccard`, `spearman` | Set overlap and rank correlation |
| `rank_delta_mean/max` | Position changes |
| `embed_ms`, `search_ms`, `rerank_ms` | Latency breakdown |
| `rerank_state` | `ok`, `timeout_fallback`, `error_fallback` |
| `vector_top5_ids`, `reranked_top5_ids` | Spot-check IDs |

**Analyze logs:**
```bash
# Extract compare metrics
docker compose logs trading-rag-svc | grep '"event": "query_compare"' | jq '{jaccard, spearman, rerank_state}'

# Find low-overlap queries (rerank is changing results)
docker compose logs trading-rag-svc | grep '"event": "query_compare"' | jq 'select(.jaccard < 0.8)'
```

### Evaluation Persistence (Optional)

Enable database persistence for long-term analytics:

```bash
EVAL_PERSIST_ENABLED=true          # Persist to query_compare_evals table
EVAL_STORE_QUESTION_PREVIEW=true   # Store first 80 chars (optional, default: hash only)
```

**Dashboard endpoints** (require admin access):

| Endpoint | Description |
|----------|-------------|
| `GET /admin/evals/query-compare/summary?workspace_id=...&since=24h` | Impact rate, latency, fallback rate |
| `GET /admin/evals/query-compare/by-config?workspace_id=...&since=7d` | Stats grouped by config fingerprint |
| `GET /admin/evals/query-compare/most-impacted?workspace_id=...&limit=20` | Lowest jaccard queries for spot-check |
| `DELETE /admin/evals/query-compare/cleanup?days=90` | Retention cleanup |

**Example response (summary):**
```json
{
  "workspace_id": "...",
  "since": "24h",
  "total": 200,
  "impacted_count": 47,
  "impact_rate": 0.235,
  "p50_rerank_ms": 145.0,
  "p95_rerank_ms": 312.0,
  "fallback_rate": 0.01,
  "timeout_rate": 0.0
}
```

## Monitoring

### Log Query

```bash
# Find all rerank fallbacks
docker compose logs trading-rag-svc | grep -E "timeout_fallback|error_fallback"

# Latency distribution
docker compose logs trading-rag-svc | grep "rerank_ms" | jq '.rerank_ms'
```

### Recommended Alerts

| Condition | Severity |
|-----------|----------|
| Fallback rate > 5% (5min) | Warning |
| Fallback rate > 20% (5min) | Critical |
| P95 rerank_ms > 500ms | Warning |
| P95 rerank_ms > 1000ms | Critical |

See [runbook.md](./runbook.md) for detailed alert configuration.
