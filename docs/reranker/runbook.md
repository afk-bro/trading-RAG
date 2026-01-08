# Cross-Encoder Reranker Runbook

Operational guide for the optional cross-encoder reranking feature.

## Quick Reference

| Action | Command/Config |
|--------|----------------|
| Enable for request | `"rerank": true` in POST /query body |
| Disable for request | `"rerank": false` or omit (default) |
| Pre-warm model | `WARMUP_RERANKER=true` in .env |
| Adjust timeout | `RERANK_TIMEOUT_S=10.0` in .env |

## How to Enable Reranking

### Per-Request (Recommended for Testing)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "your-workspace-id",
    "question": "What is Python?",
    "mode": "retrieve",
    "rerank": true,
    "retrieve_k": 50,
    "top_k": 10
  }'
```

### Per-Workspace (TODO: When DB-backed)

```sql
-- Future: Update workspace config
UPDATE workspaces
SET config = jsonb_set(config, '{rerank,enabled}', 'true')
WHERE id = 'your-workspace-id';
```

### Globally via Environment

```bash
# In .env - NOT RECOMMENDED for production
# Use per-request or per-workspace instead
WARMUP_RERANKER=true  # Pre-load model at startup
```

## How to Disable Instantly

### Per-Request Override

```json
{
  "rerank": false
}
```

Request-level `rerank: false` always takes precedence over workspace config.

### Kill Switch (Emergency)

```bash
# Restart service without warmup - model won't be loaded
unset WARMUP_RERANKER
docker compose restart trading-rag-svc

# Or set rerank: false in all requests via API gateway/proxy
```

## Interpreting Log Fields

### Structured Log Example

```json
{
  "event": "Query pipeline complete",
  "rerank_enabled": true,
  "rerank_state": "ok",
  "rerank_method": "cross_encoder",
  "rerank_model": "BAAI/bge-reranker-v2-m3",
  "rerank_timeout": false,
  "rerank_fallback": false,
  "candidates_k": 50,
  "final_k": 10,
  "rerank_ms": 127,
  "total_ms": 450
}
```

### Key Fields

| Field | Values | Meaning |
|-------|--------|---------|
| `rerank_state` | `disabled` | Reranking not requested |
| | `ok` | Reranking completed successfully |
| | `timeout_fallback` | Reranking timed out, used vector order |
| | `error_fallback` | Reranking failed, used vector order |
| `rerank_timeout` | `true/false` | Whether timeout occurred |
| `rerank_fallback` | `true/false` | Whether fell back to vector order |
| `rerank_ms` | integer | Reranking latency (null if disabled) |

## Troubleshooting: Timeout Spikes

If you see `rerank_state: timeout_fallback` spiking:

### 1. Disable Reranking (Immediate Relief)

```json
// In requests
{ "rerank": false }
```

### 2. Reduce Candidate Pool

```json
{
  "rerank": true,
  "retrieve_k": 30,  // Down from 50
  "top_k": 5         // Down from 10
}
```

### 3. Reduce Neighbor Window

Update workspace config (when DB-backed):
```json
{
  "neighbor": {
    "window": 0,      // Disable neighbors
    "pdf_window": 1   // Or reduce PDF window
  }
}
```

### 4. Increase Timeout (If GPU Can Handle It)

```bash
# In .env
RERANK_TIMEOUT_S=15.0  # Up from 10.0
```

### 5. Check GPU Contention

```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check for OOM in logs
docker compose logs trading-rag-svc | grep -i "cuda\|memory\|oom"
```

## Alert Conditions

### Recommended Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| Rerank Fallback Rate | `rate(rerank_state in [timeout_fallback, error_fallback]) > 5%` over 5min | Warning |
| Rerank Fallback Rate | `rate(rerank_state in [timeout_fallback, error_fallback]) > 20%` over 5min | Critical |
| Rerank Latency P95 | `p95(rerank_ms) > 500ms` over 5min | Warning |
| Rerank Latency P95 | `p95(rerank_ms) > 1000ms` over 5min | Critical |
| Total Query Latency | `p95(total_ms) > 2000ms` over 5min | Warning |

### Log-Based Alert (If No Metrics System)

```bash
# Count fallbacks in last 5 minutes
docker compose logs --since 5m trading-rag-svc 2>&1 | \
  grep -c '"rerank_state": "timeout_fallback\|error_fallback"'

# Alert if > 10 occurrences
```

### Prometheus Metrics (If Instrumented)

```promql
# Fallback rate
sum(rate(query_rerank_state_total{state=~"timeout_fallback|error_fallback"}[5m]))
/
sum(rate(query_rerank_state_total[5m]))

# P95 rerank latency
histogram_quantile(0.95, sum(rate(query_rerank_ms_bucket[5m])) by (le))
```

## Performance Characteristics

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Model load time | 5-15s | First request or warmup |
| Inference latency | 50-200ms | For 50 candidates on GPU |
| Memory usage | ~2GB VRAM | BGE-reranker-v2-m3 |
| Timeout default | 10s | Configurable via `RERANK_TIMEOUT_S` |

## Safety Caps (Hardcoded)

| Cap | Value | Purpose |
|-----|-------|---------|
| `MAX_CANDIDATES_K` | 200 | Prevent memory exhaustion |
| `MAX_FINAL_K` | 50 | Prevent response bloat |
| `MAX_NEIGHBOR_TOTAL` | 50 | Limit neighbor expansion |

These caps are enforced in `app/routers/query.py` and cannot be overridden by request or workspace config.
