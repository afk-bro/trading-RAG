# Operations Runbooks

Standard operating procedures for Trading RAG service.

## Table of Contents

1. [Multi-Replica Deployment Notes](#multi-replica-deployment-notes)
2. [Qdrant Collection Rebuild](#qdrant-collection-rebuild)
3. [Embedding Model Rotation](#embedding-model-rotation)
4. [Handling KB Status: None](#handling-kb-status-none)
5. [Handling KB Status: Degraded](#handling-kb-status-degraded)
6. [KB Trial Ingestion Operations](#kb-trial-ingestion-operations)
7. [Service Restart Procedure](#service-restart-procedure)
8. [Failure Modes & Recovery](#failure-modes--recovery)

---

## Multi-Replica Deployment Notes

**Important:** The current rate limiting and concurrency controls are per-replica only.

### Current Behavior (Single-Replica)

| Control | Implementation | Scope |
|---------|----------------|-------|
| Rate Limiter | In-process sliding window | Per replica |
| Workspace Semaphore | asyncio.Semaphore | Per replica |

### Multi-Replica Implications

In multi-replica deployments:

- **Rate limits become approximate.** With N replicas, effective limit is N × configured limit.
  - Example: 30 req/min configured → 60 req/min actual with 2 replicas
- **Concurrency caps multiply.** Each replica maintains its own semaphore.
  - Example: 2 concurrent/workspace configured → 4 actual with 2 replicas

### Upgrade Path (If Needed)

To achieve true distributed rate limiting:

1. **Redis-backed limiter** (recommended)
   ```python
   # Replace RateLimiter with Redis implementation
   # Upstash Redis recommended for serverless
   from upstash_ratelimit import Ratelimit
   ```

2. **Environment variables**
   ```bash
   REDIS_URL=redis://...
   RATE_LIMIT_BACKEND=redis  # Switch from 'memory' to 'redis'
   ```

3. **Distributed semaphore** (for concurrency)
   ```python
   # Use Redis-based distributed lock
   import redis.asyncio as redis
   from redis.asyncio.lock import Lock
   ```

### When to Upgrade

Upgrade to Redis-backed limiting when:
- Running 3+ replicas
- Rate limit accuracy is critical for billing/quota
- Need to prevent workspace abuse across replicas

**For single-replica or 2-replica deployments, the current implementation is sufficient.**

---

## Qdrant Collection Rebuild

When to use: Collection corruption, dimension mismatch, or after major schema changes.

### Prerequisites
- Admin token (`ADMIN_TOKEN` env var)
- Access to Qdrant management API
- Backup of existing collection (if data is valuable)

### Procedure

1. **Verify current state**
   ```bash
   # Check readiness (will show collection dimension mismatch if any)
   curl -s http://localhost:8000/ready | jq .

   # Check Qdrant collection directly
   curl -s http://qdrant:6333/collections/kb_nomic_embed_text_v1 | jq .
   ```

2. **Backup existing collection (optional)**
   ```bash
   # Create snapshot
   curl -X POST "http://qdrant:6333/collections/kb_nomic_embed_text_v1/snapshots"
   ```

3. **Delete and recreate collection**
   ```bash
   # Delete collection
   curl -X DELETE "http://qdrant:6333/collections/kb_nomic_embed_text_v1"

   # Recreate with correct dimensions (768 for nomic-embed-text)
   curl -X PUT "http://qdrant:6333/collections/kb_nomic_embed_text_v1" \
     -H "Content-Type: application/json" \
     -d '{
       "vectors": {
         "size": 768,
         "distance": "Cosine"
       },
       "on_disk_payload": true
     }'
   ```

4. **Trigger re-indexing**
   ```bash
   # Re-embed all chunks for workspace
   curl -X POST "http://localhost:8000/kb/reembed" \
     -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"workspace_id": "YOUR_WORKSPACE_ID"}'
   ```

5. **Verify rebuild**
   ```bash
   # Check /ready returns 200
   curl -s http://localhost:8000/ready | jq .

   # Test recommend endpoint
   curl -X POST "http://localhost:8000/kb/trials/recommend" \
     -H "Content-Type: application/json" \
     -d '{"workspace_id": "...", "strategy_name": "bb_reversal", "objective_type": "sharpe"}'
   ```

### Rollback
If rebuild fails, restore from snapshot:
```bash
curl -X PUT "http://qdrant:6333/collections/kb_nomic_embed_text_v1/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "http://qdrant:6333/collections/kb_nomic_embed_text_v1/snapshots/SNAPSHOT_NAME"}'
```

---

## Embedding Model Rotation

When to use: Upgrading to better embedding model, or switching providers.

### Prerequisites
- New model pulled in Ollama: `ollama pull <new-model>`
- New collection created with matching dimensions
- Workspace config updated to point to new collection

### Procedure

1. **Pull new model**
   ```bash
   docker exec ollama ollama pull nomic-embed-text:v1.5
   # Or for different model:
   docker exec ollama ollama pull mxbai-embed-large
   ```

2. **Create new collection with correct dimensions**
   ```bash
   # Check model dimensions first
   # nomic-embed-text: 768
   # mxbai-embed-large: 1024

   curl -X PUT "http://qdrant:6333/collections/kb_mxbai_embed_large_v1" \
     -H "Content-Type: application/json" \
     -d '{
       "vectors": {"size": 1024, "distance": "Cosine"},
       "on_disk_payload": true
     }'
   ```

3. **Update workspace configuration**
   ```sql
   -- Via Supabase SQL Editor or psql
   UPDATE workspaces
   SET
     default_embed_model = 'mxbai-embed-large',
     default_collection = 'kb_mxbai_embed_large_v1',
     config = config || '{"chunking": {"embed_model": "mxbai-embed-large"}}'::jsonb
   WHERE slug = 'trading';
   ```

4. **Re-embed existing documents**
   ```bash
   curl -X POST "http://localhost:8000/kb/reembed" \
     -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "workspace_id": "YOUR_WORKSPACE_ID",
       "target_model": "mxbai-embed-large",
       "target_collection": "kb_mxbai_embed_large_v1"
     }'
   ```

5. **Update service configuration**
   ```bash
   # Update .env
   EMBED_MODEL=mxbai-embed-large
   EMBED_DIM=1024
   QDRANT_COLLECTION_ACTIVE=kb_mxbai_embed_large_v1

   # Restart service
   docker compose restart trading-rag-svc
   ```

6. **Verify**
   ```bash
   curl -s http://localhost:8000/ready | jq .
   curl -s http://localhost:8000/health | jq '.embed_model, .active_collection'
   ```

### Rollback
Revert environment variables and restart service. Old collection remains intact.

---

## Handling KB Status: None

When alert fires for high `kb_status=none` rate.

### Diagnosis

1. **Check recent logs**
   ```bash
   docker compose logs trading-rag-svc --since 15m | grep kb_recommend_complete
   # Look for: status=none, after_strict=0, after_relaxed=0
   ```

2. **Identify root cause via Sentry**
   - Filter by `tag:kb_status=none`
   - Check `reasons` field in context
   - Common reasons:
     - `no_matches_after_strict_filters` - Filters too restrictive
     - `no_embedding_available` - Ollama down
     - `collection_empty` - No data indexed

3. **Check filter rejections (debug mode)**
   ```bash
   curl -X POST "http://localhost:8000/kb/trials/recommend?mode=debug" \
     -H "Content-Type: application/json" \
     -d '{
       "workspace_id": "...",
       "strategy_name": "bb_reversal",
       "objective_type": "sharpe"
     }' | jq '.filter_rejections, .recommended_relaxed_settings'
   ```

### Resolution Options

| Root Cause | Action |
|------------|--------|
| Collection empty | Run re-indexing (see Qdrant Rebuild) |
| Ollama down | Restart Ollama service |
| Filters too strict | Adjust request params or use recommended_relaxed_settings |
| No matching strategy data | Ingest more trials via `/kb/trials/ingest` |

### Relaxation Example
```bash
# Use relaxed filters in request
curl -X POST "http://localhost:8000/kb/trials/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "...",
    "strategy_name": "bb_reversal",
    "objective_type": "sharpe",
    "require_oos": false,
    "max_drawdown": 0.5,
    "min_trades": 3
  }'
```

---

## Handling KB Status: Degraded

When alert fires for high `kb_status=degraded` rate.

### Understanding Degraded Status

Degraded means recommendations are available but with caveats:
- Used relaxed filters (less strict matching)
- Used metadata-only fallback (no vector similarity)
- Low candidate count
- Missing regime context

### Diagnosis

1. **Check logs for degraded pattern**
   ```bash
   docker compose logs trading-rag-svc --since 15m | grep "status=degraded"
   # Look for: strict_to_relaxed=true, metadata_only=true
   ```

2. **Check Sentry for patterns**
   - Filter by `tag:kb_status=degraded`
   - Group by `tag:strategy` to find problematic strategies
   - Check `warnings` array in context

### Resolution Options

| Pattern | Root Cause | Action |
|---------|------------|--------|
| `strict_to_relaxed=true` | Strict filters too aggressive | Review filter thresholds |
| `metadata_only=true` | Vector search failing | Check Qdrant health |
| `repaired_params=true` | Missing/invalid params | Data quality issue |
| Low count for specific strategy | Insufficient training data | Ingest more trials |

### Monitoring Dashboard Query
```
# Degraded breakdown
tag:kb_status=degraded count() group by tag:strategy, has:strict_to_relaxed
```

---

## KB Trial Ingestion Operations

Operations for the trial ingestion pipeline that bridges backtest results to the KB.

### Dry-Run Ingestion

Preview what would be ingested without modifying Qdrant or the index:

```bash
# Via Python (service-level)
from app.services.kb.ingest import KBTrialIngester, IngestConfig

config = IngestConfig(dry_run=True)
ingester = KBTrialIngester(..., config=config)
result = await ingester.ingest_workspace(workspace_id)
# result.inserted/updated/skipped show what WOULD happen
```

### Safe Re-Ingestion

Ingestion is idempotent via content hashing. Safe to re-run anytime:

```bash
# Re-ingest all eligible trials for workspace
curl -X POST "http://localhost:8000/kb/trials/ingest" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id": "YOUR_WORKSPACE_ID"}'
```

**What happens:**
- Unchanged trials → `skipped` (content hash matches)
- Modified trials → `updated` (new hash, re-embedded)
- New trials → `inserted`
- Previously archived → `unarchived` and re-embedded

### Archive/Delete Failure Recovery

When archive fails to delete from Qdrant:

1. **Check logs for correlation ID:**
   ```bash
   docker compose logs trading-rag-svc | grep "kb_archive_qdrant_failed"
   # Look for: correlation_id, point_id, error
   ```

2. **The index is still marked archived** (with error note in reason field):
   ```sql
   SELECT id, reason, archived_at FROM kb_trial_index
   WHERE reason LIKE '%qdrant_delete_failed%';
   ```

3. **Self-healing:** Re-running ingest on the same trial will:
   - Unarchive the entry
   - Re-upsert to Qdrant (overwriting orphaned point)
   - Clear the error state

4. **Manual cleanup (if needed):**
   ```bash
   # Delete orphaned Qdrant point manually
   curl -X POST "http://qdrant:6333/collections/kb_trials/points/delete" \
     -H "Content-Type: application/json" \
     -d '{"points": ["<point_id_from_log>"]}'
   ```

### Interpreting Observability Counters

Three structured log counters for monitoring:

| Counter | Fields | Meaning |
|---------|--------|---------|
| `kb_ingest_action_total` | `action`, `count`, `workspace_id` | Batch ingestion results |
| `kb_admin_status_change_total` | `transition`, `actor`, `actor_type` | Admin status changes |
| `kb_tiebreak_applied_total` | `rule`, `workspace_id`, `strategy` | Which comparator rules are used |

**Ingest action breakdown:**
- `inserted` - New trials added to KB
- `updated` - Existing trials re-embedded (content changed)
- `skipped` - Unchanged (normal for re-runs)
- `unarchived` - Previously rejected trials restored
- `error` - Failed to process (check logs)

**Status change patterns:**
- `candidate_to_promoted` - Admin approved trial
- `candidate_to_rejected` - Admin rejected trial
- `rejected_to_promoted` - Admin restored rejected trial

**Tiebreak rules (in priority order):**
- `score` - Primary score difference > epsilon
- `status` - promoted beats candidate
- `schema` - Current schema preferred
- `promoted_at` - Recent promotion wins
- `created_at` - Newer trial wins
- `source_id` - Deterministic fallback

---

## Service Restart Procedure

Standard restart for trading-rag service.

### Pre-flight Checks
```bash
# Check current health
curl -s http://localhost:8000/health | jq .status
curl -s http://localhost:8000/ready | jq .ready

# Check for active requests (should be minimal)
docker compose logs trading-rag-svc --since 1m | grep -c kb_recommend_start
```

### Graceful Restart
```bash
# Rolling restart (if using multiple replicas)
docker compose up -d --no-deps --scale trading-rag-svc=2 trading-rag-svc
sleep 30
docker compose up -d --no-deps --scale trading-rag-svc=1 trading-rag-svc

# Single replica restart
docker compose restart trading-rag-svc
```

### Post-Restart Verification
```bash
# Wait for startup
sleep 10

# Verify health
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/ready | jq .

# Test recommend endpoint
curl -X POST "http://localhost:8000/kb/trials/recommend" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id": "...", "strategy_name": "bb_reversal", "objective_type": "sharpe"}' \
  | jq '.status, .confidence'
```

### Emergency Rollback
```bash
# If new version is broken, roll back to previous image
docker compose pull trading-rag-svc:previous-tag
docker compose up -d trading-rag-svc
```

---

## Failure Modes & Recovery

The service implements circuit breakers and retry logic for transient failures.

### Circuit Breaker Status

Check circuit breaker state via health endpoint:

```bash
curl -s http://localhost:8000/health | jq '.circuit_breakers'
# Returns: {"supabase": {"failures": 0, "is_open": false}, "qdrant": {...}}
```

### Circuit Breaker States

| State | Meaning | Behavior |
|-------|---------|----------|
| Closed | Healthy | Normal operation |
| Open | Service down | Requests fail-fast (no retry) |
| Half-Open | Testing recovery | One request allowed through |

### Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| Failure threshold | 5 | Consecutive failures to open circuit |
| Reset timeout | 30s | Time before half-open test |
| Max retry attempts | 3 | Retries per operation |
| Backoff | exponential | 0.5s base, 2x multiplier, 10s max |

### Prometheus Metrics

```
# Retry attempts by service
resilience_retries_total{service="db"}
resilience_retries_total{service="qdrant"}

# Circuit breaker state (0=closed, 1=open, 2=half_open)
circuit_breaker_state{service="db"}
circuit_breaker_state{service="qdrant"}

# Failure counts and trip events
circuit_breaker_failures_total{service="db|qdrant"}
circuit_breaker_trips_total{service="db|qdrant"}
```

### When Circuit Opens

**Symptoms:**
- Requests return 500 with "circuit breaker is open"
- `circuit_breaker_trips_total` counter increments
- `/health` shows `is_open: true`

**Diagnosis:**
```bash
# Check which service tripped
curl -s http://localhost:8000/health | jq '.circuit_breakers | to_entries | map(select(.value.is_open)) | .[].key'

# Check recent errors
docker compose logs trading-rag-svc --since 5m | grep -E "(db_retry|qdrant_retry|circuit_opened)"
```

**Resolution:**
1. **Wait for auto-recovery** - Circuit resets after 30s
2. **Fix underlying issue** - Check database/Qdrant connectivity
3. **Manual reset** (via service restart if circuit stuck)

### Transient vs Non-Transient Errors

**Retried (transient):**
- Connection refused/reset
- Timeouts
- Pool exhaustion
- Network errors

**Not retried (non-transient):**
- SQL syntax errors
- Constraint violations
- Authentication failures
- Missing collections

### Alert Thresholds

Recommended alert rules:

```yaml
# Circuit breaker opened
- alert: CircuitBreakerOpen
  expr: circuit_breaker_state > 0
  for: 1m
  labels:
    severity: warning

# High retry rate
- alert: HighRetryRate
  expr: rate(resilience_retries_total[5m]) > 0.5
  for: 5m
  labels:
    severity: warning
```

---

## Quick Reference

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe (always 200) |
| `GET /ready` | Readiness probe (503 if deps unhealthy) |
| `GET /metrics` | Prometheus metrics |
| `POST /kb/reembed` | Re-index documents (admin) |
| `POST /kb/trials/ingest` | Ingest trials (admin) |

| Environment Variable | Purpose |
|---------------------|---------|
| `ADMIN_TOKEN` | Admin endpoint authentication |
| `EMBED_MODEL` | Active embedding model |
| `EMBED_DIM` | Embedding dimensions |
| `QDRANT_COLLECTION_ACTIVE` | Active vector collection |
| `DOCS_ENABLED` | Enable/disable /docs (false in prod) |
| `CORS_ORIGINS` | Allowed CORS origins |
