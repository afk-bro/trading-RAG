# Operations Runbooks

Standard operating procedures for Trading RAG service.

## Table of Contents

1. [Qdrant Collection Rebuild](#qdrant-collection-rebuild)
2. [Embedding Model Rotation](#embedding-model-rotation)
3. [Handling KB Status: None](#handling-kb-status-none)
4. [Handling KB Status: Degraded](#handling-kb-status-degraded)
5. [Service Restart Procedure](#service-restart-procedure)

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
