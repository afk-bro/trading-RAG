# Trading RAG Pipeline - Finance Knowledge Base

A local RAG (Retrieval-Augmented Generation) pipeline for finance and trading knowledge. The system ingests YouTube transcripts and other documents via n8n orchestration, processes them through a FastAPI service with chunking, embedding, and storage capabilities.

## Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Google    │────▶│      n8n        │────▶│   FastAPI       │
│   Sheets    │     │   Orchestrator  │     │   Service       │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                    ┌────────────────────────────────┼───────────────────────────────┐
                    │                                │                               │
                    ▼                                ▼                               ▼
            ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
            │   Ollama    │                 │   Qdrant    │                 │  Supabase   │
            │ (Embeddings)│                 │  (Vectors)  │                 │  (Postgres) │
            └─────────────┘                 └─────────────┘                 └─────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | n8n (Docker) | Watch queue, call service, manage state |
| Backend | Python FastAPI | API endpoints, business logic |
| Primary DB | Supabase Postgres | Documents, chunks, metadata (source of truth) |
| Vector DB | Qdrant (Docker) | Embedding vectors, similarity search |
| Embeddings | Ollama (local) | nomic-embed-text (768 dimensions) |
| LLM | OpenRouter API | Answer generation (optional) |

## Features

- **YouTube Ingestion**: Parse URLs, fetch transcripts, extract metadata
- **Document Ingestion**: Support for PDF, article, note, transcript sources
- **Smart Chunking**: Token-aware (~512 tokens), timestamp preservation
- **Metadata Extraction**: Symbols, entities, topics, speakers
- **Semantic Search**: Qdrant vector search with payload filtering
- **Answer Generation**: Optional LLM synthesis with citations (graceful degradation)
- **Model Migration**: Re-embed support for model upgrades
- **Backtest Parameter Tuning**: Grid/random search, IS/OOS splits, overfit detection
- **Trading KB Recommend**: Strategy parameter recommendations with confidence scoring
- **Regime Fingerprints**: Materialized regime vectors for instant similarity queries
- **Pine Script Registry**: Parse, lint, and catalog Pine Script files with CLI tooling
- **Auto-Strategy Discovery**: Generate parameter specs from Pine Script inputs for backtesting
- **Coverage Triage Cockpit**: Manage weak coverage gaps with priority scoring and status workflow
- **LLM Strategy Explanation**: Generate explanations for strategy-intent matches
- **Admin UI**: Leaderboards, N-way tune comparison, ops snapshot, system health dashboard, ops alerts management
- **Ops Alerts**: Automated evaluation via pg_cron, Telegram delivery with activation/recovery/escalation notifications
- **Idempotent Notifications**: Race-safe delivery with conditional mark pattern, delivery tracking columns
- **Idempotency Hygiene**: Auto-prune via pg_cron, health page monitoring, Prometheus metrics
- **Security Hardening**: Admin auth, rate limiting, CORS allowlist, workspace isolation
- **Production Monitoring**: Sentry integration, structured logging, Prometheus alerting rules (28 alerts across 10 subsystems)

### Query Modes

| Mode | LLM Required | Description |
|------|--------------|-------------|
| `retrieve` | No | Semantic search only - returns ranked chunks |
| `answer` | Optional | LLM synthesis - returns chunks + generated answer |

When `mode=answer` is used without an LLM provider configured, the system returns retrieved chunks with a helpful message indicating that generation is disabled.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Supabase project with Postgres database
- OpenRouter API key (optional - only needed for `mode=answer` queries)
- n8n instance (optional, for automated ingestion)

### Setup

1. Clone and navigate to the project:
   ```bash
   cd trading-RAG
   ```

2. Run the setup script:
   ```bash
   ./init.sh
   ```

3. Configure environment variables in `.env`:
   ```bash
   # Required
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

   # Optional - enables LLM answer generation
   # OPENROUTER_API_KEY=your-openrouter-api-key
   ```

4. Access the services:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Qdrant Dashboard: http://localhost:6333/dashboard

## API Endpoints

### Health Check
```http
GET /health
```
Returns service status and dependency health.

### Ingest Document
```http
POST /ingest
Content-Type: application/json

{
  "workspace_id": "uuid",
  "source": {
    "url": "https://example.com/article",
    "type": "article"
  },
  "content": "Document content...",
  "metadata": {
    "title": "Article Title",
    "author": "Author Name"
  }
}
```

### Ingest YouTube
```http
POST /sources/youtube/ingest
Content-Type: application/json

{
  "workspace_id": "uuid",
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Query
```http
POST /query
Content-Type: application/json

{
  "workspace_id": "uuid",
  "question": "What is the Fed's current stance on interest rates?",
  "mode": "answer",
  "filters": {
    "source_types": ["youtube"],
    "symbols": ["SPY"],
    "topics": ["macro"]
  },
  "top_k": 5
}
```

### Re-embed
```http
POST /reembed
Content-Type: application/json

{
  "workspace_id": "uuid",
  "target_collection": "kb_new_model_v1",
  "embed_provider": "ollama",
  "embed_model": "new-model-name"
}
```

### Job Status
```http
GET /jobs/{job_id}
```

### Backtest Tuning

```http
POST /backtests/tune
Content-Type: application/json

{
  "workspace_id": "uuid",
  "strategy_entity_id": "uuid",
  "param_space": {"fast_period": [5, 10, 20], "slow_period": [20, 50, 100]},
  "objective_type": "sharpe_dd_penalty",
  "oos_ratio": 0.3
}
```

```http
GET /backtests/leaderboard?workspace_id=uuid&valid_only=true&objective_type=sharpe
```

### Trading KB Recommend
```http
POST /trading-kb/recommend
Content-Type: application/json

{
  "workspace_id": "uuid",
  "strategy_entity_id": "uuid",
  "market_regime": "trending",
  "risk_tolerance": "moderate"
}
```
Returns parameter recommendations with confidence scores based on knowledge base analysis.

### Readiness Check
```http
GET /ready
```
Deep dependency health check for Kubernetes readiness probes. Returns 200 when all dependencies (DB, Qdrant, embedder) are healthy, 503 otherwise.

### Admin UI Routes

| Route | Purpose |
|-------|---------|
| `/admin/backtests/tunes` | Filterable tune list |
| `/admin/backtests/leaderboard` | Global ranking (CSV export) |
| `/admin/backtests/compare?tune_id=A&tune_id=B` | N-way diff table (JSON export) |
| `/admin/ops/snapshot` | Go-live verification (release, config, health) |
| `/admin/system/health` | System health dashboard (status cards) |
| `/admin/system/health.json` | System health (machine-readable) |
| `/admin/coverage/cockpit` | Coverage triage cockpit UI |
| `/admin/coverage/cockpit/{run_id}` | Deep link to specific run |
| `/admin/ops-alerts` | Ops alerts management |
| `/admin/ops-alerts/{id}/acknowledge` | Acknowledge alert |
| `/admin/ops-alerts/{id}/resolve` | Resolve alert |
| `/admin/ops-alerts/{id}/reopen` | Reopen resolved alert |
| `/admin/ingest` | Ingest UI (YouTube, PDF, Pine) |

### Pine Script Registry CLI

Build a registry of Pine Script files with metadata and lint findings:

```bash
# Build registry from directory
python -m app.services.pine --build ./scripts

# Custom output directory
python -m app.services.pine --build ./scripts -o ./data

# Include additional extensions
python -m app.services.pine --build ./scripts --extensions .pine .txt

# Quiet mode (suppress info logging)
python -m app.services.pine --build ./scripts -q
```

**Output files:**
- `pine_registry.json` - Script metadata (version, type, inputs, features) with lint summaries
- `pine_lint_report.json` - Full lint findings per script

**Lint Rules:**
| Code | Severity | Description |
|------|----------|-------------|
| E001 | Error | Missing `//@version` directive |
| E002 | Error | Invalid version number |
| E003 | Error | Missing `indicator()`/`strategy()`/`library()` declaration |
| W002 | Warning | `lookahead=barmerge.lookahead_on` usage (future data leakage risk) |
| W003 | Warning | Deprecated `security()` instead of `request.security()` |
| I001 | Info | Script has exports but is not a library |
| I002 | Info | Script exceeds recommended line count |

## Project Structure

```
trading-RAG/
├── app/
│   ├── main.py               # FastAPI application
│   ├── config.py             # Configuration management
│   ├── schemas.py            # Pydantic models
│   ├── deps/
│   │   └── security.py       # Auth, rate limiting, concurrency
│   ├── routers/              # API endpoints
│   │   ├── health.py         # /health and /ready endpoints
│   │   ├── ingest.py
│   │   ├── youtube.py
│   │   ├── query.py
│   │   ├── backtests.py
│   │   └── trading_kb.py
│   ├── admin/
│   │   ├── router.py         # Admin UI and ops snapshot
│   │   └── templates/        # HTML templates
│   ├── services/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── llm.py
│   │   ├── pine/             # Pine Script registry module
│   │   └── ops_alerts/       # Telegram notifications
│   │       ├── evaluator.py  # Alert evaluation rules
│   │       └── telegram.py   # Telegram delivery
│   ├── repositories/
│   │   ├── documents.py
│   │   ├── chunks.py
│   │   └── ops_alerts.py     # Delivery tracking
│   └── jobs/
│       └── handlers/         # Job handlers
├── tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test data
├── scripts/                  # Utility scripts
├── migrations/               # SQL migrations
├── docs/
│   ├── ops/                  # Operations docs
│   │   ├── alerting-rules.md
│   │   ├── runbooks.md
│   │   └── hardening.md
│   ├── plans/                # Design documents
│   └── archive/              # Archived specs
├── dashboards/               # Grafana dashboards
├── ops/                      # Prometheus configs
├── docker-compose.rag.yml
├── Dockerfile
├── requirements.txt
├── app_spec.txt              # Application specification
├── feature_list.json         # Test feature list
└── README.md
```

## Database Schema

### documents
- Core document metadata
- Unique constraint on (workspace_id, source_type, canonical_url)
- Status: active, superseded, deleted

### chunks
- Content segments with token counts
- Timestamp tracking for YouTube
- Page tracking for PDFs
- Metadata arrays: symbols, entities, topics

### chunk_vectors
- Tracks embeddings per model/collection
- Supports model migration workflows

### backtest_tunes
- Parameter sweep sessions with objective config
- Gates policy snapshots for audit trail
- Status: pending, running, completed, canceled

### backtest_tune_runs
- Individual trials within a tune
- IS/OOS metrics and scores
- Composite objective scoring

### ops_alerts
- Operational alerts with severity levels (critical, high, medium, low)
- Status lifecycle: active, acknowledged, resolved
- Delivery tracking: `notified_at`, `recovery_notified_at`, `escalation_notified_at`
- Deduplication via `dedupe_key`
- Telegram message ID for audit trail

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Supabase project URL | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Yes |
| `OPENROUTER_API_KEY` | OpenRouter API key (enables `mode=answer`) | No |
| `QDRANT_HOST` | Qdrant host (default: qdrant) | No |
| `QDRANT_PORT` | Qdrant port (default: 6333) | No |
| `OLLAMA_HOST` | Ollama host (default: ollama) | No |
| `OLLAMA_PORT` | Ollama port (default: 11434) | No |
| `EMBED_MODEL` | Embedding model (default: nomic-embed-text) | No |
| `SERVICE_PORT` | Service port (default: 8000) | No |

**Note:** Without `OPENROUTER_API_KEY`, semantic search (`mode=retrieve`) works fully. LLM answer generation (`mode=answer`) will return retrieved chunks with a message indicating generation is disabled.

## Production Deployment

### Security Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ADMIN_TOKEN` | Required for `/admin/*` endpoints | None (endpoints disabled) |
| `DOCS_ENABLED` | Enable `/docs`, `/redoc`, `/openapi.json` | `true` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |
| `RATE_LIMIT_ENABLED` | Enable request rate limiting | `true` |
| `CONFIG_PROFILE` | Environment profile (`development`/`production`) | `development` |

### Build Metadata (set by CI/CD)

| Variable | Description |
|----------|-------------|
| `GIT_SHA` | Git commit SHA for release tracking |
| `BUILD_TIME` | ISO8601 build timestamp |

### Monitoring

| Variable | Description |
|----------|-------------|
| `SENTRY_DSN` | Sentry error tracking DSN |
| `SENTRY_ENVIRONMENT` | Environment tag for Sentry |
| `SENTRY_TRACES_SAMPLE_RATE` | Performance tracing sample rate (0.0-1.0) |

### Health Probes

- **Liveness**: `GET /health` - Basic service status
- **Readiness**: `GET /ready` - Deep dependency checks (DB, Qdrant, embedder)

Use `/ready` for Kubernetes readiness probes to prevent traffic during dependency outages.

## Development

### Run locally
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### Run tests
```bash
pytest tests/
```

### Run with Docker
```bash
docker compose -f docker-compose.rag.yml up --build
```

## License

MIT License - See LICENSE for details.
