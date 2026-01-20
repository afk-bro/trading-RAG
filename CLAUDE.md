# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Git & Communication Guidelines

**Do not reference Claude or AI in any git artifacts.** This includes commit messages, PR titles/descriptions, code comments, and branch names. Write as if a human developer authored them.

## Overview

**rag-core** is a multi-tenant Retrieval-Augmented Generation system. FastAPI service that ingests documents (YouTube transcripts, PDFs, Pine scripts), chunks them with token-awareness, generates embeddings via Ollama, stores vectors in Qdrant, and provides semantic search with LLM-powered answer generation.

Workspace-scoped: each workspace has its own configuration stored in a control-plane table.

## Architecture

```
Sources (YouTube, PDF, Pine)
              │
              ▼
    ┌─────────────────┐
    │  FastAPI Service │
    │   (rag-core)     │
    └────────┬────────┘
             │
    ┌────────┼────────┬────────────┐
    ▼        ▼        ▼            ▼
 Ollama   Qdrant   Supabase   Workspaces
(Embed)  (Vector)  (Postgres)  (Control)
```

**Data Flow**: Content → Extract → Chunk (512 tokens) → Embed (768-dim) → Vector + Postgres

**Layer Pattern**:
- `app/routers/` - API endpoints (thin controllers)
- `app/services/` - Business logic (chunker, embedder, pdf_extractor, llm, intel)
- `app/repositories/` - Data access (documents, chunks, vectors, strategy_intel)

## Common Commands

```bash
# Setup
./init.sh                                    # Creates .env, starts Docker services

# Development
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000   # Local dev server

# Docker
docker compose -f docker-compose.rag.yml up --build
docker compose -f docker-compose.rag.yml logs -f trading-rag-svc

# Testing
pytest tests/                                # All tests
pytest tests/unit/                          # Unit only
pytest tests/integration/ -m "not requires_db"  # Integration (mocked DB)

# Linting (CI runs these)
black --check app/ tests/
flake8 app/ tests/ --max-line-length=100
mypy app/ --ignore-missing-imports
```

## Configuration

**Required** (set in `.env`):
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Database credentials
- `DATABASE_URL` - PostgreSQL connection string

**Optional**:
- `OPENROUTER_API_KEY` - LLM API access (enables `mode=answer` queries)

Services default to Docker network hostnames: `qdrant:6333`, `ollama:11434`

**Database Notes**: Use Supabase transaction pooler (port 6543). asyncpg configured with `statement_cache_size=0` for pgbouncer compatibility.

## Key Architectural Details

| Principle | Details |
|-----------|---------|
| **Async-First** | All I/O uses async (asyncpg, httpx, Qdrant async). No blocking calls. |
| **Token-Aware Chunking** | tiktoken for accurate counting. ~512 tokens with overlap. Page-aware for PDFs. |
| **PDF Extraction** | Swappable backend (PyMuPDF, pdfplumber). Interface: `extract_pdf(file_bytes, config)` |
| **Idempotency** | Content hashes + idempotency keys. Unique constraint on `(workspace_id, source_type, canonical_url)` |
| **Model Migration** | `chunk_vectors` tracks embeddings per model/collection. Re-embed endpoint available. |

## Database Schema

PostgreSQL tables (via Supabase):

| Table | Purpose |
|-------|---------|
| `workspaces` | Control plane: routing defaults, config (jsonb), flags |
| `documents` | Source metadata, content hash, status lifecycle |
| `chunks` | Text segments, token counts, metadata arrays |
| `chunk_vectors` | Maps chunks to embedding model/collection |
| `strategies` | Strategy registry with tags, source refs, backtest summaries |
| `strategy_versions` | Immutable config snapshots with state machine (draft→active↔paused→retired) |
| `strategy_version_transitions` | Audit log for version state changes |
| `strategy_intel_snapshots` | Append-only time series of regime + confidence per version |
| `backtest_runs` | Strategy backtest results with metrics |
| `tune_sessions` | Parameter sweep sessions |
| `wfo_runs` | Walk-forward optimization results |
| `ops_alerts` | Operational alerts with delivery tracking |
| `jobs` | Async job queue with status tracking |

All tables FK to workspaces for multi-tenant isolation. Migrations in `migrations/`.

## API Endpoints

### Core RAG
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Generic document ingestion |
| POST | `/sources/youtube/ingest` | YouTube transcript ingestion |
| POST | `/sources/pdf/ingest` | PDF file upload (multipart) |
| POST | `/sources/pine/ingest` | Pine Script registry (admin) |
| POST | `/query` | Semantic search (`retrieve` or `answer` mode) |
| GET | `/jobs/{job_id}` | Async job status |

### Strategy Lifecycle
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/strategies/{id}/versions` | Create draft version |
| GET | `/strategies/{id}/versions` | List versions |
| GET | `/strategies/{id}/versions/{vid}` | Get version details |
| POST | `/strategies/{id}/versions/{vid}/activate` | Activate version (pauses current active) |
| POST | `/strategies/{id}/versions/{vid}/pause` | Pause active version |
| POST | `/strategies/{id}/versions/{vid}/retire` | Retire version (terminal) |
| GET | `/strategies/{id}/versions/{vid}/transitions` | Get audit trail |
| GET | `/strategies/{id}/versions/{vid}/intel/latest` | Latest intel snapshot |
| GET | `/strategies/{id}/versions/{vid}/intel` | Intel timeline (cursor pagination) |
| POST | `/strategies/{id}/versions/{vid}/intel/recompute` | Trigger intel recomputation |

### Backtests & WFO
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/backtests/tune` | Start parameter sweep |
| GET | `/backtests/tunes` | List tunes |
| POST | `/backtests/wfo` | Queue WFO job |
| GET | `/backtests/leaderboard` | Global ranking |

### Execution (admin)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/execute/intents` | Execute trade intent (paper) |
| GET | `/execute/paper/state/{ws}` | Paper trading state |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/system/health` | System health dashboard |
| GET | `/admin/coverage/cockpit` | Coverage triage UI |
| GET | `/admin/backtests/*` | Backtest admin UI |
| GET | `/admin/ops-alerts` | List ops alerts |
| POST | `/admin/ops-alerts/{id}/acknowledge` | Acknowledge alert |
| POST | `/admin/ops-alerts/{id}/resolve` | Resolve alert |
| POST | `/admin/ops-alerts/{id}/reopen` | Reopen resolved alert |
| GET | `/admin/ingest` | Ingest UI (YouTube, PDF, Pine) |

All admin endpoints require `X-Admin-Token` header.

## Testing

### Test Categories

| Category | Location | CI | Requirements |
|----------|----------|-----|--------------|
| Unit | `tests/unit/` | Always | None |
| Integration | `tests/integration/` | Always | Qdrant container |
| E2E | `tests/e2e/` | Manual | Server + browser |

### Markers

```python
@pytest.mark.requires_db      # Needs real DB - skipped normally
@pytest.mark.e2e              # Browser test - auto-skipped
@pytest.mark.slow             # Can deselect with -m "not slow"
```

### Required Env for Tests

```bash
export SUPABASE_URL=https://test.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=test-key
```

### Mypy Ratchet

`mypy.ini` has per-module ignores. Ratchet script fails CI if ignore count exceeds baseline (53). To fix: remove entry, fix errors, decrease baseline.

## Project Structure

```
trading-RAG/
├── app/                    # Main application code
│   ├── routers/            # API endpoints
│   ├── services/           # Business logic
│   │   ├── intel/          # Regime classification + confidence scoring
│   │   └── ops_alerts/     # Rule evaluation, deduplication, Telegram notifications
│   ├── repositories/       # Data access layer
│   └── jobs/               # Job handlers
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test data
├── migrations/             # SQL migrations
├── scripts/                # Utility scripts
├── docs/                   # Documentation
│   ├── ops/                # Operations docs
│   ├── plans/              # Design documents
│   └── archive/            # Archived specs
├── dashboards/             # Grafana dashboards
└── ops/                    # Prometheus configs
```

## Detailed Documentation

| Topic | Location |
|-------|----------|
| Backtest tuning & WFO | [docs/backtests.md](docs/backtests.md) |
| Pine Script system | [docs/pine-scripts.md](docs/pine-scripts.md) |
| Execution & strategy runner | [docs/execution.md](docs/execution.md) |
| Coverage triage & KB recommend | [docs/coverage.md](docs/coverage.md) |
| Operational hardening & security | [docs/ops/hardening.md](docs/ops/hardening.md) |
| Alerting rules | [docs/ops/alerting-rules.md](docs/ops/alerting-rules.md) |
| Runbooks | [docs/ops/runbooks.md](docs/ops/runbooks.md) |
| Tech debt | [docs/tech-debt.md](docs/tech-debt.md) |
| PRD v0.1 | [docs/plans/v.01-PRD.md](docs/plans/v.01-PRD.md) |

## Roadmap Context

Evolving from trading-focused RAG to general-purpose rag-core:
- Trading is the first workspace (finance knowledge base)
- PDF backend is swappable (pymupdf now, MinerU/StudyG later)
- Workspace config enables per-tenant customization without code changes
