# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Git & Communication Guidelines

**Do not reference Claude or AI in any git artifacts.** This includes commit messages, PR titles/descriptions, code comments, and branch names. Write as if a human developer authored them.

## Overview

**rag-core** is a multi-tenant Retrieval-Augmented Generation system. FastAPI service that ingests documents (YouTube transcripts, PDFs, Pine scripts, articles, text/markdown), chunks them with token-awareness, generates embeddings via Ollama, stores vectors in Qdrant, and provides semantic search with LLM-powered answer generation.

Workspace-scoped: each workspace has its own configuration stored in a control-plane table.

## Architecture

```
Sources (YouTube, PDF, Pine, Articles, Text)
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
./init.sh                                    # Creates .env, starts Docker

# Development
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Docker
docker compose -f docker-compose.rag.yml up --build

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
- `ADMIN_TOKEN` - Admin endpoint protection

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
| `workspaces` | Control plane (routing defaults, config jsonb) |
| `documents` | Source metadata, content hash, status lifecycle |
| `chunks` | Text segments with token counts, metadata arrays |
| `chunk_vectors` | Maps chunks to embedding model/collection |
| `chunk_validations` | QA status per chunk (verified, needs_review, garbage) |
| `strategy_versions` | Immutable config snapshots with state machine |
| `strategy_intel_snapshots` | Regime + confidence time series |
| `paper_equity_snapshots` | Equity tracking for paper trading |

All tables FK to workspaces. Migrations in `migrations/`.

## API Endpoints

**Health**: `GET /health`, `GET /ready`, `GET /metrics`

**RAG Core**:
- `POST /ingest/unified` - Auto-detecting unified endpoint (YouTube, PDF, article, text, Pine)
- `POST /ingest`, `/sources/youtube/ingest`, `/sources/pdf/ingest`, `/sources/pine/ingest`
- `POST /query` (modes: `retrieve`, `answer`)
- `GET /sources/pine/scripts`, `GET /sources/pine/scripts/{id}`

**Backtests** (see `docs/features/backtests.md`):
- `POST /backtests/tune`, `GET /backtests/tunes`, `GET /backtests/tunes/{id}`
- `POST /backtests/wfo`, `GET /backtests/wfo/{id}`, `GET /backtests/wfo`
- `GET /backtests/leaderboard`

**KB Recommend** (see `docs/features/kb-recommend.md`):
- `POST /kb/trials/recommend`, `/kb/trials/ingest`, `/kb/trials/upload-ohlcv`

**Execution** (see `docs/features/execution.md`):
- `POST /execute/intents`, `GET /execute/paper/state/{workspace_id}`

**Dashboards**:
- `GET /dashboards/{workspace_id}/equity` - Equity curve with drawdown
- `GET /dashboards/{workspace_id}/intel-timeline` - Confidence history
- `GET /dashboards/{workspace_id}/alerts` - Active alerts
- `GET /dashboards/{workspace_id}/summary` - Combined overview
- `GET /dashboards/{workspace_id}/backtests/{run_id}` - Run detail (`?include_coaching=true` for coaching data)
- `GET /dashboards/{workspace_id}/backtests/{run_id}/lineage` - Lineage candidates for baseline selector

**Admin** (requires `X-Admin-Token`):
- `GET /admin/system/health`, `/admin/ops/snapshot`
- `GET /admin/ops-alerts`, `POST /admin/ops-alerts/{id}/acknowledge|resolve|reopen`
- `GET /admin/coverage/cockpit`, `PATCH /admin/coverage/weak/{id}`
- `GET /admin/backtests/*`, `/admin/testing/run-plans/*`
- `GET /admin/documents/{doc_id}` - Document detail with key concepts, tickers, validation

## Testing

```bash
pytest tests/unit/ -v                        # Fast, no deps
pytest tests/integration/ -m "not requires_db"  # Needs Qdrant
pytest tests/e2e/ -m e2e -v                  # Needs running server
pytest -m slow -v                            # Slow tests (large data, skipped by default)
```

**Markers**: `@pytest.mark.requires_db`, `@pytest.mark.e2e`, `@pytest.mark.smoke`, `@pytest.mark.slow`

**Auto-skip**: `e2e`, `smoke`, and `slow` markers are auto-skipped by `tests/conftest.py` unless explicitly requested via `-m`.

**CI Jobs**: lint, unit-tests, integration-tests, integration-tests-full (nightly), smoke-test (nightly)

**Mypy Ratchet**: `mypy.ini` has per-module ignores. Baseline in `scripts/check_mypy_ratchet.sh`.

## Security & Ops

See `docs/features/ops.md` for full details.

- `require_admin_token()` - Admin endpoint protection
- `RateLimiter`, `WorkspaceSemaphore` - Rate/concurrency limiting
- Prometheus alerts in `ops/prometheus/rules/`
- Runbooks in `docs/ops/runbooks.md`

## Feature Documentation

Detailed documentation for subsystems:

- `docs/features/backtests.md` - Backtest tuning, WFO, test generator, pre-entry guards, coaching (process score, loss attribution, run lineage)
- `docs/features/orb-engine.md` - ORB v1 engine specification, event contract, v1.1 roadmap
- `docs/features/engine-protocol.md` - Reference engine protocol (interface, events, versioning, golden fixtures, consumer checklist)
- `docs/features/pine-scripts.md` - Pine Script registry, ingest, auto-strategy
- `docs/features/execution.md` - Paper execution, strategy runner
- `docs/features/coverage.md` - Coverage triage workflow
- `docs/features/kb-recommend.md` - KB recommend pipeline, regime fingerprints
- `docs/features/ops.md` - System health, security, v1.0.0 hardening
