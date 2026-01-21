# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Git & Communication Guidelines

**Do not reference Claude or AI in any git artifacts.** This includes commit messages, PR titles/descriptions, code comments, and branch names. Write as if a human authored them.

## Overview

**rag-core** is a multi-tenant RAG system. FastAPI service that ingests documents (YouTube, PDFs, Pine Scripts), chunks with token-awareness, generates embeddings via Ollama, stores vectors in Qdrant, and provides semantic search with LLM answer generation.

Workspace-scoped: each workspace has its own configuration stored in `workspaces` table.

## Architecture

```
Sources (YouTube, PDF, Pine) → FastAPI → Ollama (Embed) + Qdrant (Vector) + Supabase (Postgres)
```

**Data Flow**: Content → Extract → Chunk (512 tokens) → Embed (768-dim) → Vector + Postgres

**Layer Pattern**:
- `app/routers/` - API endpoints (thin controllers)
- `app/services/` - Business logic
- `app/repositories/` - Data access

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
pytest tests/unit/                           # Unit only
pytest tests/unit/test_pdf_extractor.py -v  # Single file

# Linting
black --check app/ tests/
flake8 app/ tests/ --max-line-length=100
mypy app/ --ignore-missing-imports
```

## Configuration

**Required** (`.env`):
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `DATABASE_URL`

**Optional**:
- `OPENROUTER_API_KEY` - LLM API (enables `mode=answer`)
- `ADMIN_TOKEN` - Admin endpoint protection

**DB Notes**: Use Supabase transaction pooler (port 6543). asyncpg uses `statement_cache_size=0` for pgbouncer.

## Key Architectural Details

- **Async-First**: All I/O uses async. Do not introduce blocking calls.
- **Token-Aware Chunking**: tiktoken for accurate counting, ~512 tokens with overlap
- **PDF Extraction**: Swappable backend (PyMuPDF, pdfplumber)
- **Idempotency**: Content hashes, unique constraint on `(workspace_id, source_type, canonical_url)`
- **Model Migration**: `chunk_vectors` tracks embeddings per model/collection

## Database Schema

PostgreSQL tables (via Supabase):

- **workspaces** - Control plane (routing defaults, config jsonb)
- **documents** - Source metadata, content hash, status lifecycle
- **chunks** - Text segments with token counts, metadata arrays
- **chunk_vectors** - Maps chunks to embedding model/collection

All tables FK to workspaces. Migrations in `migrations/`.

## API Endpoints

**Health**: `GET /health`, `GET /ready`, `GET /metrics`

**RAG Core**:
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

**Admin** (requires `X-Admin-Token`):
- `GET /admin/system/health`, `/admin/ops/snapshot`
- `GET /admin/coverage/cockpit`, `PATCH /admin/coverage/weak/{id}`
- `GET /admin/backtests/*`, `/admin/testing/run-plans/*`

## Testing

```bash
pytest tests/unit/ -v                        # Fast, no deps
pytest tests/integration/ -m "not requires_db"  # Needs Qdrant
pytest tests/e2e/ -m e2e -v                  # Needs running server
```

**Markers**: `@pytest.mark.requires_db`, `@pytest.mark.e2e`, `@pytest.mark.smoke`, `@pytest.mark.slow`

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

| Feature | Doc |
|---------|-----|
| Backtest Tuning, WFO, Test Generator | `docs/features/backtests.md` |
| Pine Script Registry, Ingest, Auto-Strategy | `docs/features/pine-scripts.md` |
| Paper Execution, Strategy Runner | `docs/features/execution.md` |
| Coverage Triage, Cockpit UI | `docs/features/coverage.md` |
| KB Recommend, Regime Fingerprints | `docs/features/kb-recommend.md` |
| System Health, Security, v1.0.0 Hardening | `docs/features/ops.md` |

## Roadmap

Evolving from trading-focused RAG to general-purpose rag-core:
- Trading is the first workspace
- PDF backend swappable (MinerU/StudyG planned)
- Workspace config enables per-tenant customization
