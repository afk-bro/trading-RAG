# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**rag-core** (née Trading RAG Pipeline) is a multi-tenant Retrieval-Augmented Generation system. FastAPI service that ingests documents (YouTube transcripts, PDFs, articles), chunks them with token-awareness, generates embeddings via Ollama, stores vectors in Qdrant, and provides semantic search with LLM-powered answer generation.

The system is workspace-scoped: each workspace has its own configuration (embedding model, chunking rules, retrieval settings) stored in a control-plane table.

## Architecture

```
Sources (YouTube, PDF, Articles)
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
- `app/services/` - Business logic (chunker, embedder, pdf_extractor, llm)
- `app/repositories/` - Data access (documents, chunks, vectors)

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
pytest tests/unit/test_pdf_extractor.py -v  # Single file
pytest tests/integration/ -m "not requires_db"  # Integration (mocked DB)

# Linting (CI runs these)
black --check app/ tests/                   # Format check
flake8 app/ tests/ --max-line-length=100    # Style
mypy app/ --ignore-missing-imports          # Types
```

## Configuration

Required environment variables (set in `.env`):
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Database credentials
- `DATABASE_URL` - PostgreSQL connection string (get from Supabase Dashboard)

Optional environment variables:
- `OPENROUTER_API_KEY` - LLM API access (enables `mode=answer` queries)

Services default to Docker network hostnames: `qdrant:6333`, `ollama:11434`

**Database Connection Notes:**
- Use Supabase transaction pooler (port 6543) for most deployments
- asyncpg configured with `statement_cache_size=0` for pgbouncer compatibility
- Direct connections (`db.[project].supabase.co`) may not be available on all plans

**Query Mode Architecture:**
```
retrieve → always available (semantic search, no LLM)
answer   → optional enhancement (LLM synthesis with graceful degradation)
```

Without `OPENROUTER_API_KEY`, `mode=answer` queries return retrieved chunks with a message indicating generation is disabled. This design avoids vendor lock-in and allows the service to start without external LLM dependencies.

## Key Architectural Details

**Async-First**: All I/O uses async (asyncpg for Postgres, httpx for HTTP, Qdrant async client). Do not introduce blocking calls.

**Token-Aware Chunking** (`app/services/chunker.py`): Uses tiktoken for accurate token counting. Chunks target ~512 tokens with overlap. Supports page-aware chunking for PDFs.

**PDF Extraction** (`app/services/pdf_extractor.py`): Swappable backend interface supporting PyMuPDF and pdfplumber. Designed for future swap to MinerU or external services without pipeline changes.

```python
# Interface contract
extract_pdf(file_bytes, config) -> PDFExtractionResult
# config: backend, max_pages, min_chars_per_page, join_pages_with, enable_ocr
```

**Idempotency**: Ingestion uses content hashes and idempotency keys. Documents have unique constraint on `(workspace_id, source_type, canonical_url)`.

**Model Migration**: `chunk_vectors` table tracks embeddings per model/collection. Re-embed endpoint supports switching embedding models without data loss.

**Workspace Control Plane**: Each workspace has routing defaults (collection, embed model, distance metric) and flexible JSON config for chunking/retrieval/pdf settings.

## Database Schema

PostgreSQL tables (via Supabase):

**workspaces** - Control plane for RAG pipelines
- `id`, `name`, `slug` (unique), `owner_id`
- Routing: `default_collection`, `default_embed_provider`, `default_embed_model`, `default_distance`
- Flags: `is_active`, `ingestion_enabled`
- `config` (jsonb) - chunking, retrieval, pdf, metadata rules

**documents** - Source metadata, content hash, status lifecycle

**chunks** - Text segments with token counts, timestamps/pages, metadata arrays (symbols, entities, topics)

**chunk_vectors** - Maps chunks to embedding model/collection

All tables have FK to workspaces for multi-tenant isolation.

Migrations in `migrations/` - applied via Supabase MCP or dashboard.

## Backtest Parameter Tuning

Research workflow for strategy optimization via parameter sweeps.

**Core Concepts**:
- `tune` - A parameter sweep session (grid/random search over param space)
- `tune_run` - A single trial within a tune (one param combination)
- IS/OOS split - In-Sample (training) and Out-of-Sample (validation) data split

**Objective Functions** (`app/services/backtest/tuner.py`):
- `sharpe` - Sharpe ratio
- `sharpe_dd_penalty` - `sharpe - λ × max_drawdown_pct`
- `return` - Return percentage
- `return_dd_penalty` - `return_pct - λ × max_drawdown_pct`
- `calmar` - `return_pct / abs(max_drawdown_pct)`

**Gates Policy**: Trials must pass gates (max drawdown ≤20%, min trades ≥5) to be considered valid. Gates are evaluated on OOS metrics when split is enabled.

**Overfit Detection**: `overfit_gap = score_is - score_oos`. Gap >0.3 indicates moderate overfit risk, >0.5 is high risk.

**Admin UI** (`app/admin/router.py`, `app/admin/templates/`):
- `/admin/backtests/tunes` - Filterable tune list with validity badges
- `/admin/backtests/tunes/{id}` - Tune detail with trial list
- `/admin/backtests/leaderboard` - Global ranking by objective score (CSV export)
- `/admin/backtests/compare?tune_id=A&tune_id=B` - N-way diff table (JSON export)

## API Endpoints

**RAG Core**:
- `GET /health` - Dependency health (Qdrant, Ollama, Supabase)
- `POST /ingest` - Generic document ingestion
- `POST /sources/youtube/ingest` - YouTube transcript ingestion
- `POST /sources/pdf/ingest` - PDF file upload ingestion (multipart form)
- `POST /query` - Semantic search (modes: `retrieve` or `answer`)
- `POST /reembed` - Migrate to new embedding model
- `GET /jobs/{job_id}` - Async job status
- `GET /metrics` - Prometheus metrics

**Backtest Tuning**:
- `POST /backtests/tune` - Start parameter sweep
- `GET /backtests/tunes` - List tunes with filters (valid_only, objective_type, oos_enabled)
- `GET /backtests/tunes/{id}` - Tune detail with trial list
- `POST /backtests/tunes/{id}/cancel` - Cancel running tune
- `GET /backtests/leaderboard` - Global ranking with best run metrics

## Testing Notes

- Unit tests mock external services (21 tests for PDF extractor alone)
- Integration tests use `@pytest.mark.requires_db` for tests needing real database
- CI runs Qdrant service container for vector DB tests

## Roadmap Context

This system is evolving from a trading-focused RAG to a general-purpose rag-core:
- Trading is the first workspace (finance knowledge base)
- PDF backend is swappable (pymupdf now, MinerU/StudyG later for OCR/tables)
- Workspace config enables per-tenant customization without code changes
