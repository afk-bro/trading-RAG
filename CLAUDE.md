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

## Pine Script Registry

Parsing and linting system for Pine Script files (`app/services/pine/`).

**Purpose**: Catalog Pine Script files with metadata extraction and static analysis for downstream RAG ingestion.

**Architecture**:
```
.pine files → Filesystem Adapter → Parser → Linter → Registry Builder → JSON artifacts
```

**Components**:
- `models.py` - Data models: `PineRegistry`, `PineScriptEntry`, `PineLintReport`, `LintFinding`
- `parser.py` - Regex-based parser extracts version, declaration, inputs, imports, features
- `linter.py` - Static analysis rules (E001-E003 errors, W002-W003 warnings, I001-I002 info)
- `registry.py` - Build orchestration + CLI entry point
- `adapters/filesystem.py` - File scanning returning `SourceFile` structured output

**CLI Usage**:
```bash
python -m app.services.pine --build ./scripts           # Build from directory
python -m app.services.pine --build ./scripts -o ./data # Custom output dir
python -m app.services.pine --build ./scripts -q        # Quiet mode
```

**Output Artifacts**:
- `pine_registry.json` - Script metadata (version, type, title, inputs, imports, features) with lint summaries
- `pine_lint_report.json` - Full lint findings per script

**Design Choices**:
- **Best-effort**: Parse errors recorded as E999 synthetic errors, build continues
- **Deterministic**: Sorted keys, consistent JSON formatting for diff stability
- **Fingerprinted**: SHA256 from raw content for change detection
- **GitHub-ready**: `root_kind` field distinguishes filesystem vs future GitHub adapter

**Lint Rules**:
| Code | Severity | Description |
|------|----------|-------------|
| E001 | Error | Missing `//@version` directive |
| E002 | Error | Invalid version number |
| E003 | Error | Missing declaration (`indicator`/`strategy`/`library`) |
| W002 | Warning | `lookahead=barmerge.lookahead_on` (future data leakage risk) |
| W003 | Warning | Deprecated `security()` instead of `request.security()` |
| I001 | Info | Script has exports but is not a library |
| I002 | Info | Script exceeds recommended line count (500) |

## API Endpoints

**Health & Readiness**:
- `GET /health` - Liveness probe (always 200), dependency health
- `GET /ready` - Readiness probe (503 if deps unhealthy), checks DB/Qdrant/embed
- `GET /metrics` - Prometheus metrics

**RAG Core**:
- `POST /ingest` - Generic document ingestion
- `POST /sources/youtube/ingest` - YouTube transcript ingestion
- `POST /sources/pdf/ingest` - PDF file upload ingestion (multipart form)
- `POST /query` - Semantic search (modes: `retrieve` or `answer`)
- `POST /reembed` - Migrate to new embedding model
- `GET /jobs/{job_id}` - Async job status

**Trading KB Recommend** (`/kb/trials/*`):
- `POST /kb/trials/recommend` - Get parameter recommendations for strategy
- `POST /kb/trials/recommend?mode=debug` - Debug mode with full candidates
- `POST /kb/trials/ingest` - Ingest trials from tune runs (admin-only)
- `POST /kb/trials/upload-ohlcv` - Upload OHLCV data for regime analysis

**Backtest Tuning**:
- `POST /backtests/tune` - Start parameter sweep
- `GET /backtests/tunes` - List tunes with filters (valid_only, objective_type, oos_enabled)
- `GET /backtests/tunes/{id}` - Tune detail with trial list
- `POST /backtests/tunes/{id}/cancel` - Cancel running tune
- `GET /backtests/leaderboard` - Global ranking with best run metrics

**Admin** (requires `X-Admin-Token` header):
- `GET /admin/ops/snapshot` - Operational health snapshot for go-live
- `GET /admin/kb/*` - KB inspection and curation endpoints
- `GET /admin/backtests/*` - Backtest admin UI

**Execution** (requires `X-Admin-Token` header):
- `POST /execute/intents` - Execute trade intent (paper mode only)
- `GET /execute/paper/state/{workspace_id}` - Get paper trading state
- `GET /execute/paper/positions/{workspace_id}` - Get open positions
- `POST /execute/paper/reconcile/{workspace_id}` - Rebuild state from journal
- `POST /execute/paper/reset/{workspace_id}` - Reset state (dev only)

## Paper Execution Adapter

Provider-agnostic broker adapter that simulates trade execution for end-to-end automation testing without exchange randomness.

**Architecture** (`app/services/execution/`):
```
TradeIntent (approved by PolicyEngine)
       │
       ▼
  ┌─────────────┐
  │ PaperBroker │ ─────► trade_events journal (ORDER_FILLED)
  └──────┬──────┘
         │
         ▼
    PaperState (in-memory, reconcilable from journal)
```

**Key Design Decisions**:
- **State persistence**: In-memory + journal rebuild (event sourcing pattern)
- **Fill simulation**: Immediate fill, caller provides `fill_price` (required)
- **Order types**: MARKET only (limit orders planned for later)
- **Position support**: Long-only, single position per symbol
- **Reconciliation**: Manual endpoint only (no auto on startup)
- **Idempotency**: Key = `(workspace_id, intent_id, mode)`, returns 409 on duplicate
- **Supported actions**: `OPEN_LONG`, `CLOSE_LONG` only (400 for others)
- **Full close only**: SELL qty must == position.qty (no partial closes)
- **Policy re-check**: Execution re-evaluates policy internally (doesn't trust caller)

**Event Flow**:
```
INTENT_EMITTED → POLICY_EVALUATED → INTENT_APPROVED
                                          │
                                          ▼
                                    ORDER_FILLED (source of truth)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                  POSITION_OPENED  POSITION_SCALED  POSITION_CLOSED
                        (observability breadcrumbs only)
```

**Cash Ledger** (`PaperState`):
- `starting_equity` - Initial capital (default 10000.0)
- `cash` - Current available cash
- `realized_pnl` - Cumulative realized profit/loss
- `positions` - Dict of symbol → `PaperPosition`

**Position Scaling Formula**:
```python
# BUY more of existing position
new_qty = old_qty + add_qty
new_avg = (old_avg * old_qty + fill_price * add_qty) / new_qty
```

**Reconciliation**:
1. Clear in-memory state (cash = starting_equity, positions = {})
2. Query `trade_events` for `ORDER_FILLED` events only
3. Dedupe by `order_id` (skip duplicates)
4. Replay fills to rebuild cash/positions
5. `POSITION_*` events are NOT replayed (observability only)

**Execute Request**:
```python
POST /execute/intents
{
    "intent": TradeIntent,   # From policy engine
    "fill_price": 50000.0,   # REQUIRED - caller provides
    "mode": "paper"          # Only paper supported
}
```

Returns 409 Conflict if intent already executed.

## Strategy Runner

The strategy runner generates TradeIntents from strategy configuration, market data, and current portfolio state. It bridges the gap between backtest research and live execution.

**Architecture** (`app/services/strategy/`):
```
ExecutionSpec (strategy instance with params)
      +
MarketSnapshot (OHLCV window)
      +
PaperState (current positions/cash)
      │
      ▼
┌─────────────────┐
│ StrategyRunner  │ ─────► list[TradeIntent]
└─────────────────┘
      │
      ▼
  PolicyEngine → PaperBroker → Journal
```

**Key Models**:
- `ExecutionSpec` - Runtime configuration for a strategy instance (strategy name, params, symbol, workspace)
- `MarketSnapshot` - Point-in-time market state with OHLCV bars (caller-provided for determinism)
- `StrategyEvaluation` - Runner output containing intents, signals, and evaluation metadata

**Key Design Decisions**:
- **Separate ExecutionSpec from StrategyRegistry**: Runtime config vs param schema definition
- **MarketSnapshot is caller-provided**: Enables deterministic testing, no internal data fetching
- **Stateless evaluation**: No internal runner state; all context passed in per call
- **Max positions only blocks entries, never exits**: Safety rule to prevent over-allocation
- **Exclude current bar from 52w high computation**: Avoid look-ahead bias
- **evaluation_id shared by all intents**: Enables end-to-end tracing from signal to fill

**Built-in Strategies**:
- `breakout_52w_high` - Entry when price exceeds 52-week high, EOD exit

**Usage Example**:
```python
from app.services.strategy import StrategyRunner, ExecutionSpec, MarketSnapshot

runner = StrategyRunner()
result = runner.evaluate(spec, snapshot, paper_state)
for intent in result.intents:
    # Execute via PaperBroker
    pass
```

## Test Generator & Run Orchestrator

Parameter sweep framework for systematic strategy testing. Takes an ExecutionSpec and generates variants automatically, then runs them through StrategyRunner + PaperBroker.

**Architecture** (`app/services/testing/`):
```
ExecutionSpec (base configuration)
        │
        ▼
┌──────────────────┐
│  Test Generator  │ ──► RunPlan with N variants
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Run Orchestrator │ ──► Execute variants, collect metrics
└──────────────────┘
        │
        ├────────────┬────────────┐
        ▼            ▼            ▼
  StrategyRunner  PaperBroker  trade_events
        │
        ▼
  RunResult (per-variant metrics)
```

**Key Design Decisions**:
- **Variant ID**: `sha256(canonical_json({base, overrides}))[:16]` - deterministic, stable across runs
- **Overrides format**: Flat dotted-path dict only (e.g., `{"entry.lookback_days": 200}`)
- **Broker isolation**: Each variant uses `uuid5(VARIANT_NS, f"{run_plan_id}:{variant_id}")` as workspace - prevents cross-contamination
- **Equity tracking**: Trade-equity points (step function) - equity at each closed trade for drawdown/sharpe
- **Persistence**: In-memory RunPlan for v0, events (RUN_STARTED, RUN_COMPLETED) in trade_events journal

**Sweepable Parameters**:
- `entry.lookback_days` - Lookback for 52w high calculation
- `risk.dollars_per_trade` - Position sizing
- `risk.max_positions` - Max concurrent positions

**Generator Output**:
1. **Baseline** variant (empty overrides) - always first
2. **Grid sweep** variants (cartesian product of sweep values)
3. **Ablation** variants (reset one param to default, relative to first grid combo)
4. Deduplication + max_variants limit

**Metrics Calculated**:
- `return_pct` - (ending_equity / starting_equity - 1) × 100
- `max_drawdown_pct` - Peak-to-trough from trade-equity curve
- `sharpe` - mean(trade_returns) / std(trade_returns), None if <2 trades
- `win_rate`, `trade_count`, `profit_factor`

**API Endpoints**:
- `POST /testing/run-plans/generate` - Generate RunPlan (no execution)
- `POST /testing/run-plans/generate-and-execute` - Generate + execute + return results (multipart form with CSV)

**Admin UI**:
- `/admin/testing/run-plans` - List page (event-driven summaries)
- `/admin/testing/run-plans/{id}` - Detail page

## Trading KB Recommend Pipeline

The `/kb/trials/recommend` endpoint provides strategy parameter recommendations based on historical backtest results.

**Response Status**:
- `ok` - High confidence recommendations available
- `degraded` - Recommendations available with caveats (used relaxed filters, low count)
- `none` - No suitable recommendations found

**Key Features**:
- Strategy-specific quality floors (sharpe ≥0.3, return ≥5%, calmar ≥0.5)
- Single-axis relaxation suggestions when `status=none`
- Confidence scoring based on candidate count and score variance
- Regime-aware filtering (volatility, trend, momentum tags)
- Debug mode with full candidate inspection

**Request Example**:
```python
POST /kb/trials/recommend
{
    "workspace_id": "uuid",
    "strategy_name": "bb_reversal",
    "objective_type": "sharpe",
    "require_oos": true,
    "max_drawdown": 0.20,
    "min_trades": 5
}
```

## Security & Operations

**Security Dependencies** (`app/deps/security.py`):
- `require_admin_token()` - Admin endpoint protection (hmac.compare_digest)
- `require_workspace_access()` - Multi-tenant authorization stub
- `RateLimiter` - Sliding window rate limiting (per-IP, per-workspace)
- `WorkspaceSemaphore` - Per-workspace concurrency caps

**Production Environment Variables**:
```bash
ADMIN_TOKEN=<secure-token>      # Required for /admin/* endpoints
DOCS_ENABLED=false              # Disable /docs in production
CORS_ORIGINS=https://app.com    # Explicit allowlist
CONFIG_PROFILE=production       # Environment tag
GIT_SHA=abc123                  # Set by CI/CD
BUILD_TIME=2025-01-09T12:00:00Z # Set by CI/CD
```

**Monitoring**:
- Sentry tags: `kb_status`, `kb_confidence`, `strategy`, `workspace_id`, `collection`, `embed_model`
- Sentry measurements: `kb.total_ms`, `kb.embed_ms`, `kb.qdrant_ms`, `kb.rerank_ms`
- 4xx errors filtered (not captured as exceptions)

**Operational Docs** (`docs/ops/`):
- `alerting-rules.md` - Sentry alert configuration
- `runbooks.md` - Qdrant rebuild, model rotation, status handling

## Testing Notes

### Test Categories

| Category | Location | Runs in CI | Requirements |
|----------|----------|------------|--------------|
| Unit | `tests/unit/` | Always | None (all mocked) |
| Contract | `tests/contract/` | Always | None (all mocked) |
| Golden | `tests/golden/` | Always | None (deterministic snapshots) |
| Integration (mocked) | `tests/integration/` | Always | Qdrant container |
| Integration (full) | `tests/integration/` | Nightly/manual | Qdrant + real services |
| Smoke | `tests/smoke/` | Nightly/manual | Full running service |
| E2E | `tests/e2e/` | Nightly/manual | Server + Playwright browser |

### Test Markers

```python
@pytest.mark.requires_db      # Needs real DB/services - skipped in normal CI
@pytest.mark.integration      # Integration test (informational)
@pytest.mark.e2e              # E2E browser test - auto-skipped unless explicit
@pytest.mark.smoke            # Smoke test - auto-skipped unless explicit
@pytest.mark.slow             # Slow test - can deselect with -m "not slow"
```

### Running Tests Locally

```bash
# All tests (e2e/smoke auto-skipped)
pytest tests/

# Unit tests only (fast, no dependencies)
pytest tests/unit/ -v

# Integration tests without DB (needs Qdrant running)
docker compose -f docker-compose.rag.yml up -d qdrant
pytest tests/integration/ -v -m "not requires_db"

# Full integration tests (needs all services)
pytest tests/integration/ -v

# Single test file
pytest tests/unit/test_chunker.py -v

# E2E tests (requires running server + browser)
# Start server: uvicorn app.main:app --port 8000
pytest tests/e2e/ -m e2e -v

# Smoke tests (requires running server)
SMOKE_TEST_URL=http://localhost:8000 pytest tests/smoke/ -m smoke -v
```

### Required Environment Variables for Tests

```bash
# Minimum for unit tests
export SUPABASE_URL=https://test.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=test-key

# For integration tests
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
```

### CI Structure

- **lint**: Black, flake8, mypy + mypy ratchet (no new ignores)
- **unit-tests**: Fast, no external deps
- **integration-tests**: Mocked services, Qdrant container
- **integration-tests-full**: All tests, nightly/manual only
- **smoke-test**: Full service, nightly/manual only

### Mypy Ratchet

`mypy.ini` has per-module ignores for legacy code. The ratchet script (`scripts/check_mypy_ratchet.sh`) fails CI if ignore count grows beyond baseline (currently 53). To fix types in a module:

1. Remove its entry from `mypy.ini`
2. Fix type errors
3. Update `BASELINE` in ratchet script (decrease is good!)

## Roadmap Context

This system is evolving from a trading-focused RAG to a general-purpose rag-core:
- Trading is the first workspace (finance knowledge base)
- PDF backend is swappable (pymupdf now, MinerU/StudyG later for OCR/tables)
- Workspace config enables per-tenant customization without code changes

## Follow-ups / Tech Debt

### Post-merge follow-ups: Results Persistence

#### 1) Idempotency for run plan creation

**Problem:** Client retries (timeouts, network blips) can create duplicate `run_plans`.

**Preferred approach:** Option A - idempotency key (client-controlled)

**Proposed schema:**
```sql
ALTER TABLE run_plans ADD COLUMN idempotency_key TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_plans_idempotency_key
    ON run_plans(idempotency_key) WHERE idempotency_key IS NOT NULL;
```

**Endpoint behavior:**
- Accept `X-Idempotency-Key` header
- If key exists → return existing `run_plan_id` (200/201 semantics consistent)
- If not → create new plan

#### 2) Retention policy for run-level trade_events

**Problem:** `trade_events` journal grows unbounded.

**Plan:** Retain 30–90 days for `RUN_*` breadcrumbs.

**Cleanup query:**
```sql
DELETE FROM trade_events
WHERE created_at < NOW() - INTERVAL '90 days'
  AND event_type IN ('run_started', 'run_completed', 'run_failed', 'run_cancelled');
```

**Implementation options:**
- pg_cron job (preferred if available)
- Scheduled endpoint invoked by external cron/GHA
