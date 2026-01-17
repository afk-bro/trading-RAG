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

## Pine Script Ingest API

Admin-only endpoint for ingesting Pine Script registries into the RAG system.

**Endpoint**: `POST /sources/pine/ingest`

**Authentication**: Requires `X-Admin-Token` header.

**Request**:
```json
{
  "workspace_id": "uuid",
  "registry_path": "/data/pine/pine_registry.json",
  "lint_path": null,
  "source_root": null,
  "include_source": true,
  "max_source_lines": 100,
  "skip_lint_errors": false,
  "update_existing": false,
  "dry_run": false
}
```

**Key Parameters**:
- `registry_path` - Server path to `pine_registry.json` (must be within `DATA_DIR`)
- `lint_path` - Optional lint report path (auto-derived if null)
- `source_root` - Directory with `.pine` files (defaults to registry parent)
- `skip_lint_errors` - Skip scripts with lint errors
- `update_existing` - Upsert if sha256 changed (false = skip changed scripts)
- `dry_run` - Validate without database writes

**Response**:
```json
{
  "status": "success",
  "scripts_processed": 50,
  "scripts_indexed": 45,
  "scripts_already_indexed": 3,
  "scripts_skipped": 2,
  "scripts_failed": 0,
  "chunks_added": 96,
  "errors": [],
  "ingest_run_id": "pine-ingest-abc12345"
}
```

**Status Values**: `success`, `partial`, `failed`, `dry_run`

**Security**:
- All paths validated against `DATA_DIR` allowlist
- Path traversal attempts return 403
- Non-.json extensions rejected with 400

## Pine Script Read APIs

Admin-only endpoints for querying indexed Pine scripts.

**List Endpoint**: `GET /sources/pine/scripts`

```
GET /sources/pine/scripts?workspace_id=<uuid>&symbol=BTC&q=breakout&limit=20
```

**Query Parameters**:
- `workspace_id` (required) - Target workspace UUID
- `symbol` - Filter by ticker symbol (uses GIN index on chunks)
- `status` - Filter by status: `active` (default), `superseded`, `deleted`, `all`
- `q` - Free-text search on title
- `order_by` - Sort field: `updated_at` (default), `created_at`, `title`
- `order_dir` - Sort direction: `desc` (default), `asc`
- `limit` - Results per page (1-100, default 20)
- `offset` - Pagination offset (default 0)

**List Response**:
```json
{
  "items": [
    {
      "id": "uuid",
      "canonical_url": "pine://local/strategies/breakout.pine",
      "rel_path": "strategies/breakout.pine",
      "title": "52W Breakout Strategy",
      "script_type": "strategy",
      "pine_version": "5",
      "symbols": ["BTC", "ETH"],
      "lint_summary": {"errors": 0, "warnings": 1, "info": 2},
      "lint_available": true,
      "sha256": "abc123...",
      "chunk_count": 3,
      "created_at": "2025-01-15T...",
      "updated_at": "2025-01-15T...",
      "status": "active"
    }
  ],
  "total": 45,
  "limit": 20,
  "offset": 0,
  "has_more": true,
  "next_offset": 20
}
```

**Detail Endpoint**: `GET /sources/pine/scripts/{doc_id}`

```
GET /sources/pine/scripts/<uuid>?workspace_id=<uuid>&include_chunks=true&include_lint_findings=true
```

**Query Parameters**:
- `workspace_id` (required) - Target workspace UUID
- `include_chunks` - Include chunk content (default false)
- `chunk_limit` - Chunks per page (1-200, default 50)
- `chunk_offset` - Chunk pagination offset (default 0)
- `include_lint_findings` - Include lint findings array (default false, capped at 200)

**Detail Response** includes all list fields plus:
- `inputs` - Input parameter definitions
- `imports` - Library imports
- `features` - Feature flags (uses_alerts, is_repainting, etc.)
- `chunks` - Optional chunk array with pagination
- `lint_findings` - Optional lint findings array

**Authentication**: Both endpoints require `X-Admin-Token` header.

## Auto-Strategy Discovery

Automatic parameter spec generation from Pine Script inputs for backtesting automation (`app/services/pine/spec_generator.py`).

**Pipeline**:
```
Pine Script → Parser → PineInput[] → SpecGenerator → StrategySpec
```

**Key Components**:
- `ParamSpec` - Parameter specification with bounds, step, options, sweepable flag, priority
- `StrategySpec` - Complete strategy spec with params list and auto-generated sweep config
- `pine_input_to_param_spec()` - Converts parsed Pine inputs to ParamSpec
- `generate_strategy_spec()` - Generates full StrategySpec from PineScriptEntry

**Sweepable Detection**:
- Bool inputs: Always sweepable (true/false)
- Int/Float with bounds: Sweepable if `min_value` and `max_value` defined
- Options array: Sweepable if `options` length > 1
- Source/color/session: Generally not sweepable

**Priority Scoring** (higher = more likely to affect strategy):
- Base priority by type: int/float = 10, bool = 5
- Keywords boost: `length`, `period`, `threshold`, `atr`, `rsi` = +10
- Keywords penalty: `color`, `style`, `display`, `show` = -10
- Bounds present: +15 (indicates optimization intent)

**Usage**:
```python
from app.services.pine.spec_generator import generate_strategy_spec

spec = generate_strategy_spec(pine_entry)
sweepable = spec.sweepable_params  # Only params suitable for optimization
sweep_config = spec.sweep_config   # Auto-generated grid for tuning
```

## API Endpoints

**Health & Readiness**:
- `GET /health` - Liveness probe (always 200), dependency health
- `GET /ready` - Readiness probe (503 if deps unhealthy), checks DB/Qdrant/embed
- `GET /metrics` - Prometheus metrics

**RAG Core**:
- `POST /ingest` - Generic document ingestion
- `POST /sources/youtube/ingest` - YouTube transcript ingestion
- `POST /sources/pdf/ingest` - PDF file upload ingestion (multipart form)
- `POST /sources/pine/ingest` - Pine Script registry ingestion (admin-only)
- `GET /sources/pine/scripts` - List indexed Pine scripts (admin-only)
- `GET /sources/pine/scripts/{doc_id}` - Pine script details (admin-only)
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
- `GET /admin/system/health` - System health dashboard (HTML)
- `GET /admin/system/health.json` - System health status (JSON)
- `GET /admin/kb/*` - KB inspection and curation endpoints
- `GET /admin/backtests/*` - Backtest admin UI
- `GET /admin/coverage/weak` - List weak coverage runs for triage
- `PATCH /admin/coverage/weak/{run_id}` - Update coverage status (open/acknowledged/resolved)
- `POST /admin/coverage/explain` - LLM-powered strategy match explanation
- `GET /admin/coverage/cockpit` - Coverage triage cockpit UI
- `GET /admin/coverage/cockpit/{run_id}` - Deep link to specific coverage run

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

## Coverage Triage Workflow

Admin endpoints for managing coverage gaps in the cockpit UI.

**Architecture** (`app/admin/coverage.py`, `app/services/coverage_gap/repository.py`):
```
Match Run (weak_coverage=true)
       │
       ▼
┌─────────────────┐
│ Coverage Status │ ──► open → acknowledged → resolved
└─────────────────┘
       │
       ▼
   Priority Score (deterministic ranking)
```

**Status Lifecycle**:
- `open` - New coverage gap, needs attention (default)
- `acknowledged` - Someone is investigating
- `resolved` - Gap addressed (strategy added, false positive, etc.)

**Priority Score Formula** (higher = more urgent):
| Component | Value | Condition |
|-----------|-------|-----------|
| Base | `0.5 - best_score` | Clamped to [0, 0.5] |
| No results | +0.2 | `num_above_threshold == 0` |
| NO_MATCHES | +0.15 | Reason code present |
| NO_STRONG_MATCHES | +0.1 | Reason code present |
| Recency | +0.05 | Created in last 24h |

**Key Endpoints**:
- `GET /admin/coverage/weak?workspace_id=...&status=open` - List weak coverage runs
  - `status`: `open` (default), `acknowledged`, `resolved`, `all`
  - `include_candidate_cards=true` (default) - Hydrate strategy cards
  - Results sorted by `priority_score` descending
- `PATCH /admin/coverage/weak/{run_id}` - Update status
  - Body: `{"status": "acknowledged|resolved", "note": "optional resolution note"}`
  - Tracks `acknowledged_at/by`, `resolved_at/by`, `resolution_note`

**Response Fields**:
- `coverage_status` - Current triage status
- `priority_score` - Computed ranking score (0.0 to ~1.0)
- `strategy_cards_by_id` - Hydrated candidate strategy cards
- `missing_strategy_ids` - IDs of deleted/archived strategies

**Resolution Guard**:
Cannot mark a run as `resolved` without at least one of:
- `candidate_strategy_ids` present (strategies were recommended)
- `resolution_note` provided (explains why resolved)

Returns 400 if guard fails. Prevents silent "resolved but nothing changed" states.

**Auto-Resolve on Success**:
When `/youtube/match-pine` produces `weak_coverage=false`:
1. Find all `open`/`acknowledged` runs with same `intent_signature`
2. Auto-resolve them with `resolved_by='system'`
3. Set `resolution_note='Auto-resolved by successful match'`

This closes coverage gaps automatically when a matching strategy is added.

**LLM-Powered Strategy Explanation**:
- `POST /admin/coverage/explain` - Generate explanation of why a strategy matches an intent
  - Request: `{run_id, strategy_id}` + `workspace_id` query param
  - Response: `{explanation, model, provider, latency_ms}`
- Builds prompts from:
  - Intent: archetypes, indicators, timeframes, symbols, risk terms
  - Strategy: name, description, tags, backtest summary
  - Overlap: matched tags and similarity score
- Generates 2-4 sentence practical explanation
- Requires LLM configuration (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`)
- Returns 503 if LLM not configured, 404 if run/strategy not found

**Cockpit UI** (`/admin/coverage/cockpit`):
- Two-panel layout: queue (left) + detail (right)
- Status tabs: Open, Acknowledged, Resolved, All
- Priority badges: P1 (>=0.75), P2 (>=0.40), P3 (<0.40)
- Strategy cards with tags, backtest status, OOS score
- "Explain Match" button generates LLM explanation per candidate
- Deep link support: `/admin/coverage/cockpit/{run_id}`
- Triage controls: Acknowledge, Resolve, Reopen with optional notes

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

## Regime Fingerprints

Materialized regime fingerprints for instant similarity queries (Migration 056).

**Problem**: Regime similarity queries were recomputing vectors on every request.

**Solution**: Precompute and store regime hashes + vectors at tune time for O(1) lookup.

**Schema** (`regime_fingerprints` table):
- `fingerprint` (BYTEA) - 32-byte SHA256 hash for exact matching
- `regime_vector` (FLOAT8[]) - 6-dim vector: [atr_norm, rsi, bb_width, efficiency, trend_strength, zscore]
- `trend_tag`, `vol_tag`, `efficiency_tag` - Denormalized tags for SQL filtering
- `regime_schema_version` - Schema version for compatibility (default: `regime_v1_1`)

**SQL Functions**:
- `compute_regime_fingerprint(FLOAT8[])` - Compute SHA256 from vector (rounds to 4 decimals)
- `regime_distance(FLOAT8[], FLOAT8[])` - Euclidean distance between vectors

**Indexes**:
- Hash index on `fingerprint` for O(1) exact matching
- B-tree on `tune_id` for tune-based lookups
- Composite index on tags for SQL filtering
- GIN on `regime_vector` for array operators

**Usage**:
```sql
-- Find all trials with exact same regime
SELECT * FROM regime_fingerprints
WHERE fingerprint = compute_regime_fingerprint(ARRAY[0.014, 45.2, 0.023, 0.65, 0.78, -0.52]);

-- Find similar regimes by distance
SELECT *, regime_distance(regime_vector, ARRAY[...]) as dist
FROM regime_fingerprints
ORDER BY dist LIMIT 10;
```

## System Health Dashboard

Single-page operational health view for "what's broken?" diagnostics (`app/admin/system_health.py`).

**Endpoints**:
- `GET /admin/system/health` - HTML dashboard with status cards
- `GET /admin/system/health.json` - Machine-readable JSON

**Component Health Checks**:
- **Database**: Pool size, available connections, acquire latency P95, connection errors (5m)
- **Qdrant**: Vector count, segment count, collection status, last error
- **LLM**: Provider configured, degraded count (1h), error count (1h), last success/error
- **SSE**: Subscriber count, events published (1h), queue drops (1h)
- **Ingestion**: Last success/failure per source type (YouTube, PDF, Pine), pending jobs

**Status Values**: `ok`, `degraded`, `error`, `unknown`

**Design Decisions**:
- No external dependencies (queries internal state only)
- Decision-grade metrics (actionable, not just informational)
- Sub-second response time (cached where appropriate)
- Answers "should I wake someone up?" without log access

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

## v1.0.0 - Operational Hardening (Jan 2026)

Tagged release implementing four phases of operational safety for production readiness.

### Phase 1: Idempotency
- `X-Idempotency-Key` header for `/backtests/tune` endpoint
- Prevents duplicate tunes on client retries (corrupts leaderboards)
- Atomic claim pattern with `INSERT ON CONFLICT DO NOTHING`
- Pending request polling with configurable timeout

**Migration required**: `053_idempotency_keys.sql`

### Phase 2: Retention
- SQL functions for batch deletes: `retention_prune_trade_events()`, `retention_prune_job_runs()`, `retention_prune_match_runs()`
- pg_cron scheduling at 3:15/3:20/3:25 AM (when available)
- Admin endpoints with `dry_run` support for preview
- Retention job logging in `retention_job_log` table

**Migration required**: `054_retention_functions.sql`, `055_retention_pgcron_schedule.sql`

### Phase 3: LLM Fallback
- Graceful degradation when LLM times out or errors
- `StrategyExplanation` response includes `degraded`, `reason_code`, `model`, `provider`
- Reason codes: `llm_timeout`, `llm_unconfigured`, `llm_error`, `llm_rate_limit`
- Full stack traces logged internally, user-safe messages returned

### Phase 4: SSE (Server-Sent Events)
- Real-time updates for admin coverage cockpit at `/admin/events/stream`
- `InMemoryEventBus` with abstract `EventBus` interface for future Redis/PgNotify
- Topic-based filtering: `coverage`, `backtests`
- Workspace-scoped event delivery
- `Last-Event-ID` reconnection support with event buffer

### Canary Tests
Located in `tests/integration/test_operational_hardening.py`:
- `TestIdempotencyConcurrentRetry` - Same key returns same tune_id
- `TestRetentionDryRun` - Dry run returns count without deletion
- `TestLLMTimeoutFallback` - Timeout triggers fallback response
- `TestSSEEventDelivery` - Event publish/subscribe verification

**Note**: DB tests gracefully skip if migrations not applied.

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
