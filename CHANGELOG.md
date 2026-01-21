# Changelog

All notable changes to the Trading RAG Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Ops Alerts Admin UI** - Admin page for viewing and managing operational alerts
  - `GET /admin/ops/alerts` - HTML list page with filtering, pagination, severity badges
  - `POST /admin/ops/alerts/{id}/acknowledge` - Mark alert as acknowledged
  - `POST /admin/ops/alerts/{id}/resolve` - Mark alert as resolved
  - `POST /admin/ops/alerts/{id}/reopen` - Reopen a resolved alert
  - Status workflow: `firing` → `acknowledged` → `resolved`
  - Styled table with severity colors (critical=red, warning=yellow, info=blue)

- **Webhook Sinks for Alerts** - Deliver alerts to external services
  - `SlackWebhookSink` - Format alerts as Slack messages with severity colors
  - `GenericWebhookSink` - POST JSON payload to any HTTP endpoint
  - Retry logic with exponential backoff (3 attempts, 1s/2s/4s delays)
  - Fire-and-forget delivery (non-blocking)
  - Configuration via `SLACK_WEBHOOK_URL`, `ALERT_WEBHOOK_URL`, `ALERT_WEBHOOK_HEADERS`
  - Integrated with `AlertTransitionManager` for automatic delivery on state changes

- **Documentation Refactor** - CLAUDE.md reduced from 1060 to 148 lines (86% reduction)
  - Detailed feature docs extracted to `docs/features/`:
    - `backtests.md` - Backtest tuning, WFO, test generator
    - `pine-scripts.md` - Pine registry, ingest, auto-strategy
    - `execution.md` - Paper execution, strategy runner
    - `coverage.md` - Coverage triage workflow
    - `kb-recommend.md` - KB pipeline, regime fingerprints
    - `ops.md` - System health, security, v1.0.0 hardening
  - CLAUDE.md now contains essential quick-reference with pointers to detailed docs

- **Idempotency Hygiene Monitoring** - Full observability for idempotency key lifecycle
  - Health page card: total keys, expired pending, pending requests, oldest ages
  - Prometheus metrics: `idempotency_keys_total`, `idempotency_expired_pending_total`, etc.
  - Alert rules: ExpiredPending (warn >100), ExpiredCritical (crit >1000), PruneStale (>48h)
  - Thresholds match between health page and Prometheus for consistency
  - Metrics updated during health checks via `set_idempotency_metrics()`

- **System Health Dashboard** - Single-page operational health view (`/admin/system/health`)
  - HTML dashboard with status cards answering "what's broken?" without opening logs
  - JSON endpoint (`/admin/system/health.json`) for machine-readable status
  - Component health checks: Database pool, Qdrant vectors, LLM providers, SSE bus
  - Decision-grade metrics: pool acquire latency, connection errors, degraded counts
  - Ingestion pipeline status: YouTube, PDF, Pine last success/failure times

- **Regime Fingerprint Materialization** - Instant regime similarity queries (Migration 056)
  - `regime_fingerprints` table stores precomputed regime vectors per tune run
  - SHA256 fingerprint for O(1) exact regime matching via hash index
  - Raw 6-dim vectors: [atr_norm, rsi, bb_width, efficiency, trend_strength, zscore]
  - SQL functions: `compute_regime_fingerprint()`, `regime_distance()`
  - Denormalized tags (trend, vol, efficiency) for SQL filtering
  - Eliminates per-request regime computation overhead

- **Auto-Strategy Discovery from Pine Scripts** - Parameter spec generation
  - `spec_generator.py` converts Pine Script inputs to `StrategySpec`
  - Identifies sweepable parameters (has min/max bounds or options)
  - Priority scoring based on keywords (length, period, threshold = high; color, style = low)
  - Generates `sweep_config` for automated parameter sweeps
  - `ParamSpec` dataclass with full metadata (bounds, step, tooltip, group)

- **Prometheus Alerting Rules** - Ready-to-deploy alert configuration
  - Critical alerts: 5xx error rate, service down, DB pool exhausted, Qdrant errors
  - Warning alerts: P95 latency, LLM degraded rate, KB weak coverage, tune failures
  - Info alerts: Low confidence recommendations, high tune duration
  - Runbook URLs linked in annotations
  - Sentry tag/measurement reference for dashboards

- **Operational Hardening Canary Tests** - Integration tests for Q1 2026 safety features
  - Phase 1: Idempotency concurrent retry test (same key = same tune_id)
  - Phase 2: Retention dry-run test (prune without delete)
  - Phase 3: LLM timeout fallback test (graceful degradation)
  - Phase 4: SSE event delivery test (publish/subscribe verification)
  - Marked with `@pytest.mark.integration` and `@pytest.mark.requires_db`

- **LLM-Powered Strategy Explanation** - Generate explanations of why strategies match user intent
  - `POST /admin/coverage/explain` - API endpoint for on-demand explanations
    - Takes `run_id` + `strategy_id` + `verbosity` (short/detailed)
    - Builds prompts from intent (archetypes, indicators, timeframes) + strategy (tags, description)
    - Shows matched tags and similarity score context
    - Returns model, provider, latency, cache_hit, and confidence_qualifier
  - **Explanation caching** - Stored in `match_runs.explanations_cache` JSONB
    - Cache key: `strategy_id:verbosity`
    - Invalidated if strategy.updated_at > cached.strategy_updated_at
    - Avoids redundant LLM calls when toggling verbosity
  - **Verbosity toggle** - "Short / Detailed" button in cockpit UI
    - Short: 2-4 sentences (300 tokens max)
    - Detailed: 2-3 paragraphs (600 tokens max)
  - **Confidence qualifier** - Deterministic line appended to explanations
    - Formula: 40% tag overlap + 40% match score + 20% backtest status
    - High (≥0.7), Medium (≥0.4), Low (<0.4)
    - Shows reasoning: "based on 3 matching tags, validated backtest"
  - "Explain Match" button with toggle behavior, loading state, error retry
  - Cache hit indicator (⚡ cached / 🔄 fresh) in UI
  - Requires LLM configuration (ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY)
  - Returns 503 if LLM not configured, graceful error handling
  - Migration: `052_match_runs_explanations_cache.sql`

- **Pine Script Read APIs** - Admin endpoints for querying indexed Pine scripts
  - `GET /sources/pine/scripts` - List scripts with filtering (symbol, status, free-text)
  - `GET /sources/pine/scripts/{doc_id}` - Script details with chunks and lint findings
  - `GET /sources/pine/scripts/lookup` - Find script by rel_path (filesystem linking)
  - `GET /sources/pine/scripts/match` - Semantic search with ranked results
    - Searches title, path, input names, and chunk content
    - Returns match score (0-1), match reasons, snippets, input preview
    - Filters: `symbol`, `script_type`, `lint_ok`
  - `pine_metadata` JSONB column stores structured metadata (inputs, imports, features, lint)
  - Symbol filtering uses existing GIN index on chunks for efficient queries
  - Pagination with `has_more`/`next_offset` for both scripts and chunks
  - Literal types for constrained values (script_type, pine_version, status)
- **Pine Script Registry Module** - Complete parsing and linting system for Pine Script files
  - Regex-based parser extracts version, declaration, inputs, imports, and feature flags
  - Static analysis linter with rules: E001-E003 (errors), W002-W003 (warnings), I001-I002 (info)
  - Registry builder produces `pine_registry.json` (metadata + lint summaries) and `pine_lint_report.json` (full findings)
  - CLI entry point: `python -m app.services.pine --build ./scripts`
  - Best-effort parsing: failures recorded as E999 synthetic errors, build continues
  - Deterministic output: sorted keys, SHA256 fingerprinting for change detection
  - Designed for future GitHub adapter parity (root_kind field)
- **Pine Script Ingest API** - Admin endpoints for ingesting Pine Script registries into RAG
  - `POST /sources/pine/ingest` - Ingest scripts from registry file
  - `POST /sources/pine/rebuild-and-ingest` - Build registry + ingest in single action
    - Scans `scripts_root` for .pine files, builds registry, ingests to workspace
    - Returns build stats (files_scanned, parse_errors, lint_errors) and ingest stats
    - Ideal for cron/CI automation of Pine knowledge base updates
  - Path validation against `DATA_DIR` allowlist (prevents path traversal attacks)
  - Auto-derives lint report path from registry location
  - `dry_run` mode for validation without database writes
  - Detailed response: `scripts_indexed`, `scripts_already_indexed`, `scripts_skipped`, `scripts_failed`
  - Requires `X-Admin-Token` header for authentication
  - Supports `skip_lint_errors` to filter scripts with lint errors
  - `update_existing` controls upsert behavior for changed scripts (sha256-based)

- **Cockpit Data Plumbing** - Admin endpoints for coverage gap inspection
  - `GET /admin/coverage/weak` - List weak coverage runs shaped for cockpit UI
    - Returns: `run_id`, `created_at`, `intent_signature`, `script_type`
    - Coverage data: `weak_reason_codes[]`, `best_score`, `num_above_threshold`
    - Candidates: `candidate_strategy_ids[]`, `candidate_scores`
    - Display helpers: `query_preview` (120 chars), `source_ref` (youtube:ID or doc:UUID)
  - Candidate persistence at match time (point-in-time snapshots):
    - `candidate_strategy_ids UUID[]` - Strategy IDs with tag overlap
    - `candidate_scores JSONB` - Detailed scores per strategy
  - Partial indexes for efficient weak coverage queries
  - Migration: `050_match_runs_candidates.sql`

- **Cockpit UX Endpoints** - Bulk fetching to avoid N+1 queries
  - `GET /strategies/cards?ids=uuid,uuid,...` - Bulk fetch strategy cards by IDs
    - Returns `{uuid: StrategyCard}` map for efficient lookups
    - Max 100 IDs per request, missing IDs silently omitted
  - `GET /admin/coverage/weak?include_candidate_cards=true` - Hydrate candidates (default: true)
    - Collects all candidate IDs across items in one query
    - Returns `strategy_cards_by_id` alongside items
    - Single call for weak items + all candidate cards
    - Hydration capped at 300 unique IDs to prevent payload bloat
    - Returns `missing_strategy_ids[]` for deleted/archived strategies
  - `StrategyCard` schema (lightweight for UI):
    - `id`, `name`, `slug`, `engine`, `status`, `tags`
    - `backtest_status`, `last_backtest_at`, `best_oos_score`, `max_drawdown`

- **Coverage Triage Workflow** - Actionable coverage gap management
  - `PATCH /admin/coverage/weak/{run_id}` - Update coverage status
    - Status transitions: `open` → `acknowledged` → `resolved`
    - Tracks `acknowledged_at/by`, `resolved_at/by`, `resolution_note`
  - Status filter: `?status=open|acknowledged|resolved|all` (default: open)
  - Priority scoring for triage ranking (higher = more urgent):
    - Base: `(0.5 - best_score)` clamped to [0, 0.5]
    - +0.2 if `num_above_threshold == 0`
    - +0.15 for `NO_MATCHES` reason code
    - +0.1 for `NO_STRONG_MATCHES` reason code
    - +0.05 recency bonus (last 24h)
  - Results sorted by `priority_score` descending (most actionable first)
  - Migration: `051_match_runs_triage.sql`

- **Strategy Registry v1** - Multi-engine strategy catalog with coverage integration
  - Supports engines: `pine`, `python`, `vectorbt`, `backtesting_py`
  - JSONB columns for flexible schema: `source_ref`, `tags`, `backtest_summary`
  - MatchIntent-compatible tags for coverage gap routing
  - API Endpoints (`/strategies`):
    - `GET /` - List strategies with filters (engine, status, review_status, search)
    - `GET /{id}` - Full details with source_ref and backtest_summary
    - `POST /` - Create strategy in draft status (auto-generates slug)
    - `PATCH /{id}` - Update status, tags, backtest_summary
    - `GET /candidates/by-intent` - Find strategies by tag overlap with scoring
  - Coverage Integration:
    - `CoverageResponse.intent_signature` - SHA256 hash for candidate lookup
    - `CoverageResponse.candidate_strategies` - Matching strategies when weak=true
    - YouTube match-pine endpoint surfaces candidates on coverage gaps
  - Database: `strategies` table with GIN index on tags for efficient overlap queries
  - Migration: `049_strategies.sql`

- **Cross-Encoder Reranking with Neighbor Expansion**
  - Optional two-stage retrieval: vector search → cross-encoder rerank
  - BGE-reranker-v2-m3 model (local inference, no external API)
  - Neighbor expansion: fetches adjacent chunks for context continuity
  - **Disabled by default** - enable via `rerank: true` in request or workspace config
  - Safety caps: `retrieve_k ≤ 200`, `top_k ≤ 50`, timeout fallback to vector order
  - New `QueryMeta` fields for observability:
    - `rerank_state`: `disabled`, `ok`, `timeout_fallback`, `error_fallback`
    - `rerank_ms`, `rerank_method`, `rerank_model`, `rerank_timeout`, `rerank_fallback`
  - `debug: true` request flag exposes per-chunk `vector_score`, `rerank_score`, `rerank_rank`
  - Configurable via `Settings`: `warmup_reranker`, `rerank_timeout_s`
  - ML dependencies pinned in `constraints.txt` for reproducibility
- **Backtest Parameter Tuning System** - Complete research workflow for strategy optimization
  - Grid/random search over strategy parameter spaces
  - IS/OOS (In-Sample/Out-of-Sample) split validation with configurable ratio
  - Composite objective functions: `sharpe`, `sharpe_dd_penalty`, `return`, `return_dd_penalty`, `calmar`
  - Gates policy enforcement (max drawdown, min trades) with audit snapshots
  - Overfit diagnostics via IS-OOS score gap analysis
- **Admin UI for Tuning**
  - `/admin/backtests/tunes` - Filterable list with validity badges
  - `/admin/backtests/leaderboard` - Global ranking by objective score
  - `/admin/backtests/compare` - N-way diff table with highlighting
  - CSV export for leaderboard, JSON export for compare
  - "Compare Selected" multi-select from leaderboard
- **Tuning API Endpoints**
  - `POST /backtests/tune` - Run parameter sweep
  - `GET /backtests/tunes` - List with filters (valid_only, objective_type, oos_enabled)
  - `GET /backtests/leaderboard` - Ranked tunes with best run metrics
  - `GET /backtests/tunes/{id}` - Tune detail with trial list
  - `POST /backtests/tunes/{id}/cancel` - Cancel running tune
- **Database Migrations** (012-016)
  - Tune run failed_reason tracking
  - IS/OOS split columns (score_is, score_oos, oos_ratio)
  - Metrics JSONB persistence (metrics_is, metrics_oos)
  - Composite objective support (objective_type, objective_params, objective_score)
  - Gates JSONB for policy snapshots

### Changed
- **BREAKING**: `OPENROUTER_API_KEY` is now optional (was required)
  - Service starts and runs without LLM configuration
  - `mode=retrieve` (semantic search) always works
  - `mode=answer` returns retrieved chunks with helpful message when LLM not configured
  - Enables graceful degradation and avoids vendor lock-in

### Fixed
- **Pine Script router registration** - Router was missing from `app/routers/__init__.py` exports
- **JSONB string serialization** - Handle asyncpg returning JSONB columns as strings instead of dicts
- **Source type constraint** - Add `pine_script` to `documents.source_type` check constraint (migration 047)

### Added
- `LLMNotConfiguredError` exception for clear error handling when LLM is not available
- Graceful degradation for `mode=answer` queries without OpenRouter API key
- Initial release of Trading RAG Pipeline
- Document ingestion endpoint (`POST /ingest`)
- YouTube transcript ingestion (`POST /sources/youtube/ingest`)
- Semantic search with filtering (`POST /query`)
- Re-embedding for model migration (`POST /reembed`)
- Job status tracking (`GET /jobs/{job_id}`)
- Health check with dependency status (`GET /health`)
- Qdrant vector storage with 8 payload indexes
- Ollama embedding with nomic-embed-text model
- Token-aware chunking (512 tokens max)
- Metadata extraction (symbols, entities, topics)
- YouTube transcript normalization
- Timestamp-aware chunking for video content
- OpenAPI documentation (Swagger UI and ReDoc)
- Docker Compose configuration for local development
- Structured JSON logging with request context

### Technical Details
- FastAPI 0.109.2 with Pydantic v2
- Qdrant vector database (768-dim, Cosine distance)
- Supabase for document/chunk storage
- tiktoken for accurate token counting
- Request ID middleware for tracing
- CORS configuration for cross-origin requests

## [0.1.0] - 2025-01-03

### Added
- Initial project structure
- Core API endpoints
- Database schema design
- Docker configuration
- Unit tests for extractor and chunker
- Development requirements separation

---

[Unreleased]: https://github.com/username/trading-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/trading-rag/releases/tag/v0.1.0
