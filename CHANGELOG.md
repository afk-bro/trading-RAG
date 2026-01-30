# Changelog

All notable changes to the Trading RAG Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **ICT Unicorn Model Strategy** - Discretionary trading strategy implementation
  - 8-criteria checklist (3 mandatory + 5 scored) for trade qualification
  - HTF/LTF multi-timeframe analysis with session filtering
  - Volatility-normalized position sizing via ATR
  - Direction filter support (long-only, short-only, both)
  - Time-stop exit for trades not hitting TP/SL within session
  - Look-ahead bias fixes in entry logic
  - **Wick guard** (`max_wick_ratio`): skip entries where signal bar adverse wick exceeds threshold
  - **Range guard** (`max_range_atr_mult`): skip entries where signal bar range exceeds ATR multiple
  - **Displacement guard** (`min_displacement_atr`): skip entries where MSS displacement is below ATR threshold — filters low-conviction structure shifts. Validated across 5 market regimes (2021-2025), sub-window splits, and ATR(10/14/21) normalizations using deterministic worst-case intrabar execution. Recommended production value: 0.5x ATR
  - MSS match order fix: prefer newest matching shift (reversed iteration) for both `check_criteria()` and `analyze_unicorn_setup()`

- **Strategy Backtest UI** - Interactive backtesting interface at `/admin/backtests/strategy-test`
  - Strategy and symbol selection (ICT Unicorn Model, ES/NQ futures)
  - Date range picker with data availability validation
  - Direction filter dropdown (long-only recommended based on analysis)
  - Min criteria score selector (3-5 of 5)
  - Trading platform selector with realistic commission rates:
    - Apex (Rithmic): $3.98/RT
    - Apex (Tradovate): $3.10/RT
    - Topstep: $2.80/RT
    - Custom: user-defined
  - Friction settings: slippage (ticks), commission, intrabar policy
  - Loading animation with chart bars and cycling status messages
  - Rich results display:
    - **R-Metrics card** (highlighted): Avg R, Total R, Max Win, Max Loss
    - **Expectancy Analysis**: Visual formula E = (Win% × Avg Win R) − (Loss% × |Avg Loss R|)
    - **R Distribution**: Horizontal bar chart with P(≥+1R) and P(≥+2R) badges
    - "No +2R+ wins" warning badge when fat tails missing
    - Session breakdown chart (London, NY, Asia)
    - Exit reasons pie chart (target, stop_loss, time_stop, session_end)
    - Criteria bottleneck analysis
    - MFE/MAE analysis
    - Confidence buckets correlation
  - HTMX-powered form submission with real-time updates

- **Databento Integration** - CME futures historical data
  - API client for batch data requests
  - Local CSV loading with front-month contract filtering
  - OHLCV-1m data for ES and NQ futures (2021-2026)
  - Zstd-compressed storage with manifest/metadata files

- **Document Detail Page** - Comprehensive document viewer at `/admin/documents/{doc_id}`
  - Document metadata display (title, author, source type, timestamps)
  - Stats bar: chunks, tokens, concepts, tickers
  - Key trading concepts extraction (40+ terms: breakout, support, resistance, VWAP, etc.)
  - Ticker symbol detection with comprehensive exclusion list (200+ non-tickers)
  - Chunk validation UI (Verified/Needs Review/Garbage buttons)
  - `chunk_validations` table for tracking QA status
  - Link from ingest results to document detail page

- **Transcript Preprocessing Filters** - Clean YouTube transcripts before chunking
  - Sponsor segment removal (sponsored by, today's sponsor, use code, etc.)
  - Engagement phrase removal (subscribe, like button, bell notification, etc.)
  - Preserves legitimate content ("like water through a channel")
  - 13 new unit tests for filter coverage

- **Ticker Detection Improvements** - Reduced false positives
  - Fixed regex to prevent mid-word matches (RRENT from CURRENT, TICLE from ARTICLE)
  - Added 70+ trading indicator exclusions (EMA, SMA, VWAP, RSI, MACD, ATR, etc.)
  - Added finance/regulatory term exclusions (PDT, FX, FINRA, CMT, CPA, CFTC, etc.)
  - Added futures contract exclusions (ES, NQ, YM, CL, GC, ZB)
  - Added options term exclusions (IV, HV, OI, DTE, ITM, OTM, ATM)

- **Unified Ingestion Endpoint** - Single endpoint for all content types with auto-detection
  - `POST /ingest/unified` - Multipart form-data endpoint
  - Auto-detects: YouTube URLs, PDF URLs, article URLs, PDF files, text/markdown files, Pine files
  - Source type override via `source_type` parameter
  - Detection logic in `app/services/ingest/detection.py`
  - Service layer refactor: youtube.py, pdf.py, text.py, article.py under `app/services/ingest/`

- **Cookie-Based Admin Authentication** - Seamless admin UI navigation
  - `GET /admin/login` - Login page with token input
  - `POST /admin/auth/login` - Authenticate and set httponly cookie
  - `GET /admin/auth/logout` - Clear auth cookie and redirect
  - `GET /admin/auth/check` - Verify authentication status
  - 30-day cookie expiry with "Remember me" option
  - Token accepted from: header (`X-Admin-Token`), cookie (`admin_token`), or query param

- **Landing Page** - Public system overview at root URL
  - `GET /` - HTML landing page describing Trading RAG system
  - Feature highlights, architecture overview, quick links
  - No authentication required

- **Article Extractor Service** - Web content extraction for article URLs
  - `app/services/article_extractor.py` - Fetch and parse web articles
  - Uses trafilatura for content extraction
  - Extracts: title, text, author, published date
  - Response size guard (10MB limit)

- **Admin Ingest UI Enhancements** - Extended content type support
  - Article URL tab with title override
  - Text/Markdown file upload tab
  - All tabs use unified endpoint via FormData

- **Webhook Alert Configuration** - Settings for alert delivery
  - `webhook_enabled` - Master switch for webhook alerts
  - `slack_webhook_url` - Slack incoming webhook URL
  - `alert_webhook_url` - Generic webhook URL
  - `alert_webhook_headers` - Custom headers for webhook requests

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

- **Paper Equity Snapshots & Drawdown Alerts (PR9)** - Equity tracking with automated drawdown monitoring
  - `paper_equity_snapshots` table: append-only equity time series per workspace
  - Dual time axes: `snapshot_ts` (market time), `computed_at` (wall clock)
  - Equity components: cash, positions_value, realized_pnl
  - Deduplication via SHA256 `inputs_hash` to prevent snapshot spam
  - `PaperEquityRepository` with insert_snapshot, list_window, compute_drawdown
  - Paper broker integration: automatic equity snapshot recording after executions
  - `WORKSPACE_DRAWDOWN_HIGH` alert rule: WARN at 12%, CRITICAL at 20%
  - Hysteresis to prevent flapping: clear WARN at 10%, clear CRITICAL at 16%
  - Prometheus gauge `workspace_drawdown_pct` for Grafana dashboards
  - Runbook for drawdown alert response
  - Migration: 081_paper_equity_snapshots.sql

- **Auto-Pause Guardrail (PR9b)** - Safety feature for CRITICAL alerts
  - `auto_pause_enabled` config flag (default: false)
  - Auto-pauses active strategy versions when CRITICAL drawdown (>20%) fires
  - Auto-pauses specific version when CRITICAL confidence (<0.20) fires
  - Prometheus counter `ops_alert_auto_pause_total` for tracking
  - `EvalResult` extended with `versions_auto_paused`, `auto_paused_version_ids`
  - Manual unpause required via API

- **Read-Only Dashboard Endpoints (PR9b)** - Trust-building visualization data
  - `GET /dashboards/{workspace_id}/equity` - Equity curve with drawdown overlay
  - `GET /dashboards/{workspace_id}/intel-timeline` - Confidence & regime history
  - `GET /dashboards/{workspace_id}/alerts` - Active alerts by severity
  - `GET /dashboards/{workspace_id}/summary` - Combined overview for dashboard cards

- **Strategy Lifecycle v0.5** - Version management and state machine for strategies
  - `strategy_versions` table with immutable config snapshots (SHA256 hash)
  - State machine: `draft` → `active` ↔ `paused` → `retired` (retired is terminal)
  - One-active constraint via partial unique index
  - API endpoints: create version, list versions, get version, activate, pause, retire

- **Strategy Intelligence Snapshots (v1.5)** - Append-only time series for regime + confidence
  - `strategy_intel_snapshots` table for tracking intel per strategy version
  - Dual time axes: `as_of_ts` (market time) and `computed_at` (wall clock)
  - Core intel: `regime` (TEXT), `confidence_score` [0,1], `confidence_components` (JSONB)
  - Live computation via `app/services/intel/confidence.py` and `runner.py`
  - API endpoints for latest/timeline/recompute
  - Migration: 080_strategy_intel_snapshots.sql

- **Strategy Confidence Alert Rule** - Ops alert for low confidence scores
  - `STRATEGY_CONFIDENCE_LOW` rule type for detecting degraded strategy confidence
  - Thresholds: MEDIUM <0.35, HIGH <0.20, with persistence gate (2 consecutive)
  - Auto-resolution with hysteresis

- **Idempotent Telegram Notification Delivery** - Race-safe notification system for ops alerts

- **Idempotency Hygiene Monitoring** - Full observability for idempotency key lifecycle
  - Health page card: total keys, expired pending, pending requests, oldest ages
  - Prometheus metrics and alert rules

- **System Health Dashboard** - Single-page operational health view (`/admin/system/health`)

- **Regime Fingerprint Materialization** - Instant regime similarity queries (Migration 056)

- **Auto-Strategy Discovery from Pine Scripts** - Parameter spec generation

- **Prometheus Alerting Rules** - Ready-to-deploy alert configuration

- **Operational Hardening Canary Tests** - Integration tests for Q1 2026 safety features
