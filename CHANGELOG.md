# Changelog

All notable changes to the Trading RAG Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
