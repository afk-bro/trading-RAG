# Changelog

All notable changes to the Trading RAG Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Extracted dashboards SQL into `DashboardsRepository` layer
- Consolidated duplicate `fmtPct`, `fmtNum`, `fmtPnl` into `@/lib/chart-utils`
- Consolidated duplicate `Field` component into `@/components/ui/Field`
- Persistent `httpx.AsyncClient` in embedder service (connection reuse)
- Sanitized error responses — internal details logged server-side only
- Moved reranker docs into standard `docs/features/` and `docs/ops/` layout

### Fixed
- Timing-safe API key comparison via `hmac.compare_digest()`
- Localhost admin bypass now checks `request.client.host` (not spoofable `Host` header)
- React hooks ordering in `BacktestRunPage` (useState before conditional return)
- Batch chunk INSERT uses single multi-row query (was N+1)
- PDF extraction offloaded to thread pool (`asyncio.to_thread`)

### Added
- ARIA attributes across dashboard (dialogs, tables, tabs, workspace switcher)
- Admin token required on all `/debug/*` endpoints
- Playwright E2E tests for React dashboard (shell, backtests, workspace switcher, tabs)
- `docs/README.md` central documentation index
- `migrations/README.md` migration process guide

### Security
- Docker ports bound to `127.0.0.1` (Qdrant, Ollama)
- Pinned Docker images: Qdrant v1.12.4, Ollama 0.15.6
- Removed unused `recharts` dependency (~200KB bundle reduction)

## [1.2.0] - 2026-02-09

### Added
- **ORB v1 Hardening** — Frozen preset, golden test fixtures, no-trade reason tracking, live smoke tests
- **Engine Protocol** specification — Reference interface for backtest engines with versioning and consumer checklist
- **Backtest Coaching UX** — Process score, loss attribution, run lineage with baseline selector
  - `DeltaKpiStrip` with trajectory sparklines
  - `WhatYouLearnedCard` for param diff summaries
  - `ProcessScoreCard` with component-level grading
  - `LossAttributionPanel` with time/size clusters and counterfactuals
  - `BaselineSelector` for manual baseline comparison
  - 500ms coaching budget with `asyncio.wait()` partial timeout recovery
  - `coaching_partial` flag for degraded data signaling

### Changed
- Coaching system documentation added to `docs/features/backtests.md`

## [1.1.0] - 2026-02-08

### Added
- **React Dashboard SPA** — Regime-aware equity charts, alerts panel, RAG-backed trade context
- **Backtest Results Viewer** — Equity charts (LWC v4.2), trade tables, CSV/JSON export
- **Strategy Version Compare** — Side-by-side KPI deltas, equity overlay, config diff
- **Workspace Switcher** — Multi-tenant workspace scoping for all dashboard views
- **ORB v1 Preset** — Opening Range Breakout engine with WFO segments, run events, replay UI
- **ORB Summary Panel** — Session stats, signal distribution, entry timing analysis
- Dashboard skeletons, `ErrorBoundary`, and empty states
- Dev proxy routes for workspaces, strategies, execute, health

## [1.0.0] - 2026-01-29

Operational hardening release for production readiness. See `docs/ops/hardening.md`.

### Added
- **Paper Equity Snapshots & Drawdown Alerts (PR9)** — Equity tracking with automated drawdown monitoring
  - `paper_equity_snapshots` table: append-only equity time series per workspace
  - Dual time axes: `snapshot_ts` (market time), `computed_at` (wall clock)
  - Deduplication via SHA256 `inputs_hash`
  - `WORKSPACE_DRAWDOWN_HIGH` alert rule: WARN at 12%, CRITICAL at 20%
  - Hysteresis to prevent flapping: clear WARN at 10%, clear CRITICAL at 16%
  - Prometheus gauge `workspace_drawdown_pct`
- **Auto-Pause Guardrail (PR9b)** — Safety feature for CRITICAL alerts
  - Auto-pauses active strategy versions when CRITICAL drawdown (>20%) fires
  - Auto-pauses specific version when CRITICAL confidence (<0.20) fires
  - Manual unpause required via API
- **Read-Only Dashboard Endpoints (PR9b)** — Trust-building visualization data
  - `GET /dashboards/{workspace_id}/equity` — Equity curve with drawdown overlay
  - `GET /dashboards/{workspace_id}/intel-timeline` — Confidence & regime history
  - `GET /dashboards/{workspace_id}/alerts` — Active alerts by severity
  - `GET /dashboards/{workspace_id}/summary` — Combined overview
- **Strategy Confidence Alert Rule** — MEDIUM <0.35, HIGH <0.20, with persistence gate
- **Webhook Sinks for Alerts** — Slack and generic webhook delivery with retry logic
- **Ops Alerts Admin UI** — Alert management page with filtering, pagination, status workflow
- **Idempotent Telegram Notification Delivery** — Race-safe notification system
- **Idempotency Hygiene Monitoring** — Health page card, Prometheus metrics, alert rules
- **System Health Dashboard** — Single-page operational health view (`/admin/system/health`)
- **Prometheus Alerting Rules** — Ready-to-deploy alert configuration
- **Operational Hardening Canary Tests** — Integration tests for safety features

## [0.9.0] - 2026-01-15

Initial feature set establishing the RAG pipeline, strategy management, and backtest infrastructure.

### Added
- **ICT Unicorn Model Strategy** — Discretionary trading strategy implementation
  - 8-criteria checklist (3 mandatory + 5 scored) for trade qualification
  - HTF/LTF multi-timeframe analysis with session filtering
  - Volatility-normalized position sizing via ATR
  - Direction filter support (long-only, short-only, both)
  - Time-stop exit for trades not hitting TP/SL within session
  - Look-ahead bias fixes in entry logic
  - Wick guard, range guard, displacement guard
  - NY_OPEN session profile, setup session diagnostics
  - Session sensitivity sweep validation
- **Strategy Backtest UI** — Interactive backtesting at `/admin/backtests/strategy-test`
  - Strategy/symbol selection, date range picker, direction filter
  - Trading platform selector with realistic commission rates
  - R-Metrics, expectancy analysis, R distribution, session breakdown
  - Exit reasons, criteria bottleneck, MFE/MAE analysis
- **Databento Integration** — CME futures historical data (ES/NQ 2021-2026)
- **Document Detail Page** — Document viewer with key concepts, tickers, chunk validation
- **Transcript Preprocessing Filters** — Sponsor/engagement phrase removal
- **Ticker Detection Improvements** — Reduced false positives with 200+ exclusions
- **Unified Ingestion Endpoint** — `POST /ingest/unified` with auto-detection
- **Cookie-Based Admin Authentication** — 30-day httponly cookie with multi-source token acceptance
- **Landing Page** — Public system overview at root URL
- **Article Extractor Service** — Web content extraction via trafilatura
- **Webhook Alert Configuration** — Settings for Slack and generic webhook delivery
- **Documentation Refactor** — CLAUDE.md reduced from 1060 to 148 lines (86% reduction)
- **Strategy Lifecycle v0.5** — Version management with state machine (draft → active ↔ paused → retired)
- **Strategy Intelligence Snapshots** — Append-only time series for regime + confidence
- **Regime Fingerprint Materialization** — Instant regime similarity queries (Migration 056)
- **Auto-Strategy Discovery from Pine Scripts** — Parameter spec generation

### Fixed
- `check_criteria()` in backtest loop now receives config (session profile was silently ignored)
