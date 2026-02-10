# Documentation

Central index for trading-RAG project documentation.

## Features

- [Backtests](features/backtests.md) - Parameter tuning, WFO, coaching, eval mode, ICT Unicorn strategy
- [Backtest Fill Semantics](features/backtest-fill-semantics.md) - Entry/exit execution model and intrabar policy
- [ORB Engine](features/orb-engine.md) - Opening Range Breakout engine specification and events
- [Engine Protocol](features/engine-protocol.md) - Reference protocol for backtest engines (interface, events, versioning, fixtures)
- [Execution](features/execution.md) - Paper trading adapter, strategy runner, execution gating
- [KB Recommend](features/kb-recommend.md) - Strategy parameter recommendations and regime fingerprints
- [Coverage](features/coverage.md) - Coverage triage workflow and cockpit UI
- [Pine Scripts](features/pine-scripts.md) - Pine Script registry, linting, ingestion, auto-strategy discovery
- [Reranker](features/reranker.md) - Cross-encoder reranking for two-stage retrieval
- [Ops & Security](features/ops.md) - System health dashboard, security, alert webhooks, v1.0.0 hardening

## Operations

- [Runbooks](ops/runbooks.md) - Standard operating procedures (restarts, collection rebuild, model rotation, failure recovery)
- [Alerting Rules](ops/alerting-rules.md) - Prometheus and Sentry alert definitions with thresholds
- [Hardening](ops/hardening.md) - v1.0.0 operational hardening (idempotency, retention, LLM fallback, SSE)
- [Reranker Runbook](ops/reranker-runbook.md) - Reranker operational guide, troubleshooting, alert conditions

## Integrations

- [n8n Workflows](n8n/README.md) - n8n workflow patterns for YouTube ingestion and API orchestration

## Other

- [Constraints](constraints.txt) - Dependency version pins for torch, transformers, tokenizers

## Assets

- `dashboard/` - SVG diagrams for dashboard features (equity, KPI alerts, trade explorer)
- `historical_data/` - Sample OHLCV data files for backtesting
