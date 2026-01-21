# Product Requirements Document (PRD)
## Trading RAG Platform — **Beta v0.1**

**Status:** Draft  
**Owner:** You  
**Audience:** Internal + Early Beta Users  
**Target:** Private Beta (Technical Users)

---

## 1. Purpose

The goal of **Beta v0.1** is to validate the platform as a **reliable research, backtesting, and strategy evaluation system** for crypto trading strategies.

This release is **not** intended for live trading.  
It is intended to prove correctness, reproducibility, observability, and usability for serious strategy research.

---

## 2. Goals

### Primary Goals
- Enable users to **ingest historical crypto market data**
- Allow users to **define, backtest, and tune trading strategies**
- Provide **clear, reproducible results** with visual inspection
- Surface **operational health and alerts** automatically
- Support **strategy confidence and coverage analysis**

### Non-Goals (Explicit)
- ❌ Live trading or order execution
- ❌ Paper trading with real accounts
- ❌ AI-generated strategies
- ❌ Public user onboarding / billing
- ❌ Mobile UI

---

## 3. Target Users

### Beta Users
- Technically proficient traders / quants
- Comfortable with backtesting concepts
- Understand simulation limitations
- Expect transparency over polish

### User Needs
- Trust the data
- Reproduce results
- Understand *why* a strategy works or fails
- Detect system issues early (alerts)

---

## 4. System Scope (Beta v0.1)

### 4.1 Market Data

#### Requirements
- Crypto OHLCV historical data via **CCXT**
- Supported venues:
  - KuCoin (required)
  - Binance (optional, later)
- Supported symbols (initial):
  - BTC/USDT
  - ETH/USDT
- Supported timeframes:
  - 15m
  - 1h

#### Data Guarantees
- All timestamps normalized to UTC
- Candles stored in canonical schema
- `(venue, symbol, timeframe, ts)` uniqueness enforced
- Missing candles logged (not silently ignored)

#### Out of Scope
- Tick data
- Order book data
- Funding rates (for beta)

---

## 5. Backtesting & Strategy Engine

### 5.1 Strategy Lifecycle

#### Strategy Definition
- Strategy logic stored as versioned code
- Parameter sets stored separately
- Immutable strategy versions (hash or version tag)

#### Strategy Controls
- Enable / disable per workspace
- Clear distinction between:
  - Strategy logic
  - Parameter configuration
  - Backtest run

---

### 5.2 Backtest Engine Requirements

#### Simulation Rules
- Deterministic execution (same inputs → same outputs)
- Explicit fee model (maker/taker)
- Explicit slippage model (fixed bps acceptable)
- Clear position sizing rules

#### Metrics (Required)
- Total return
- Max drawdown
- Sharpe ratio
- Win rate
- Profit factor
- Trade count

#### Benchmarks
- Buy & Hold comparison for each asset

---

## 6. Walk-Forward Optimization (WFO)

### Requirements
- Fixed rolling windows
- Separate train / test segments
- Per-fold metrics stored
- Aggregate selection logic documented

### Out of Scope
- Dynamic regime-adaptive WFO
- Live parameter switching

---

## 7. Visualization & Reporting

### 7.1 Required Visuals

#### Backtest Report
- Equity curve (strategy vs benchmark)
- Drawdown curve
- Candlestick chart with:
  - Entry markers
  - Exit markers
  - Stop loss / take profit
- Trade list table

#### Formats
- Interactive HTML report (Plotly or similar)
- CSV trade export
- JSON metrics snapshot

---

## 8. Alerts & Observability

### 8.1 Ops Alerts

#### Alert Types
- Health degradation (DB, vector store, services)
- Coverage weakness
- Confidence drop
- Drift spike

#### Alert Features
- Automated evaluation via scheduled jobs
- Telegram delivery with:
  - Activation notifications
  - Recovery notifications
  - Escalation notifications
- Delivery idempotency (no duplicates)

---

### 8.2 Admin Visibility

#### Admin UI Pages
- Ops Alerts management
- System health dashboard
- Coverage cockpit
- Ingest UI
- Backtest tuning sessions
- Backtest detail pages

#### Required Controls
- Acknowledge alerts
- Resolve alerts
- Reopen alerts
- Severity filtering
- Workspace scoping

---

## 9. Security & Isolation

### Requirements
- Workspace-scoped data isolation
- Admin token authentication
- Read-only admin mode (optional)
- No cross-workspace data access

### Out of Scope
- Public authentication
- OAuth
- User self-service onboarding

---

## 10. Reliability & Reproducibility

### Requirements
- All jobs idempotent
- All backtests reproducible via:
  - Data revision
  - Strategy version
  - Parameter set
- Job retries with bounded attempts
- Alerting on job failures

---

## 11. Documentation (Required for Beta)

### Required Docs
- System overview
- Backtesting assumptions
- Strategy lifecycle explanation
- Alert severity definitions
- Known limitations
- “What this system does NOT do”

---

## 12. Success Criteria (Beta Exit)

Beta v0.1 is considered successful if:

- A user can ingest data, run a backtest, and review results end-to-end
- Backtests are reproducible across runs
- Alerts fire automatically and deliver to Telegram
- Admin UI allows full inspection and triage
- No silent failures in data, jobs, or alerts

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|----|----|
| Data gaps | Explicit logging + visibility |
| Overconfidence in results | Clear disclaimers + benchmarks |
| Alert spam | Deduplication + severity rules |
| Scope creep | Explicit non-goals |

---

## 14. Post-Beta (Not in v0.1)

- Live trading
- Paper trading
- Advanced regime detection
- Public dashboards
- User billing

---

## 15. Summary

**Beta v0.1** delivers a **serious, transparent, and reproducible trading research platform**.

It prioritizes:
- Correctness over polish
- Observability over automation
- Trust over hype

Anything not required to validate those principles is intentionally deferred.
