# Execution System

Trade execution adapters, strategy runners, and testing frameworks.

## Paper Execution Adapter

Provider-agnostic broker adapter that simulates trade execution for end-to-end automation testing without exchange randomness.

**Source**: `app/services/execution/`

### Architecture

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

### Key Design Decisions

| Decision | Details |
|----------|---------|
| State persistence | In-memory + journal rebuild (event sourcing) |
| Fill simulation | Immediate fill, caller provides `fill_price` (required) |
| Order types | MARKET only (limit orders planned later) |
| Position support | Long-only, single position per symbol |
| Reconciliation | Manual endpoint only (no auto on startup) |
| Idempotency | Key = `(workspace_id, intent_id, mode)`, returns 409 on duplicate |
| Supported actions | `OPEN_LONG`, `CLOSE_LONG` only (400 for others) |
| Full close only | SELL qty must == position.qty (no partial closes) |
| Policy re-check | Execution re-evaluates policy internally |

### Event Flow

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

### Cash Ledger (`PaperState`)

| Field | Description |
|-------|-------------|
| `starting_equity` | Initial capital (default 10000.0) |
| `cash` | Current available cash |
| `realized_pnl` | Cumulative realized profit/loss |
| `positions` | Dict of symbol → `PaperPosition` |

### Position Scaling

```python
new_qty = old_qty + add_qty
new_avg = (old_avg * old_qty + fill_price * add_qty) / new_qty
```

### Reconciliation Process

1. Clear in-memory state (cash = starting_equity, positions = {})
2. Query `trade_events` for `ORDER_FILLED` events only
3. Dedupe by `order_id` (skip duplicates)
4. Replay fills to rebuild cash/positions
5. `POSITION_*` events are NOT replayed (observability only)

### API Endpoints

```
POST /execute/intents                        - Execute trade intent (paper mode only)
GET  /execute/paper/state/{workspace_id}     - Get paper trading state
GET  /execute/paper/positions/{workspace_id} - Get open positions
POST /execute/paper/reconcile/{workspace_id} - Rebuild state from journal
POST /execute/paper/reset/{workspace_id}     - Reset state (dev only)
```

All require `X-Admin-Token` header.

### Execute Request

```python
POST /execute/intents
{
    "intent": TradeIntent,   # From policy engine
    "fill_price": 50000.0,   # REQUIRED - caller provides
    "mode": "paper"          # Only paper supported
}
```

Returns 409 Conflict if intent already executed.

---

## Strategy Runner

Generates TradeIntents from strategy configuration, market data, and current portfolio state. Bridges backtest research and live execution.

**Source**: `app/services/strategy/`

### Architecture

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

### Key Models

| Model | Description |
|-------|-------------|
| `ExecutionSpec` | Runtime configuration for strategy instance (name, params, symbol, workspace) |
| `MarketSnapshot` | Point-in-time market state with OHLCV bars (caller-provided for determinism) |
| `StrategyEvaluation` | Runner output containing intents, signals, and evaluation metadata |

### Key Design Decisions

- **Separate ExecutionSpec from StrategyRegistry**: Runtime config vs param schema definition
- **MarketSnapshot is caller-provided**: Enables deterministic testing, no internal data fetching
- **Stateless evaluation**: No internal runner state; all context passed in per call
- **Max positions only blocks entries, never exits**: Safety rule to prevent over-allocation
- **Exclude current bar from 52w high computation**: Avoid look-ahead bias
- **evaluation_id shared by all intents**: Enables end-to-end tracing from signal to fill

### Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `breakout_52w_high` | Entry when price exceeds 52-week high, EOD exit |

### Usage

```python
from app.services.strategy import StrategyRunner, ExecutionSpec, MarketSnapshot

runner = StrategyRunner()
result = runner.evaluate(spec, snapshot, paper_state)
for intent in result.intents:
    # Execute via PaperBroker
    pass
```

---

## Test Generator & Run Orchestrator

Parameter sweep framework for systematic strategy testing. Takes an ExecutionSpec and generates variants automatically, then runs them through StrategyRunner + PaperBroker.

**Source**: `app/services/testing/`

### Architecture

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

### Key Design Decisions

| Decision | Implementation |
|----------|----------------|
| Variant ID | `sha256(canonical_json({base, overrides}))[:16]` - deterministic, stable |
| Overrides format | Flat dotted-path dict only (e.g., `{"entry.lookback_days": 200}`) |
| Broker isolation | Each variant uses `uuid5(VARIANT_NS, f"{run_plan_id}:{variant_id}")` as workspace |
| Equity tracking | Trade-equity points (step function) - equity at each closed trade |
| Persistence | In-memory RunPlan for v0, events in trade_events journal |

### Sweepable Parameters

- `entry.lookback_days` - Lookback for 52w high calculation
- `risk.dollars_per_trade` - Position sizing
- `risk.max_positions` - Max concurrent positions

### Generator Output

1. **Baseline** variant (empty overrides) - always first
2. **Grid sweep** variants (cartesian product of sweep values)
3. **Ablation** variants (reset one param to default, relative to first grid combo)
4. Deduplication + max_variants limit

### Metrics Calculated

| Metric | Formula |
|--------|---------|
| `return_pct` | (ending_equity / starting_equity - 1) × 100 |
| `max_drawdown_pct` | Peak-to-trough from trade-equity curve |
| `sharpe` | mean(trade_returns) / std(trade_returns), None if <2 trades |
| `win_rate` | wins / total_trades |
| `trade_count` | Total closed trades |
| `profit_factor` | gross_profit / gross_loss |

### API Endpoints

```
POST /testing/run-plans/generate             - Generate RunPlan (no execution)
POST /testing/run-plans/generate-and-execute - Generate + execute + return results (multipart CSV)
```

### Admin UI

- `/admin/testing/run-plans` - List page (event-driven summaries)
- `/admin/testing/run-plans/{id}` - Detail page
