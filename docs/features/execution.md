# Execution System

Paper trading execution for end-to-end automation testing.

## Paper Execution Adapter (`app/services/execution/`)

Provider-agnostic broker adapter that simulates trade execution without exchange randomness.

**Architecture**:
```
TradeIntent (approved by PolicyEngine)
       │
       ▼
  PaperBroker ────► trade_events journal (ORDER_FILLED)
       │
       ▼
  PaperState (in-memory, reconcilable from journal)
```

**Key Design Decisions**:
- **State persistence**: In-memory + journal rebuild (event sourcing)
- **Fill simulation**: Immediate fill, caller provides `fill_price` (required)
- **Order types**: MARKET only (limit orders planned)
- **Position support**: Long-only, single position per symbol
- **Reconciliation**: Manual endpoint only
- **Idempotency**: Key = `(workspace_id, intent_id, mode)`, returns 409 on duplicate
- **Supported actions**: `OPEN_LONG`, `CLOSE_LONG` only
- **Full close only**: SELL qty must == position.qty

## Execution Gating (Safety)

The paper broker enforces "paused means zero orders" at the last possible decision point.

**Gating Flow**:
```
execute_intent()
    ├─ 1. Validate action supported
    ├─ 2. Idempotency check
    ├─ 3. STRATEGY STATE CHECK ← blocks if no active version
    ├─ 4. Re-evaluate policy
    └─ 5. Execute fill
```

**Strategy State Check** (`paper_broker.py:117-153`):
- Queries `strategies` table by `strategy_entity_id`
- Returns `STRATEGY_PAUSED` error if `active_version_id IS NULL`
- Journals `INTENT_REJECTED` event for audit trail

**Integration with Auto-Pause**:
When `AUTO_PAUSE_ENABLED=true` and a CRITICAL alert fires (drawdown >20%, confidence <0.20):
1. Auto-pause sets `strategy_versions.state = 'paused'`
2. This clears `strategies.active_version_id`
3. Next execution attempt is blocked with `STRATEGY_PAUSED`

**Error Codes**:
| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `STRATEGY_PAUSED` | 200 (success=false) | Strategy has no active version |
| `POLICY_REJECTED` | 200 (success=false) | Policy rule rejected intent |
| `ALREADY_EXECUTED` | 409 | Idempotency hit |

**Backward Compatibility**:
If `version_repo` is not passed to PaperBroker, the check is skipped (legacy mode).

**Event Flow**:
```
INTENT_EMITTED → POLICY_EVALUATED → INTENT_APPROVED → ORDER_FILLED
                                                           │
                          ┌────────────────────────────────┼────────────────┐
                          ▼                                ▼                ▼
                  POSITION_OPENED            POSITION_SCALED     POSITION_CLOSED
                       (observability breadcrumbs only)
```

**Cash Ledger** (`PaperState`):
- `starting_equity` - Initial capital (default 10000.0)
- `cash` - Current available cash
- `realized_pnl` - Cumulative realized P&L
- `positions` - Dict of symbol → `PaperPosition`

**Position Scaling**:
```python
new_qty = old_qty + add_qty
new_avg = (old_avg * old_qty + fill_price * add_qty) / new_qty
```

**Reconciliation**:
1. Clear state (cash = starting_equity, positions = {})
2. Query `ORDER_FILLED` events only
3. Dedupe by `order_id`
4. Replay fills to rebuild state

**Execute Request**:
```python
POST /execute/intents
{"intent": TradeIntent, "fill_price": 50000.0, "mode": "paper"}
```

## Strategy Runner (`app/services/strategy/`)

Generates TradeIntents from strategy configuration, market data, and portfolio state.

**Architecture**:
```
ExecutionSpec + MarketSnapshot + PaperState
       │
       ▼
  StrategyRunner ────► list[TradeIntent]
       │
       ▼
  PolicyEngine → PaperBroker → Journal
```

**Key Models**:
- `ExecutionSpec` - Runtime configuration (strategy name, params, symbol, workspace)
- `MarketSnapshot` - Point-in-time market state with OHLCV bars (caller-provided)
- `StrategyEvaluation` - Runner output with intents, signals, and metadata

**Key Design Decisions**:
- Separate ExecutionSpec from StrategyRegistry (runtime vs schema)
- MarketSnapshot is caller-provided (deterministic testing)
- Stateless evaluation (all context passed in per call)
- Max positions only blocks entries, never exits
- Exclude current bar from 52w high (avoid look-ahead bias)
- evaluation_id shared by all intents (end-to-end tracing)

**Built-in Strategies**:
- `breakout_52w_high` - Entry when price exceeds 52-week high, EOD exit

**Usage**:
```python
from app.services.strategy import StrategyRunner, ExecutionSpec, MarketSnapshot
runner = StrategyRunner()
result = runner.evaluate(spec, snapshot, paper_state)
```
