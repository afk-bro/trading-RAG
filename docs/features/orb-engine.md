# ORB Engine

Opening Range Breakout engine for intraday strategies.

## v1.0 (current)

### Event Types

| Type | Phase | Payload |
|------|-------|---------|
| `orb_range_update` | OR_BUILD | `orb_high`, `orb_low`, `or_minutes`, `or_start_index` |
| `orb_range_locked` | BREAKOUT_SCAN | `high`, `low`, `range`, `or_minutes`, `or_start_index`, `or_lock_index`, `or_bar_count_needed` |
| `setup_valid` | ENTRY | `direction`, `level`, `confirm_mode`, `trigger_price` |
| `entry_signal` | TRADE_MGMT | `side`, `price`, `stop`, `target`, `size`, `risk_points` |

All events carry common keys: `type`, `bar_index`, `ts`, `session_date`, `phase`, `schema_version`.

### Schema Versioning

- Current schema version: **1.0.0**
- Events carry `schema_version` on every record
- Contract validation via `validate_events()` in `contracts.py`

### State Machine

```
PREMARKET → OR_BUILD → BREAKOUT_SCAN → ENTRY → TRADE_MGMT → EXIT → LOCKOUT
                                  ↑                              |
                                  └──────── (if max_trades > 1) ─┘
```

### Preset Immutability

`NY_AM_ORB_V1` is frozen (`version="1.0"`, `schema_version="1.0.0"`).
Do not mutate — clone to `ny-am-orb-v1.1` for changes.

### Regression Canary

Golden fixture at `tests/golden/fixtures/orb_v1_golden_run.json` captures exact event
and trade output for a deterministic synthetic session. If the fixture breaks, ORB
semantics changed.

## v1.1 Roadmap

### New Event: `position_closed`

Fires when a position exits (target, stop, session_close, eod).

| Field | Type | Description |
|-------|------|-------------|
| `exit_reason` | string | `"target"`, `"stop"`, `"session_close"`, `"eod"` |
| `exit_price` | float | Execution price |
| `pnl` | float | Net P&L after commission |
| `side` | string | `"long"` or `"short"` |
| `entry_bar` | int | Bar index of entry |
| `exit_bar` | int | Bar index of exit |

Makes replay and coaching cleaner — consumers can react to position lifecycle
without parsing the trades list.

### New Event: `gate_rejected`

Fires when a pre-entry gate filters out a candidate trade.

| Field | Type | Description |
|-------|------|-------------|
| `gate_name` | string | Gate identifier (e.g. `"regime_filter"`, `"max_drawdown"`) |
| `detail` | string | Human-readable rejection reason |
| `direction` | string | The rejected direction |

Explains why entries were filtered, visible in replay and no-trade summaries.

### Extension Rules

| Change | Version Bump |
|--------|-------------|
| Optional keys added to existing events | None |
| New event type added | Minor (1.0.0 → 1.1.0) |
| Required key added to existing event | Major (1.0.0 → 2.0.0) |
| Required key removed or renamed | Major |
| Semantic change to existing key | Major |

### Consumer Checklist

Before bumping schema version, verify these consumers still work:

| Consumer | Location | Impact |
|----------|----------|--------|
| ReplayPanel | `dashboard/src/components/backtests/ReplayPanel.tsx` | Renders events as timeline |
| ORBSummaryPanel | `dashboard/src/components/backtests/ORBSummaryPanel.tsx` | Derives summary + no-trade reasons |
| ORRangeDisplay | `dashboard/src/components/backtests/ORRangeDisplay.tsx` | Visual OR state from events |
| process_score | `app/services/backtest/process_score.py` | Scores rule adherence |
| loss_attribution | `app/services/backtest/loss_attribution.py` | Attributes losses to event gaps |
| run_events repo | `app/repositories/run_events.py` | Persists and queries events |
| Golden fixture | `tests/golden/test_orb_golden.py` | Regression canary |
| Live smoke | `tests/golden/test_orb_live_smoke.py` | Real-data validation |
