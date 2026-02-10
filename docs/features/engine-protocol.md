# Engine Protocol

Reference specification for backtest engines. Every engine in
`app/services/backtest/engines/` must satisfy this protocol to work with
the runner, replay UI, coaching pipeline, and strategy presets system.

ORB v1 is the reference implementation. When in doubt, match its patterns.

## 1. Interface

Engines implement `BacktestEngine` (structural typing via `Protocol`):

```python
class BacktestEngine(Protocol):
    def run(
        self,
        ohlcv_df: pd.DataFrame,
        config: dict[str, Any],
        params: dict[str, Any],
        initial_cash: float = 10000,
        commission_bps: float = 10,
        slippage_bps: float = 0,
    ) -> BacktestResult: ...
```

Every engine must also expose a `name` class attribute matching the
`StrategyPreset.engine` value (e.g., `name = "orb"`).

### BacktestResult

Engines return a `BacktestResult` containing:

| Field | Type | Description |
|-------|------|-------------|
| `return_pct` | float | Total return % |
| `max_drawdown_pct` | float | Max drawdown % |
| `sharpe_ratio` | float \| None | Annualized Sharpe |
| `win_rate` | float | Win rate [0, 1] |
| `num_trades` | int | Closed trade count |
| `equity_curve` | list[dict] | `[{"t": iso_str, "equity": float}]` |
| `trades` | list[dict] | Trade records (engine-specific fields allowed) |
| `events` | list[dict] | Run events for replay (see Event Contract below) |
| `warnings` | list[str] | Runtime warnings |

## 2. Event Contract

Every engine that emits events must follow this contract.

### Common Keys (required on every event)

| Key | Type | Description |
|-----|------|-------------|
| `type` | string | Event type identifier |
| `bar_index` | int | Bar position in the input DataFrame |
| `ts` | string | UTC ISO 8601 timestamp |
| `session_date` | string | ET date (YYYY-MM-DD) |
| `phase` | string | Engine state machine phase |
| `schema_version` | string | Semver string (e.g., `"1.0.0"`) |

### Type-Specific Keys

Each event type declares its own required payload keys beyond the common
set. These are defined in a `contracts.py` module co-located with the engine.

### Validation

Every engine must provide a `validate_events(events) -> list[str]` function
that checks all events against the contract. This function must:

- Return an empty list for valid events
- Return human-readable error strings for violations
- Be cheap enough to run in test/debug mode
- Be called under `if __debug__:` in the engine's `run()` method

Reference: `app/services/backtest/engines/orb/contracts.py`

## 3. Schema Versioning

Every engine tracks its event schema version.

### Version Format

Semver: `MAJOR.MINOR.PATCH`

### Bump Rules

| Change | Bump | Example |
|--------|------|---------|
| Optional key added to existing event | None | Adding `latency_ms` to `entry_signal` |
| New event type added | Minor | Adding `position_closed` type |
| Required key added to existing type | Major | Adding required `slippage` to `entry_signal` |
| Required key removed or renamed | Major | Renaming `price` to `fill_price` |
| Semantic change to existing key | Major | Changing `pnl` from gross to net |

### Version Constant

Define at module level:

```python
MY_ENGINE_EVENT_SCHEMA_VERSION = "1.0.0"
```

Stamp on every event in the `_evt_base()` factory.

## 4. State Machine

Event-driven engines must define a clear state machine with:

- An `Enum` of phases (e.g., `ORBPhase`)
- A dataclass for mutable session state (e.g., `ORBSessionState`) with a `reset()` method
- Phase transitions logged via events

The phase value is included on every event for replay consumers.

Reference: `app/services/backtest/engines/orb/types.py`

## 5. Strategy Preset

Every engine needs at least one `StrategyPreset` in `app/strategies/presets.py`.

### Required Fields

| Field | Description |
|-------|-------------|
| `slug` | URL-safe unique identifier (e.g., `"ny-am-orb-v1"`) |
| `name` | Human-readable name |
| `engine` | Must match engine's `name` attribute |
| `version` | Preset version (e.g., `"1.0"`) |
| `schema_version` | Event schema version (e.g., `"1.0.0"`) |
| `param_schema` | Dict of `ParamDef` objects |
| `default_params` | Default parameter values |
| `events` | List of event types this engine emits |

### Immutability Policy

Once a preset is released:

1. Mark it `FROZEN` with a comment
2. Never mutate its fields
3. Clone to a new slug for changes (e.g., `ny-am-orb-v1` → `ny-am-orb-v1.1`)
4. The frozen preset's tests must continue to pass forever

## 6. Golden Fixtures

Every engine must have at least one golden fixture test.

### Fixture Format

JSON file in `tests/golden/fixtures/`:

```json
{
  "name": "my_engine_golden_run",
  "description": "What this fixture captures",
  "engine_params": {},
  "session_df_args": {},
  "bar_overrides": [],
  "expected_summary": {"num_trades": 1, "win_rate": 1.0},
  "expected_events": [],
  "expected_trades": []
}
```

### Test Requirements

- Compare events field-by-field with `pytest.approx(abs=1e-6)` for floats
- Assert event count, type sequence, and all key values
- Assert trade count, fields, and summary metrics
- Run `validate_events()` on the result
- Include doc comment: "If this fixture changes, you changed [engine] semantics."

### Coverage Targets

At minimum, cover:

| Scenario | Rationale |
|----------|-----------|
| Long-side winner | Primary happy path |
| Short-side winner | Asymmetric code paths |
| Losser (stop hit or EOD close) | Exit logic |
| Alternate confirm mode | Mode-specific branches |

Reference: `tests/golden/test_orb_golden.py` (parametrized across 3 fixtures)

## 7. Live-Data Smoke Tests

Optional but strongly recommended. Use `@pytest.mark.slow` (auto-skipped in CI).

### Requirements

- Skip gracefully if data file is missing (`pytest.skip`)
- Validate events pass contract
- Check timing/range plausibility (engine-specific)
- Verify idempotent rerun (same input → same events)

Reference: `tests/golden/test_orb_live_smoke.py`

## 8. Consumer Checklist

Before shipping a new engine, verify integration with all consumers:

| Consumer | What to Check |
|----------|--------------|
| `BacktestRunner._resolve_engine()` | Engine class registered |
| `StrategyPreset` in `presets.py` | Preset defined with correct engine name |
| `ReplayPanel` | Events render in timeline (or add engine-specific panel) |
| `process_score` | Events work with `_score_rule_adherence()` |
| `loss_attribution` | Trade records compatible |
| `run_events` repository | Events persist and query correctly |
| Golden fixtures | At least one fixture per scenario |
| `to_config_snapshot()` | Version fields propagate to strategy_versions table |

## 9. File Layout

Follow this directory structure:

```
app/services/backtest/engines/my_engine/
├── __init__.py          # exports MyEngine
├── engine.py            # MyEngine class implementing BacktestEngine
├── contracts.py         # event schema, validate_events()
└── types.py             # Enums, dataclasses, params
```

Register in `app/services/backtest/engines/__init__.py` and in
`BacktestRunner._resolve_engine()`.

## Reference Implementation

ORB v1 is the canonical example:

- Engine: `app/services/backtest/engines/orb/engine.py`
- Contract: `app/services/backtest/engines/orb/contracts.py`
- Types: `app/services/backtest/engines/orb/types.py`
- Preset: `app/strategies/presets.py` (`NY_AM_ORB_V1`)
- Golden tests: `tests/golden/test_orb_golden.py`
- Live smoke: `tests/golden/test_orb_live_smoke.py`
- Feature doc: `docs/features/orb-engine.md`
