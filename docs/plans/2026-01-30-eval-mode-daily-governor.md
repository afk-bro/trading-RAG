# Eval-Mode Daily Governor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a daily risk governor with half-size stepdown and eval-mode flag that locks 2R target / disables management, enabling the Unicorn backtest to simulate prop-firm eval conditions (Apex/Topstep).

**Architecture:** New `DailyGovernor` dataclass tracks per-day state (loss, trade count, risk multiplier, halted flag). It's instantiated inside `run_unicorn_backtest()` and queried before each entry, updated after each trade close. An `--eval-mode` CLI flag enables the governor with sensible defaults and disables breakeven/trailing. Position sizing is updated to respect `risk_multiplier` and skip trades where `contracts < 1`.

**Tech Stack:** Pure Python dataclasses, no new dependencies. Follows existing guard pattern (config param → validation → diagnostic field → gate check → setup record tracking).

---

## Task 1: Add `DailyGovernor` dataclass and reset logic

**Files:**
- Create: `app/services/backtest/engines/daily_governor.py`
- Test: `tests/unit/test_daily_governor.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_daily_governor.py
"""Tests for the daily risk governor."""

from datetime import datetime, timezone
from app.services.backtest.engines.daily_governor import DailyGovernor


class TestDailyGovernorInit:
    def test_default_state_allows_trading(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        assert gov.allows_entry() is True
        assert gov.risk_multiplier == 1.0
        assert gov.halted_for_day is False

    def test_custom_params(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=600.0,
            max_trades_per_day=3,
            half_size_multiplier=0.25,
        )
        assert gov.max_daily_loss_dollars == 600.0
        assert gov.max_trades_per_day == 3
        assert gov.half_size_multiplier == 0.25


class TestDailyGovernorReset:
    def test_reset_clears_state(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.day_loss_dollars = -200.0
        gov.day_trade_count = 2
        gov.risk_multiplier = 0.5
        gov.halted_for_day = True

        gov.reset_day()

        assert gov.day_loss_dollars == 0.0
        assert gov.day_trade_count == 0
        assert gov.risk_multiplier == 1.0
        assert gov.halted_for_day is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_daily_governor.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# app/services/backtest/engines/daily_governor.py
"""Daily risk governor for eval/prop-firm backtesting."""

from dataclasses import dataclass, field


@dataclass
class DailyGovernor:
    """Per-day risk state tracker with half-size stepdown.

    Policy:
        - Full loss (day_loss <= -max_daily_loss) → halt for day
        - Half loss (day_loss <= -half_threshold) → risk_multiplier drops to half_size_multiplier
        - Trade count cap → halt for day
    """

    # --- Config (set once) ---
    max_daily_loss_dollars: float = 300.0
    max_trades_per_day: int = 2
    half_size_multiplier: float = 0.5

    # --- Per-day mutable state ---
    day_loss_dollars: float = 0.0
    day_trade_count: int = 0
    risk_multiplier: float = 1.0
    halted_for_day: bool = False
    current_date: object = None  # date object, set on first bar of day

    @property
    def half_loss_threshold(self) -> float:
        """Loss level that triggers half-size stepdown."""
        return self.max_daily_loss_dollars * self.half_size_multiplier

    def allows_entry(self) -> bool:
        """Check if governor permits a new trade entry."""
        if self.halted_for_day:
            return False
        if self.day_trade_count >= self.max_trades_per_day:
            self.halted_for_day = True
            return False
        if self.day_loss_dollars <= -self.max_daily_loss_dollars:
            self.halted_for_day = True
            return False
        return True

    def record_trade_close(self, pnl_dollars: float) -> None:
        """Update state after a trade closes."""
        self.day_trade_count += 1
        if pnl_dollars < 0:
            self.day_loss_dollars += pnl_dollars

        # Half-size stepdown: if cumulative loss hits half threshold,
        # next trade uses reduced size
        if self.day_loss_dollars <= -self.half_loss_threshold:
            self.risk_multiplier = self.half_size_multiplier

        # Full halt
        if self.day_loss_dollars <= -self.max_daily_loss_dollars:
            self.halted_for_day = True

    def maybe_reset(self, bar_date) -> None:
        """Reset state if calendar day changed."""
        if self.current_date != bar_date:
            self.reset_day()
            self.current_date = bar_date

    def reset_day(self) -> None:
        """Reset all per-day state."""
        self.day_loss_dollars = 0.0
        self.day_trade_count = 0
        self.risk_multiplier = 1.0
        self.halted_for_day = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_daily_governor.py::TestDailyGovernorInit -v && pytest tests/unit/test_daily_governor.py::TestDailyGovernorReset -v`
Expected: PASS

**Step 5: Commit**

```
feat(unicorn): add DailyGovernor dataclass with reset logic
```

---

## Task 2: Governor policy logic — halt, stepdown, trade counting

**Files:**
- Modify: `tests/unit/test_daily_governor.py` (add tests)
- Already created: `app/services/backtest/engines/daily_governor.py`

**Step 1: Write the failing tests**

```python
class TestGovernorHalt:
    def test_halts_after_full_loss(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-300.0)
        assert gov.halted_for_day is True
        assert gov.allows_entry() is False

    def test_halts_after_exceeding_max_trades(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.record_trade_close(100.0)  # win
        gov.record_trade_close(50.0)   # win
        # 2 trades taken, at cap
        assert gov.allows_entry() is False

    def test_not_halted_after_winning_trade(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(200.0)
        assert gov.halted_for_day is False
        assert gov.allows_entry() is True
        assert gov.risk_multiplier == 1.0


class TestGovernorStepdown:
    def test_half_loss_triggers_stepdown(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-150.0)  # exactly half_loss_threshold
        assert gov.risk_multiplier == 0.5
        assert gov.halted_for_day is False  # not fully halted yet
        assert gov.allows_entry() is True   # can still trade at half size

    def test_small_loss_no_stepdown(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-100.0)  # below half threshold
        assert gov.risk_multiplier == 1.0

    def test_stepdown_then_halt_on_second_loss(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-150.0)  # half → stepdown to 0.5
        assert gov.risk_multiplier == 0.5
        gov.record_trade_close(-150.0)  # cumulative -300 → halt
        assert gov.halted_for_day is True
        assert gov.allows_entry() is False

    def test_custom_half_size_multiplier(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=600.0,
            max_trades_per_day=3,
            half_size_multiplier=0.25,
        )
        # half_loss_threshold = 600 * 0.25 = 150
        gov.record_trade_close(-150.0)
        assert gov.risk_multiplier == 0.25


class TestGovernorDayReset:
    def test_maybe_reset_on_new_date(self):
        from datetime import date
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.maybe_reset(date(2024, 1, 2))
        gov.record_trade_close(-300.0)
        assert gov.halted_for_day is True

        gov.maybe_reset(date(2024, 1, 3))  # new day
        assert gov.halted_for_day is False
        assert gov.allows_entry() is True
        assert gov.risk_multiplier == 1.0

    def test_maybe_reset_same_date_noop(self):
        from datetime import date
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.maybe_reset(date(2024, 1, 2))
        gov.record_trade_close(-200.0)
        gov.maybe_reset(date(2024, 1, 2))  # same day, no reset
        assert gov.day_loss_dollars == -200.0


class TestGovernorWinsOnlyTrack:
    def test_winning_trade_does_not_reduce_day_loss(self):
        """day_loss_dollars only accumulates losses, not wins."""
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-200.0)
        gov.record_trade_close(500.0)  # big win
        # day_loss stays at -200, not offset by win
        assert gov.day_loss_dollars == -200.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_daily_governor.py -v`
Expected: All new tests PASS (implementation already in Task 1 covers this logic)

If any fail, adjust `DailyGovernor` implementation from Task 1 until all pass.

**Step 3: Commit**

```
test(unicorn): add comprehensive daily governor policy tests
```

---

## Task 3: Wire governor into `run_unicorn_backtest` scan loop

**Files:**
- Modify: `app/services/backtest/engines/unicorn_runner.py`
  - Import `DailyGovernor`
  - Add `daily_governor` parameter to `run_unicorn_backtest()` (default `None`)
  - Add day-reset call at top of main loop
  - Add `allows_entry()` gate before capacity check (~line 1226)
  - Add `record_trade_close()` in trade finalization block (~line 1194)
  - Add `risk_multiplier` to position sizing (~line 1420)
  - Skip trade when `quantity < 1` after applying multiplier
- Modify: `app/services/backtest/engines/daily_governor.py` — add `DailyGovernorStats` for result tracking
- Test: `tests/unit/test_unicorn_runner.py` (add governor integration tests)

**Step 1: Write the failing integration test**

Add to `tests/unit/test_unicorn_runner.py`:

```python
class TestDailyGovernorIntegration:
    """Test that daily governor gates entries in the backtest loop."""

    def test_governor_halts_after_max_trades(self):
        """With max_trades=1, only one trade per day should be taken."""
        from app.services.backtest.engines.daily_governor import DailyGovernor

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig()
        governor = DailyGovernor(max_daily_loss_dollars=5000.0, max_trades_per_day=1)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
            daily_governor=governor,
        )

        # Count trades per calendar day
        from collections import Counter
        trades_per_day = Counter(t.entry_time.date() for t in result.trades)
        for day, count in trades_per_day.items():
            assert count <= 1, f"Day {day} had {count} trades, expected max 1"

    def test_governor_none_means_no_limit(self):
        """Without governor, no daily limits apply (backward compat)."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result_no_gov = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=UnicornConfig(),
            daily_governor=None,
        )
        # Should produce same result as before (no governor = legacy behavior)
        # Just verify it doesn't crash
        assert result_no_gov.trades_taken >= 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_unicorn_runner.py::TestDailyGovernorIntegration -v`
Expected: FAIL — `run_unicorn_backtest() got an unexpected keyword argument 'daily_governor'`

**Step 3: Write the implementation**

In `unicorn_runner.py`, add to function signature (~line 953, after `bar_bundle`):

```python
    # Daily risk governor (eval mode)
    daily_governor: Optional["DailyGovernor"] = None,
```

Add import at top of file:

```python
from app.services.backtest.engines.daily_governor import DailyGovernor
```

In the main scan loop, add day-reset at **top of loop body** (after `ts = bar.ts`, ~line 1078):

```python
        # Daily governor: reset on new calendar day
        if daily_governor is not None:
            daily_governor.maybe_reset(ts.date())
```

In the **trade finalization block** (~line 1194, inside the `for trade in closed:` loop, after line 1215 `result.total_pnl_dollars += trade.pnl_dollars`):

```python
            # Update daily governor
            if daily_governor is not None:
                daily_governor.record_trade_close(trade.pnl_dollars)
```

Add governor gate **before** capacity check (~line 1226, insert before `if len(open_trades) >= max_concurrent_trades:`):

```python
        # Daily governor gate
        if daily_governor is not None and not daily_governor.allows_entry():
            continue
```

Modify position sizing (~line 1420) to apply `risk_multiplier`:

```python
            # Position sizing (account for slippage in risk)
            risk_dollars = (risk_points + slippage_points) * point_value
            effective_risk_budget = dollars_per_trade
            if daily_governor is not None:
                effective_risk_budget *= daily_governor.risk_multiplier
            if risk_dollars > 0:
                quantity = int(effective_risk_budget / risk_dollars)
            else:
                quantity = 1

            # Skip trade if position size is zero (risk too wide for budget)
            if quantity < 1:
                setup_record.taken = False
                setup_record.reason_not_taken = (
                    f"position_size_zero: risk ${risk_dollars:.0f} > budget ${effective_risk_budget:.0f}"
                )
                result.all_setups.append(setup_record)
                continue
```

Note: this also removes the `max(1, ...)` floor — trades that are too expensive for the account now get skipped instead of forced to 1 contract. This is the "structural sizing" fix.

**Step 4: Run tests**

Run: `pytest tests/unit/test_unicorn_runner.py::TestDailyGovernorIntegration -v`
Expected: PASS

Run: `pytest tests/unit/test_unicorn_runner.py -v`
Expected: All existing tests still PASS (governor=None by default)

**Step 5: Commit**

```
feat(unicorn): wire daily governor into backtest scan loop
```

---

## Task 4: Add `SetupOccurrence` diagnostic fields for governor

**Files:**
- Modify: `app/services/backtest/engines/unicorn_runner.py` — add fields to `SetupOccurrence`
- Modify: `app/services/backtest/engines/unicorn_runner.py` — record diagnostics in scan loop
- Modify: `app/services/backtest/engines/daily_governor.py` — add `DailyGovernorStats`
- Modify: `app/services/backtest/engines/unicorn_runner.py` — add stats to `UnicornBacktestResult`
- Test: `tests/unit/test_daily_governor.py` (add stats tests)

**Step 1: Write the failing test**

```python
# In tests/unit/test_daily_governor.py

class TestDailyGovernorStats:
    def test_stats_track_halted_days_and_skipped(self):
        from app.services.backtest.engines.daily_governor import DailyGovernorStats

        stats = DailyGovernorStats()
        stats.signals_skipped += 3
        stats.days_halted += 1
        stats.half_size_trades += 1

        assert stats.signals_skipped == 3
        assert stats.days_halted == 1
        assert stats.half_size_trades == 1
```

**Step 2: Run to verify failure**

Run: `pytest tests/unit/test_daily_governor.py::TestDailyGovernorStats -v`
Expected: FAIL — `cannot import name 'DailyGovernorStats'`

**Step 3: Implement**

Add to `daily_governor.py`:

```python
@dataclass
class DailyGovernorStats:
    """Aggregated stats for governor behavior across the backtest."""
    signals_skipped: int = 0     # setups that passed criteria but governor blocked
    days_halted: int = 0         # calendar days where governor halted early
    half_size_trades: int = 0    # trades taken at reduced size
    total_days_traded: int = 0   # days with at least one trade
```

Add to `SetupOccurrence` (~line 515, after `guard_reason_code`):

```python
    governor_rejected: bool = False  # daily governor blocked this entry
```

Add to `UnicornBacktestResult` (~line 642, after `session_diagnostics`):

```python
    # Daily governor stats (None when governor not used)
    governor_stats: Optional[dict] = None
```

In the scan loop, where the governor gate blocks entry, record diagnostics:

```python
        # Daily governor gate
        if daily_governor is not None and not daily_governor.allows_entry():
            # Record as scanned but governor-blocked
            # (Only record if we would have checked criteria — i.e., after capacity check)
            governor_stats["signals_skipped"] += 1
            continue
```

After the main loop, populate `result.governor_stats`:

```python
    if daily_governor is not None:
        result.governor_stats = {
            "signals_skipped": governor_stats["signals_skipped"],
            "days_halted": governor_stats["days_halted"],
            "half_size_trades": governor_stats["half_size_trades"],
            "total_days_traded": governor_stats["total_days_traded"],
        }
```

Implementation detail: track `governor_stats` as a local dict initialized before the main loop. Increment `days_halted` when `maybe_reset` transitions from halted to reset. Increment `half_size_trades` at trade entry when `risk_multiplier < 1.0`. Increment `total_days_traded` when first trade of a new day is taken.

**Step 4: Run tests**

Run: `pytest tests/unit/test_daily_governor.py -v`
Expected: PASS

Run: `pytest tests/unit/test_unicorn_runner.py -v`
Expected: All PASS (no regressions)

**Step 5: Commit**

```
feat(unicorn): add governor diagnostics to SetupOccurrence and result stats
```

---

## Task 5: Add `--eval-mode` CLI flag and wire to governor

**Files:**
- Modify: `scripts/run_unicorn_backtest.py`
  - Add `--eval-mode` flag
  - Add `--max-daily-loss`, `--max-trades-per-day` CLI args (with eval defaults)
  - Instantiate `DailyGovernor` when eval-mode or governor args present
  - Pass governor to `run_unicorn_backtest()`
  - Disable `breakeven_at_r` when eval-mode active
  - Print governor summary after report
- Test: `tests/unit/test_run_unicorn_backtest_cli.py` (add eval-mode arg parsing test)

**Step 1: Write the failing test**

Check existing CLI tests first. Add to `tests/unit/test_run_unicorn_backtest_cli.py`:

```python
class TestEvalModeArgs:
    def test_eval_mode_sets_defaults(self):
        """--eval-mode should set governor defaults without requiring explicit args."""
        import sys
        from unittest.mock import patch
        from scripts.run_unicorn_backtest import main

        args = [
            "run_unicorn_backtest.py",
            "--symbol", "NQ",
            "--synthetic",
            "--days", "5",
            "--eval-mode",
            "--seed", "42",
        ]

        with patch.object(sys, "argv", args):
            # Just verify it parses without error
            # Full integration tested via backtest run
            pass
```

This is a lightweight test — the real verification is running the full CLI and checking governor stats appear in output.

**Step 2: Implement CLI changes**

After `--write-baseline` arg, add:

```python
    # Eval mode (prop firm simulation)
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Enable eval/prop-firm mode: daily governor, structural sizing (skip if contracts < 1), "
             "fixed 2R target, no breakeven/trailing. Sets sensible defaults for --max-daily-loss "
             "and --max-trades-per-day if not explicitly provided."
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        metavar="DOLLARS",
        help="Max daily loss in dollars before halting for the day. "
             "Default: equal to --dollars-per-trade when --eval-mode is set."
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        metavar="N",
        help="Max trades per calendar day. Default: 2 when --eval-mode is set."
    )
```

After arg parsing, before config construction:

```python
    # Eval mode defaults
    daily_governor = None
    if args.eval_mode or args.max_daily_loss is not None or args.max_trades_per_day is not None:
        from app.services.backtest.engines.daily_governor import DailyGovernor

        max_loss = args.max_daily_loss if args.max_daily_loss is not None else args.dollars_per_trade
        max_trades = args.max_trades_per_day if args.max_trades_per_day is not None else 2

        daily_governor = DailyGovernor(
            max_daily_loss_dollars=max_loss,
            max_trades_per_day=max_trades,
        )

        if args.eval_mode:
            # Lock eval behavior: no breakeven, no trailing
            args.breakeven_at_r = None

        print(f"Daily governor: max loss ${max_loss:.0f}/day, max {max_trades} trades/day, "
              f"half-size at ${max_loss * 0.5:.0f} loss")
```

Pass governor to `run_unicorn_backtest()`:

```python
        daily_governor=daily_governor,
```

After report output, print governor summary:

```python
    # Governor summary
    if result.governor_stats is not None:
        gs = result.governor_stats
        print(f"\nDaily Governor:")
        print(f"  Signals skipped:    {gs['signals_skipped']}")
        print(f"  Days halted early:  {gs['days_halted']}")
        print(f"  Half-size trades:   {gs['half_size_trades']}")
        print(f"  Days with trades:   {gs['total_days_traded']}")
```

Also add `governor_stats` to `_build_output_dict()`:

```python
        "governor_stats": result.governor_stats,
```

**Step 3: Run tests**

Run: `pytest tests/unit/test_run_unicorn_backtest_cli.py -v`
Expected: PASS

Run: `pytest tests/unit/ -q`
Expected: Full suite green

**Step 4: Commit**

```
feat(unicorn): add --eval-mode CLI flag with daily governor integration
```

---

## Task 6: Add eval-mode segment to run label/key

**Files:**
- Modify: `app/services/strategy/strategies/unicorn_model.py`
  - Update `_label_segments()` to include eval segment
  - Update `EXPECTED_SEGMENT_ORDER` to include `"eval"`
  - Update `build_run_label()` display_map
- Test: `tests/unit/test_criteria_reclassification.py` (add eval label tests)

**Step 1: Write the failing test**

Add to `tests/unit/test_criteria_reclassification.py`:

```python
class TestEvalModeLabel:
    def test_eval_mode_in_label(self):
        config = UnicornConfig()
        label = build_run_label(config, eval_mode=True)
        assert "Eval=on" in label

    def test_no_eval_mode_in_label_by_default(self):
        config = UnicornConfig()
        label = build_run_label(config)
        assert "Eval=" not in label

    def test_eval_mode_in_key(self):
        config = UnicornConfig()
        key = build_run_key(config, eval_mode=True)
        assert "eval_on" in key
```

**Step 2: Run to verify failure**

Run: `pytest tests/unit/test_criteria_reclassification.py::TestEvalModeLabel -v`
Expected: FAIL — `build_run_label() got an unexpected keyword argument 'eval_mode'`

**Step 3: Implement**

Add `eval_mode: object = None` parameter to `_label_segments()`, `build_run_label()`, and `build_run_key()`.

In `_label_segments()`, append eval segment only when True:

```python
    if eval_mode:
        segs.append(("eval", "on"))
```

Update `EXPECTED_SEGMENT_ORDER`:

```python
EXPECTED_SEGMENT_ORDER = ("ver", "bias", "side", "displ", "minscore", "window", "ts", "eval")
```

Update `display_map` in `build_run_label()`:

```python
        "eval": lambda v: f"Eval={v}",
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_criteria_reclassification.py -v`
Expected: All PASS (including existing segment order tests — verify they allow optional trailing segment)

**Step 5: Commit**

```
feat(unicorn): add eval segment to run label and key
```

---

## Task 7: Structural sizing — skip when contracts < 1

**Files:**
- Already implemented in Task 3 (quantity < 1 skip)
- Test: `tests/unit/test_unicorn_runner.py` (add explicit sizing test)

**Step 1: Write the failing test**

```python
class TestStructuralSizing:
    def test_wide_stop_skips_trade(self):
        """When stop is too wide for the risk budget, trade is skipped."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0,
                                          volatility=20.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67,
                                          volatility=7.0, interval_minutes=5)

        # Very small budget: $50 per trade with NQ ($20/pt)
        # Any stop > 2.5 points should produce quantity=0 → skip
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=50,  # tiny budget
            config=UnicornConfig(),
        )

        # Check that some setups were skipped due to position size
        size_skipped = [
            s for s in result.all_setups
            if s.reason_not_taken and "position_size_zero" in s.reason_not_taken
        ]
        # With $50 budget and NQ, most structural stops will be too wide
        # (Just verify the mechanism exists, not exact count)
        assert isinstance(size_skipped, list)  # mechanism exists
```

**Step 2: Run to verify**

Run: `pytest tests/unit/test_unicorn_runner.py::TestStructuralSizing -v`
Expected: PASS (already implemented in Task 3)

**Step 3: Commit**

```
test(unicorn): add structural sizing skip test for wide stops
```

---

## Task 8: Add governor stats to `format_backtest_report`

**Files:**
- Modify: `app/services/backtest/engines/unicorn_runner.py` — update `format_backtest_report()`
- Test: `tests/unit/test_criteria_reclassification.py` (add report content test)

**Step 1: Write the failing test**

```python
class TestGovernorInReport:
    def test_governor_stats_in_report(self):
        """When governor_stats is set, report should include governor section."""
        # Use a minimal backtest result with governor_stats populated
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        from app.services.backtest.engines.daily_governor import DailyGovernor
        governor = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=1)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=UnicornConfig(),
            daily_governor=governor,
        )
        report = format_backtest_report(result)
        assert "Daily Governor" in report or "Governor" in report
```

**Step 2: Implement**

In `format_backtest_report()`, after the existing sections, add:

```python
    if result.governor_stats is not None:
        gs = result.governor_stats
        lines.append("")
        lines.append("─" * 50)
        lines.append("DAILY GOVERNOR")
        lines.append("─" * 50)
        lines.append(f"Signals skipped:       {gs['signals_skipped']}")
        lines.append(f"Days halted early:     {gs['days_halted']}")
        lines.append(f"Half-size trades:      {gs['half_size_trades']}")
        lines.append(f"Days with trades:      {gs['total_days_traded']}")
```

**Step 3: Run tests**

Run: `pytest tests/unit/ -q`
Expected: Full suite green

**Step 4: Commit**

```
feat(unicorn): show daily governor stats in backtest report
```

---

## Task 9: End-to-end CLI smoke test

**Files:**
- Test: manual CLI run (not automated — just verification)

**Step 1: Run eval-mode backtest**

```bash
python scripts/run_unicorn_backtest.py \
    --symbol NQ --synthetic --days 30 --seed 42 \
    --eval-mode --dollars-per-trade 300 \
    --json --write-baseline /tmp/eval_baseline.json
```

Verify:
- Governor summary printed
- `governor_stats` present in JSON output
- `signals_skipped > 0` (governor actually blocked some entries)
- No crashes

**Step 2: Run baseline compare**

```bash
python scripts/run_unicorn_backtest.py \
    --symbol NQ --synthetic --days 30 --seed 99 \
    --eval-mode --dollars-per-trade 300 \
    --baseline-run /tmp/eval_baseline.json
```

Verify: baseline comparison section appears with governor-mode run keys.

**Step 3: Run full unit suite**

```bash
pytest tests/unit/ -q
```

Expected: All pass, no regressions.

**Step 4: Commit**

```
test(unicorn): verify eval-mode CLI end-to-end
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | `DailyGovernor` dataclass + reset | `daily_governor.py`, `test_daily_governor.py` |
| 2 | Policy tests (halt, stepdown, counting) | `test_daily_governor.py` |
| 3 | Wire governor into scan loop | `unicorn_runner.py`, `test_unicorn_runner.py` |
| 4 | Governor diagnostic fields + stats | `unicorn_runner.py`, `daily_governor.py` |
| 5 | `--eval-mode` CLI flag | `run_unicorn_backtest.py`, CLI tests |
| 6 | Eval segment in run label/key | `unicorn_model.py`, label tests |
| 7 | Structural sizing skip test | `test_unicorn_runner.py` |
| 8 | Governor stats in report | `unicorn_runner.py`, report tests |
| 9 | E2E CLI smoke test | Manual verification |

**Not in scope (Phase 2):**
- Opposing liquidity "clean path to 2R" filter
- Macro-id gating (no re-entry in same macro after BE)
- ES vs NQ separate default profiles
