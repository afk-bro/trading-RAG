"""Unit tests for the ORB (Opening Range Breakout) backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.engines.orb.engine import ORBEngine
from app.services.backtest.engines.orb.contracts import (
    ALL_REQUIRED_KEYS,
    COMMON_REQUIRED_KEYS,
    ORB_EVENT_SCHEMA_VERSION,
    ORB_EVENT_TYPES,
    validate_events,
)
from app.services.backtest.engines.base import BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_session_df(
    n_bars: int = 150,
    base_price: float = 100.0,
    or_high: float | None = None,
    or_low: float | None = None,
    breakout_bar: int | None = None,
    breakout_dir: str = "long",
    target_bar: int | None = None,
    stop_bar: int | None = None,
    session_start_str: str = "2024-01-02 14:30",  # 09:30 ET = 14:30 UTC
    freq: str = "1min",
) -> pd.DataFrame:
    """Generate synthetic 1-min OHLCV for a single NY AM session.

    All timestamps are UTC. NY AM session (09:30-12:00 ET) = 14:30-17:00 UTC.
    Default OR range builds from bars 0-29 (first 30 minutes).
    """
    idx = pd.date_range(session_start_str, periods=n_bars, freq=freq, tz="UTC")
    np.random.seed(42)

    opens = np.full(n_bars, base_price, dtype=float)
    highs = np.full(n_bars, base_price + 0.3, dtype=float)
    lows = np.full(n_bars, base_price - 0.3, dtype=float)
    closes = np.full(n_bars, base_price, dtype=float)

    # Build OR: bars 0-29 establish the range
    if or_high is not None and or_low is not None:
        # Bar 5 sets the high
        highs[5] = or_high
        opens[5] = or_high - 0.2
        closes[5] = or_high - 0.1
        # Bar 10 sets the low
        lows[10] = or_low
        opens[10] = or_low + 0.2
        closes[10] = or_low + 0.1
        # Rest stay in range
        for i in range(30):
            if i not in (5, 10):
                highs[i] = min(highs[i], or_high - 0.05)
                lows[i] = max(lows[i], or_low + 0.05)
                closes[i] = (or_high + or_low) / 2

    # Breakout bar
    if breakout_bar is not None and or_high is not None and or_low is not None:
        if breakout_dir == "long":
            closes[breakout_bar] = or_high + 0.15
            highs[breakout_bar] = or_high + 0.20
            opens[breakout_bar] = or_high - 0.05
            lows[breakout_bar] = or_high - 0.10
        else:
            closes[breakout_bar] = or_low - 0.15
            lows[breakout_bar] = or_low - 0.20
            opens[breakout_bar] = or_low + 0.05
            highs[breakout_bar] = or_low + 0.10

    # Target bar (price hits target)
    if target_bar is not None and or_high is not None and or_low is not None:
        if breakout_dir == "long":
            entry_approx = or_high + 0.15
            risk = entry_approx - or_low
            target_price = entry_approx + risk * 1.5
            highs[target_bar] = target_price + 0.1
            closes[target_bar] = target_price
            opens[target_bar] = entry_approx + 0.1
            lows[target_bar] = entry_approx
        else:
            entry_approx = or_low - 0.15
            risk = or_high - entry_approx
            target_price = entry_approx - risk * 1.5
            lows[target_bar] = target_price - 0.1
            closes[target_bar] = target_price
            opens[target_bar] = entry_approx - 0.1
            highs[target_bar] = entry_approx

    # Stop bar (price hits stop)
    if stop_bar is not None and or_high is not None and or_low is not None:
        if breakout_dir == "long":
            lows[stop_bar] = or_low - 0.1
            closes[stop_bar] = or_low
            opens[stop_bar] = base_price
            highs[stop_bar] = base_price + 0.1
        else:
            highs[stop_bar] = or_high + 0.1
            closes[stop_bar] = or_high
            opens[stop_bar] = base_price
            lows[stop_bar] = base_price - 0.1

    volume = np.random.randint(100, 1000, n_bars)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
        index=idx,
    )
    return df


def _run_engine(
    df: pd.DataFrame,
    params: dict | None = None,
    config: dict | None = None,
    initial_cash: float = 10000,
) -> BacktestResult:
    """Convenience: run ORBEngine with defaults."""
    engine = ORBEngine()
    return engine.run(
        ohlcv_df=df,
        config=config or {},
        params=params or {"or_minutes": 30, "confirm_mode": "close-beyond"},
        initial_cash=initial_cash,
    )


def _events_of_type(result: BacktestResult, typ: str) -> list[dict]:
    return [e for e in result.events if e.get("type") == typ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoTradesOutsideSession:
    """Test 1: Bars entirely outside session produce no trades."""

    def test_no_trades_outside_session(self):
        # 07:00-09:00 ET = 12:00-14:00 UTC — before NY AM
        df = _build_session_df(
            n_bars=120,
            session_start_str="2024-01-02 12:00",  # 07:00 ET
        )
        result = _run_engine(df)
        assert result.num_trades == 0
        assert len(_events_of_type(result, "orb_range_locked")) == 0


class TestORBuildEmitsRangeUpdates:
    """Test 2: 30 bars from 09:30 emit orb_range_update events."""

    def test_or_build_emits_range_updates(self):
        df = _build_session_df(
            n_bars=35,
            or_high=101.0,
            or_low=99.0,
        )
        result = _run_engine(df)
        updates = _events_of_type(result, "orb_range_update")
        # Should have 30 updates (one per OR bar)
        assert len(updates) == 30
        # Last update should have correct range
        last = updates[-1]
        assert last["orb_high"] == 101.0
        assert last["orb_low"] == 99.0


class TestORRangeLocked30m:
    """Test 3: or_minutes=30, 1m bars → locked at bar 30."""

    def test_or_range_locked_at_correct_bar_30m(self):
        df = _build_session_df(n_bars=40, or_high=101.0, or_low=99.0)
        result = _run_engine(df, params={"or_minutes": 30})
        locked = _events_of_type(result, "orb_range_locked")
        assert len(locked) == 1
        evt = locked[0]
        assert evt["high"] == 101.0
        assert evt["low"] == 99.0
        assert evt["range"] == pytest.approx(2.0, abs=0.01)
        assert evt["or_bar_count_needed"] == 30
        # Lock happens at bar_index 29 (0-indexed, 30th bar)
        assert evt["bar_index"] == 29


class TestORRangeLocked15m:
    """Test 4: or_minutes=15 → locked at bar 15."""

    def test_or_range_locked_15m(self):
        df = _build_session_df(n_bars=40, or_high=101.0, or_low=99.0)
        result = _run_engine(df, params={"or_minutes": 15})
        locked = _events_of_type(result, "orb_range_locked")
        assert len(locked) == 1
        assert locked[0]["bar_index"] == 14  # 15th bar, 0-indexed
        assert locked[0]["or_bar_count_needed"] == 15


class TestORRangeLocked60m:
    """Test 5: or_minutes=60 → locked at bar 60."""

    def test_or_range_locked_60m(self):
        df = _build_session_df(n_bars=70, or_high=101.0, or_low=99.0)
        result = _run_engine(df, params={"or_minutes": 60})
        locked = _events_of_type(result, "orb_range_locked")
        assert len(locked) == 1
        assert locked[0]["bar_index"] == 59  # 60th bar
        assert locked[0]["or_bar_count_needed"] == 60


class TestCloseBeyondLongBreakout:
    """Test 6: Bar close > OR high → long breakout."""

    def test_close_beyond_long_breakout(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
        )
        result = _run_engine(df)
        setups = _events_of_type(result, "setup_valid")
        assert len(setups) >= 1
        assert setups[0]["direction"] == "long"
        entries = _events_of_type(result, "entry_signal")
        assert len(entries) >= 1
        assert entries[0]["side"] == "buy"


class TestCloseBeyondShortBreakout:
    """Test 7: Bar close < OR low → short breakout."""

    def test_close_beyond_short_breakout(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="short",
        )
        result = _run_engine(df)
        setups = _events_of_type(result, "setup_valid")
        assert len(setups) >= 1
        assert setups[0]["direction"] == "short"
        entries = _events_of_type(result, "entry_signal")
        assert len(entries) >= 1
        assert entries[0]["side"] == "sell"


class TestRetestConfirmMode:
    """Test 8: Retest confirm mode — initial break then retest confirm."""

    def test_retest_confirm_mode(self):
        or_high, or_low = 101.0, 99.0
        df = _build_session_df(
            n_bars=50,
            or_high=or_high,
            or_low=or_low,
        )
        # Bar 31: initial break above OR high
        df.iloc[31, df.columns.get_loc("High")] = or_high + 0.5
        df.iloc[31, df.columns.get_loc("Close")] = or_high + 0.3
        # Bar 33: retest — low touches OR high, close above
        df.iloc[33, df.columns.get_loc("Low")] = or_high - 0.05
        df.iloc[33, df.columns.get_loc("Close")] = or_high + 0.2
        df.iloc[33, df.columns.get_loc("High")] = or_high + 0.3

        result = _run_engine(
            df,
            params={
                "or_minutes": 30,
                "confirm_mode": "retest",
            },
        )
        setups = _events_of_type(result, "setup_valid")
        assert len(setups) >= 1
        assert setups[0]["direction"] == "long"
        assert setups[0]["confirm_mode"] == "retest"


class TestNoBreakoutNoTrade:
    """Test 9: Bars stay in range → 0 trades."""

    def test_no_breakout_no_trade(self):
        df = _build_session_df(
            n_bars=150,
            or_high=101.0,
            or_low=99.0,
            # No breakout bar set — prices stay flat at base_price=100
        )
        result = _run_engine(df)
        assert result.num_trades == 0


class TestStopHitOROpposite:
    """Test 10: Long entry, low <= OR low → stopped out."""

    def test_stop_hit_or_opposite(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
            stop_bar=35,
        )
        result = _run_engine(df)
        assert result.num_trades >= 1
        trade = result.trades[0]
        assert trade["exit_reason"] == "stop"
        assert trade["pnl"] < 0


class TestTargetHit:
    """Test 11: Long entry, high >= target → target hit."""

    def test_target_hit(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
            target_bar=40,
        )
        result = _run_engine(df)
        assert result.num_trades >= 1
        trade = result.trades[0]
        assert trade["exit_reason"] == "target"
        assert trade["pnl"] > 0


class TestFixedTicksStopMode:
    """Test 12: stop_mode=fixed-ticks → stop = entry - tick_value."""

    def test_fixed_ticks_stop_mode(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
        )
        result = _run_engine(
            df,
            params={
                "or_minutes": 30,
                "confirm_mode": "close-beyond",
                "stop_mode": "fixed-ticks",
                "fixed_ticks": 50,
                "target_r": 1.5,
            },
        )
        entries = _events_of_type(result, "entry_signal")
        assert len(entries) >= 1
        entry = entries[0]
        expected_stop = entry["price"] - 0.50  # 50 ticks * 0.01
        assert entry["stop"] == pytest.approx(expected_stop, abs=0.01)


class TestMaxTradesLockout:
    """Test 13: max_trades=1, second breakout after close → no second entry."""

    def test_max_trades_lockout(self):
        or_high, or_low = 101.0, 99.0
        df = _build_session_df(
            n_bars=80,
            or_high=or_high,
            or_low=or_low,
            breakout_bar=32,
            breakout_dir="long",
            stop_bar=35,
        )
        # Second breakout at bar 45
        df.iloc[45, df.columns.get_loc("Close")] = or_high + 0.2
        df.iloc[45, df.columns.get_loc("High")] = or_high + 0.3

        result = _run_engine(
            df,
            params={
                "or_minutes": 30,
                "max_trades": 1,
            },
        )
        # Only 1 trade should be recorded (stopped out), not 2
        assert result.num_trades == 1


class TestSessionEndForceClose:
    """Test 14: Position open at 12:00 ET → closed with session_close."""

    def test_session_end_force_close(self):
        # Build exactly 150 bars from 09:30 ET, covers 09:30-12:00
        df = _build_session_df(
            n_bars=155,  # extends slightly past 12:00 ET
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
            # No target/stop bar — position stays open until session end
        )
        result = _run_engine(df)
        assert result.num_trades >= 1
        # Find the trade — it should be closed by session
        trade = result.trades[0]
        assert trade["exit_reason"] in ("session_close", "eod")


class TestEventsCompatibleWithProcessScore:
    """Test 15: Events work with _score_rule_adherence()."""

    def test_events_compatible_with_process_score(self):
        from app.services.backtest.process_score import _score_rule_adherence

        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
        )
        result = _run_engine(df)
        score = _score_rule_adherence(result.events)
        # Should return a numeric score (setup_valid precedes entry_signal)
        assert score is not None
        assert isinstance(score, float)
        assert score >= 0


class TestBacktestResultMetrics:
    """Test 16: Full run populates return_pct, win_rate, etc."""

    def test_backtest_result_metrics(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
            target_bar=40,
        )
        result = _run_engine(df)
        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 1
        assert result.return_pct != 0
        assert result.win_rate >= 0
        assert len(result.equity_curve) > 0
        assert len(result.trades) >= 1


class TestEventsHaveRequiredKeys:
    """Test 17: Every event has type, bar_index, ts, session_date, phase."""

    def test_events_have_required_keys(self):
        df = _build_session_df(
            n_bars=50,
            or_high=101.0,
            or_low=99.0,
            breakout_bar=32,
            breakout_dir="long",
        )
        result = _run_engine(df)
        required = {"type", "bar_index", "ts", "session_date", "phase"}
        for evt in result.events:
            missing = required - set(evt.keys())
            assert not missing, f"Event {evt.get('type')} missing keys: {missing}"


class TestRunnerResolvesORBEngine:
    """Test 18: _resolve_engine returns ORBEngine for engine='orb'."""

    def test_runner_resolves_orb_engine(self):
        from unittest.mock import MagicMock

        from app.services.backtest.runner import BacktestRunner

        runner = BacktestRunner(kb_repo=MagicMock(), backtest_repo=MagicMock())
        engine = runner._resolve_engine({"engine": "orb"})
        assert isinstance(engine, ORBEngine)
        assert engine.name == "orb"


class TestInsufficientBars:
    """Engine raises clear error for < 3 bars."""

    def test_insufficient_bars(self):
        idx = pd.date_range("2024-01-02 14:30", periods=2, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100, 100],
                "High": [101, 101],
                "Low": [99, 99],
                "Close": [100, 100],
                "Volume": [100, 100],
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="Insufficient bars"):
            _run_engine(df)


class TestTzNaiveLocalized:
    """Tz-naive index gets localized to UTC automatically."""

    def test_tz_naive_localized(self):
        idx = pd.date_range("2024-01-02 14:30", periods=50, freq="1min")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Open": np.full(50, 100.0),
                "High": np.full(50, 100.3),
                "Low": np.full(50, 99.7),
                "Close": np.full(50, 100.0),
                "Volume": np.random.randint(100, 1000, 50),
            },
            index=idx,
        )
        # Should not raise
        result = _run_engine(df)
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Invariant tests (event semantic contracts for ReplayPanel + consumers)
# ---------------------------------------------------------------------------


def _full_session_result(
    breakout_dir: str = "long",
    confirm_mode: str = "close-beyond",
) -> BacktestResult:
    """Run a full single-session scenario that produces a trade."""
    df = _build_session_df(
        n_bars=50,
        or_high=101.0,
        or_low=99.0,
        breakout_bar=32,
        breakout_dir=breakout_dir,
        target_bar=40,
    )
    return _run_engine(
        df,
        params={
            "or_minutes": 30,
            "confirm_mode": confirm_mode,
        },
    )


class TestInvariantOneLockPerSession:
    """Invariant 1: Exactly one orb_range_locked per session,
    appearing after the last orb_range_update."""

    def test_single_session_one_lock(self):
        result = _full_session_result()
        locked = _events_of_type(result, "orb_range_locked")
        assert len(locked) == 1

    def test_lock_follows_last_update(self):
        result = _full_session_result()
        updates = _events_of_type(result, "orb_range_update")
        locked = _events_of_type(result, "orb_range_locked")
        assert len(locked) == 1
        last_update_idx = updates[-1]["bar_index"]
        lock_idx = locked[0]["bar_index"]
        # Lock should be on the same bar as the last update
        assert lock_idx == last_update_idx

    def test_multi_session_one_lock_each(self):
        """Two consecutive trading days each get exactly one lock."""
        # Day 1: 09:30-12:00 ET = 14:30-17:00 UTC
        # Day 2: next trading day
        day1 = pd.date_range("2024-01-02 14:30", periods=150, freq="1min", tz="UTC")
        day2 = pd.date_range("2024-01-03 14:30", periods=150, freq="1min", tz="UTC")
        idx = day1.append(day2)
        np.random.seed(42)
        n = len(idx)
        df = pd.DataFrame(
            {
                "Open": np.full(n, 100.0),
                "High": np.full(n, 100.3),
                "Low": np.full(n, 99.7),
                "Close": np.full(n, 100.0),
                "Volume": np.random.randint(100, 1000, n),
            },
            index=idx,
        )
        result = _run_engine(df)
        locked = _events_of_type(result, "orb_range_locked")
        assert len(locked) == 2
        # Different session dates
        dates = {e["session_date"] for e in locked}
        assert len(dates) == 2

    def test_lock_or_minutes_is_config_value(self):
        """orb_range_locked.or_minutes should be the config param, not bar count."""
        result = _full_session_result()
        locked = _events_of_type(result, "orb_range_locked")
        assert locked[0]["or_minutes"] == 30


class TestInvariantSetupEntryWindow:
    """Invariant 2: setup_valid.bar_index <= entry_signal.bar_index
    <= setup_valid.bar_index + 5 (process_score MAX_SETUP_WINDOW_BARS)."""

    def test_entry_within_window_close_beyond(self):
        result = _full_session_result(confirm_mode="close-beyond")
        setups = _events_of_type(result, "setup_valid")
        entries = _events_of_type(result, "entry_signal")
        assert len(setups) >= 1
        assert len(entries) >= 1
        for entry in entries:
            # Find the most recent setup_valid before this entry
            prior = [s for s in setups if s["bar_index"] <= entry["bar_index"]]
            assert prior, "entry_signal has no prior setup_valid"
            gap = entry["bar_index"] - prior[-1]["bar_index"]
            assert 0 <= gap <= 5, f"Gap {gap} exceeds process_score window"

    def test_entry_within_window_retest(self):
        """Retest mode: entry still within the 5-bar window."""
        or_high, or_low = 101.0, 99.0
        df = _build_session_df(n_bars=50, or_high=or_high, or_low=or_low)
        # Bar 31: initial break
        df.iloc[31, df.columns.get_loc("High")] = or_high + 0.5
        df.iloc[31, df.columns.get_loc("Close")] = or_high + 0.3
        # Bar 33: retest confirm
        df.iloc[33, df.columns.get_loc("Low")] = or_high - 0.05
        df.iloc[33, df.columns.get_loc("Close")] = or_high + 0.2
        df.iloc[33, df.columns.get_loc("High")] = or_high + 0.3

        result = _run_engine(
            df,
            params={
                "or_minutes": 30,
                "confirm_mode": "retest",
            },
        )
        setups = _events_of_type(result, "setup_valid")
        entries = _events_of_type(result, "entry_signal")
        if setups and entries:
            gap = entries[0]["bar_index"] - setups[0]["bar_index"]
            assert 0 <= gap <= 5


class TestInvariantSetupOnlyFromBreakoutScan:
    """Invariant 3: setup_valid only fires when engine is in BREAKOUT_SCAN."""

    def test_setup_only_in_breakout_scan(self):
        """The BREAKOUT_SCAN branch is the only code path that emits setup_valid.
        Verify no setup_valid events appear before orb_range_locked."""
        result = _full_session_result()
        locked = _events_of_type(result, "orb_range_locked")
        setups = _events_of_type(result, "setup_valid")
        if locked and setups:
            lock_bar = locked[0]["bar_index"]
            for s in setups:
                assert s["bar_index"] > lock_bar, (
                    f"setup_valid at bar {s['bar_index']} before "
                    f"orb_range_locked at bar {lock_bar}"
                )


class TestInvariantEntryRequiresPriorSetup:
    """Invariant 4: entry_signal never fires without a prior setup_valid
    for the same direction."""

    def test_entry_has_matching_setup(self):
        result = _full_session_result(breakout_dir="long")
        setups = _events_of_type(result, "setup_valid")
        entries = _events_of_type(result, "entry_signal")
        dir_map = {"buy": "long", "sell": "short"}
        for entry in entries:
            entry_dir = dir_map.get(entry["side"])
            prior = [
                s
                for s in setups
                if s["bar_index"] <= entry["bar_index"] and s["direction"] == entry_dir
            ]
            assert prior, (
                f"entry_signal(side={entry['side']}) at bar {entry['bar_index']} "
                f"has no matching setup_valid(direction={entry_dir})"
            )

    def test_entry_has_matching_setup_short(self):
        result = _full_session_result(breakout_dir="short")
        setups = _events_of_type(result, "setup_valid")
        entries = _events_of_type(result, "entry_signal")
        dir_map = {"buy": "long", "sell": "short"}
        for entry in entries:
            entry_dir = dir_map.get(entry["side"])
            prior = [
                s
                for s in setups
                if s["bar_index"] <= entry["bar_index"] and s["direction"] == entry_dir
            ]
            assert prior


class TestInvariantEntryPayloadComplete:
    """Invariant 5: entry_signal payload has everything the UI needs:
    price, stop, target, side, size."""

    def test_entry_payload_keys(self):
        result = _full_session_result()
        entries = _events_of_type(result, "entry_signal")
        assert len(entries) >= 1
        required_keys = {"price", "stop", "target", "side", "size", "risk_points"}
        for entry in entries:
            missing = required_keys - set(entry.keys())
            assert not missing, f"entry_signal missing UI keys: {missing}"

    def test_entry_payload_values_sensible(self):
        result = _full_session_result()
        entry = _events_of_type(result, "entry_signal")[0]
        assert entry["price"] > 0
        assert entry["stop"] > 0
        assert entry["target"] > 0
        assert entry["size"] >= 1.0
        assert entry["risk_points"] > 0
        assert entry["side"] in ("buy", "sell")
        # Target should be on correct side of entry
        if entry["side"] == "buy":
            assert entry["target"] > entry["price"]
            assert entry["stop"] < entry["price"]
        else:
            assert entry["target"] < entry["price"]
            assert entry["stop"] > entry["price"]


# ---------------------------------------------------------------------------
# Contract validation tests (contracts.py)
# ---------------------------------------------------------------------------


class TestContractSchemaVersion:
    """Schema version is well-formed semver string."""

    def test_version_format(self):
        parts = ORB_EVENT_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()


class TestContractDefinitions:
    """Contract constants are consistent and complete."""

    def test_all_event_types_in_registry(self):
        expected = {
            "orb_range_update",
            "orb_range_locked",
            "setup_valid",
            "entry_signal",
        }
        assert set(ORB_EVENT_TYPES.keys()) == expected

    def test_common_keys_are_frozenset(self):
        assert isinstance(COMMON_REQUIRED_KEYS, frozenset)

    def test_type_keys_are_frozensets(self):
        for keys in ORB_EVENT_TYPES.values():
            assert isinstance(keys, frozenset)

    def test_all_required_keys_is_union(self):
        manual_union = COMMON_REQUIRED_KEYS | frozenset().union(
            *ORB_EVENT_TYPES.values()
        )
        assert ALL_REQUIRED_KEYS == manual_union


class TestValidateEventsFunction:
    """validate_events() catches missing keys and unknown types."""

    def test_valid_orb_range_update(self):
        evt = {
            "type": "orb_range_update",
            "bar_index": 0,
            "ts": "2024-01-02T14:30:00+00:00",
            "session_date": "2024-01-02",
            "phase": "or_build",
            "orb_high": 101.0,
            "orb_low": 99.0,
            "or_minutes": 30,
            "or_start_index": 0,
        }
        assert validate_events([evt]) == []

    def test_missing_common_key(self):
        evt = {
            "type": "orb_range_update",
            # missing bar_index, ts, session_date, phase
            "orb_high": 101.0,
            "orb_low": 99.0,
            "or_minutes": 30,
            "or_start_index": 0,
        }
        errors = validate_events([evt])
        assert len(errors) >= 1
        assert "missing common keys" in errors[0]

    def test_missing_type_specific_key(self):
        evt = {
            "type": "entry_signal",
            "bar_index": 36,
            "ts": "2024-01-02T15:06:00+00:00",
            "session_date": "2024-01-02",
            "phase": "trade_mgmt",
            "side": "buy",
            "price": 101.0,
            # missing: stop, target, size, risk_points
        }
        errors = validate_events([evt])
        assert len(errors) == 1
        assert "missing payload keys" in errors[0]

    def test_unknown_event_type(self):
        evt = {
            "type": "bogus_type",
            "bar_index": 0,
            "ts": "2024-01-02T14:30:00+00:00",
            "session_date": "2024-01-02",
            "phase": "or_build",
        }
        errors = validate_events([evt])
        assert len(errors) == 1
        assert "unknown type" in errors[0]

    def test_empty_events_list_valid(self):
        assert validate_events([]) == []

    def test_extra_keys_ignored(self):
        """Extra keys beyond required are allowed (forward compat)."""
        evt = {
            "type": "orb_range_update",
            "bar_index": 0,
            "ts": "2024-01-02T14:30:00+00:00",
            "session_date": "2024-01-02",
            "phase": "or_build",
            "orb_high": 101.0,
            "orb_low": 99.0,
            "or_minutes": 30,
            "or_start_index": 0,
            "future_field": "anything",
        }
        assert validate_events([evt]) == []


class TestEngineEventsPassContract:
    """All events produced by the engine pass validate_events()."""

    def test_full_trade_events_valid(self):
        result = _full_session_result()
        errors = validate_events(result.events)
        assert errors == [], f"Contract violations: {errors}"

    def test_no_trade_events_valid(self):
        """Session with no breakout still produces valid events."""
        df = _build_session_df(n_bars=50, or_high=101.0, or_low=99.0)
        result = _run_engine(df)
        errors = validate_events(result.events)
        assert errors == [], f"Contract violations: {errors}"

    def test_short_trade_events_valid(self):
        result = _full_session_result(breakout_dir="short")
        errors = validate_events(result.events)
        assert errors == [], f"Contract violations: {errors}"

    def test_retest_events_valid(self):
        """Retest confirm mode events also pass contract."""
        or_high, or_low = 101.0, 99.0
        df = _build_session_df(n_bars=50, or_high=or_high, or_low=or_low)
        df.iloc[31, df.columns.get_loc("High")] = or_high + 0.5
        df.iloc[31, df.columns.get_loc("Close")] = or_high + 0.3
        df.iloc[33, df.columns.get_loc("Low")] = or_high - 0.05
        df.iloc[33, df.columns.get_loc("Close")] = or_high + 0.2
        df.iloc[33, df.columns.get_loc("High")] = or_high + 0.3
        result = _run_engine(df, params={"or_minutes": 30, "confirm_mode": "retest"})
        errors = validate_events(result.events)
        assert errors == [], f"Contract violations: {errors}"


class TestSchemaVersion:
    """Events carry schema_version for forward compatibility."""

    def test_all_events_have_schema_version(self):
        result = _full_session_result()
        for evt in result.events:
            assert "schema_version" in evt, (
                f"Event {evt.get('type')} at bar {evt.get('bar_index')} "
                f"missing schema_version"
            )

    def test_schema_version_matches_contract(self):
        result = _full_session_result()
        for evt in result.events:
            assert evt["schema_version"] == ORB_EVENT_SCHEMA_VERSION

    def test_schema_version_is_semver(self):
        result = _full_session_result()
        if result.events:
            ver = result.events[0]["schema_version"]
            parts = ver.split(".")
            assert len(parts) == 3
            for p in parts:
                assert p.isdigit()
