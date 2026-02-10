"""Golden fixture regression tests for ORB v1 engine.

If any fixture changes, you changed ORB semantics. Either the change
is intentional (update the fixture) or you introduced a regression.

Fixtures cover three scenarios:
  - orb_v1_golden_run:    close-beyond long, target hit (winner)
  - orb_v1_golden_short:  close-beyond short, target hit (winner)
  - orb_v1_golden_retest: retest-confirm long (loser)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.engines.orb.engine import ORBEngine
from app.services.backtest.engines.orb.contracts import validate_events

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURE_FILES = [
    "orb_v1_golden_run.json",
    "orb_v1_golden_short.json",
    "orb_v1_golden_retest.json",
]


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
    session_start_str: str = "2024-01-02 14:30",
    freq: str = "1min",
) -> pd.DataFrame:
    """Deterministic synthetic session builder (copied from engine tests)."""
    idx = pd.date_range(session_start_str, periods=n_bars, freq=freq, tz="UTC")
    np.random.seed(42)

    opens = np.full(n_bars, base_price, dtype=float)
    highs = np.full(n_bars, base_price + 0.3, dtype=float)
    lows = np.full(n_bars, base_price - 0.3, dtype=float)
    closes = np.full(n_bars, base_price, dtype=float)

    if or_high is not None and or_low is not None:
        highs[5] = or_high
        opens[5] = or_high - 0.2
        closes[5] = or_high - 0.1
        lows[10] = or_low
        opens[10] = or_low + 0.2
        closes[10] = or_low + 0.1
        for i in range(30):
            if i not in (5, 10):
                highs[i] = min(highs[i], or_high - 0.05)
                lows[i] = max(lows[i], or_low + 0.05)
                closes[i] = (or_high + or_low) / 2

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
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
        index=idx,
    )


def _load_and_run(fixture_data: dict) -> object:
    """Build DF from fixture args, apply bar overrides, run engine."""
    args = dict(fixture_data["session_df_args"])
    df = _build_session_df(**args)

    # Apply bar overrides (used by retest fixture)
    for override in fixture_data.get("bar_overrides", []):
        bar = override["bar"]
        for col, val in override.items():
            if col != "bar":
                df.iloc[bar, df.columns.get_loc(col)] = val

    engine = ORBEngine()
    return engine.run(
        ohlcv_df=df,
        config={},
        params=fixture_data["engine_params"],
        initial_cash=10000,
    )


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


def _fixture_id(path: str) -> str:
    return path.replace("orb_v1_golden_", "").replace(".json", "")


@pytest.fixture(
    scope="module",
    params=FIXTURE_FILES,
    ids=[_fixture_id(f) for f in FIXTURE_FILES],
)
def golden_pair(request):
    """Load fixture JSON and run engine, return (fixture_data, result)."""
    path = FIXTURES_DIR / request.param
    with open(path) as f:
        data = json.load(f)
    result = _load_and_run(data)
    return data, result


# ---------------------------------------------------------------------------
# Tests — run once per fixture
# ---------------------------------------------------------------------------


class TestGoldenEventCount:
    """Event count must match the frozen fixture exactly."""

    def test_event_count_matches(self, golden_pair):
        data, result = golden_pair
        assert len(result.events) == len(data["expected_events"]), (
            f"[{data['name']}] event count: "
            f"{len(result.events)} != {len(data['expected_events'])}"
        )


class TestGoldenEventFields:
    """Every event field must match the fixture within float tolerance."""

    def test_events_match_field_by_field(self, golden_pair):
        data, result = golden_pair
        for i, (actual, expected) in enumerate(
            zip(result.events, data["expected_events"])
        ):
            assert set(actual.keys()) == set(expected.keys()), (
                f"[{data['name']}] Event {i} key mismatch: "
                f"extra={set(actual.keys()) - set(expected.keys())}, "
                f"missing={set(expected.keys()) - set(actual.keys())}"
            )
            for key in expected:
                exp_val = expected[key]
                act_val = actual[key]
                if isinstance(exp_val, float):
                    assert act_val == pytest.approx(exp_val, abs=1e-6), (
                        f"[{data['name']}] Event {i} ({expected['type']}).{key}: "
                        f"{act_val} != {exp_val}"
                    )
                else:
                    assert act_val == exp_val, (
                        f"[{data['name']}] Event {i} ({expected['type']}).{key}: "
                        f"{act_val!r} != {exp_val!r}"
                    )


class TestGoldenEventTypes:
    """Event type sequence must match exactly."""

    def test_event_type_sequence(self, golden_pair):
        data, result = golden_pair
        actual_types = [e["type"] for e in result.events]
        expected_types = [e["type"] for e in data["expected_events"]]
        assert actual_types == expected_types


class TestGoldenTradeCount:
    """Trade count and summary must match."""

    def test_trade_count(self, golden_pair):
        data, result = golden_pair
        assert result.num_trades == data["expected_summary"]["num_trades"]

    def test_win_rate(self, golden_pair):
        data, result = golden_pair
        assert result.win_rate == pytest.approx(
            data["expected_summary"]["win_rate"], abs=1e-6
        )


class TestGoldenTradeFields:
    """Trade records must match the fixture within float tolerance."""

    def test_trades_match_field_by_field(self, golden_pair):
        data, result = golden_pair
        assert len(result.trades) == len(data["expected_trades"])
        for i, (actual, expected) in enumerate(
            zip(result.trades, data["expected_trades"])
        ):
            for key in expected:
                exp_val = expected[key]
                act_val = actual[key]
                if isinstance(exp_val, float):
                    assert act_val == pytest.approx(
                        exp_val, abs=1e-6
                    ), f"[{data['name']}] Trade {i}.{key}: {act_val} != {exp_val}"
                else:
                    assert (
                        act_val == exp_val
                    ), f"[{data['name']}] Trade {i}.{key}: {act_val!r} != {exp_val!r}"


class TestGoldenContractValid:
    """Golden events must pass contract validation."""

    def test_contract_passes(self, golden_pair):
        data, result = golden_pair
        errors = validate_events(result.events)
        assert errors == [], f"[{data['name']}] Contract violations: {errors}"
