"""Live-data smoke tests for ORB v1 engine on NQ 1-minute bars.

Requires the GLBX CSV at docs/historical_data/. Skipped by default;
run with: pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.backtest.engines.orb.engine import ORBEngine
from app.services.backtest.engines.orb.contracts import validate_events

CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "historical_data"
    / "GLBX-20260129-JNB8PDSQ7C"
    / "glbx-mdp3-20210128-20260127.ohlcv-1m.csv"
)

pytestmark = pytest.mark.slow


def _load_nq_day(date_str: str, symbol: str) -> pd.DataFrame:
    """Load a single trading day of NQ 1m bars from the GLBX CSV.

    Filters by symbol and ET date, renames columns to OHLCV convention.
    """
    if not CSV_PATH.exists():
        pytest.skip(f"GLBX CSV not found at {CSV_PATH}")

    # Read only needed columns to reduce memory
    df = pd.read_csv(
        CSV_PATH,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={"symbol": "str"},
    )
    df = df[df["symbol"] == symbol].copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

    # Filter to target date in ET
    df["et"] = df["ts_event"].dt.tz_convert("America/New_York")
    df = df[df["et"].dt.date.astype(str) == date_str].copy()

    if df.empty:
        pytest.skip(f"No data for {symbol} on {date_str}")

    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df = df.set_index("ts_event").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def _run_orb(df: pd.DataFrame, or_minutes: int = 30) -> object:
    engine = ORBEngine()
    return engine.run(
        ohlcv_df=df,
        config={},
        params={"or_minutes": or_minutes, "confirm_mode": "close-beyond"},
        initial_cash=100_000,
    )


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def nq_day():
    return _load_nq_day("2024-01-02", "NQH4")


@pytest.fixture(scope="module")
def result_30m(nq_day):
    return _run_orb(nq_day, or_minutes=30)


@pytest.fixture(scope="module")
def result_15m(nq_day):
    return _run_orb(nq_day, or_minutes=15)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestDataLoaded:
    """Verify NQ data is present and has enough bars."""

    def test_at_least_60_bars(self, nq_day):
        assert len(nq_day) >= 60, f"Only {len(nq_day)} bars loaded"


class TestEngineRunsWithoutCrash:
    """Engine completes on real NQ data without exceptions."""

    def test_engine_completes_30m(self, result_30m):
        assert result_30m is not None

    def test_engine_completes_15m(self, result_15m):
        assert result_15m is not None


class TestEventsPassContract:
    """All emitted events pass contract validation."""

    def test_events_valid_30m(self, result_30m):
        errors = validate_events(result_30m.events)
        assert errors == [], f"Contract violations: {errors}"

    def test_events_valid_15m(self, result_15m):
        errors = validate_events(result_15m.events)
        assert errors == [], f"Contract violations: {errors}"


class TestORLockTiming:
    """OR locks at the expected time relative to session open."""

    def test_or_lock_timing_30m(self, result_30m):
        locked = [e for e in result_30m.events if e["type"] == "orb_range_locked"]
        assert len(locked) >= 1
        lock_ts = pd.Timestamp(locked[0]["ts"])
        lock_et = lock_ts.tz_convert("America/New_York")
        # 09:30 + 30m = 10:00 ET; lock should be near 09:59 ET
        assert lock_et.hour == 9, f"Lock at {lock_et}, expected ~09:59 ET"
        assert lock_et.minute >= 55, f"Lock at {lock_et}, expected minute >= 55"

    def test_or_lock_timing_15m(self, result_15m):
        locked = [e for e in result_15m.events if e["type"] == "orb_range_locked"]
        assert len(locked) >= 1
        lock_ts = pd.Timestamp(locked[0]["ts"])
        lock_et = lock_ts.tz_convert("America/New_York")
        # 09:30 + 15m = 09:45 ET; lock should be near 09:44 ET
        assert lock_et.hour == 9, f"Lock at {lock_et}, expected ~09:44 ET"
        assert lock_et.minute >= 40, f"Lock at {lock_et}, expected minute >= 40"


class TestORRangePlausible:
    """OR range values are in a reasonable NQ range."""

    def test_or_range_plausible(self, result_30m):
        locked = [e for e in result_30m.events if e["type"] == "orb_range_locked"]
        assert len(locked) >= 1
        orb = locked[0]
        high = orb["high"]
        low = orb["low"]
        rng = orb["range"]
        assert high > 10000, f"OR high={high} too low for NQ"
        assert low > 10000, f"OR low={low} too low for NQ"
        assert 5 <= rng <= 1000, f"OR range={rng} outside plausible NQ range"


class TestIdempotentRerun:
    """Running the engine twice on the same data produces identical events."""

    def test_idempotent_rerun(self, nq_day):
        r1 = _run_orb(nq_day)
        r2 = _run_orb(nq_day)
        assert len(r1.events) == len(r2.events)
        for e1, e2 in zip(r1.events, r2.events):
            assert e1 == e2
