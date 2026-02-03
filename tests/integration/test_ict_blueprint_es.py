"""Integration test for ICT Blueprint engine with real ES data."""

import os

import pandas as pd
import numpy as np
import pytest

from app.services.backtest.engines.ict_blueprint.engine import ICTBlueprintEngine
from app.services.backtest.engines.ict_blueprint.htf_provider import DefaultHTFProvider
from app.services.backtest.engines.ict_blueprint.types import Bias, ICTBlueprintParams
from app.services.backtest.engines.base import BacktestResult

ES_DAILY_PATH = os.path.join("docs", "historical_data", "ES_daily.csv")
ES_H1_PATH = os.path.join("docs", "historical_data", "ES_h1.csv")


def _normalize_cols(df: pd.DataFrame) -> None:
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":
            col_map[c] = "Open"
        elif cl == "high":
            col_map[c] = "High"
        elif cl == "low":
            col_map[c] = "Low"
        elif cl == "close":
            col_map[c] = "Close"
        elif cl == "volume":
            col_map[c] = "Volume"
    if col_map:
        df.rename(columns=col_map, inplace=True)


@pytest.mark.slow
class TestICTBlueprintES:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not os.path.exists(ES_DAILY_PATH) or not os.path.exists(ES_H1_PATH):
            pytest.skip("ES CSV files not available")

    def test_runs_with_real_data(self):
        daily_df = pd.read_csv(ES_DAILY_PATH, parse_dates=True, index_col=0)
        h1_df = pd.read_csv(ES_H1_PATH, parse_dates=True, index_col=0)
        _normalize_cols(daily_df)
        _normalize_cols(h1_df)

        engine = ICTBlueprintEngine()
        result = engine.run(
            ohlcv_df=h1_df,
            config={"htf_df": daily_df, "point_value": 50.0, "risk_pct": 0.01},
            params={},
            initial_cash=100000,
            commission_bps=10,
            slippage_bps=5,
        )

        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 1, f"Expected at least 1 trade, got {result.num_trades}"
        assert len(result.equity_curve) > 0
        assert len(result.trades) == result.num_trades

        # No NaN in equity curve
        for point in result.equity_curve:
            assert not np.isnan(point["equity"])

        # All BacktestResult fields populated
        assert result.return_pct is not None
        assert result.max_drawdown_pct is not None
        assert result.win_rate is not None
        assert result.buy_hold_return_pct is not None
        assert result.avg_trade_pct is not None

        # Trade fields
        for trade in result.trades:
            assert trade["entry_price"] > 0
            assert trade["exit_price"] > 0
            assert trade["side"] in ("long", "short")
            assert trade["exit_reason"] in ("stop_loss", "take_profit", "eod_close")


# ---------------------------------------------------------------------------
# No-lookahead verification (unit-level, no real data needed)
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_first_h1_of_day_does_not_see_that_days_daily_close(self):
        """The first H1 bar of day D must only see daily state through day D-1.

        Regression guard against the midnight-timestamp lookahead bug.
        """
        # Create 3 daily bars: Jan 2, Jan 3, Jan 4
        daily_df = pd.DataFrame(
            {
                "Open": [100.0, 110.0, 120.0],
                "High": [105.0, 115.0, 125.0],
                "Low": [95.0, 105.0, 115.0],
                "Close": [102.0, 112.0, 122.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

        params = ICTBlueprintParams()
        provider = DefaultHTFProvider(daily_df, params, session_close_hour=16)

        # H1 bar at 10:00 on Jan 3 — should only see Jan 2's daily close
        h1_ts_jan3_10am = int(pd.Timestamp("2024-01-03 10:00:00").value)
        snap = provider.get_state_at(h1_ts_jan3_10am)
        # Provider should have processed at most daily bar 0 (Jan 2, closes at 16:00 Jan 2)
        # Jan 3's daily close is at 16:00 Jan 3, which is AFTER 10:00 Jan 3
        # So the state should reflect Jan 2 only
        # We verify by checking the number of processed bars
        assert provider._processed_up_to <= 0  # Only Jan 2 (index 0) or nothing

    def test_h1_after_daily_close_sees_that_day(self):
        """An H1 bar after the session close should see that day's daily bar."""
        daily_df = pd.DataFrame(
            {
                "Open": [100.0, 110.0],
                "High": [105.0, 115.0],
                "Low": [95.0, 105.0],
                "Close": [102.0, 112.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

        params = ICTBlueprintParams()
        provider = DefaultHTFProvider(daily_df, params, session_close_hour=16)

        # H1 bar at 17:00 on Jan 2 — should see Jan 2's daily close
        h1_ts_jan2_5pm = int(pd.Timestamp("2024-01-02 17:00:00").value)
        snap = provider.get_state_at(h1_ts_jan2_5pm)
        assert provider._processed_up_to == 0  # Jan 2 processed

        # H1 bar at 17:00 on Jan 3 — should see Jan 3's daily close
        h1_ts_jan3_5pm = int(pd.Timestamp("2024-01-03 17:00:00").value)
        snap = provider.get_state_at(h1_ts_jan3_5pm)
        assert provider._processed_up_to == 1  # Jan 3 processed

    def test_date_only_index_gets_session_close_offset(self):
        """Date-only daily index should have 16h offset applied."""
        daily_df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        params = ICTBlueprintParams()
        provider = DefaultHTFProvider(daily_df, params, session_close_hour=16)

        expected_offset = 16 * 3600 * 1_000_000_000  # 16h in ns
        midnight = int(pd.Timestamp("2024-01-02").value)
        assert provider._daily_close_ts[0] == midnight + expected_offset

    def test_datetime_index_gets_no_offset(self):
        """If daily index already has time component, no offset is added."""
        daily_df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0]},
            index=pd.to_datetime(["2024-01-02 16:00:00"]),
        )
        params = ICTBlueprintParams()
        provider = DefaultHTFProvider(daily_df, params, session_close_hour=16)

        expected = int(pd.Timestamp("2024-01-02 16:00:00").value)
        assert provider._daily_close_ts[0] == expected  # No offset

    def test_warns_on_hour_mismatch(self, caplog):
        """Warn if daily timestamps carry a time that doesn't match session_close_hour."""
        daily_df = pd.DataFrame(
            {
                "Open": [100.0, 110.0],
                "High": [105.0, 115.0],
                "Low": [95.0, 105.0],
                "Close": [102.0, 112.0],
            },
            index=pd.to_datetime(["2024-01-02 17:00:00", "2024-01-03 17:00:00"]),
        )
        params = ICTBlueprintParams()
        import logging

        with caplog.at_level(logging.WARNING):
            DefaultHTFProvider(daily_df, params, session_close_hour=16)
        assert "session_close_hour" in caplog.text

    def test_warns_on_duplicate_dates(self, caplog):
        """Warn if daily DataFrame has duplicate calendar dates (sub-daily data)."""
        daily_df = pd.DataFrame(
            {
                "Open": [100.0, 110.0],
                "High": [105.0, 115.0],
                "Low": [95.0, 105.0],
                "Close": [102.0, 112.0],
            },
            index=pd.to_datetime(["2024-01-02 10:00:00", "2024-01-02 11:00:00"]),
        )
        params = ICTBlueprintParams()
        import logging

        with caplog.at_level(logging.WARNING):
            DefaultHTFProvider(daily_df, params, session_close_hour=16)
        assert "unique dates" in caplog.text
