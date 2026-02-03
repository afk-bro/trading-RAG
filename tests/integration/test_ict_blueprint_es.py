"""Integration test for ICT Blueprint engine with real ES data."""

import os

import pandas as pd
import numpy as np
import pytest

from app.services.backtest.engines.ict_blueprint.engine import ICTBlueprintEngine
from app.services.backtest.engines.base import BacktestResult

ES_DAILY_PATH = os.path.join("docs", "historical_data", "ES_daily.csv")
ES_H1_PATH = os.path.join("docs", "historical_data", "ES_h1.csv")


@pytest.mark.slow
class TestICTBlueprintES:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not os.path.exists(ES_DAILY_PATH) or not os.path.exists(ES_H1_PATH):
            pytest.skip("ES CSV files not available")

    def test_runs_with_real_data(self):
        daily_df = pd.read_csv(ES_DAILY_PATH, parse_dates=True, index_col=0)
        h1_df = pd.read_csv(ES_H1_PATH, parse_dates=True, index_col=0)

        # Normalize columns
        for df in [daily_df, h1_df]:
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
