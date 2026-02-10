"""Unit tests for ICT Blueprint engine with synthetic data."""

import pandas as pd
import numpy as np
import pytest

from app.services.backtest.engines.ict_blueprint.engine import ICTBlueprintEngine
from app.services.backtest.engines.base import BacktestResult


def _make_daily_df(n=60) -> pd.DataFrame:
    """Create a synthetic daily OHLCV DataFrame with a clear trend structure.

    Generates an up-down-up pattern to produce MSBs, ranges, and OBs.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = np.zeros(n)
    close[0] = 4500.0

    # Phase 1: up (0 to n/3)
    p1 = n // 3
    p2 = 2 * n // 3
    for i in range(1, p1):
        close[i] = close[i - 1] + np.random.uniform(5, 20)
    # Phase 2: down (n/3 to 2n/3)
    for i in range(max(p1, 1), p2):
        close[i] = close[i - 1] - np.random.uniform(5, 15)
    # Phase 3: up again (2n/3 to n)
    for i in range(max(p2, 1), n):
        close[i] = close[i - 1] + np.random.uniform(5, 20)

    high = close + np.random.uniform(5, 15, n)
    low = close - np.random.uniform(5, 15, n)
    open_ = close + np.random.uniform(-10, 10, n)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.randint(1000, 5000, n),
        },
        index=dates,
    )
    return df


def _make_h1_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic H1 bars covering the same period as daily.

    7 H1 bars per trading day, roughly following the daily range.
    """
    rows = []
    for day_ts, day_row in daily_df.iterrows():
        d_open = day_row["Open"]
        d_high = day_row["High"]
        d_low = day_row["Low"]
        d_close = day_row["Close"]
        d_range = d_high - d_low
        if d_range <= 0:
            d_range = 1.0

        # Generate 7 hourly bars
        np.random.seed(hash(str(day_ts)) % (2**31))
        h1_closes = np.linspace(d_open, d_close, 8)  # 8 points = 7 intervals
        for j in range(7):
            ts = pd.Timestamp(day_ts) + pd.Timedelta(hours=9 + j + 1)
            c = h1_closes[j + 1]
            o = h1_closes[j]
            h = max(o, c) + np.random.uniform(0, d_range * 0.1)
            low = min(o, c) - np.random.uniform(0, d_range * 0.1)
            rows.append(
                {"Open": o, "High": h, "Low": low, "Close": c, "Volume": 500, "ts": ts}
            )

    h1_df = pd.DataFrame(rows).set_index("ts")
    h1_df.index.name = None
    return h1_df


class TestICTBlueprintEngine:
    def test_returns_backtest_result(self):
        daily_df = _make_daily_df(60)
        h1_df = _make_h1_df(daily_df)

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
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.trades, list)
        assert result.return_pct is not None
        assert result.max_drawdown_pct is not None
        assert result.buy_hold_return_pct is not None

    def test_equity_curve_not_empty(self):
        daily_df = _make_daily_df(60)
        h1_df = _make_h1_df(daily_df)

        engine = ICTBlueprintEngine()
        result = engine.run(
            ohlcv_df=h1_df,
            config={"htf_df": daily_df, "point_value": 50.0},
            params={},
            initial_cash=100000,
        )
        assert len(result.equity_curve) > 0

    def test_no_nan_in_equity_curve(self):
        daily_df = _make_daily_df(60)
        h1_df = _make_h1_df(daily_df)

        engine = ICTBlueprintEngine()
        result = engine.run(
            ohlcv_df=h1_df,
            config={"htf_df": daily_df, "point_value": 50.0},
            params={},
            initial_cash=100000,
        )
        for point in result.equity_curve:
            assert not np.isnan(
                point["equity"]
            ), f"NaN found in equity curve at {point['t']}"

    def test_empty_df_returns_empty_result(self):
        daily_df = _make_daily_df(5)
        h1_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        h1_df.index = pd.DatetimeIndex([])

        engine = ICTBlueprintEngine()
        result = engine.run(
            ohlcv_df=h1_df,
            config={"htf_df": daily_df},
            params={},
        )
        assert result.num_trades == 0
        assert result.equity_curve == []

    def test_trades_have_required_fields(self):
        daily_df = _make_daily_df(60)
        h1_df = _make_h1_df(daily_df)

        engine = ICTBlueprintEngine()
        result = engine.run(
            ohlcv_df=h1_df,
            config={"htf_df": daily_df, "point_value": 50.0},
            params={},
            initial_cash=100000,
        )
        for trade in result.trades:
            assert "entry_time" in trade
            assert "exit_time" in trade
            assert "side" in trade
            assert "entry_price" in trade
            assert "exit_price" in trade
            assert "pnl" in trade
            assert "exit_reason" in trade

    def test_custom_params(self):
        daily_df = _make_daily_df(60)
        h1_df = _make_h1_df(daily_df)

        engine = ICTBlueprintEngine()
        result = engine.run(
            ohlcv_df=h1_df,
            config={"htf_df": daily_df, "point_value": 50.0},
            params={
                "swing_lookback": 2,
                "ob_candles": 2,
                "entry_mode": "msb_close",
                "min_rr": 1.5,
                "require_sweep": False,
            },
            initial_cash=100000,
        )
        assert isinstance(result, BacktestResult)

    def test_engine_name(self):
        engine = ICTBlueprintEngine()
        assert engine.name == "ict_blueprint"

    def test_missing_htf_data_raises(self):
        h1_df = _make_h1_df(_make_daily_df(10))
        engine = ICTBlueprintEngine()
        with pytest.raises(ValueError, match="htf_data_path"):
            engine.run(ohlcv_df=h1_df, config={}, params={})
