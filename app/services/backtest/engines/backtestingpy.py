"""Backtesting.py engine adapter.

This adapter wraps the backtesting.py library to run strategy backtests.

Note: For v1, this creates a generic mean-reversion strategy based on
the compiled config parameters. Future versions could:
- Parse entry_rules text to generate strategy logic
- Use LLM to generate strategy code from pseudocode
- Support multiple strategy templates
"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np
import structlog

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from app.services.backtest.engines.base import BacktestResult, TradeRecord

logger = structlog.get_logger(__name__)


def _create_strategy_class(
    config: dict[str, Any],
    params: dict[str, Any],
) -> type[Strategy]:
    """
    Create a Strategy class based on compiled config and params.

    For v1: Creates a simple mean-reversion strategy using SMA.
    The strategy buys when price crosses below lower band and sells when it crosses above SMA.
    """
    # Extract parameters with defaults
    period = params.get("period", 20)
    threshold = params.get("threshold", params.get("std_dev", 2.0))

    class GenericStrategy(Strategy):
        """Generic strategy generated from KB spec.

        Uses Bollinger Bands logic for v1:
        - Buy when price < lower band (SMA - n*std)
        - Sell when price > SMA (mean reversion target)
        """

        # Class-level parameters (can be optimized)
        n_period = period
        n_std = threshold

        def init(self):
            close = self.data.Close

            # Calculate SMA
            self.sma = self.I(
                lambda x: pd.Series(x).rolling(self.n_period).mean(),
                close,
                name=f"SMA({self.n_period})",
            )

            # Calculate standard deviation
            self.std = self.I(
                lambda x: pd.Series(x).rolling(self.n_period).std(),
                close,
                name=f"STD({self.n_period})",
            )

            # Calculate Bollinger Bands
            self.upper = self.I(
                lambda: self.sma + self.n_std * self.std,
                name="Upper Band",
            )
            self.lower = self.I(
                lambda: self.sma - self.n_std * self.std,
                name="Lower Band",
            )

        def next(self):
            # Skip if not enough data
            if len(self.data) < self.n_period:
                return

            price = self.data.Close[-1]
            sma = self.sma[-1]
            lower = self.lower[-1]

            # Entry: price below lower band (oversold)
            if not self.position:
                if price < lower:
                    self.buy()

            # Exit: price reverts to mean (SMA)
            elif self.position:
                if price > sma:
                    self.position.close()

    # Set strategy name from config
    strategy_name = config.get("strategy_name", "GenericStrategy")
    GenericStrategy.__name__ = strategy_name
    GenericStrategy.__qualname__ = strategy_name

    return GenericStrategy


class BacktestingPyEngine:
    """Engine adapter for backtesting.py library."""

    name = "backtesting.py"

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        config: dict[str, Any],
        params: dict[str, Any],
        initial_cash: float = 10000,
        commission_bps: float = 10,
        slippage_bps: float = 0,
    ) -> BacktestResult:
        """
        Run backtest using backtesting.py.

        Args:
            ohlcv_df: OHLCV DataFrame with datetime index
            config: Compiled backtest config
            params: Validated parameters
            initial_cash: Starting capital
            commission_bps: Commission (10 = 0.1%)
            slippage_bps: Slippage (not directly supported, included in commission)

        Returns:
            BacktestResult with metrics and trades
        """
        warnings = []

        # Convert commission from bps to decimal
        commission = (commission_bps + slippage_bps) / 10000

        logger.info(
            "Running backtest",
            engine=self.name,
            data_rows=len(ohlcv_df),
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            params=params,
        )

        # Create strategy class
        strategy_class = _create_strategy_class(config, params)

        # Create and run backtest
        bt = Backtest(
            ohlcv_df,
            strategy_class,
            cash=initial_cash,
            commission=commission,
            exclusive_orders=True,
            trade_on_close=True,
        )

        try:
            stats = bt.run()
        except Exception as e:
            logger.error("Backtest failed", error=str(e))
            raise RuntimeError(f"Backtest execution failed: {e}")

        # Extract equity curve
        equity_curve = self._extract_equity_curve(stats)

        # Extract trades
        trades = self._extract_trades(stats)

        # Calculate additional metrics (handle NaN values)
        return_pct_raw = stats.get("Return [%]", 0)
        return_pct = float(return_pct_raw) if return_pct_raw and not np.isnan(return_pct_raw) else 0.0
        max_dd_raw = stats.get("Max. Drawdown [%]", 0)
        max_dd = float(max_dd_raw) if max_dd_raw and not np.isnan(max_dd_raw) else 0.0
        sharpe = stats.get("Sharpe Ratio")
        if sharpe is not None and not np.isnan(sharpe):
            sharpe = float(sharpe)
        else:
            sharpe = None
            warnings.append("Sharpe ratio could not be calculated")

        win_rate_raw = stats.get("Win Rate [%]", 0)
        win_rate = float(win_rate_raw) / 100 if win_rate_raw and not np.isnan(win_rate_raw) else 0.0
        num_trades = int(stats.get("# Trades", 0))
        buy_hold_raw = stats.get("Buy & Hold Return [%]", 0)
        buy_hold = float(buy_hold_raw) if buy_hold_raw and not np.isnan(buy_hold_raw) else 0.0
        avg_trade_raw = stats.get("Avg. Trade [%]", 0)
        avg_trade = float(avg_trade_raw) if avg_trade_raw and not np.isnan(avg_trade_raw) else 0.0
        max_duration_raw = stats.get("Max. Trade Duration", 0)
        # Max Trade Duration can be a timedelta or NaN
        if hasattr(max_duration_raw, "total_seconds"):
            max_duration = int(max_duration_raw.total_seconds() / 3600)
        elif max_duration_raw and not np.isnan(max_duration_raw):
            max_duration = int(max_duration_raw)
        else:
            max_duration = 0

        profit_factor = stats.get("Profit Factor")
        if profit_factor is not None and not np.isnan(profit_factor):
            profit_factor = float(profit_factor)
        else:
            profit_factor = None

        if num_trades == 0:
            warnings.append("No trades were executed - strategy may need parameter adjustment")

        logger.info(
            "Backtest completed",
            return_pct=return_pct,
            max_drawdown=max_dd,
            num_trades=num_trades,
            sharpe=sharpe,
        )

        return BacktestResult(
            return_pct=return_pct,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            num_trades=num_trades,
            buy_hold_return_pct=buy_hold,
            avg_trade_pct=avg_trade,
            max_trade_duration=max_duration,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            trades=trades,
            warnings=warnings,
        )

    def _extract_equity_curve(self, stats) -> list[dict[str, Any]]:
        """Extract equity curve from backtest stats."""
        equity_curve = []

        try:
            # stats._equity_curve is a DataFrame with equity values
            equity_df = stats._equity_curve
            if equity_df is not None and "Equity" in equity_df.columns:
                for idx, row in equity_df.iterrows():
                    equity_curve.append({
                        "t": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "equity": float(row["Equity"]),
                    })
        except Exception as e:
            logger.warning("Could not extract equity curve", error=str(e))

        return equity_curve

    def _extract_trades(self, stats) -> list[dict[str, Any]]:
        """Extract trade list from backtest stats."""
        trades = []

        try:
            trades_df = stats._trades
            if trades_df is not None and len(trades_df) > 0:
                for _, trade in trades_df.iterrows():
                    entry_time = trade.get("EntryTime")
                    exit_time = trade.get("ExitTime")

                    trades.append({
                        "entry_time": entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time),
                        "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
                        "side": "long" if trade.get("Size", 0) > 0 else "short",
                        "entry_price": float(trade.get("EntryPrice", 0)),
                        "exit_price": float(trade.get("ExitPrice", 0)),
                        "size": abs(float(trade.get("Size", 0))),
                        "pnl": float(trade.get("PnL", 0)),
                        "return_pct": float(trade.get("ReturnPct", 0)) * 100,
                        "duration_bars": int(trade.get("Duration", 0).total_seconds() / 3600) if hasattr(trade.get("Duration"), "total_seconds") else 0,
                    })
        except Exception as e:
            logger.warning("Could not extract trades", error=str(e))

        return trades
