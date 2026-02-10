"""Base interface for backtest engines."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol

import pandas as pd


@dataclass
class TradeRecord:
    """Single trade record from backtest."""

    entry_time: datetime
    exit_time: datetime
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    duration_bars: int


@dataclass
class BacktestResult:
    """Result from running a backtest."""

    # Summary metrics
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    win_rate: float
    num_trades: int

    # Additional metrics
    buy_hold_return_pct: float
    avg_trade_pct: float
    max_trade_duration: int
    profit_factor: Optional[float]

    # Equity curve: list of {"t": iso_datetime, "equity": float}
    equity_curve: list[dict[str, Any]]

    # Trades: list of trade records as dicts
    trades: list[dict[str, Any]]

    # Run events for replay (ORB engine, etc.)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Warnings generated during backtest
    warnings: list[str] = field(default_factory=list)


class BacktestEngine(Protocol):
    """Protocol for backtest engine adapters.

    Implement this interface to add support for new backtesting frameworks.

    Usage:
        engine = BacktestingPyEngine()
        result = engine.run(
            ohlcv_df=df,
            config=compiled_backtest_config,
            params=validated_params,
            initial_cash=10000,
            commission_bps=10,
        )
    """

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
        Run a backtest with the given data and configuration.

        Args:
            ohlcv_df: DataFrame with OHLCV data (datetime index, columns: Open, High, Low, Close, Volume)  # noqa: E501
            config: Compiled backtest configuration from strategy spec
            params: Validated parameters (with defaults applied)
            initial_cash: Starting capital
            commission_bps: Commission in basis points (10 = 0.1%)
            slippage_bps: Slippage in basis points

        Returns:
            BacktestResult with metrics, equity curve, and trades
        """
        ...
