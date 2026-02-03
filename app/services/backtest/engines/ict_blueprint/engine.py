"""ICT Blueprint backtest engine — main loop implementing BacktestEngine protocol."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.services.backtest.engines.base import BacktestResult

from .htf_bias import SwingDetector, is_in_discount, is_in_premium
from .htf_provider import DefaultHTFProvider
from .ltf_entry import (
    advance_setup,
    check_ob_zone_entry,
    create_setup_for_ob,
    get_active_setup_keys,
    select_candidate_setups,
)
from .risk_manager import (
    bps_to_points,
    check_entry_exit_collision,
    check_exit,
    check_rr_gate,
    close_position,
    compute_position_size,
    compute_stop_price,
    compute_target_price,
    process_derisk,
)
from .types import (
    Bias,
    ClosedTrade,
    HTFStateSnapshot,
    ICTBlueprintParams,
    LTFSetup,
    Position,
    SetupPhase,
    Side,
    SwingPoint,
)


class ICTBlueprintEngine:
    """Backtest engine for the ICT Blueprint dual-timeframe strategy."""

    name = "ict_blueprint"

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        config: dict[str, Any],
        params: dict[str, Any],
        initial_cash: float = 10000,
        commission_bps: float = 10,
        slippage_bps: float = 0,
    ) -> BacktestResult:
        """Run the ICT Blueprint backtest.

        ohlcv_df: H1 DataFrame (primary timeframe).
        config must contain 'htf_data_path' pointing to daily CSV,
               or 'htf_df' with a pre-loaded daily DataFrame.
        """
        warnings: list[str] = []

        # Build params
        bp = ICTBlueprintParams.from_dict(params)
        point_value = config.get("point_value", bp.point_value)
        risk_pct = config.get("risk_pct", bp.risk_pct)

        # Load HTF data
        htf_df = self._load_htf_data(config, warnings)

        # Build provider
        provider = DefaultHTFProvider(htf_df, bp)

        # Build H1 bar tuples
        h1_bars = self._build_bar_tuples(ohlcv_df)
        if not h1_bars:
            return self._empty_result(warnings)

        # Slippage in points
        slippage_pts = bps_to_points(h1_bars[0][5], slippage_bps)

        # State
        equity = initial_cash
        peak_equity = initial_cash
        max_dd = 0.0
        position: Optional[Position] = None
        closed_trades: list[ClosedTrade] = []
        equity_curve: list[dict[str, Any]] = []
        active_setups: list[LTFSetup] = []

        # LTF swing detector
        ltf_swing_det = SwingDetector(lookback=bp.ltf_swing_lookback)
        ltf_swing_highs: list[SwingPoint] = []
        ltf_swing_lows: list[SwingPoint] = []

        # Partial PnL accumulator (from half-off de-risk)
        partial_pnl_dollars = 0.0

        for bar_pos, bar in enumerate(h1_bars):
            bar_idx, bar_ts, bar_open, bar_high, bar_low, bar_close = bar

            # Update LTF swings
            swings = ltf_swing_det.push(bar_idx, bar_ts, bar_open, bar_high, bar_low, bar_close)
            for sp in swings:
                if sp.is_high:
                    ltf_swing_highs.append(sp)
                else:
                    ltf_swing_lows.append(sp)

            # 1. Get HTF state (causal)
            htf_snap = provider.get_state_at(bar_ts)

            # 2. If position open: de-risk + exit check
            if position is not None:
                # De-risk
                partial = process_derisk(
                    position, bar_high, bar_low, bp.derisk_mode, bp.derisk_trigger_rr
                )
                if partial is not None:
                    partial_pnl_dollars += partial * point_value

                # Exit check
                exit_result = check_exit(
                    position, bar_high, bar_low, bar_close, bar_ts, slippage_pts
                )
                if exit_result is not None:
                    exit_price, reason = exit_result
                    trade = close_position(
                        position, exit_price, bar_ts, reason,
                        point_value, commission_bps, slippage_bps,
                    )
                    trade.pnl_dollars += partial_pnl_dollars
                    closed_trades.append(trade)
                    equity += trade.pnl_dollars
                    partial_pnl_dollars = 0.0

                    # Track attempt on OB
                    if position.ob_id is not None:
                        for s in active_setups:
                            if s.ob.ob_id == position.ob_id:
                                s.ob.attempts_used += 1
                                s.last_exit_bar_index = bar_idx
                                # Reset to scanning for re-entry
                                if s.ob.attempts_used < bp.max_attempts_per_ob:
                                    s.phase = SetupPhase.SCANNING
                                    s.l0 = None
                                    s.h0 = None
                                    s.sweep_low = None
                                    s.msb_bar_index = -1
                                    s.breaker = None
                                    s.fvg = None
                                else:
                                    s.phase = SetupPhase.TIMED_OUT
                                break

                    position = None

                # Record equity and continue if still in position
                self._record_equity(equity_curve, bar_ts, equity)
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = max(max_dd, dd)
                if position is not None:
                    continue

            else:
                # 3. Record equity (no position)
                self._record_equity(equity_curve, bar_ts, equity)
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = max(max_dd, dd)

            # 4. Skip if neutral bias
            if htf_snap.bias == Bias.NEUTRAL:
                continue

            # 5. Premium/discount eligibility gate
            if htf_snap.current_range is None:
                continue

            h1_mid = (bar_high + bar_low) / 2.0
            if htf_snap.bias == Bias.BULLISH:
                if not is_in_discount(h1_mid, htf_snap.current_range, bp.discount_threshold):
                    # Not in discount — don't activate new setups, but advance existing
                    self._advance_existing_setups(
                        active_setups, bar, ltf_swing_lows, ltf_swing_highs,
                        h1_bars[:bar_pos + 1], bp, htf_snap,
                    )
                    continue
            else:
                if not is_in_premium(h1_mid, htf_snap.current_range, bp.discount_threshold):
                    self._advance_existing_setups(
                        active_setups, bar, ltf_swing_lows, ltf_swing_highs,
                        h1_bars[:bar_pos + 1], bp, htf_snap,
                    )
                    continue

            # 6. Activate new setups for OBs we don't already track
            existing_ob_ids = get_active_setup_keys(active_setups)
            for ob in htf_snap.active_obs:
                if ob.invalidated:
                    continue
                if ob.ob_id in existing_ob_ids:
                    continue
                if ob.attempts_used >= bp.max_attempts_per_ob:
                    continue
                # Check if H1 bar enters OB zone
                if check_ob_zone_entry(
                    bar_open, bar_high, bar_low, bar_close, ob,
                    bp.ob_zone_entry_requirement, bp.ob_zone_overlap_pct,
                ):
                    setup = create_setup_for_ob(ob, htf_snap.bias)
                    active_setups.append(setup)

            # 7-8. Select candidates and advance
            candidates = select_candidate_setups(active_setups)
            for setup in candidates:
                entry_price = advance_setup(
                    setup, bar, ltf_swing_lows, ltf_swing_highs,
                    h1_bars[:bar_pos + 1], bp, htf_snap,
                )
                if entry_price is not None:
                    # Compute stop
                    stop = compute_stop_price(setup, bp.stop_mode, setup.side, bp.stop_buffer_ticks)
                    if stop is None:
                        continue

                    # Entry-exit collision check
                    if check_entry_exit_collision(entry_price, stop, bar_high, bar_low, setup.side):
                        setup.ob.attempts_used += 1
                        setup.last_exit_bar_index = bar_idx
                        if setup.ob.attempts_used >= bp.max_attempts_per_ob:
                            setup.phase = SetupPhase.TIMED_OUT
                        else:
                            setup.phase = SetupPhase.SCANNING
                            setup.l0 = None
                            setup.h0 = None
                            setup.sweep_low = None
                            setup.msb_bar_index = -1
                            setup.breaker = None
                            setup.fvg = None
                        continue

                    # Compute target
                    target = compute_target_price(
                        entry_price, stop, setup.side, bp.tp_mode,
                        htf_snap, bp.fixed_rr,
                    )
                    if target is None:
                        continue

                    # R:R gate
                    if not check_rr_gate(entry_price, stop, target, bp.min_rr, setup.side):
                        continue

                    # Position sizing
                    size = compute_position_size(
                        equity, risk_pct, entry_price, stop, point_value,
                    )
                    if size <= 0:
                        continue

                    risk_pts = abs(entry_price - stop)

                    position = Position(
                        entry_time=bar_ts,
                        entry_price=entry_price,
                        stop_price=stop,
                        target_price=target,
                        side=setup.side,
                        size=size,
                        risk_points=risk_pts,
                        ob_id=setup.ob.ob_id,
                    )
                    partial_pnl_dollars = 0.0
                    break  # One entry per bar

            # 9. Garbage-collect
            active_setups = [
                s for s in active_setups
                if s.phase not in (SetupPhase.TIMED_OUT,)
                and not s.ob.invalidated
            ]

        # End of data: force close
        if position is not None and h1_bars:
            last_bar = h1_bars[-1]
            trade = close_position(
                position, last_bar[5], last_bar[1], "eod_close",
                point_value, commission_bps, slippage_bps,
            )
            trade.pnl_dollars += partial_pnl_dollars
            closed_trades.append(trade)
            equity += trade.pnl_dollars

        return self._build_result(
            closed_trades, equity_curve, initial_cash, equity, max_dd,
            h1_bars, point_value, warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_htf_data(config: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
        """Load daily DataFrame from config."""
        if "htf_df" in config:
            return config["htf_df"]

        path = config.get("htf_data_path")
        if not path:
            raise ValueError("Config must include 'htf_data_path' or 'htf_df'")

        df = pd.read_csv(path, parse_dates=True, index_col=0)
        # Normalize column names
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
            df = df.rename(columns=col_map)

        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Daily CSV missing columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df

    @staticmethod
    def _build_bar_tuples(
        df: pd.DataFrame,
    ) -> list[tuple[int, int, float, float, float, float]]:
        """Convert DataFrame to list of (index, ts_ns, open, high, low, close)."""
        bars = []
        for i, (ts_val, row) in enumerate(df.iterrows()):
            ts_ns = int(pd.Timestamp(ts_val).value)
            bars.append((
                i, ts_ns,
                float(row["Open"]), float(row["High"]),
                float(row["Low"]), float(row["Close"]),
            ))
        return bars

    @staticmethod
    def _record_equity(
        curve: list[dict[str, Any]], ts_ns: int, equity: float
    ) -> None:
        dt = pd.Timestamp(ts_ns, unit="ns").isoformat()
        curve.append({"t": dt, "equity": equity})

    @staticmethod
    def _advance_existing_setups(
        setups: list[LTFSetup],
        bar: tuple[int, int, float, float, float, float],
        ltf_lows: list[SwingPoint],
        ltf_highs: list[SwingPoint],
        h1_bars: list[tuple[int, int, float, float, float, float]],
        params: ICTBlueprintParams,
        htf_snap: HTFStateSnapshot,
    ) -> None:
        """Advance existing setups even when premium/discount gate blocks new activations."""
        candidates = select_candidate_setups(setups)
        for setup in candidates:
            advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf_snap)

    @staticmethod
    def _empty_result(warnings: list[str]) -> BacktestResult:
        return BacktestResult(
            return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=None,
            win_rate=0.0,
            num_trades=0,
            buy_hold_return_pct=0.0,
            avg_trade_pct=0.0,
            max_trade_duration=0,
            profit_factor=None,
            equity_curve=[],
            trades=[],
            warnings=warnings,
        )

    @staticmethod
    def _build_result(
        closed_trades: list[ClosedTrade],
        equity_curve: list[dict[str, Any]],
        initial_cash: float,
        final_equity: float,
        max_dd: float,
        h1_bars: list[tuple[int, int, float, float, float, float]],
        point_value: float,
        warnings: list[str],
    ) -> BacktestResult:
        num_trades = len(closed_trades)
        return_pct = ((final_equity - initial_cash) / initial_cash * 100.0) if initial_cash > 0 else 0.0

        # Buy & hold
        if h1_bars:
            bh_return = ((h1_bars[-1][5] - h1_bars[0][5]) / h1_bars[0][5] * 100.0) if h1_bars[0][5] != 0 else 0.0
        else:
            bh_return = 0.0

        # Per-trade stats
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        total_return_pct = 0.0
        max_duration = 0
        trade_dicts: list[dict[str, Any]] = []

        for t in closed_trades:
            pct = (t.pnl_dollars / initial_cash * 100.0) if initial_cash > 0 else 0.0
            total_return_pct += pct
            if t.pnl_dollars > 0:
                wins += 1
                gross_profit += t.pnl_dollars
            elif t.pnl_dollars < 0:
                gross_loss += abs(t.pnl_dollars)

            # Duration in H1 bars (approx)
            dur_ns = t.exit_time - t.entry_time
            dur_hours = max(1, int(dur_ns / 3_600_000_000_000))
            max_duration = max(max_duration, dur_hours)

            entry_dt = pd.Timestamp(t.entry_time, unit="ns")
            exit_dt = pd.Timestamp(t.exit_time, unit="ns")

            trade_dicts.append({
                "entry_time": entry_dt.isoformat(),
                "exit_time": exit_dt.isoformat(),
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl_dollars,
                "pnl_points": t.pnl_points,
                "return_pct": pct,
                "exit_reason": t.exit_reason,
                "duration_hours": dur_hours,
            })

        win_rate = (wins / num_trades) if num_trades > 0 else 0.0
        avg_trade_pct = (total_return_pct / num_trades) if num_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        # Sharpe ratio from equity curve returns
        sharpe = None
        if len(equity_curve) > 1:
            equities = [e["equity"] for e in equity_curve]
            returns = []
            for i in range(1, len(equities)):
                if equities[i - 1] > 0:
                    returns.append((equities[i] - equities[i - 1]) / equities[i - 1])
            if returns:
                arr = np.array(returns)
                std = arr.std()
                if std > 0:
                    # Annualize: ~252 trading days * ~7 H1 bars/day
                    sharpe = float(arr.mean() / std * math.sqrt(252 * 7))

        return BacktestResult(
            return_pct=return_pct,
            max_drawdown_pct=max_dd * 100.0,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            num_trades=num_trades,
            buy_hold_return_pct=bh_return,
            avg_trade_pct=avg_trade_pct,
            max_trade_duration=max_duration,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            trades=trade_dicts,
            warnings=warnings,
        )
