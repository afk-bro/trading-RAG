"""ORB (Opening Range Breakout) backtest engine."""

from __future__ import annotations

import math
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.services.backtest.engines.base import BacktestResult
from app.utils.time import to_eastern_time

from .contracts import ORB_EVENT_SCHEMA_VERSION, validate_events
from .types import (
    ORBClosedTrade,
    ORBParams,
    ORBPhase,
    ORBPosition,
    ORBRange,
    ORBSessionState,
    SESSION_WINDOWS,
)

_ET = ZoneInfo("America/New_York")


class ORBEngine:
    """Backtest engine for Opening Range Breakout strategies."""

    name = "orb"

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        config: dict[str, Any],
        params: dict[str, Any],
        initial_cash: float = 10000,
        commission_bps: float = 10,
        slippage_bps: float = 0,
    ) -> BacktestResult:
        warnings: list[str] = []

        # Step 0: Timestamp normalization
        df = ohlcv_df.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if len(df) < 3:
            raise ValueError("Insufficient bars to infer interval (need >= 3)")

        # Step 1: Setup
        p = ORBParams.from_dict(params)
        p.point_value = float(config.get("point_value", p.point_value))
        p.risk_pct = float(config.get("risk_pct", p.risk_pct))

        bar_interval = self._infer_interval(df)
        bar_minutes = bar_interval.total_seconds() / 60.0
        if bar_minutes <= 0:
            raise ValueError(f"Invalid bar interval: {bar_interval}")
        or_bar_count_needed = max(1, round(p.or_minutes / bar_minutes))

        session_window = SESSION_WINDOWS.get(p.session)
        if session_window is None:
            raise ValueError(
                f"Unknown session '{p.session}'. "
                f"Valid: {list(SESSION_WINDOWS.keys())}"
            )

        # State
        state = ORBSessionState()
        equity = initial_cash
        peak_equity = initial_cash
        max_dd = 0.0
        closed_trades: list[ORBClosedTrade] = []
        equity_curve: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        # Step 2: Bar loop
        for bar_index, (ts_val, row) in enumerate(df.iterrows()):
            ts = pd.Timestamp(ts_val)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            bar_open = float(row["Open"])
            bar_high = float(row["High"])
            bar_low = float(row["Low"])
            bar_close = float(row["Close"])
            ts_utc_iso = ts.isoformat()

            # Convert to ET for session logic
            et_time = to_eastern_time(ts.to_pydatetime())
            et_dt = ts.to_pydatetime().astimezone(_ET)
            et_date = et_dt.date()
            session_date_str = et_date.isoformat()

            # Session boundary check: date changed or past session end
            if state.session_date is not None and (
                et_date != state.session_date or et_time >= session_window.end
            ):
                # Force close if in TRADE_MGMT
                if state.phase == ORBPhase.TRADE_MGMT and state.position is not None:
                    trade = self._close_position(
                        state.position,
                        bar_open,  # close at session boundary bar open
                        ts_utc_iso,
                        bar_index,
                        "session_close",
                        p,
                        commission_bps,
                        slippage_bps,
                    )
                    closed_trades.append(trade)
                    equity += trade.pnl
                    state.position = None
                state.reset()

            # Record equity every bar
            equity_curve.append({"t": ts_utc_iso, "equity": equity})
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity
                max_dd = max(max_dd, dd)

            # Common event payload
            def _evt_base(evt_type: str, phase: ORBPhase) -> dict[str, Any]:
                return {
                    "type": evt_type,
                    "bar_index": bar_index,
                    "ts": ts_utc_iso,
                    "session_date": session_date_str,
                    "phase": phase.value,
                    "schema_version": ORB_EVENT_SCHEMA_VERSION,
                }

            # PREMARKET -> OR_BUILD
            if state.phase == ORBPhase.PREMARKET:
                if et_time >= session_window.start and et_time < session_window.end:
                    state.phase = ORBPhase.OR_BUILD
                    state.session_date = et_date
                    state.orb_range = ORBRange(
                        high=bar_high,
                        low=bar_low,
                        start_bar_index=bar_index,
                    )
                    state.or_bar_count = 1

                    evt = _evt_base("orb_range_update", ORBPhase.OR_BUILD)
                    evt.update(
                        {
                            "orb_high": bar_high,
                            "orb_low": bar_low,
                            "or_minutes": p.or_minutes,
                            "or_start_index": bar_index,
                        }
                    )
                    events.append(evt)

                    # Check if locked immediately (single-bar OR)
                    if state.or_bar_count >= or_bar_count_needed:
                        self._lock_range(
                            state,
                            bar_index,
                            or_bar_count_needed,
                            p.or_minutes,
                            events,
                            _evt_base,
                            ts_utc_iso,
                            session_date_str,
                        )
                continue

            # OR_BUILD -> BREAKOUT_SCAN
            if state.phase == ORBPhase.OR_BUILD:
                orb = state.orb_range
                assert orb is not None
                orb.high = max(orb.high, bar_high)
                orb.low = min(orb.low, bar_low)
                state.or_bar_count += 1

                evt = _evt_base("orb_range_update", ORBPhase.OR_BUILD)
                evt.update(
                    {
                        "orb_high": orb.high,
                        "orb_low": orb.low,
                        "or_minutes": p.or_minutes,
                        "or_start_index": orb.start_bar_index,
                    }
                )
                events.append(evt)

                if state.or_bar_count >= or_bar_count_needed:
                    self._lock_range(
                        state,
                        bar_index,
                        or_bar_count_needed,
                        p.or_minutes,
                        events,
                        _evt_base,
                        ts_utc_iso,
                        session_date_str,
                    )
                continue

            # BREAKOUT_SCAN
            if state.phase == ORBPhase.BREAKOUT_SCAN:
                direction = self._check_breakout(
                    state,
                    bar_index,
                    bar_high,
                    bar_low,
                    bar_close,
                    p,
                    or_bar_count_needed,
                )
                if direction is not None:
                    orb = state.orb_range
                    assert orb is not None
                    level = orb.high if direction == "long" else orb.low
                    evt = _evt_base("setup_valid", ORBPhase.ENTRY)
                    evt.update(
                        {
                            "direction": direction,
                            "level": level,
                            "confirm_mode": p.confirm_mode,
                            "trigger_price": bar_close,
                        }
                    )
                    events.append(evt)
                    state.phase = ORBPhase.ENTRY

                    # Immediate entry on the same bar
                    entered = self._try_entry(
                        state,
                        bar_index,
                        bar_close,
                        ts_utc_iso,
                        session_date_str,
                        direction,
                        p,
                        equity,
                        events,
                        _evt_base,
                    )
                    if not entered:
                        # Failed entry (size=0), back to scanning
                        state.phase = ORBPhase.BREAKOUT_SCAN
                continue

            # ENTRY (shouldn't normally reach here — entry happens same bar
            # as breakout confirm, but handle for robustness)
            if state.phase == ORBPhase.ENTRY:
                state.phase = ORBPhase.BREAKOUT_SCAN
                continue

            # TRADE_MGMT
            if state.phase == ORBPhase.TRADE_MGMT:
                pos = state.position
                assert pos is not None

                exit_price: Optional[float] = None
                exit_reason: Optional[str] = None

                if pos.side == "long":
                    if bar_low <= pos.stop:
                        exit_price = pos.stop
                        exit_reason = "stop"
                    elif bar_high >= pos.target:
                        exit_price = pos.target
                        exit_reason = "target"
                else:  # short
                    if bar_high >= pos.stop:
                        exit_price = pos.stop
                        exit_reason = "stop"
                    elif bar_low <= pos.target:
                        exit_price = pos.target
                        exit_reason = "target"

                if exit_price is not None and exit_reason is not None:
                    trade = self._close_position(
                        pos,
                        exit_price,
                        ts_utc_iso,
                        bar_index,
                        exit_reason,
                        p,
                        commission_bps,
                        slippage_bps,
                    )
                    closed_trades.append(trade)
                    equity += trade.pnl
                    state.position = None
                    state.phase = ORBPhase.EXIT

                    # Immediate transition from EXIT
                    if state.trade_count < p.max_trades:
                        state.phase = ORBPhase.BREAKOUT_SCAN
                        # Reset retest FSM
                        state.pending_dir = None
                        state.pending_break_index = None
                        state.retest_confirmed = False
                    else:
                        state.phase = ORBPhase.LOCKOUT
                continue

            # EXIT (immediate transition handled above, but just in case)
            if state.phase == ORBPhase.EXIT:
                if state.trade_count < p.max_trades:
                    state.phase = ORBPhase.BREAKOUT_SCAN
                else:
                    state.phase = ORBPhase.LOCKOUT
                continue

            # LOCKOUT — no action
            if state.phase == ORBPhase.LOCKOUT:
                continue

        # End of data: force close open position
        if state.position is not None and len(df) > 0:
            last_ts = pd.Timestamp(df.index[-1])
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            last_close = float(df.iloc[-1]["Close"])
            trade = self._close_position(
                state.position,
                last_close,
                last_ts.isoformat(),
                len(df) - 1,
                "eod",
                p,
                commission_bps,
                slippage_bps,
            )
            closed_trades.append(trade)
            equity += trade.pnl
            state.position = None

        # Contract validation (runs under assert — free in production with -O)
        if __debug__:
            errors = validate_events(events)
            assert not errors, f"Event contract violations: {errors}"

        # Step 3: Build result
        return self._build_result(
            closed_trades,
            equity_curve,
            events,
            initial_cash,
            equity,
            max_dd,
            df,
            p,
            warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_interval(df: pd.DataFrame) -> pd.Timedelta:
        """Infer bar interval from median of first 10 consecutive deltas."""
        idx = df.index[:11]
        if len(idx) < 2:
            raise ValueError("Not enough bars to infer interval")
        deltas = pd.Series(idx[1:]) - pd.Series(idx[:-1])
        return deltas.median()

    @staticmethod
    def _lock_range(
        state: ORBSessionState,
        bar_index: int,
        or_bar_count_needed: int,
        or_minutes: int,
        events: list[dict[str, Any]],
        _evt_base: Any,
        ts_utc_iso: str,
        session_date_str: str,
    ) -> None:
        orb = state.orb_range
        assert orb is not None
        orb.locked = True
        orb.lock_bar_index = bar_index
        state.phase = ORBPhase.BREAKOUT_SCAN

        evt = _evt_base("orb_range_locked", ORBPhase.BREAKOUT_SCAN)
        evt.update(
            {
                "high": orb.high,
                "low": orb.low,
                "range": round(orb.high - orb.low, 6),
                "or_minutes": or_minutes,
                "or_start_index": orb.start_bar_index,
                "or_lock_index": bar_index,
                "or_bar_count_needed": or_bar_count_needed,
            }
        )
        events.append(evt)

    @staticmethod
    def _check_breakout(
        state: ORBSessionState,
        bar_index: int,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        p: ORBParams,
        or_bar_count_needed: int,
    ) -> Optional[str]:
        """Check for breakout. Returns 'long', 'short', or None."""
        orb = state.orb_range
        if orb is None or not orb.locked:
            return None

        if p.confirm_mode == "close-beyond":
            if bar_close > orb.high:
                return "long"
            if bar_close < orb.low:
                return "short"
            return None

        elif p.confirm_mode == "retest":
            # Step 1: detect initial break
            if state.pending_dir is None:
                if bar_high > orb.high:
                    state.pending_dir = "long"
                    state.pending_break_index = bar_index
                    state.pending_expires_at_index = bar_index + or_bar_count_needed
                    state.retest_confirmed = False
                elif bar_low < orb.low:
                    state.pending_dir = "short"
                    state.pending_break_index = bar_index
                    state.pending_expires_at_index = bar_index + or_bar_count_needed
                    state.retest_confirmed = False
                return None

            # Check expiry
            if bar_index > state.pending_expires_at_index:
                state.pending_dir = None
                state.pending_break_index = None
                state.retest_confirmed = False
                return None

            # Step 2: confirm retest
            if state.pending_dir == "long":
                # Retest: low touches OR high from above, close above
                if bar_low <= orb.high and bar_close > orb.high:
                    direction = "long"
                    state.pending_dir = None
                    state.pending_break_index = None
                    state.retest_confirmed = False
                    return direction
            elif state.pending_dir == "short":
                # Retest: high touches OR low from below, close below
                if bar_high >= orb.low and bar_close < orb.low:
                    direction = "short"
                    state.pending_dir = None
                    state.pending_break_index = None
                    state.retest_confirmed = False
                    return direction

            return None

        return None

    def _try_entry(
        self,
        state: ORBSessionState,
        bar_index: int,
        bar_close: float,
        ts_utc_iso: str,
        session_date_str: str,
        direction: str,
        p: ORBParams,
        equity: float,
        events: list[dict[str, Any]],
        _evt_base: Any,
    ) -> bool:
        """Attempt to open a position. Returns True if entered."""
        orb = state.orb_range
        assert orb is not None

        entry_price = bar_close
        stop = self._compute_stop(entry_price, direction, orb, p)
        risk_points = abs(entry_price - stop)

        if risk_points < 1e-9:
            return False

        target = self._compute_target(entry_price, direction, risk_points, p)
        size = self._compute_size(equity, risk_points, p)

        if size <= 0:
            return False

        state.position = ORBPosition(
            entry_bar=bar_index,
            entry_time=ts_utc_iso,
            entry_price=entry_price,
            stop=stop,
            target=target,
            side=direction,
            size=size,
            risk_points=risk_points,
        )
        state.trade_count += 1
        state.phase = ORBPhase.TRADE_MGMT

        evt = _evt_base("entry_signal", ORBPhase.TRADE_MGMT)
        evt.update(
            {
                "side": "buy" if direction == "long" else "sell",
                "price": entry_price,
                "stop": stop,
                "target": target,
                "size": size,
                "risk_points": risk_points,
            }
        )
        events.append(evt)
        return True

    @staticmethod
    def _compute_stop(
        entry: float,
        direction: str,
        orb: ORBRange,
        p: ORBParams,
    ) -> float:
        if p.stop_mode == "fixed-ticks":
            tick_value = p.fixed_ticks * 0.01  # ticks to price
            if direction == "long":
                return entry - tick_value
            return entry + tick_value
        # or-opposite (default)
        if direction == "long":
            return orb.low
        return orb.high

    @staticmethod
    def _compute_target(
        entry: float,
        direction: str,
        risk_points: float,
        p: ORBParams,
    ) -> float:
        offset = risk_points * p.target_r
        if direction == "long":
            return entry + offset
        return entry - offset

    @staticmethod
    def _compute_size(
        equity: float,
        risk_points: float,
        p: ORBParams,
    ) -> float:
        if risk_points < 1e-9:
            return 0.0
        raw = (equity * p.risk_pct) / (risk_points * p.point_value)
        return max(1.0, raw) if raw > 0 else 0.0

    @staticmethod
    def _close_position(
        pos: ORBPosition,
        exit_price: float,
        exit_time: str,
        exit_bar: int,
        exit_reason: str,
        p: ORBParams,
        commission_bps: float,
        slippage_bps: float,
    ) -> ORBClosedTrade:
        if pos.side == "long":
            raw_pnl = (exit_price - pos.entry_price) * pos.size * p.point_value
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.size * p.point_value

        # Commission: applied on entry + exit notional
        cost_bps = (commission_bps + slippage_bps) / 10000.0
        cost = (pos.entry_price + exit_price) * pos.size * cost_bps
        pnl = raw_pnl - cost

        return ORBClosedTrade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            side=pos.side,
            size=pos.size,
            pnl=pnl,
            exit_reason=exit_reason,
            entry_bar=pos.entry_bar,
            exit_bar=exit_bar,
        )

    @staticmethod
    def _build_result(
        closed_trades: list[ORBClosedTrade],
        equity_curve: list[dict[str, Any]],
        events: list[dict[str, Any]],
        initial_cash: float,
        final_equity: float,
        max_dd: float,
        df: pd.DataFrame,
        p: ORBParams,
        warnings: list[str],
    ) -> BacktestResult:
        num_trades = len(closed_trades)
        return_pct = (
            ((final_equity - initial_cash) / initial_cash * 100.0)
            if initial_cash > 0
            else 0.0
        )

        # Buy & hold
        if len(df) >= 2:
            first_close = float(df.iloc[0]["Close"])
            last_close = float(df.iloc[-1]["Close"])
            bh_return = (
                ((last_close - first_close) / first_close * 100.0)
                if first_close != 0
                else 0.0
            )
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
            pct = (t.pnl / initial_cash * 100.0) if initial_cash > 0 else 0.0
            total_return_pct += pct
            if t.pnl > 0:
                wins += 1
                gross_profit += t.pnl
            elif t.pnl < 0:
                gross_loss += abs(t.pnl)

            dur_bars = max(1, t.exit_bar - t.entry_bar)
            max_duration = max(max_duration, dur_bars)

            trade_dicts.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "pnl": t.pnl,
                    "return_pct": pct,
                    "duration_bars": dur_bars,
                    "exit_reason": t.exit_reason,
                }
            )

        win_rate = (wins / num_trades) if num_trades > 0 else 0.0
        avg_trade_pct = (total_return_pct / num_trades) if num_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        # Sharpe from equity returns
        sharpe: Optional[float] = None
        if len(equity_curve) > 1:
            equities = [e["equity"] for e in equity_curve]
            returns = []
            for i in range(1, len(equities)):
                if equities[i - 1] > 0:
                    returns.append((equities[i] - equities[i - 1]) / equities[i - 1])
            if returns:
                arr = np.array(returns)
                std = float(arr.std())
                if std > 0:
                    # Annualize: ~252 days * ~390 min / bar_minutes bars/day
                    bars_per_day = 150  # approx for 1-min intraday
                    sharpe = float(arr.mean() / std * math.sqrt(252 * bars_per_day))

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
            events=events,
            warnings=warnings,
        )
