#!/usr/bin/env python3
"""
Session profile sensitivity sweep.

Runs 5 windows × 4 profiles × 2 instruments = 40 backtests.
Produces decision-grade tables for docs.

Usage:
    python scripts/run_session_sweep.py \
        --csv docs/historical_data/GLBX-20260129-JNB8PDSQ7C/glbx-mdp3-20210128-20260127.ohlcv-1m.csv
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest.data import DatabentoFetcher
from app.services.backtest.engines.unicorn_runner import (
    IntrabarPolicy,
    run_unicorn_backtest,
    TradeOutcome,
)
from app.services.strategy.strategies.unicorn_model import (
    SessionProfile,
    UnicornConfig,
)

# ── Sweep matrix ──────────────────────────────────────────────────────────────

PROFILES = [
    SessionProfile.WIDE,
    SessionProfile.NORMAL,
    SessionProfile.STRICT,
    SessionProfile.NY_OPEN,
]

WINDOWS = [
    ("W1", "2021-02-01", "2021-07-31", "Recovery"),
    ("W2", "2022-04-01", "2022-09-30", "Bear"),
    ("W3", "2023-01-01", "2023-06-30", "AI rally"),
    ("W4", "2024-01-01", "2024-06-30", "Bull continuation"),
    ("W5", "2024-10-01", "2025-03-31", "Recent"),
]

INSTRUMENTS = ["NQ", "ES"]

# Production guards
GUARD_CONFIG = dict(
    max_wick_ratio=0.6,
    min_displacement_atr=0.5,
)


@dataclass
class RunResult:
    window: str
    profile: str
    symbol: str
    trades: int
    wins: int
    win_pct: float
    expectancy: float  # pts/trade
    pnl: float  # dollars
    profit_factor: float
    macro_rejected_pct: float  # % of setups where macro_window=False
    # Per-session breakdown (WIDE only)
    session_expectancy: dict  # session -> expectancy pts/trade (taken trades)
    session_pnl: dict  # session -> total PnL pts (taken trades)
    session_trades: dict  # session -> trade count
    session_setups: dict  # session -> total setup count (all qualifying)


def run_single(symbol, htf_bars, ltf_bars, profile):
    """Run one backtest with production guards."""
    config = UnicornConfig(
        session_profile=profile,
        **GUARD_CONFIG,
    )

    result = run_unicorn_backtest(
        symbol=symbol,
        htf_bars=htf_bars,
        ltf_bars=ltf_bars,
        dollars_per_trade=1000.0,
        config=config,
        intrabar_policy=IntrabarPolicy.WORST,
    )

    # Macro rejection rate from setup diagnostics
    total_setups = len(result.all_setups)
    macro_rejected = sum(
        1 for s in result.all_setups if not s.setup_in_macro_window
    )
    macro_rejected_pct = (
        macro_rejected / total_setups * 100 if total_setups > 0 else 0.0
    )

    # Per-session breakdown (from setup_session diagnostic)
    session_trades = defaultdict(int)
    session_pnl = defaultdict(float)
    session_setups = defaultdict(int)

    for setup in result.all_setups:
        sess = setup.setup_session or "unknown"
        session_setups[sess] += 1

    for trade in result.trades:
        # Match trade back to session via entry time
        matching = [
            s for s in result.all_setups
            if s.timestamp == trade.entry_time and s.taken
        ]
        if matching:
            sess = matching[0].setup_session or "unknown"
        else:
            sess = trade.session.value
        session_trades[sess] += 1
        session_pnl[sess] += trade.pnl_points

    session_expectancy = {}
    for sess, count in session_trades.items():
        if count > 0:
            session_expectancy[sess] = session_pnl[sess] / count
        else:
            session_expectancy[sess] = 0.0

    return RunResult(
        window="",
        profile=profile.value,
        symbol=symbol,
        trades=result.trades_taken,
        wins=result.wins,
        win_pct=result.win_rate * 100,
        expectancy=result.expectancy_points,
        pnl=result.total_pnl_dollars,
        profit_factor=result.profit_factor,
        macro_rejected_pct=macro_rejected_pct,
        session_expectancy=dict(session_expectancy),
        session_pnl=dict(session_pnl),
        session_trades=dict(session_trades),
        session_setups=dict(session_setups),
    )


def print_separator(char="=", width=100):
    print(char * width)


def print_main_table(results, symbol):
    """Print per-window × profile table for one instrument."""
    print()
    print_separator()
    print(f"  SESSION PROFILE SWEEP — {symbol}")
    print_separator()
    print(
        f"{'Window':<6} {'Profile':<10} {'Trades':>7} {'Win%':>7} "
        f"{'Expect':>9} {'PnL($)':>11} {'PF':>7} {'MacroRej%':>10}"
    )
    print_separator("-")

    for wname, _, _, wchar in WINDOWS:
        for profile in PROFILES:
            key = (wname, profile.value, symbol)
            r = results.get(key)
            if r is None:
                continue
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
            print(
                f"{wname:<6} {r.profile:<10} {r.trades:>7} {r.win_pct:>6.1f}% "
                f"{r.expectancy:>+9.2f} {r.pnl:>+11.2f} {pf_str:>7} {r.macro_rejected_pct:>9.1f}%"
            )
        print()  # blank between windows


def print_wide_session_breakdown(results, symbol):
    """Print per-session breakdown for WIDE runs."""
    print()
    print_separator()
    print(f"  WIDE PER-SESSION BREAKDOWN — {symbol} (taken trades)")
    print_separator()
    print(
        f"{'Window':<6} {'Session':<12} {'Trades':>7} {'Expect':>9} {'PnL(pts)':>10}"
    )
    print_separator("-")

    sessions_order = ["ny_am", "london", "ny_pm", "asia", "off_hours"]

    for wname, _, _, _ in WINDOWS:
        key = (wname, "wide", symbol)
        r = results.get(key)
        if r is None:
            continue
        for sess in sessions_order:
            tc = r.session_trades.get(sess, 0)
            exp = r.session_expectancy.get(sess, 0.0)
            pnl = r.session_pnl.get(sess, 0.0)
            if tc == 0:
                continue
            print(
                f"{wname:<6} {sess:<12} {tc:>7} {exp:>+9.2f} {pnl:>+10.2f}"
            )
        print()

    # Setup distribution (all qualifying, not just taken)
    print()
    print_separator()
    print(f"  WIDE PER-SESSION SETUP DISTRIBUTION — {symbol} (all qualifying setups)")
    print_separator()
    print(f"{'Window':<6} {'Session':<12} {'Setups':>7}")
    print_separator("-")
    for wname, _, _, _ in WINDOWS:
        key = (wname, "wide", symbol)
        r = results.get(key)
        if r is None:
            continue
        for sess in sessions_order:
            sc = r.session_setups.get(sess, 0)
            if sc == 0:
                continue
            print(f"{wname:<6} {sess:<12} {sc:>7}")
        print()


def print_expectancy_delta(results, symbol):
    """Print expectancy delta vs WIDE baseline."""
    print()
    print_separator()
    print(f"  EXPECTANCY DELTA vs WIDE — {symbol}")
    print_separator()
    print(f"{'Window':<6} {'WIDE':>9} {'NORMAL':>9} {'STRICT':>9} {'NY_OPEN':>9}")
    print_separator("-")

    for wname, _, _, _ in WINDOWS:
        wide_key = (wname, "wide", symbol)
        wide_r = results.get(wide_key)
        if wide_r is None:
            continue
        baseline = wide_r.expectancy

        cols = [f"{baseline:>+9.2f}"]
        for profile in [SessionProfile.NORMAL, SessionProfile.STRICT, SessionProfile.NY_OPEN]:
            key = (wname, profile.value, symbol)
            r = results.get(key)
            if r is None:
                cols.append(f"{'N/A':>9}")
            else:
                delta = r.expectancy - baseline
                cols.append(f"{delta:>+9.2f}")
        print(f"{wname:<6} {'  '.join(cols)}")


def apply_decision_rules(results, symbol):
    """Apply mechanical decision rules and print verdict."""
    print()
    print_separator()
    print(f"  DECISION RULES — {symbol}")
    print_separator()

    # STRICT vs NORMAL
    strict_wins_exp = 0
    normal_total_trades = 0
    strict_total_trades = 0
    for wname, _, _, _ in WINDOWS:
        n_key = (wname, "normal", symbol)
        s_key = (wname, "strict", symbol)
        n = results.get(n_key)
        s = results.get(s_key)
        if n is None or s is None:
            continue
        if s.expectancy >= n.expectancy:
            strict_wins_exp += 1
        normal_total_trades += n.trades
        strict_total_trades += s.trades

    trade_retention = (
        strict_total_trades / normal_total_trades * 100
        if normal_total_trades > 0
        else 0
    )
    strict_pass = strict_wins_exp >= 4 and trade_retention >= 60

    print(f"STRICT vs NORMAL:")
    print(f"  Expectancy >= NORMAL in {strict_wins_exp}/5 windows (need >=4): {'PASS' if strict_wins_exp >= 4 else 'FAIL'}")
    print(f"  Trade retention: {trade_retention:.1f}% (need >=60%): {'PASS' if trade_retention >= 60 else 'FAIL'}")
    print(f"  --> {'STRICT replaces NORMAL' if strict_pass else 'Keep NORMAL'}")
    print()

    # NY_OPEN vs STRICT
    ny_wins_exp = 0
    strict_total2 = 0
    ny_total = 0
    for wname, _, _, _ in WINDOWS:
        s_key = (wname, "strict", symbol)
        ny_key = (wname, "ny_open", symbol)
        s = results.get(s_key)
        ny = results.get(ny_key)
        if s is None or ny is None:
            continue
        if ny.expectancy >= s.expectancy:
            ny_wins_exp += 1
        strict_total2 += s.trades
        ny_total += ny.trades

    ny_retention = (
        ny_total / strict_total2 * 100 if strict_total2 > 0 else 0
    )
    ny_pass = ny_wins_exp >= 4 and ny_retention >= 70

    print(f"NY_OPEN vs STRICT:")
    print(f"  Expectancy >= STRICT in {ny_wins_exp}/5 windows (need >=4): {'PASS' if ny_wins_exp >= 4 else 'FAIL'}")
    print(f"  Trade retention: {ny_retention:.1f}% (need >=70%): {'PASS' if ny_retention >= 70 else 'FAIL'}")
    print(f"  --> {'NY_OPEN replaces STRICT' if ny_pass else 'Keep STRICT'}")
    print()

    # Final recommendation
    if ny_pass:
        rec = "NY_OPEN"
    elif strict_pass:
        rec = "STRICT"
    else:
        rec = "NORMAL (no change)"
    print(f"  RECOMMENDATION for {symbol}: --session-profile {rec.lower().split()[0]}")


def main():
    parser = argparse.ArgumentParser(description="Session profile sensitivity sweep")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to Databento OHLCV 1m CSV (covers 2021-2025)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=INSTRUMENTS,
        help="Instruments to sweep (default: NQ ES)",
    )
    args = parser.parse_args()

    fetcher = DatabentoFetcher()
    results = {}  # (window, profile, symbol) -> RunResult

    total_runs = len(args.symbols) * len(WINDOWS) * len(PROFILES)
    run_count = 0

    for symbol in args.symbols:
        for wname, wstart, wend, wchar in WINDOWS:
            # Load data once per (symbol, window)
            print(f"\n--- Loading {symbol} {wname} ({wstart} → {wend}, {wchar}) ---")
            htf_bars, ltf_bars = fetcher.load_from_csv(
                csv_path=args.csv,
                symbol=symbol,
                start_date=wstart,
                end_date=wend,
                htf_interval="15m",
                ltf_interval="5m",
            )
            print(f"  {len(htf_bars)} HTF bars, {len(ltf_bars)} LTF bars")

            if len(htf_bars) < 50 or len(ltf_bars) < 30:
                print(f"  SKIP: insufficient bars")
                continue

            for profile in PROFILES:
                run_count += 1
                print(
                    f"  [{run_count}/{total_runs}] {symbol} {wname} {profile.value}...",
                    end="",
                    flush=True,
                )
                r = run_single(symbol, htf_bars, ltf_bars, profile)
                r.window = wname
                key = (wname, profile.value, symbol)
                results[key] = r
                print(
                    f" {r.trades} trades, {r.win_pct:.1f}% WR, "
                    f"{r.expectancy:+.2f} exp, ${r.pnl:+,.0f}"
                )

    # ── Reports ──────────────────────────────────────────────────────────────
    for symbol in args.symbols:
        print_main_table(results, symbol)
        print_expectancy_delta(results, symbol)
        print_wide_session_breakdown(results, symbol)
        apply_decision_rules(results, symbol)

    print()
    print_separator("=")
    print("  SWEEP COMPLETE")
    print_separator("=")


if __name__ == "__main__":
    main()
