#!/usr/bin/env python3
"""
CLI script to run Unicorn Model backtests.

Usage:
    python scripts/run_unicorn_backtest.py --symbol NQ --htf data/nq_15m.csv --ltf data/nq_5m.csv

CSV format expected:
    date,open,high,low,close,volume
    2024-01-02 09:30:00,17000.00,17005.00,16995.00,17003.00,1000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest.data import parse_ohlcv_csv
from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    run_unicorn_backtest,
    format_backtest_report,
)
from app.services.strategy.strategies.unicorn_model import (
    UnicornConfig,
    SessionProfile,
)


def load_bars_from_csv(filepath: str) -> list[OHLCVBar]:
    """Load OHLCVBars from a CSV file."""
    with open(filepath, "rb") as f:
        content = f.read()

    result = parse_ohlcv_csv(content)

    bars = []
    for idx, row in result.df.iterrows():
        bar = OHLCVBar(
            ts=idx.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]) if row["Volume"] else 0,
        )
        bars.append(bar)

    return bars


def generate_sample_data(symbol: str, days: int = 30) -> tuple[list[OHLCVBar], list[OHLCVBar]]:
    """Generate synthetic data for testing when no CSV is provided."""
    import random

    print(f"Generating {days} days of synthetic {symbol} data...")

    # Start price based on symbol
    if "NQ" in symbol.upper():
        base_price = 17500.0
        tick_size = 0.25
    else:  # ES
        base_price = 4800.0
        tick_size = 0.25

    htf_bars = []
    ltf_bars = []

    # Generate data for each trading day
    start_date = datetime(2024, 1, 2, 0, 0)
    price = base_price

    for day in range(days):
        current_date = start_date.replace(day=start_date.day + day)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Generate 15m bars (HTF) for the day: 6:00 AM to 5:00 PM = ~44 bars
        for hour in range(6, 17):
            for minute in [0, 15, 30, 45]:
                ts = current_date.replace(hour=hour, minute=minute)

                # Add trend and noise
                trend = random.gauss(0, 0.5)  # Small random walk
                volatility = random.uniform(5, 20)

                open_ = price
                close = price + trend
                high = max(open_, close) + random.uniform(0, volatility)
                low = min(open_, close) - random.uniform(0, volatility)

                htf_bars.append(OHLCVBar(
                    ts=ts,
                    open=round(open_ / tick_size) * tick_size,
                    high=round(high / tick_size) * tick_size,
                    low=round(low / tick_size) * tick_size,
                    close=round(close / tick_size) * tick_size,
                    volume=random.randint(500, 5000),
                ))

                price = close

        # Generate 5m bars (LTF) - 3x as many
        price = base_price + (day * random.gauss(0, 10))

        for hour in range(6, 17):
            for minute in range(0, 60, 5):
                ts = current_date.replace(hour=hour, minute=minute)

                trend = random.gauss(0, 0.2)
                volatility = random.uniform(2, 8)

                open_ = price
                close = price + trend
                high = max(open_, close) + random.uniform(0, volatility)
                low = min(open_, close) - random.uniform(0, volatility)

                ltf_bars.append(OHLCVBar(
                    ts=ts,
                    open=round(open_ / tick_size) * tick_size,
                    high=round(high / tick_size) * tick_size,
                    low=round(low / tick_size) * tick_size,
                    close=round(close / tick_size) * tick_size,
                    volume=random.randint(100, 1500),
                ))

                price = close

    return htf_bars, ltf_bars


def main():
    parser = argparse.ArgumentParser(
        description="Run ICT Unicorn Model backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with CSV files
    python scripts/run_unicorn_backtest.py --symbol NQ --htf data/nq_15m.csv --ltf data/nq_5m.csv

    # Run with synthetic data (for testing)
    python scripts/run_unicorn_backtest.py --symbol NQ --synthetic --days 60

    # Customize risk parameters
    python scripts/run_unicorn_backtest.py --symbol ES --synthetic --dollars-per-trade 2000
        """
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (NQ, ES, MNQ, MES)"
    )
    parser.add_argument(
        "--htf",
        help="Path to HTF (15m) CSV file"
    )
    parser.add_argument(
        "--ltf",
        help="Path to LTF (5m) CSV file"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of loading from CSV"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of synthetic data to generate (default: 30)"
    )
    parser.add_argument(
        "--dollars-per-trade",
        type=float,
        default=1000.0,
        help="Risk amount per trade in dollars (default: 1000)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent positions (default: 1)"
    )
    parser.add_argument(
        "--no-eod-exit",
        action="store_true",
        help="Disable automatic EOD exit"
    )
    parser.add_argument(
        "--output",
        help="Output file for report (default: stdout)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text report"
    )
    parser.add_argument(
        "--min-criteria",
        type=int,
        default=6,
        help="Minimum criteria score for soft scoring (default: 6 of 8)"
    )
    parser.add_argument(
        "--session-profile",
        choices=["strict", "normal", "wide"],
        default="normal",
        help="Session profile: strict (NY AM only), normal (London+NY AM), wide (all sessions)"
    )
    parser.add_argument(
        "--fvg-atr-mult",
        type=float,
        default=0.3,
        help="FVG minimum size as ATR multiple (default: 0.3)"
    )
    parser.add_argument(
        "--stop-atr-mult",
        type=float,
        default=3.0,
        help="Max stop distance as ATR multiple (default: 3.0)"
    )

    args = parser.parse_args()

    # Load or generate data
    if args.synthetic:
        htf_bars, ltf_bars = generate_sample_data(args.symbol, args.days)
        print(f"Generated {len(htf_bars)} HTF bars and {len(ltf_bars)} LTF bars")
    else:
        if not args.htf or not args.ltf:
            parser.error("--htf and --ltf are required unless using --synthetic")

        print(f"Loading HTF data from {args.htf}...")
        htf_bars = load_bars_from_csv(args.htf)
        print(f"Loaded {len(htf_bars)} HTF bars")

        print(f"Loading LTF data from {args.ltf}...")
        ltf_bars = load_bars_from_csv(args.ltf)
        print(f"Loaded {len(ltf_bars)} LTF bars")

    # Build config
    config = UnicornConfig(
        min_criteria_score=args.min_criteria,
        session_profile=SessionProfile(args.session_profile),
        fvg_min_atr_mult=args.fvg_atr_mult,
        stop_max_atr_mult=args.stop_atr_mult,
    )

    # Run backtest
    print(f"\nRunning Unicorn Model backtest for {args.symbol}...")
    print(f"Risk per trade: ${args.dollars_per_trade:,.2f}")
    print(f"Max concurrent positions: {args.max_concurrent}")
    print(f"Min criteria score: {args.min_criteria}/8 (soft scoring)")
    print(f"Session profile: {args.session_profile}")
    print(f"FVG ATR mult: {args.fvg_atr_mult}, Stop ATR mult: {args.stop_atr_mult}")
    print("")

    result = run_unicorn_backtest(
        symbol=args.symbol,
        htf_bars=htf_bars,
        ltf_bars=ltf_bars,
        dollars_per_trade=args.dollars_per_trade,
        max_concurrent_trades=args.max_concurrent,
        eod_exit=not args.no_eod_exit,
        min_criteria_score=args.min_criteria,
        config=config,
    )

    # Format output
    if args.json:
        import json
        from dataclasses import asdict

        # Convert to JSON-serializable format
        output = {
            "symbol": result.symbol,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "total_bars": result.total_bars,
            "total_setups_scanned": result.total_setups_scanned,
            "partial_setups": result.partial_setups,
            "valid_setups": result.valid_setups,
            "trades_taken": result.trades_taken,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_pnl_handles": result.total_pnl_handles,
            "total_pnl_dollars": result.total_pnl_dollars,
            "expectancy_handles": result.expectancy_handles,
            "avg_mfe": result.avg_mfe,
            "avg_mae": result.avg_mae,
            "mfe_capture_rate": result.mfe_capture_rate,
            "avg_r_multiple": result.avg_r_multiple,
            "confidence_win_correlation": result.confidence_win_correlation,
            "criteria_bottlenecks": [
                {"criterion": b.criterion, "fail_rate": b.fail_rate}
                for b in result.criteria_bottlenecks
            ],
            "session_stats": {
                s.value: {
                    "total_setups": result.session_stats[s].total_setups,
                    "valid_setups": result.session_stats[s].valid_setups,
                    "trades_taken": result.session_stats[s].trades_taken,
                    "win_rate": result.session_stats[s].win_rate,
                    "total_pnl": result.session_stats[s].total_pnl_handles,
                }
                for s in result.session_stats
            },
            "confidence_buckets": [
                {
                    "range": f"{b.min_confidence:.1f}-{b.max_confidence:.1f}",
                    "trades": b.trade_count,
                    "win_rate": b.win_rate,
                    "avg_r": b.avg_r_multiple,
                }
                for b in result.confidence_buckets
            ],
        }
        report = json.dumps(output, indent=2)
    else:
        report = format_backtest_report(result)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Quick summary to stderr if outputting to file
    if args.output and not args.json:
        print(f"\nQuick Summary:")
        print(f"  Trades: {result.trades_taken} | Win Rate: {result.win_rate*100:.1f}%")
        print(f"  PnL: {result.total_pnl_handles:+.2f} handles (${result.total_pnl_dollars:+,.2f})")
        print(f"  Expectancy: {result.expectancy_handles:+.2f} handles/trade")
        print(f"  Top Bottleneck: {result.criteria_bottlenecks[0].criterion if result.criteria_bottlenecks else 'N/A'}")


if __name__ == "__main__":
    main()
