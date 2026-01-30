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
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest.data import parse_ohlcv_csv
from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    run_unicorn_backtest,
    format_backtest_report,
    IntrabarPolicy,
)
from app.services.strategy.strategies.unicorn_model import (
    UnicornConfig,
    SessionProfile,
    BiasDirection,
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


def generate_sample_data(
    symbol: str,
    days: int = 30,
    profile: str = "realistic",
) -> tuple[list[OHLCVBar], list[OHLCVBar]]:
    """
    Generate synthetic data for testing with different realism profiles.

    Profiles:
        easy: Clean Gaussian random walk. Patterns are easy to detect, exits
              hit reliably. Good for sanity checking logic, NOT for validation.

        realistic: More realistic market microstructure:
            - Fat tails (Student-t distribution for shocks)
            - Vol clustering (GARCH-ish: high vol begets high vol)
            - Regime switching (trend vs range periods)
            - Occasional large wicks

        evil: Adversarial conditions that break naive strategies:
            - Everything in realistic, plus:
            - Overnight gaps (jump process)
            - Extreme wicks that hunt stops
            - More frequent regime changes
            - Fakeouts and failed patterns
    """
    import random
    import math

    print(f"Generating {days} days of synthetic {symbol} data (profile: {profile})...")

    # Start price based on symbol
    if "NQ" in symbol.upper():
        base_price = 17500.0
        tick_size = 0.25
        base_vol = 15.0  # NQ is more volatile
    else:  # ES
        base_price = 4800.0
        tick_size = 0.25
        base_vol = 5.0

    # Profile-specific parameters
    if profile == "easy":
        # Clean gaussian, low vol, no regime switching
        fat_tail_df = 100  # Effectively normal (high df = normal)
        vol_persistence = 0.0  # No vol clustering
        regime_switch_prob = 0.0
        gap_prob = 0.0
        wick_prob = 0.0
        wick_mult = 1.0
    elif profile == "realistic":
        # Student-t with ~5 df (fat tails), GARCH-ish vol
        fat_tail_df = 5
        vol_persistence = 0.7  # Vol clusters
        regime_switch_prob = 0.05  # ~5% chance to switch trend/range
        gap_prob = 0.02  # 2% overnight gap
        wick_prob = 0.08  # 8% large wick bars
        wick_mult = 2.5
    else:  # evil
        fat_tail_df = 3  # Very fat tails
        vol_persistence = 0.85  # Strong vol clustering
        regime_switch_prob = 0.12  # Frequent regime changes
        gap_prob = 0.08  # More gaps
        wick_prob = 0.15  # Many stop-hunting wicks
        wick_mult = 4.0

    def student_t_sample(df: float) -> float:
        """Sample from Student-t distribution (fat tails)."""
        if df >= 100:
            return random.gauss(0, 1)
        # Use ratio of normals approximation
        chi2 = sum(random.gauss(0, 1) ** 2 for _ in range(int(df))) / df
        return random.gauss(0, 1) / math.sqrt(chi2) if chi2 > 0 else 0

    htf_bars = []
    ltf_bars = []

    # State variables
    start_date = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    price = base_price
    current_vol = base_vol
    trend_bias = 0.0  # Positive = bullish, negative = bearish
    in_trend = random.random() < 0.5  # Start in trend or range

    for day in range(days):
        current_date = start_date.replace(day=start_date.day + day)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Overnight gap (jump process) at day start
        if random.random() < gap_prob:
            gap_size = student_t_sample(fat_tail_df) * base_vol * 3
            price += gap_size
            if profile != "easy":
                print(f"  Gap at {current_date.date()}: {gap_size:+.2f}")

        # Regime switching
        if random.random() < regime_switch_prob:
            in_trend = not in_trend
            if in_trend:
                trend_bias = random.choice([-1, 1]) * random.uniform(0.3, 0.8)
            else:
                trend_bias = 0.0

        # Generate 15m bars (HTF) for the day: 6:00 AM to 5:00 PM
        for hour in range(6, 17):
            for minute in [0, 15, 30, 45]:
                ts = current_date.replace(hour=hour, minute=minute)

                # GARCH-ish vol clustering
                vol_shock = abs(student_t_sample(fat_tail_df))
                current_vol = (vol_persistence * current_vol +
                              (1 - vol_persistence) * base_vol +
                              vol_shock * 0.5)
                current_vol = max(base_vol * 0.5, min(base_vol * 3, current_vol))

                # Price move with fat tails
                noise = student_t_sample(fat_tail_df) * current_vol * 0.1
                move = trend_bias + noise

                open_ = price
                close = price + move

                # Normal high/low
                high = max(open_, close) + random.uniform(0, current_vol * 0.3)
                low = min(open_, close) - random.uniform(0, current_vol * 0.3)

                # Occasional large wicks (stop hunting)
                if random.random() < wick_prob:
                    if random.random() < 0.5:
                        # Wick down then recover
                        low = min(open_, close) - current_vol * wick_mult
                    else:
                        # Wick up then drop
                        high = max(open_, close) + current_vol * wick_mult

                htf_bars.append(OHLCVBar(
                    ts=ts,
                    open=round(open_ / tick_size) * tick_size,
                    high=round(high / tick_size) * tick_size,
                    low=round(low / tick_size) * tick_size,
                    close=round(close / tick_size) * tick_size,
                    volume=random.randint(500, 5000),
                ))

                price = close

        # Generate 5m bars (LTF) - track HTF price but with more noise
        ltf_price = htf_bars[-44].open if len(htf_bars) >= 44 else price

        for hour in range(6, 17):
            for minute in range(0, 60, 5):
                ts = current_date.replace(hour=hour, minute=minute)

                # LTF has same vol characteristics but smaller moves
                noise = student_t_sample(fat_tail_df) * current_vol * 0.03
                move = (trend_bias * 0.33) + noise

                open_ = ltf_price
                close = ltf_price + move

                high = max(open_, close) + random.uniform(0, current_vol * 0.1)
                low = min(open_, close) - random.uniform(0, current_vol * 0.1)

                # LTF wicks
                if random.random() < wick_prob * 0.5:
                    if random.random() < 0.5:
                        low = min(open_, close) - current_vol * wick_mult * 0.5
                    else:
                        high = max(open_, close) + current_vol * wick_mult * 0.5

                ltf_bars.append(OHLCVBar(
                    ts=ts,
                    open=round(open_ / tick_size) * tick_size,
                    high=round(high / tick_size) * tick_size,
                    low=round(low / tick_size) * tick_size,
                    close=round(close / tick_size) * tick_size,
                    volume=random.randint(100, 1500),
                ))

                ltf_price = close

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

    # Fetch real NQ data from Databento API (requires DATABENTO_API_KEY env var)
    python scripts/run_unicorn_backtest.py --symbol NQ --databento --start-date 2024-01-01 --end-date 2024-01-31

    # Estimate Databento API cost before fetching
    python scripts/run_unicorn_backtest.py --symbol NQ --databento --start-date 2024-01-01 --end-date 2024-01-31 --cost-only

    # Load from local Databento CSV (no API key needed)
    python scripts/run_unicorn_backtest.py --symbol NQ --databento-csv path/to/glbx-data.csv --start-date 2024-01-01 --end-date 2024-01-31

    # Realistic friction settings
    python scripts/run_unicorn_backtest.py --symbol NQ --synthetic --slippage-ticks 2 --commission 4.50 --intrabar-policy worst
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
    # Databento real data options
    parser.add_argument(
        "--databento",
        action="store_true",
        help="Fetch real data from Databento API (requires DATABENTO_API_KEY)"
    )
    parser.add_argument(
        "--databento-csv",
        help="Load from local Databento CSV file instead of API (can be .csv or .csv.zst)"
    )
    parser.add_argument(
        "--start-date",
        help="Start date for Databento data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="End date for Databento data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cost-only",
        action="store_true",
        help="Only estimate Databento API cost, don't fetch data"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local cache for Databento data"
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
        default=3,
        help="Minimum SCORED criteria (out of 5, not 8). Mandatory criteria always required. (default: 3)"
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
    # Friction parameters (critical for realistic backtesting)
    parser.add_argument(
        "--slippage-ticks",
        type=float,
        default=1.0,
        help="Slippage per side in ticks (default: 1 tick each way)"
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=2.50,
        help="Round-trip commission per contract in dollars (default: $2.50)"
    )
    parser.add_argument(
        "--intrabar-policy",
        choices=["worst", "best", "random", "ohlc_path"],
        default="worst",
        help="How to resolve stop/target ambiguity when both hit in same bar. "
             "worst=stop first (conservative), best=target first, random=50/50, "
             "ohlc_path=deterministic O→H→L→C / O→L→H→C (default: worst)"
    )
    # Synthetic data profiles
    parser.add_argument(
        "--synthetic-profile",
        choices=["easy", "realistic", "evil"],
        default="realistic",
        help="Synthetic data profile: easy (clean gaussian), realistic (fat tails, vol clustering), "
             "evil (extreme conditions, gaps, wicks). (default: realistic)"
    )
    # Direction filter (diagnostics showed longs profitable, shorts bleed)
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Only take long trades. Diagnostics showed longs are profitable, shorts bleed."
    )
    # Time-stop: cut grindy losers early
    parser.add_argument(
        "--time-stop",
        type=int,
        metavar="MINUTES",
        help="Exit if not at +0.25R within N minutes. ICT entries that work tend to move fast."
    )
    parser.add_argument(
        "--time-stop-threshold",
        type=float,
        default=0.25,
        help="R-multiple threshold for time-stop (default: 0.25R)"
    )
    parser.add_argument(
        "--max-wick-ratio", type=float, default=None,
        help="Skip entry if signal bar adverse wick ratio exceeds this (0-1). None=disabled."
    )
    parser.add_argument(
        "--max-range-atr", type=float, default=None,
        help="Skip entry if signal bar range exceeds this ATR multiple. None=disabled."
    )

    args = parser.parse_args()

    # Load or generate data
    if args.databento_csv:
        # Load from local Databento CSV file
        from app.services.backtest.data import DatabentoFetcher

        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required when using --databento-csv")

        fetcher = DatabentoFetcher()

        print(f"Loading {args.symbol} data from {args.databento_csv}...")
        print(f"  Date range: {args.start_date} to {args.end_date}")
        htf_bars, ltf_bars = fetcher.load_from_csv(
            csv_path=args.databento_csv,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            htf_interval="15m",
            ltf_interval="5m",
        )
        print(f"Loaded {len(htf_bars)} HTF (15m) bars and {len(ltf_bars)} LTF (5m) bars")

    elif args.databento:
        # Fetch real data from Databento API
        from app.services.backtest.data import DatabentoFetcher, get_continuous_symbols

        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required when using --databento")

        fetcher = DatabentoFetcher()

        if args.cost_only:
            # Just estimate cost and exit
            from datetime import datetime
            start_dt = datetime.fromisoformat(args.start_date)
            end_dt = datetime.fromisoformat(args.end_date)
            contracts = get_continuous_symbols(args.symbol, start_dt, end_dt)

            total_cost = 0.0
            print(f"\nContracts needed for {args.symbol} {args.start_date} to {args.end_date}:")
            for contract, period_start, period_end in contracts:
                cost = fetcher.estimate_cost(
                    symbol=contract,
                    start_date=period_start.strftime("%Y-%m-%d"),
                    end_date=period_end.strftime("%Y-%m-%d"),
                )
                print(f"  {contract}: {period_start.date()} to {period_end.date()} - ${cost:.2f}")
                total_cost += cost

            print(f"\nTotal estimated cost: ${total_cost:.2f}")
            print("(Note: New accounts get $125 free credits)")
            sys.exit(0)

        print(f"Fetching {args.symbol} data from Databento ({args.start_date} to {args.end_date})...")
        htf_bars, ltf_bars = fetcher.fetch_futures_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            htf_interval="15m",
            ltf_interval="5m",
            use_cache=not args.no_cache,
        )
        print(f"Fetched {len(htf_bars)} HTF (15m) bars and {len(ltf_bars)} LTF (5m) bars")

    elif args.synthetic:
        htf_bars, ltf_bars = generate_sample_data(
            args.symbol, args.days, profile=args.synthetic_profile
        )
        print(f"Generated {len(htf_bars)} HTF bars and {len(ltf_bars)} LTF bars")

    else:
        # Load from CSV files
        if not args.htf or not args.ltf:
            parser.error("--htf and --ltf are required unless using --synthetic or --databento")

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
        max_wick_ratio=args.max_wick_ratio,
        max_range_atr_mult=args.max_range_atr,
    )

    # Run backtest
    print(f"\nRunning Unicorn Model backtest for {args.symbol}...")
    print(f"Risk per trade: ${args.dollars_per_trade:,.2f}")
    print(f"Max concurrent positions: {args.max_concurrent}")
    print(f"Criteria: 3 mandatory + {args.min_criteria}/5 scored (guardrailed soft scoring)")
    print(f"Session profile: {args.session_profile}")
    print(f"FVG ATR mult: {args.fvg_atr_mult}, Stop ATR mult: {args.stop_atr_mult}")
    print(f"Friction: {args.slippage_ticks} ticks slippage, ${args.commission:.2f} commission")
    print(f"Intrabar policy: {args.intrabar_policy}")
    if args.long_only:
        print(f"Direction filter: LONG ONLY")
    if args.time_stop:
        print(f"Time-stop: exit if not at +{args.time_stop_threshold}R within {args.time_stop} minutes")
    if args.max_wick_ratio is not None:
        print(f"Wick guard: max adverse wick ratio = {args.max_wick_ratio}")
    if args.max_range_atr is not None:
        print(f"Range guard: max signal bar range = {args.max_range_atr}x ATR")
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
        slippage_ticks=args.slippage_ticks,
        commission_per_contract=args.commission,
        intrabar_policy=IntrabarPolicy(args.intrabar_policy),
        direction_filter=BiasDirection.BULLISH if args.long_only else None,
        time_stop_minutes=args.time_stop,
        time_stop_r_threshold=args.time_stop_threshold,
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
