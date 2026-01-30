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
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest.data import parse_ohlcv_csv
from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    run_unicorn_backtest,
    format_backtest_report,
    format_trade_trace,
    IntrabarPolicy,
    BiasState,
    BarBundle,
)
from app.services.strategy.strategies.unicorn_model import (
    UnicornConfig,
    SessionProfile,
    BiasDirection,
)
from app.services.strategy.indicators.tf_bias import compute_tf_bias


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
    multi_tf: bool = False,
):
    """
    Generate synthetic data for testing with different realism profiles.

    Args:
        symbol: Root symbol (NQ, ES)
        days: Number of trading days
        profile: "easy", "realistic", or "evil"
        multi_tf: If True, return BarBundle with all TFs (1m base).
                  If False, return legacy (htf_bars, ltf_bars) tuple.

    Returns:
        BarBundle when multi_tf=True, tuple[list[OHLCVBar], list[OHLCVBar]] otherwise.

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

    if not multi_tf:
        return htf_bars, ltf_bars

    # Multi-TF mode: generate 1m bars from the same price path, then resample
    print("  Generating 1m base bars for multi-TF resampling...")
    m1_bars = []
    m1_price = htf_bars[0].open if htf_bars else base_price

    for day in range(days):
        current_date = start_date.replace(day=start_date.day + day)
        if current_date.weekday() >= 5:
            continue

        for hour in range(6, 17):
            for minute in range(0, 60):
                ts = current_date.replace(hour=hour, minute=minute)

                noise = student_t_sample(fat_tail_df) * current_vol * 0.01
                move = (trend_bias * 0.067) + noise

                open_ = m1_price
                close = m1_price + move
                high = max(open_, close) + random.uniform(0, current_vol * 0.03)
                low = min(open_, close) - random.uniform(0, current_vol * 0.03)

                if random.random() < wick_prob * 0.3:
                    if random.random() < 0.5:
                        low = min(open_, close) - current_vol * wick_mult * 0.2
                    else:
                        high = max(open_, close) + current_vol * wick_mult * 0.2

                m1_bars.append(OHLCVBar(
                    ts=ts,
                    open=round(open_ / tick_size) * tick_size,
                    high=round(high / tick_size) * tick_size,
                    low=round(low / tick_size) * tick_size,
                    close=round(close / tick_size) * tick_size,
                    volume=random.randint(50, 500),
                ))

                m1_price = close

    # Resample from 1m to all TFs
    from app.services.backtest.data import DatabentoFetcher
    fetcher = DatabentoFetcher()

    bundle = BarBundle(
        h4=fetcher._resample_bars(m1_bars, "4h"),
        h1=fetcher._resample_bars(m1_bars, "1h"),
        m15=fetcher._resample_bars(m1_bars, "15m"),
        m5=fetcher._resample_bars(m1_bars, "5m"),
        m1=m1_bars,
    )
    print(f"  Multi-TF: {len(bundle.h4)} 4H, {len(bundle.h1)} 1H, "
          f"{len(bundle.m15)} 15m, {len(bundle.m5)} 5m, {len(bundle.m1)} 1m bars")

    return bundle


# ---------------------------------------------------------------------------
# Reference-symbol bias series builder
# ---------------------------------------------------------------------------

REF_HTF_LOOKBACK = 100  # HTF bars fed to compute_tf_bias per evaluation point
REF_LTF_LOOKBACK = 60   # LTF bars (optional) fed per evaluation point


def _sort_and_normalize_tz(bars: list[OHLCVBar]) -> list[OHLCVBar]:
    """Sort bars by timestamp and ensure all are UTC-aware."""
    sorted_bars = sorted(bars, key=lambda b: b.ts)
    normalized = []
    for b in sorted_bars:
        ts = b.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts != b.ts:
            b = OHLCVBar(ts=ts, open=b.open, high=b.high, low=b.low,
                         close=b.close, volume=b.volume)
        normalized.append(b)
    return normalized


def load_ref_data(
    args: argparse.Namespace,
    ref_htf_path: str | None,
) -> tuple[list[OHLCVBar], list[OHLCVBar], str]:
    """
    Load reference symbol data using the same source as the primary symbol.

    Returns (ref_htf_bars, ref_ltf_bars, source_label).
    LTF may be empty — build_reference_bias_series tolerates that.
    """
    ref_symbol = args.ref_symbol

    if ref_htf_path:
        # Explicit CSV path for reference HTF
        print(f"Loading reference HTF data from {ref_htf_path}...")
        ref_htf = load_bars_from_csv(ref_htf_path)
        print(f"  Loaded {len(ref_htf)} reference HTF bars")
        return _sort_and_normalize_tz(ref_htf), [], "csv"

    if args.databento_csv:
        from app.services.backtest.data import DatabentoFetcher

        fetcher = DatabentoFetcher()
        print(f"Loading reference {ref_symbol} from Databento CSV {args.databento_csv}...")
        ref_htf, ref_ltf = fetcher.load_from_csv(
            csv_path=args.databento_csv,
            symbol=ref_symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            htf_interval="15m",
            ltf_interval="5m",
        )
        ref_htf = _sort_and_normalize_tz(ref_htf)
        ref_ltf = _sort_and_normalize_tz(ref_ltf)
        print(f"  Loaded {len(ref_htf)} HTF + {len(ref_ltf)} LTF reference bars")
        return ref_htf, ref_ltf, "databento-csv"

    if args.databento:
        from app.services.backtest.data import DatabentoFetcher

        fetcher = DatabentoFetcher()
        print(f"Fetching reference {ref_symbol} from Databento API...")
        ref_htf, ref_ltf = fetcher.fetch_futures_data(
            symbol=ref_symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            htf_interval="15m",
            ltf_interval="5m",
            use_cache=not args.no_cache,
        )
        ref_htf = _sort_and_normalize_tz(ref_htf)
        ref_ltf = _sort_and_normalize_tz(ref_ltf)
        print(f"  Fetched {len(ref_htf)} HTF + {len(ref_ltf)} LTF reference bars")
        return ref_htf, ref_ltf, "databento"

    if args.synthetic:
        ref_htf, ref_ltf = generate_sample_data(
            ref_symbol, args.days, profile=args.synthetic_profile
        )
        ref_htf = _sort_and_normalize_tz(ref_htf)
        ref_ltf = _sort_and_normalize_tz(ref_ltf)
        print(f"  Generated {len(ref_htf)} HTF + {len(ref_ltf)} LTF reference bars (synthetic)")
        return ref_htf, ref_ltf, "synthetic"

    # Fallback: require explicit --ref-htf
    raise SystemExit(
        "error: --ref-symbol requires either --ref-htf <csv>, "
        "--databento, --databento-csv, or --synthetic to load reference data"
    )


def build_reference_bias_series(
    ref_htf_bars: list[OHLCVBar],
    ref_ltf_bars: list[OHLCVBar],
    ref_h4_bars: Optional[list[OHLCVBar]] = None,
    ref_h1_bars: Optional[list[OHLCVBar]] = None,
) -> list[BiasState]:
    """
    Build a causal BiasState series from reference symbol bars.

    For each HTF bar timestamp, compute bias using only data
    available *up to* that point (lookback windows).
    LTF, h4, h1 bars are optional — None is tolerated.
    """
    from bisect import bisect_right
    from datetime import timedelta

    series: list[BiasState] = []

    # Pre-build h4/h1 completion timestamps for causal alignment
    h4_completed_ts: list[datetime] = []
    h1_completed_ts: list[datetime] = []
    if ref_h4_bars:
        h4_completed_ts = [b.ts + timedelta(hours=4) for b in ref_h4_bars]
    if ref_h1_bars:
        h1_completed_ts = [b.ts + timedelta(hours=1) for b in ref_h1_bars]

    for i, bar in enumerate(ref_htf_bars):
        # Causal window: only bars up to and including current
        htf_start = max(0, i + 1 - REF_HTF_LOOKBACK)
        htf_window = ref_htf_bars[htf_start:i + 1]

        # LTF causal window: bars with ts <= current HTF bar ts
        if ref_ltf_bars:
            ltf_window = [
                b for b in ref_ltf_bars
                if b.ts <= bar.ts
            ][-REF_LTF_LOOKBACK:]
        else:
            ltf_window = []

        # Causal h4/h1 windows (only completed bars)
        causal_h4 = None
        causal_h1 = None
        if ref_h4_bars and h4_completed_ts:
            n = bisect_right(h4_completed_ts, bar.ts)
            if n > 0:
                causal_h4 = ref_h4_bars[:n][-25:]
        if ref_h1_bars and h1_completed_ts:
            n = bisect_right(h1_completed_ts, bar.ts)
            if n > 0:
                causal_h1 = ref_h1_bars[:n][-100:]

        bias = compute_tf_bias(
            h4_bars=causal_h4,
            h1_bars=causal_h1,
            m15_bars=htf_window,
            m5_bars=ltf_window if ltf_window else None,
            timestamp=bar.ts,
        )

        series.append(BiasState(
            ts=bar.ts,
            direction=bias.final_direction,
            confidence=bias.final_confidence,
        ))

    return series


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

    # Intermarket reference: tag NQ trades with ES bias (observability only)
    python scripts/run_unicorn_backtest.py --symbol NQ --synthetic --ref-symbol ES

    # Reference from explicit CSV
    python scripts/run_unicorn_backtest.py --symbol NQ --htf data/nq_15m.csv --ltf data/nq_5m.csv --ref-symbol ES --ref-htf data/es_15m.csv
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
        choices=["ny_open", "strict", "normal", "wide"],
        default="normal",
        help="Session profile: ny_open (9:30-10:30), strict (NY AM 9:30-11:00), "
             "normal (London+NY AM), wide (all sessions)"
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
    parser.add_argument(
        "--min-displacement-atr", type=float, default=None,
        help="Skip entry if MSS displacement < this ATR multiple. None=disabled."
    )
    # Intermarket reference symbol
    parser.add_argument(
        "--ref-symbol",
        help="Reference symbol for intermarket bias (e.g., ES when trading NQ). "
             "Data loaded from same source as primary unless --ref-htf is given."
    )
    parser.add_argument(
        "--ref-htf",
        help="Explicit CSV path for reference HTF bars (overrides source auto-detect)"
    )
    # Multi-timeframe execution
    parser.add_argument(
        "--multi-tf",
        action="store_true",
        help="Enable multi-TF mode: full bias stack (4H/1H/15m/5m) + 1m trade management. "
             "Synthetic: generates 1m base and resamples. Databento: uses fetch_multi_tf()."
    )
    # Trace mode (post-run trade replay)
    parser.add_argument(
        "--trace-trade-index",
        type=int,
        metavar="N",
        help="Trace trade at index N (0-based). Prints full replay after report."
    )
    parser.add_argument(
        "--trace-verbose",
        action="store_true",
        help="Print all bars in trace management path (default: first 5 + exit bar)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Seed for synthetic data generation (for reproducibility)."
    )

    args = parser.parse_args()

    # Validate ref args
    if args.ref_htf and not args.ref_symbol:
        parser.error("--ref-htf requires --ref-symbol")
    if args.ref_symbol and args.ref_symbol.upper() == args.symbol.upper():
        parser.error("--ref-symbol must differ from --symbol")

    # Apply seed for reproducibility
    if args.seed is not None:
        import random
        random.seed(args.seed)

    # Load or generate data
    bar_bundle = None  # Set when --multi-tf is active

    if args.databento_csv:
        # Load from local Databento CSV file
        from app.services.backtest.data import DatabentoFetcher

        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required when using --databento-csv")

        fetcher = DatabentoFetcher()

        print(f"Loading {args.symbol} data from {args.databento_csv}...")
        print(f"  Date range: {args.start_date} to {args.end_date}")

        if args.multi_tf:
            bar_bundle = fetcher.load_multi_tf_from_csv(
                csv_path=args.databento_csv,
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            htf_bars = bar_bundle.m15
            ltf_bars = bar_bundle.m5
            print(f"Loaded multi-TF: {len(bar_bundle.h4)} 4H, {len(bar_bundle.h1)} 1H, "
                  f"{len(htf_bars)} 15m, {len(ltf_bars)} 5m, {len(bar_bundle.m1)} 1m bars")
        else:
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

        if args.multi_tf:
            bar_bundle = fetcher.fetch_multi_tf(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                use_cache=not args.no_cache,
            )
            htf_bars = bar_bundle.m15
            ltf_bars = bar_bundle.m5
            print(f"Fetched multi-TF: {len(bar_bundle.h4)} 4H, {len(bar_bundle.h1)} 1H, "
                  f"{len(htf_bars)} 15m, {len(ltf_bars)} 5m, {len(bar_bundle.m1)} 1m bars")
        else:
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
        if args.multi_tf:
            bar_bundle = generate_sample_data(
                args.symbol, args.days, profile=args.synthetic_profile, multi_tf=True
            )
            htf_bars = bar_bundle.m15
            ltf_bars = bar_bundle.m5
            print(f"Generated multi-TF: {len(bar_bundle.h4)} 4H, {len(bar_bundle.h1)} 1H, "
                  f"{len(htf_bars)} 15m, {len(ltf_bars)} 5m, {len(bar_bundle.m1)} 1m bars")
        else:
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
        min_scored_criteria=args.min_criteria,
        session_profile=SessionProfile(args.session_profile),
        fvg_min_atr_mult=args.fvg_atr_mult,
        stop_max_atr_mult=args.stop_atr_mult,
        max_wick_ratio=args.max_wick_ratio,
        max_range_atr_mult=args.max_range_atr,
        min_displacement_atr=args.min_displacement_atr,
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
    if args.min_displacement_atr is not None:
        print(f"Displacement guard: min MSS displacement = {args.min_displacement_atr:.2f}x ATR")

    # Build reference bias series if requested
    reference_bias_series = None
    reference_symbol = None
    if args.ref_symbol:
        reference_symbol = args.ref_symbol.upper()
        ref_htf_bars, ref_ltf_bars, ref_source = load_ref_data(args, args.ref_htf)
        print(f"Reference symbol: {reference_symbol} (source: {ref_source})")
        reference_bias_series = build_reference_bias_series(
            ref_htf_bars, ref_ltf_bars,
        )
        # Coverage: what fraction of primary HTF range is covered
        if reference_bias_series:
            ref_start = reference_bias_series[0].ts
            ref_end = reference_bias_series[-1].ts
            print(f"  Bias points: {len(reference_bias_series)} "
                  f"({ref_start.date()} to {ref_end.date()})")
            print(f"  LTF bars used: {len(ref_ltf_bars)}")
            if htf_bars:
                primary_start = htf_bars[0].ts
                primary_end = htf_bars[-1].ts
                covered = sum(
                    1 for b in reference_bias_series
                    if primary_start <= b.ts <= primary_end
                )
                coverage_pct = covered / len(htf_bars) * 100 if htf_bars else 0.0
                print(f"  Coverage of primary range: {coverage_pct:.1f}%")
        else:
            print("  WARNING: no reference bias points generated")

    print("")

    if args.multi_tf:
        print(f"Multi-TF mode: full bias stack (4H/1H/15m/5m) + 1m execution")

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
        reference_bias_series=reference_bias_series,
        reference_symbol=reference_symbol,
        bar_bundle=bar_bundle,
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
            "total_pnl_points": result.total_pnl_points,
            "total_pnl_dollars": result.total_pnl_dollars,
            "expectancy_points": result.expectancy_points,
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
                    "total_pnl": result.session_stats[s].total_pnl_points,
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
            "session_diagnostics": result.session_diagnostics,
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

    # Post-run reference diagnostics
    if reference_bias_series is not None and hasattr(result, "session_diagnostics"):
        diag = result.session_diagnostics or {}
        ia = diag.get("intermarket_agreement")
        if ia and "by_agreement" in ia:
            missing = ia["by_agreement"].get("missing_ref", {})
            missing_count = missing.get("trades", 0)
            total = result.trades_taken or 1
            print(f"\nRef-symbol coverage: {missing_count}/{total} trades "
                  f"({missing_count / total * 100:.1f}%) had missing_ref")

    # Trace mode: replay a single trade
    if args.trace_trade_index is not None:
        idx = args.trace_trade_index
        if idx < 0 or idx >= len(result.trades):
            print(f"\nError: --trace-trade-index {idx} out of range "
                  f"(0–{len(result.trades) - 1}, {len(result.trades)} trades)",
                  file=sys.stderr)
            sys.exit(1)

        tick_size = 0.25
        slippage_pts = args.slippage_ticks * tick_size
        trace_output = format_trade_trace(
            trade=result.trades[idx],
            trade_index=idx,
            bar_bundle=bar_bundle,
            result=result,
            intrabar_policy=IntrabarPolicy(args.intrabar_policy),
            slippage_points=slippage_pts,
            verbose=args.trace_verbose,
        )
        print("\n" + trace_output)

    # Quick summary to stderr if outputting to file
    if args.output and not args.json:
        print(f"\nQuick Summary:")
        print(f"  Trades: {result.trades_taken} | Win Rate: {result.win_rate*100:.1f}%")
        print(f"  PnL: {result.total_pnl_points:+.2f} points (${result.total_pnl_dollars:+,.2f})")
        print(f"  Expectancy: {result.expectancy_points:+.2f} points/trade")
        print(f"  Top Bottleneck: {result.criteria_bottlenecks[0].criterion if result.criteria_bottlenecks else 'N/A'}")


if __name__ == "__main__":
    main()
