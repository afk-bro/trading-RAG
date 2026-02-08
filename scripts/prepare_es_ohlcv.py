"""
Prepare continuous ES futures OHLCV from Databento 1-minute bars.

Reads the raw CSV in chunks, filters for ES outright contracts,
builds a continuous front-month series (highest daily volume),
then resamples to Daily (RTH 09:30-16:00 ET) and H1 bars.

Outputs:
  docs/historical_data/ES_daily.csv
  docs/historical_data/ES_h1.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_CSV = (
    ROOT
    / "docs/historical_data/GLBX-20260129-JNB8PDSQ7C"
    / "glbx-mdp3-20210128-20260127.ohlcv-1m.csv"
)
OUT_DIR = ROOT / "docs/historical_data"
CHUNK_SIZE = 500_000

# Regex: outright ES contract = ES + month code + 1-digit year (e.g. ESH1, ESZ5)
ES_OUTRIGHT = re.compile(r"^ES[HMUZ]\d$")


def load_es_minutes() -> pd.DataFrame:
    """Load only ES outright 1-min bars from the large CSV in chunks."""
    print(f"Reading {SRC_CSV.name} in chunks of {CHUNK_SIZE:,} rows ...")
    frames = []
    total_read = 0
    total_kept = 0

    reader = pd.read_csv(
        SRC_CSV,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        },
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        total_read += len(chunk)
        # Keep only ES outright contracts
        mask = chunk["symbol"].str.match(ES_OUTRIGHT, na=False)
        filtered = chunk.loc[mask].copy()
        total_kept += len(filtered)
        if not filtered.empty:
            frames.append(filtered)
        if (i + 1) % 20 == 0:
            print(f"  ... processed {total_read:,} rows, kept {total_kept:,} ES rows")

    print(f"  Total rows read: {total_read:,}, ES outright rows kept: {total_kept:,}")
    df = pd.concat(frames, ignore_index=True)

    # Parse timestamps, localize to UTC then convert to US/Eastern
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    df["ts_et"] = df["ts"].dt.tz_convert("US/Eastern")
    df.drop(columns=["ts_event", "ts"], inplace=True)
    df.set_index("ts_et", inplace=True)
    df.sort_index(inplace=True)
    print(f"  Date range: {df.index.min()} -> {df.index.max()}")
    return df


def build_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build continuous front-month series.

    For each calendar day, pick the contract with the highest total volume.
    Then keep only bars from that contract for that day.
    """
    print("Building continuous front-month series (volume-based roll) ...")
    df["trade_date"] = df.index.date

    # Daily volume per contract per day
    daily_vol = df.groupby(["trade_date", "symbol"])["volume"].sum()
    # For each day, the contract with max volume is the front month
    front = daily_vol.groupby(level="trade_date").idxmax().apply(lambda x: x[1])
    front.name = "front_symbol"
    front_df = front.reset_index()

    # Map each row's trade_date to front symbol, filter matches
    df = df.reset_index()
    df = df.merge(front_df, on="trade_date", how="left")
    df = df[df["symbol"] == df["front_symbol"]].copy()
    df.set_index("ts_et", inplace=True)
    df.drop(columns=["trade_date", "front_symbol", "symbol"], inplace=True)
    df.sort_index(inplace=True)

    n_days = len(front)
    print(f"  Continuous series: {n_days} trading days, {len(df):,} minute bars")
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only RTH bars: 09:30:00 <= time < 16:00:00 Eastern."""
    t = df.index.time
    import datetime as _dt

    mask = (t >= _dt.time(9, 30)) & (t < _dt.time(16, 0))
    rth = df.loc[mask].copy()
    print(f"  RTH filter: {len(df):,} -> {len(rth):,} bars")
    return rth


def resample_daily(rth: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RTH minute bars into daily OHLCV."""
    daily = rth.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    daily = daily.dropna(subset=["open"])
    daily.index = daily.index.date
    daily.index.name = "date"
    return daily


def resample_h1(rth: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RTH minute bars into 1-hour OHLCV."""
    h1 = rth.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    h1 = h1.dropna(subset=["open"])
    h1.index.name = "datetime"
    return h1


def main():
    if not SRC_CSV.exists():
        print(f"ERROR: source CSV not found: {SRC_CSV}")
        sys.exit(1)

    df = load_es_minutes()
    continuous = build_continuous(df)
    rth = filter_rth(continuous)

    # --- Daily ---
    daily = resample_daily(rth)
    daily_path = OUT_DIR / "ES_daily.csv"
    daily.to_csv(daily_path)
    print(f"\nDaily bars written to {daily_path}")
    print(f"  Rows: {len(daily)}")
    print(f"  NaN check: {daily.isna().sum().to_dict()}")
    print("  First 5 rows:")
    print(daily.head().to_string())
    print("  Last 5 rows:")
    print(daily.tail().to_string())

    # --- H1 ---
    h1 = resample_h1(rth)
    # Format datetime without timezone for clean CSV
    h1.index = h1.index.strftime("%Y-%m-%d %H:%M:%S")
    h1_path = OUT_DIR / "ES_h1.csv"
    h1.to_csv(h1_path)
    print(f"\nH1 bars written to {h1_path}")
    print(f"  Rows: {len(h1)}")
    print(f"  NaN check (before fmt): n/a (dropped above)")
    print("  First 5 rows:")
    print(h1.head().to_string())
    print("  Last 5 rows:")
    print(h1.tail().to_string())

    # --- Sanity checks ---
    print("\n--- Sanity Checks ---")
    print(f"Daily rows: {len(daily)} (expect ~1250 for 5 years)")
    print(f"H1 rows:    {len(h1)} (expect ~7-8x daily = ~8750-10000)")
    expected_h1_per_day = len(h1) / len(daily) if len(daily) else 0
    print(f"Avg H1 bars per day: {expected_h1_per_day:.1f} (expect ~7: 09:30,10:30,...,15:30)")


if __name__ == "__main__":
    main()
