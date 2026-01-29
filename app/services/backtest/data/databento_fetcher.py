"""
Databento API data fetcher for futures backtesting.

Fetches historical OHLCV data for NQ/ES futures from Databento's CME Globex feed.

Usage:
    from app.services.backtest.data import DatabentoFetcher

    fetcher = DatabentoFetcher()  # Uses DATABENTO_API_KEY env var
    htf_bars, ltf_bars = fetcher.fetch_futures_data(
        symbol="NQ",
        start_date="2024-01-01",
        end_date="2024-01-31",
        htf_interval="15m",
        ltf_interval="5m",
    )

Environment Variables:
    DATABENTO_API_KEY: Your Databento API key

References:
    - https://databento.com/docs/schemas-and-data-formats/ohlcv
    - https://databento.com/datasets/GLBX.MDP3
"""

import os
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

# Databento dataset for CME Globex
GLBX_DATASET = "GLBX.MDP3"

# Contract month codes
MONTH_CODES = {
    1: "F",   # January
    2: "G",   # February
    3: "H",   # March
    4: "J",   # April
    5: "K",   # May
    6: "M",   # June
    7: "N",   # July
    8: "Q",   # August
    9: "U",   # September
    10: "V",  # October
    11: "X",  # November
    12: "Z",  # December
}

# NQ and ES are quarterly contracts (H, M, U, Z)
QUARTERLY_MONTHS = [3, 6, 9, 12]

# Schema mapping for intervals
INTERVAL_TO_SCHEMA = {
    "1s": "ohlcv-1s",
    "1m": "ohlcv-1m",
    "5m": "ohlcv-1m",   # Will be resampled from 1m
    "15m": "ohlcv-1m",  # Will be resampled from 1m
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}


def get_front_month_symbol(root: str, as_of_date: datetime) -> str:
    """
    Get the front-month futures contract symbol.

    Args:
        root: Contract root (e.g., "NQ", "ES")
        as_of_date: Date to determine front month

    Returns:
        Symbol like "NQH4" (NQ March 2024)

    Note:
        NQ/ES roll to next contract ~8 days before expiration (3rd Friday).
        This is a simplified version - production should use actual roll dates.
    """
    year = as_of_date.year
    month = as_of_date.month
    day = as_of_date.day

    # Find the next quarterly month
    for q_month in QUARTERLY_MONTHS:
        if q_month > month or (q_month == month and day < 10):
            # This is the front month
            month_code = MONTH_CODES[q_month]
            year_code = str(year)[-1]  # Single digit year
            return f"{root}{month_code}{year_code}"

    # Rollover to next year's first quarterly month
    month_code = MONTH_CODES[3]  # March
    year_code = str(year + 1)[-1]
    return f"{root}{month_code}{year_code}"


def get_continuous_symbols(
    root: str,
    start_date: datetime,
    end_date: datetime,
) -> list[tuple[str, datetime, datetime]]:
    """
    Get list of contract symbols needed to cover a date range.

    Returns list of (symbol, start, end) tuples for each contract period.
    This handles contract rolls for continuous backtesting.

    Args:
        root: Contract root (e.g., "NQ", "ES")
        start_date: Start of date range
        end_date: End of date range

    Returns:
        List of (symbol, period_start, period_end) tuples
    """
    contracts = []
    current_date = start_date
    prev_symbol = None
    max_iterations = 20  # Safety limit (5 years of quarterly contracts)

    for _ in range(max_iterations):
        if current_date > end_date:
            break

        symbol = get_front_month_symbol(root, current_date)

        # If we got the same symbol as before, advance by one day and try again
        # This handles edge cases around rollover dates
        if symbol == prev_symbol:
            current_date = current_date + timedelta(days=1)
            continue

        prev_symbol = symbol

        # Find when this contract expires (simplified: 3rd Friday of expiry month)
        # Extract month/year from symbol
        month_code = symbol[-2]
        year_digit = symbol[-1]

        # Reverse lookup month
        contract_month = None
        for m, code in MONTH_CODES.items():
            if code == month_code:
                contract_month = m
                break

        if contract_month is None:
            raise ValueError(f"Invalid month code in symbol: {symbol}")

        # Determine year - handle year rollover
        base_year = current_date.year
        year_suffix = int(year_digit)
        current_suffix = base_year % 10

        if year_suffix >= current_suffix:
            contract_year = (base_year // 10) * 10 + year_suffix
        else:
            # Year rolled over (e.g., December 2024 -> March 2025)
            contract_year = (base_year // 10 + 1) * 10 + year_suffix

        # Approximate expiry: 3rd Friday of contract month
        # Roll date: ~10 days before expiry (conservative)
        first_day = datetime(contract_year, contract_month, 1)
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)
        roll_date = third_friday - timedelta(days=10)

        # Period for this contract
        period_start = current_date
        period_end = min(roll_date, end_date)

        if period_start <= period_end:
            contracts.append((symbol, period_start, period_end))

        # Move to next contract period (day after roll)
        current_date = roll_date + timedelta(days=1)

    return contracts


@dataclass
class FetchResult:
    """Result of fetching data from Databento."""
    symbol: str
    interval: str
    start_date: datetime
    end_date: datetime
    bars: list  # List of OHLCVBar
    cost_usd: float
    from_cache: bool


class DatabentoFetcher:
    """
    Fetch historical futures data from Databento API.

    Supports NQ/ES futures with automatic contract rolling and local caching.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the fetcher.

        Args:
            api_key: Databento API key (defaults to DATABENTO_API_KEY env var)
            cache_dir: Directory for caching data (defaults to .databento_cache)
        """
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        if not self.api_key:
            logger.warning(
                "No DATABENTO_API_KEY found. Set environment variable or pass api_key."
            )

        self.cache_dir = cache_dir or Path(".databento_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None

    @property
    def client(self):
        """Lazy-load Databento client."""
        if self._client is None:
            try:
                import databento as db
                self._client = db.Historical(self.api_key)
            except ImportError:
                raise ImportError(
                    "databento package not installed. "
                    "Install with: pip install databento"
                )
        return self._client

    def _get_cache_key(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> str:
        """Generate cache key for a data request."""
        key_data = f"{symbol}:{interval}:{start}:{end}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached data."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[list[dict]]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                logger.info("Loaded from cache", cache_key=cache_key)
                return data
            except Exception as e:
                logger.warning("Cache read failed", error=str(e))
        return None

    def _save_to_cache(self, cache_key: str, data: list[dict]) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
            logger.info("Saved to cache", cache_key=cache_key)
        except Exception as e:
            logger.warning("Cache write failed", error=str(e))

    def estimate_cost(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        schema: str = "ohlcv-1m",
    ) -> float:
        """
        Estimate the cost of a data request before fetching.

        Args:
            symbol: Contract symbol (e.g., "NQH4")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            schema: Data schema

        Returns:
            Estimated cost in USD
        """
        try:
            cost = self.client.metadata.get_cost(
                dataset=GLBX_DATASET,
                symbols=[symbol],
                schema=schema,
                start=start_date,
                end=end_date,
            )
            return cost
        except Exception as e:
            logger.warning("Cost estimation failed", error=str(e))
            return 0.0

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        schema: str = "ohlcv-1m",
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Fetch OHLCV bars from Databento.

        Args:
            symbol: Contract symbol (e.g., "NQH4", "ESM4")
            start_date: Start date (YYYY-MM-DD or ISO format)
            end_date: End date (YYYY-MM-DD or ISO format)
            schema: OHLCV schema (ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d)
            use_cache: Whether to use local cache

        Returns:
            List of bar dictionaries with keys: ts, open, high, low, close, volume
        """
        # Check cache first
        cache_key = self._get_cache_key(symbol, schema, start_date, end_date)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Fetch from API
        logger.info(
            "Fetching from Databento",
            symbol=symbol,
            start=start_date,
            end=end_date,
            schema=schema,
        )

        try:
            data = self.client.timeseries.get_range(
                dataset=GLBX_DATASET,
                symbols=[symbol],
                schema=schema,
                start=start_date,
                end=end_date,
            )

            # Convert to DataFrame then to list of dicts
            df = data.to_df()

            bars = []
            for idx, row in df.iterrows():
                bar = {
                    "ts": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                }
                bars.append(bar)

            logger.info(
                "Fetched from Databento",
                symbol=symbol,
                bar_count=len(bars),
            )

            # Cache the result
            if use_cache:
                self._save_to_cache(cache_key, bars)

            return bars

        except Exception as e:
            logger.error("Databento fetch failed", error=str(e), symbol=symbol)
            raise

    def fetch_futures_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        htf_interval: str = "15m",
        ltf_interval: str = "5m",
        use_cache: bool = True,
    ) -> tuple[list, list]:
        """
        Fetch futures data for backtesting with HTF and LTF bars.

        Handles continuous contracts automatically by fetching and stitching
        multiple contract months together.

        Args:
            symbol: Root symbol (e.g., "NQ", "ES")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            htf_interval: Higher timeframe interval (15m, 1h)
            ltf_interval: Lower timeframe interval (1m, 5m)
            use_cache: Whether to use local cache

        Returns:
            Tuple of (htf_bars, ltf_bars) as lists of OHLCVBar objects
        """
        from app.services.strategy.models import OHLCVBar

        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Get contract symbols needed for date range
        contracts = get_continuous_symbols(symbol, start_dt, end_dt)

        logger.info(
            "Fetching continuous futures data",
            symbol=symbol,
            contracts=[c[0] for c in contracts],
            start=start_date,
            end=end_date,
        )

        # Fetch 1-minute data and resample
        all_1m_bars = []
        for contract_symbol, period_start, period_end in contracts:
            bars = self.fetch_ohlcv(
                symbol=contract_symbol,
                start_date=period_start.strftime("%Y-%m-%d"),
                end_date=period_end.strftime("%Y-%m-%d"),
                schema="ohlcv-1m",
                use_cache=use_cache,
            )
            all_1m_bars.extend(bars)

        if not all_1m_bars:
            raise ValueError(f"No data fetched for {symbol} {start_date} to {end_date}")

        # Convert to OHLCVBar objects
        ohlcv_bars = []
        for bar in all_1m_bars:
            ts = datetime.fromisoformat(bar["ts"].replace("Z", "+00:00"))
            ohlcv_bars.append(OHLCVBar(
                ts=ts,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
            ))

        # Sort by timestamp
        ohlcv_bars.sort(key=lambda b: b.ts)

        # Resample to HTF and LTF
        htf_bars = self._resample_bars(ohlcv_bars, htf_interval)
        ltf_bars = self._resample_bars(ohlcv_bars, ltf_interval)

        logger.info(
            "Resampled futures data",
            htf_interval=htf_interval,
            htf_count=len(htf_bars),
            ltf_interval=ltf_interval,
            ltf_count=len(ltf_bars),
        )

        return htf_bars, ltf_bars

    def _resample_bars(
        self,
        bars: list,
        interval: str,
    ) -> list:
        """
        Resample 1-minute bars to a different interval.

        Args:
            bars: List of OHLCVBar objects (1-minute)
            interval: Target interval (1m, 5m, 15m, 1h)

        Returns:
            Resampled list of OHLCVBar objects
        """
        from app.services.strategy.models import OHLCVBar

        if not bars:
            return []

        # Parse interval
        if interval == "1m":
            return bars
        elif interval == "5m":
            minutes = 5
        elif interval == "15m":
            minutes = 15
        elif interval == "1h":
            minutes = 60
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Group bars by interval
        resampled = []
        current_group = []
        current_bucket = None

        for bar in bars:
            # Calculate which bucket this bar belongs to
            bar_minutes = bar.ts.hour * 60 + bar.ts.minute
            bucket = (bar_minutes // minutes) * minutes

            bucket_key = (bar.ts.date(), bucket)

            if current_bucket is None:
                current_bucket = bucket_key

            if bucket_key != current_bucket:
                # Finalize previous bucket
                if current_group:
                    resampled.append(self._aggregate_bars(current_group))
                current_group = [bar]
                current_bucket = bucket_key
            else:
                current_group.append(bar)

        # Don't forget the last group
        if current_group:
            resampled.append(self._aggregate_bars(current_group))

        return resampled

    def _aggregate_bars(self, bars: list):
        """Aggregate multiple bars into one OHLCV bar."""
        from app.services.strategy.models import OHLCVBar

        if not bars:
            raise ValueError("Cannot aggregate empty bar list")

        return OHLCVBar(
            ts=bars[0].ts,  # Use first bar's timestamp
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
        )


def main():
    """CLI for testing Databento fetcher."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch futures data from Databento")
    parser.add_argument("--symbol", required=True, help="Root symbol (NQ, ES)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--htf", default="15m", help="HTF interval (default: 15m)")
    parser.add_argument("--ltf", default="5m", help="LTF interval (default: 5m)")
    parser.add_argument("--cost-only", action="store_true", help="Only estimate cost")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")

    args = parser.parse_args()

    fetcher = DatabentoFetcher()

    if args.cost_only:
        # Get contracts and estimate total cost
        start_dt = datetime.fromisoformat(args.start)
        end_dt = datetime.fromisoformat(args.end)
        contracts = get_continuous_symbols(args.symbol, start_dt, end_dt)

        total_cost = 0.0
        print(f"\nContracts needed for {args.symbol} {args.start} to {args.end}:")
        for contract, period_start, period_end in contracts:
            cost = fetcher.estimate_cost(
                symbol=contract,
                start_date=period_start.strftime("%Y-%m-%d"),
                end_date=period_end.strftime("%Y-%m-%d"),
            )
            print(f"  {contract}: {period_start.date()} to {period_end.date()} - ${cost:.2f}")
            total_cost += cost

        print(f"\nTotal estimated cost: ${total_cost:.2f}")
        return

    # Fetch data
    htf_bars, ltf_bars = fetcher.fetch_futures_data(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        htf_interval=args.htf,
        ltf_interval=args.ltf,
        use_cache=not args.no_cache,
    )

    print(f"\nFetched {len(htf_bars)} HTF ({args.htf}) bars")
    print(f"Fetched {len(ltf_bars)} LTF ({args.ltf}) bars")

    if htf_bars:
        print(f"\nHTF range: {htf_bars[0].ts} to {htf_bars[-1].ts}")
        print(f"Sample HTF bar: {htf_bars[0]}")

    if ltf_bars:
        print(f"\nLTF range: {ltf_bars[0].ts} to {ltf_bars[-1].ts}")


if __name__ == "__main__":
    main()
