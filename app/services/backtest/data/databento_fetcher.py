"""
Databento API data fetcher for futures backtesting.

Fetches historical OHLCV data for NQ/ES futures from Databento's CME Globex feed.
Supports both live API fetching and loading from downloaded CSV files.

Usage (API):
    from app.services.backtest.data import DatabentoFetcher

    fetcher = DatabentoFetcher()  # Uses DATABENTO_API_KEY env var
    htf_bars, ltf_bars = fetcher.fetch_futures_data(
        symbol="NQ",
        start_date="2024-01-01",
        end_date="2024-01-31",
        htf_interval="15m",
        ltf_interval="5m",
    )

Usage (Local CSV):
    from app.services.backtest.data import DatabentoFetcher

    fetcher = DatabentoFetcher()
    htf_bars, ltf_bars = fetcher.load_from_csv(
        csv_path="path/to/glbx-mdp3-data.csv",
        symbol="NQ",  # Filter by symbol root
        start_date="2024-01-01",
        end_date="2024-01-31",
        htf_interval="15m",
        ltf_interval="5m",
    )

Environment Variables:
    DATABENTO_API_KEY: Your Databento API key (optional for local files)

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
from typing import Any, Optional

from app.utils.instruments import data_root
import structlog

logger = structlog.get_logger(__name__)

# Databento dataset for CME Globex
GLBX_DATASET = "GLBX.MDP3"

# Contract month codes
MONTH_CODES = {
    1: "F",  # January
    2: "G",  # February
    3: "H",  # March
    4: "J",  # April
    5: "K",  # May
    6: "M",  # June
    7: "N",  # July
    8: "Q",  # August
    9: "U",  # September
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
    "5m": "ohlcv-1m",  # Will be resampled from 1m
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
            htf_interval: Higher timeframe interval (5m, 15m, 1h, 4h)
            ltf_interval: Lower timeframe interval (1m, 5m, 15m)
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
            ohlcv_bars.append(
                OHLCVBar(
                    ts=ts,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                )
            )

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
            interval: Target interval (1m, 5m, 15m, 1h, 4h)

        Returns:
            Resampled list of OHLCVBar objects
        """
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
        elif interval == "4h":
            minutes = 240
        elif interval in ("1d", "1w"):
            return self._resample_bars_calendar(bars, interval)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Group bars by interval
        resampled = []
        current_group: list[Any] = []
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

    def _resample_bars_calendar(self, bars: list, interval: str) -> list:
        """Resample 1m bars to daily or weekly using calendar boundaries.

        Daily: group by date.
        Weekly: group by ISO week (Mon–Sun).
        """
        from collections import OrderedDict

        groups: OrderedDict = OrderedDict()
        for bar in bars:
            if interval == "1d":
                key = bar.ts.date()
            else:  # 1w
                iso = bar.ts.isocalendar()
                key = (iso[0], iso[1])  # (year, week)
            groups.setdefault(key, []).append(bar)

        return [self._aggregate_bars(g) for g in groups.values()]

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

    def load_from_csv(
        self,
        csv_path: str | Path,
        symbol: str,
        start_date: str,
        end_date: str,
        htf_interval: str = "15m",
        ltf_interval: str = "5m",
        front_month_only: bool = True,
    ) -> tuple[list, list]:
        """
        Load futures data from a local Databento CSV file.

        Handles the standard Databento CSV format with columns:
        ts_event, rtype, publisher_id, instrument_id, open, high, low, close, volume, symbol

        Args:
            csv_path: Path to the CSV file (can be .csv or .csv.zst)
            symbol: Root symbol to filter (e.g., "NQ", "ES")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            htf_interval: Higher timeframe interval (5m, 15m, 1h, 4h)
            ltf_interval: Lower timeframe interval (1m, 5m, 15m)
            front_month_only: If True, only load front-month contracts (handles rolls)

        Returns:
            Tuple of (htf_bars, ltf_bars) as lists of OHLCVBar objects
        """
        from app.services.strategy.models import OHLCVBar

        csv_path = Path(csv_path)

        # Handle zstd compressed files
        if csv_path.suffix == ".zst":
            csv_path = self._decompress_zst(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Resolve micro → full-size root for data loading (MNQ→NQ, MES→ES)
        csv_root = data_root(symbol)

        # Build front-month contract mapping if needed
        front_month_contracts: dict[str, str] = {}  # date_str -> contract symbol
        if front_month_only:
            contracts = get_continuous_symbols(csv_root, start_dt, end_dt)
            for contract_symbol, period_start, period_end in contracts:
                # Map each date in the period to this contract
                current = period_start
                while current <= period_end:
                    front_month_contracts[current.strftime("%Y-%m-%d")] = (
                        contract_symbol
                    )
                    current += timedelta(days=1)

            logger.info(
                "Front-month contracts for period",
                contracts=[
                    (c[0], str(c[1].date()), str(c[2].date())) for c in contracts
                ],
            )

        logger.info(
            "Loading from local CSV",
            path=str(csv_path),
            symbol=symbol,
            start=start_date,
            end=end_date,
            front_month_only=front_month_only,
        )

        # Read and filter CSV
        ohlcv_bars = []
        rows_processed = 0
        rows_matched = 0
        rows_skipped_contract = 0

        with open(csv_path, "r") as f:
            # Read header
            header = f.readline().strip().split(",")
            col_idx = {name: i for i, name in enumerate(header)}

            # Validate required columns
            required = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
            missing = [c for c in required if c not in col_idx]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for line in f:
                rows_processed += 1
                parts = line.strip().split(",")

                # Filter by symbol - match symbol root (NQ, ES)
                sym = parts[col_idx["symbol"]]

                # Symbol format is like "NQH4", "ESM5", "NQH1-NQM1" (spread)
                # Skip spreads (contain hyphen)
                if "-" in sym:
                    continue

                # Check if symbol starts with our root
                if not sym.startswith(csv_root):
                    continue

                # Parse timestamp
                ts_str = parts[col_idx["ts_event"]]
                try:
                    # Handle Databento timestamp format: 2021-01-28T00:00:00.000000000Z
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                # Filter by date range
                if ts.date() < start_dt.date() or ts.date() > end_dt.date():
                    continue

                # Filter by front-month contract if enabled
                if front_month_only:
                    date_str = ts.strftime("%Y-%m-%d")
                    expected_contract = front_month_contracts.get(date_str)
                    if expected_contract and sym != expected_contract:
                        rows_skipped_contract += 1
                        continue

                # Parse OHLCV values
                try:
                    bar = OHLCVBar(
                        ts=ts,
                        open=float(parts[col_idx["open"]]),
                        high=float(parts[col_idx["high"]]),
                        low=float(parts[col_idx["low"]]),
                        close=float(parts[col_idx["close"]]),
                        volume=float(parts[col_idx["volume"]]),
                    )
                    ohlcv_bars.append(bar)
                    rows_matched += 1
                except (ValueError, IndexError):
                    continue

        logger.info(
            "Loaded from CSV",
            rows_processed=rows_processed,
            rows_matched=rows_matched,
            rows_skipped_wrong_contract=rows_skipped_contract,
            bars_loaded=len(ohlcv_bars),
        )

        if not ohlcv_bars:
            raise ValueError(
                f"No data found for {symbol} between {start_date} and {end_date}"
            )

        # Sort by timestamp
        ohlcv_bars.sort(key=lambda b: b.ts)

        # Resample to HTF and LTF
        htf_bars = self._resample_bars(ohlcv_bars, htf_interval)
        ltf_bars = self._resample_bars(ohlcv_bars, ltf_interval)

        logger.info(
            "Resampled local data",
            htf_interval=htf_interval,
            htf_count=len(htf_bars),
            ltf_interval=ltf_interval,
            ltf_count=len(ltf_bars),
        )

        return htf_bars, ltf_bars

    def fetch_multi_tf(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ):
        """
        Fetch futures data and resample to all timeframes for multi-TF backtesting.

        Fetches 1m bars and resamples to 4h, 1h, 15m, 5m, returning a BarBundle
        alongside the raw 1m bars.

        Args:
            symbol: Root symbol (e.g., "NQ", "ES")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use local cache

        Returns:
            BarBundle with all timeframes populated
        """
        from app.services.strategy.models import OHLCVBar

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        contracts = get_continuous_symbols(symbol, start_dt, end_dt)

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

        ohlcv_bars = []
        for bar in all_1m_bars:
            ts = datetime.fromisoformat(bar["ts"].replace("Z", "+00:00"))
            ohlcv_bars.append(
                OHLCVBar(
                    ts=ts,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                )
            )
        ohlcv_bars.sort(key=lambda b: b.ts)

        return self._resample_to_bundle(ohlcv_bars)

    def load_multi_tf_from_csv(
        self,
        csv_path: str | Path,
        symbol: str,
        start_date: str,
        end_date: str,
        front_month_only: bool = True,
    ):
        """
        Load futures data from a local Databento CSV and resample to all timeframes.

        Args:
            csv_path: Path to the CSV file
            symbol: Root symbol to filter (e.g., "NQ", "ES")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            front_month_only: If True, only load front-month contracts

        Returns:
            BarBundle with all timeframes populated
        """
        from app.services.strategy.models import OHLCVBar

        csv_path = Path(csv_path)
        if csv_path.suffix == ".zst":
            csv_path = self._decompress_zst(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Resolve micro → full-size root for data loading (MNQ→NQ, MES→ES)
        csv_root = data_root(symbol)

        front_month_contracts: dict[str, str] = {}
        if front_month_only:
            contracts = get_continuous_symbols(csv_root, start_dt, end_dt)
            for contract_symbol, period_start, period_end in contracts:
                current = period_start
                while current <= period_end:
                    front_month_contracts[current.strftime("%Y-%m-%d")] = (
                        contract_symbol
                    )
                    current += timedelta(days=1)

        ohlcv_bars = []
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(",")
            col_idx = {name: i for i, name in enumerate(header)}
            required = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
            missing = [c for c in required if c not in col_idx]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for line in f:
                parts = line.strip().split(",")
                sym = parts[col_idx["symbol"]]
                if "-" in sym or not sym.startswith(csv_root):
                    continue
                ts_str = parts[col_idx["ts_event"]]
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if ts.date() < start_dt.date() or ts.date() > end_dt.date():
                    continue
                if front_month_only:
                    date_str = ts.strftime("%Y-%m-%d")
                    expected_contract = front_month_contracts.get(date_str)
                    if expected_contract and sym != expected_contract:
                        continue
                try:
                    ohlcv_bars.append(
                        OHLCVBar(
                            ts=ts,
                            open=float(parts[col_idx["open"]]),
                            high=float(parts[col_idx["high"]]),
                            low=float(parts[col_idx["low"]]),
                            close=float(parts[col_idx["close"]]),
                            volume=float(parts[col_idx["volume"]]),
                        )
                    )
                except (ValueError, IndexError):
                    continue

        if not ohlcv_bars:
            raise ValueError(
                f"No data found for {symbol} between {start_date} and {end_date}"
            )

        ohlcv_bars.sort(key=lambda b: b.ts)
        return self._resample_to_bundle(ohlcv_bars)

    def _resample_to_bundle(self, m1_bars: list):
        """Resample 1m bars into a BarBundle with all timeframes."""
        from app.services.backtest.engines.unicorn_runner import BarBundle

        return BarBundle(
            h4=self._resample_bars(m1_bars, "4h"),
            h1=self._resample_bars(m1_bars, "1h"),
            m15=self._resample_bars(m1_bars, "15m"),
            m5=self._resample_bars(m1_bars, "5m"),
            m1=m1_bars,
            daily=self._resample_bars(m1_bars, "1d"),
            weekly=self._resample_bars(m1_bars, "1w"),
        )

    def _decompress_zst(self, zst_path: Path) -> Path:
        """Decompress a .zst file if the uncompressed version doesn't exist."""
        csv_path = zst_path.with_suffix("")  # Remove .zst extension

        if csv_path.exists():
            logger.info("Using existing decompressed file", path=str(csv_path))
            return csv_path

        logger.info("Decompressing zst file", src=str(zst_path), dst=str(csv_path))

        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "zstandard package not installed. "
                "Install with: pip install zstandard"
            )

        dctx = zstd.ZstdDecompressor()
        with open(zst_path, "rb") as ifh, open(csv_path, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)

        logger.info(
            "Decompressed zst file",
            size_mb=csv_path.stat().st_size / 1024 / 1024,
        )

        return csv_path


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
            print(
                f"  {contract}: {period_start.date()} to {period_end.date()} - ${cost:.2f}"
            )
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
