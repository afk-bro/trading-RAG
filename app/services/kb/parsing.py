"""
Enhanced OHLCV parsing for the Trading Knowledge Base.

Extends the base backtest parser with:
- Content fingerprinting (SHA256) for deduplication
- Timeframe detection
- Instrument extraction from filename/data
- ParsedDataset wrapper with full metadata
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from app.services.backtest.data import (
    parse_ohlcv_csv,
    OHLCVParseResult,
    OHLCVParseError,
)


# =============================================================================
# ParsedDataset
# =============================================================================


@dataclass
class ParsedDataset:
    """
    Parsed OHLCV dataset with full metadata for KB operations.

    Contains the DataFrame plus all metadata needed for regime computation,
    filtering, and deduplication.
    """

    df: pd.DataFrame  # ts, open, high, low, close, volume (lowercase)
    n_bars: int
    ts_start: datetime
    ts_end: datetime
    instrument: Optional[str]
    timeframe: Optional[str]
    fingerprint: str  # SHA256 of canonical representation
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_days(self) -> float:
        """Total duration in days."""
        if self.ts_start and self.ts_end:
            return (self.ts_end - self.ts_start).total_seconds() / 86400
        return 0.0


# =============================================================================
# Main Parsing Function
# =============================================================================


def parse_ohlcv_for_kb(
    file_content: bytes,
    filename: str = "data.csv",
    instrument_hint: Optional[str] = None,
    timeframe_hint: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> ParsedDataset:
    """
    Parse OHLCV data for KB operations.

    Extends base parser with fingerprinting and metadata extraction.

    Args:
        file_content: Raw CSV bytes
        filename: Original filename (used for instrument/timeframe extraction)
        instrument_hint: Optional instrument override
        timeframe_hint: Optional timeframe override
        date_from: Optional date filter start
        date_to: Optional date filter end

    Returns:
        ParsedDataset with full metadata

    Raises:
        OHLCVParseError: If data is invalid
    """
    warnings = []

    # Use base parser
    result = parse_ohlcv_csv(
        file_content=file_content,
        filename=filename,
        date_from=date_from,
        date_to=date_to,
    )

    warnings.extend(result.warnings)

    # Convert to lowercase columns for consistency
    df = result.df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Detect or use provided metadata
    instrument = instrument_hint or _extract_instrument(filename, df)
    timeframe = timeframe_hint or _detect_timeframe(df)

    if timeframe is None:
        warnings.append("timeframe_undetected")

    # Compute fingerprint
    fingerprint = _compute_fingerprint(df)

    return ParsedDataset(
        df=df,
        n_bars=result.row_count,
        ts_start=result.date_min,
        ts_end=result.date_max,
        instrument=instrument,
        timeframe=timeframe,
        fingerprint=fingerprint,
        warnings=warnings,
    )


# =============================================================================
# Fingerprinting
# =============================================================================


def _compute_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute SHA256 fingerprint of OHLCV data.

    Uses canonical representation for stability:
    - Sorted by timestamp
    - Fixed decimal precision
    - Deterministic column order

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        SHA256 hex digest
    """
    # Ensure sorted (should already be from parser)
    df_sorted = df.sort_index()

    # Build canonical string representation
    # Format: timestamp,open,high,low,close,volume per row
    # Use fixed precision to avoid float representation issues
    lines = []

    for idx, row in df_sorted.iterrows():
        # Format timestamp
        if hasattr(idx, "isoformat"):
            ts_str = idx.isoformat()
        else:
            ts_str = str(idx)

        # Format OHLCV with consistent precision
        line = (
            f"{ts_str},"
            f"{row['open']:.8f},"
            f"{row['high']:.8f},"
            f"{row['low']:.8f},"
            f"{row['close']:.8f},"
            f"{row.get('volume', 0):.2f}"
        )
        lines.append(line)

    canonical = "\n".join(lines)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_fingerprint_from_bytes(file_content: bytes) -> str:
    """
    Compute fingerprint directly from file bytes.

    Faster but less stable than parsed fingerprint.
    Use for quick dedup checks before full parsing.

    Args:
        file_content: Raw file bytes

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(file_content).hexdigest()


# =============================================================================
# Metadata Extraction
# =============================================================================


def _extract_instrument(filename: str, df: pd.DataFrame) -> Optional[str]:
    """
    Extract instrument/symbol from filename or data.

    Patterns matched:
    - BTCUSD_1h.csv -> BTCUSD
    - btc_usdt_4h_data.csv -> BTCUSDT
    - SPY_daily.csv -> SPY
    - data_AAPL.csv -> AAPL

    Args:
        filename: Original filename
        df: Parsed DataFrame (for column-based detection)

    Returns:
        Instrument string or None
    """
    if not filename:
        return None

    # Remove extension and common suffixes
    name = filename.lower()
    name = re.sub(r"\.(csv|txt|data)$", "", name)
    name = re.sub(r"_(data|ohlcv|candles?|history)$", "", name)

    # Common timeframe patterns to remove
    name = re.sub(r"_(1[mhd]|5m|15m|30m|1h|4h|1d|daily|hourly|minute)$", "", name)

    # Try to extract known patterns

    # Pattern: BTCUSD or BTC_USD or btcusdt
    crypto_pattern = r"(btc|eth|sol|xrp|ada|doge|bnb|ltc|dot|link)[-_]?(usd[t]?|usdc|busd|eur|gbp)"
    crypto_match = re.search(crypto_pattern, name, re.IGNORECASE)
    if crypto_match:
        return crypto_match.group().upper().replace("-", "").replace("_", "")

    # Pattern: Stock symbols (2-5 uppercase letters at start or end)
    stock_pattern = r"^([a-z]{2,5})[-_]|[-_]([a-z]{2,5})$"
    stock_match = re.search(stock_pattern, name, re.IGNORECASE)
    if stock_match:
        symbol = stock_match.group(1) or stock_match.group(2)
        return symbol.upper()

    # Pattern: Just the name if short enough
    clean_name = re.sub(r"[^a-z]", "", name)
    if 2 <= len(clean_name) <= 6:
        return clean_name.upper()

    return None


def _detect_timeframe(df: pd.DataFrame) -> Optional[str]:
    """
    Detect timeframe from timestamp intervals.

    Args:
        df: DataFrame with datetime index

    Returns:
        Timeframe string (e.g., "1m", "5m", "1h", "1d") or None
    """
    if df.empty or len(df) < 2:
        return None

    # Get timestamp differences
    if isinstance(df.index, pd.DatetimeIndex):
        diffs = df.index.to_series().diff().dropna()
    else:
        return None

    if len(diffs) == 0:
        return None

    # Get median difference (robust to gaps/weekends)
    median_diff = diffs.median()

    # Convert to seconds
    median_seconds = median_diff.total_seconds()

    # Map to standard timeframes
    timeframe_map = [
        (60, "1m"),
        (300, "5m"),
        (900, "15m"),
        (1800, "30m"),
        (3600, "1h"),
        (14400, "4h"),
        (86400, "1d"),
        (604800, "1w"),
    ]

    # Find closest match (within 10% tolerance)
    for seconds, tf in timeframe_map:
        if 0.9 * seconds <= median_seconds <= 1.1 * seconds:
            return tf

    # For daily+ data with gaps (weekends), be more lenient
    if 43200 <= median_seconds <= 259200:  # 12h to 3d
        return "1d"

    return None


def detect_timeframe_from_filename(filename: str) -> Optional[str]:
    """
    Extract timeframe hint from filename.

    Args:
        filename: Original filename

    Returns:
        Timeframe string or None
    """
    if not filename:
        return None

    name = filename.lower()

    # Direct matches
    patterns = [
        (r"[_-]1m\b|[_-]1min", "1m"),
        (r"[_-]5m\b|[_-]5min", "5m"),
        (r"[_-]15m\b|[_-]15min", "15m"),
        (r"[_-]30m\b|[_-]30min", "30m"),
        (r"[_-]1h\b|[_-]1hour|[_-]hourly", "1h"),
        (r"[_-]4h\b|[_-]4hour", "4h"),
        (r"[_-]1d\b|[_-]daily|[_-]day", "1d"),
        (r"[_-]1w\b|[_-]weekly|[_-]week", "1w"),
    ]

    for pattern, tf in patterns:
        if re.search(pattern, name):
            return tf

    return None


# =============================================================================
# Validation
# =============================================================================


def validate_for_regime(df: pd.DataFrame, min_bars: int = 200) -> list[str]:
    """
    Validate dataset is suitable for regime computation.

    Args:
        df: DataFrame to validate
        min_bars: Minimum required bars

    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []

    if len(df) < min_bars:
        warnings.append(f"insufficient_bars_{len(df)}_need_{min_bars}")

    # Check for large gaps (could indicate bad data or weekends)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()

        # Gaps more than 10x median
        large_gaps = (diffs > median_diff * 10).sum()
        if large_gaps > 0:
            warnings.append(f"large_gaps_{large_gaps}")

    # Check for zero volume (might indicate incomplete data)
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > len(df) * 0.1:  # More than 10%
            warnings.append(f"high_zero_volume_{zero_vol}")

    return warnings


# Re-export for convenience
__all__ = [
    "ParsedDataset",
    "parse_ohlcv_for_kb",
    "compute_fingerprint_from_bytes",
    "detect_timeframe_from_filename",
    "validate_for_regime",
    "OHLCVParseError",
]
