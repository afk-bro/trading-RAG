"""OHLCV data parsing and validation for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Required columns (case-insensitive)
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

# Column aliases mapping (lowercase)
COLUMN_ALIASES = {
    "timestamp": "date",
    "datetime": "date",
    "time": "date",
    "adj_close": "close",
    "adjusted_close": "close",
    "adj close": "close",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
}


@dataclass
class OHLCVParseResult:
    """Result of parsing OHLCV CSV data."""

    df: pd.DataFrame
    row_count: int
    date_min: datetime
    date_max: datetime
    warnings: list[str] = field(default_factory=list)


class OHLCVParseError(Exception):
    """Error parsing OHLCV data."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def parse_ohlcv_csv(
    file_content: bytes,
    filename: str = "data.csv",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    max_rows: int = 2_000_000,
    max_file_size_mb: int = 25,
) -> OHLCVParseResult:
    """
    Parse OHLCV CSV data into a DataFrame suitable for backtesting.

    Args:
        file_content: Raw CSV bytes
        filename: Original filename (for error messages)
        date_from: Optional filter - only include rows >= this date
        date_to: Optional filter - only include rows <= this date
        max_rows: Maximum allowed rows
        max_file_size_mb: Maximum file size in MB

    Returns:
        OHLCVParseResult with DataFrame and metadata

    Raises:
        OHLCVParseError: If data is invalid
    """
    warnings = []

    # Check file size
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise OHLCVParseError(
            f"File too large: {file_size_mb:.1f}MB (max {max_file_size_mb}MB)",
            {"file_size_mb": file_size_mb, "max_mb": max_file_size_mb},
        )

    # Parse CSV
    try:
        # Try to decode as UTF-8
        content_str = file_content.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
    except UnicodeDecodeError:
        try:
            content_str = file_content.decode("latin-1")
            df = pd.read_csv(StringIO(content_str))
            warnings.append("File decoded as latin-1 (non-UTF8)")
        except Exception as e:
            raise OHLCVParseError(f"Failed to decode CSV: {e}")
    except pd.errors.EmptyDataError:
        raise OHLCVParseError("CSV file is empty")
    except Exception as e:
        raise OHLCVParseError(f"Failed to parse CSV: {e}")

    if len(df) == 0:
        raise OHLCVParseError("CSV has no data rows")

    # Check row limit
    if len(df) > max_rows:
        raise OHLCVParseError(
            f"Too many rows: {len(df)} (max {max_rows})",
            {"row_count": len(df), "max_rows": max_rows},
        )

    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip()

    # Apply column aliases
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
            warnings.append(f"Column '{col}' mapped to '{COLUMN_ALIASES[col]}'")
    if rename_map:
        df = df.rename(columns=rename_map)

    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise OHLCVParseError(
            f"Missing required columns: {', '.join(sorted(missing))}",
            {"missing_columns": list(missing), "found_columns": list(df.columns)},
        )

    # Parse date column
    try:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except Exception as e:
        raise OHLCVParseError(
            f"Failed to parse date column: {e}",
            {"sample_values": df["date"].head(5).tolist()},
        )

    # Validate numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            raise OHLCVParseError(f"Failed to parse {col} as numeric: {e}")

    # Check for NaNs
    nan_counts = df[numeric_cols].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        # Drop rows with NaN in OHLCV columns
        original_len = len(df)
        df = df.dropna(subset=numeric_cols)
        dropped = original_len - len(df)
        warnings.append(f"Dropped {dropped} rows with NaN values in OHLCV columns")

    if len(df) == 0:
        raise OHLCVParseError("No valid rows after removing NaN values")

    # Check for negative prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] < 0).any():
            raise OHLCVParseError(
                f"Column '{col}' contains negative values",
                {"min_value": float(df[col].min())},
            )

    # Check OHLC consistency (high >= low, etc.)
    invalid_ohlc = (df["high"] < df["low"]).sum()
    if invalid_ohlc > 0:
        raise OHLCVParseError(
            f"{invalid_ohlc} rows have high < low",
            {"invalid_rows": invalid_ohlc},
        )

    # Sort by date ascending
    original_order = df["date"].tolist()
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    sorted_order = df["date"].tolist()
    if original_order != sorted_order:
        warnings.append("Data was sorted by date (was not in chronological order)")

    # Remove duplicates on date
    original_len = len(df)
    df = df.drop_duplicates(subset=["date"], keep="last")
    duplicates_removed = original_len - len(df)
    if duplicates_removed > 0:
        warnings.append(f"Removed {duplicates_removed} duplicate timestamps")

    # Apply date filters
    if date_from:
        if not isinstance(date_from, pd.Timestamp):
            date_from = pd.Timestamp(date_from, tz="UTC")
        original_len = len(df)
        df = df[df["date"] >= date_from]
        filtered = original_len - len(df)
        if filtered > 0:
            warnings.append(f"Filtered out {filtered} rows before {date_from}")

    if date_to:
        if not isinstance(date_to, pd.Timestamp):
            date_to = pd.Timestamp(date_to, tz="UTC")
        original_len = len(df)
        df = df[df["date"] <= date_to]
        filtered = original_len - len(df)
        if filtered > 0:
            warnings.append(f"Filtered out {filtered} rows after {date_to}")

    if len(df) == 0:
        raise OHLCVParseError("No rows remaining after date filtering")

    # Minimum data requirement
    if len(df) < 10:
        raise OHLCVParseError(
            f"Insufficient data: only {len(df)} rows (minimum 10 required)",
            {"row_count": len(df)},
        )

    # Set date as index (required by backtesting.py)
    df = df.set_index("date")

    # Ensure proper column order and types
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Rename to title case for backtesting.py
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    logger.info(
        "Parsed OHLCV data",
        filename=filename,
        row_count=len(df),
        date_min=str(df.index.min()),
        date_max=str(df.index.max()),
        warnings_count=len(warnings),
    )

    return OHLCVParseResult(
        df=df,
        row_count=len(df),
        date_min=df.index.min().to_pydatetime(),
        date_max=df.index.max().to_pydatetime(),
        warnings=warnings,
    )
