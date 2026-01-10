"""
Regime computation integration for backtest tuning.

Provides functions to compute regime snapshots for IS/OOS windows
and embed them in metrics dictionaries.
"""

from datetime import datetime
from typing import Optional

import structlog

from app.services.backtest.data import parse_ohlcv_csv, OHLCVParseError

# Import directly from submodules to avoid circular import
from app.services.kb.types import RegimeSnapshot
from app.services.kb.regime import compute_regime_snapshot

logger = structlog.get_logger(__name__)


def compute_regime_for_window(
    file_content: bytes,
    filename: str = "data.csv",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    source: str = "live",
    timeframe: Optional[str] = None,
    instrument: Optional[str] = None,
) -> Optional[RegimeSnapshot]:
    """
    Compute regime snapshot for a data window.

    Parses the OHLCV data with date filters and computes regime features.

    Args:
        file_content: Raw CSV bytes
        filename: Original filename
        date_from: Start of window (inclusive)
        date_to: End of window (inclusive)
        source: Computation source ("live", "backfill", "query")
        timeframe: Optional timeframe hint
        instrument: Optional instrument hint

    Returns:
        RegimeSnapshot or None if computation fails
    """
    try:
        # Parse OHLCV for the window
        result = parse_ohlcv_csv(
            file_content=file_content,
            filename=filename,
            date_from=date_from,
            date_to=date_to,
        )

        df = result.df

        # Check minimum data
        if len(df) < 50:  # Need at least 50 bars for meaningful regime
            logger.debug(
                "Insufficient bars for regime computation",
                n_bars=len(df),
                date_from=str(date_from) if date_from else None,
                date_to=str(date_to) if date_to else None,
            )
            return None

        # Convert column names to lowercase for regime computation
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]

        # Compute regime snapshot
        snapshot = compute_regime_snapshot(
            df=df_lower,
            source=source,
            instrument=instrument,
            timeframe=timeframe,
        )

        return snapshot

    except OHLCVParseError as e:
        logger.warning(
            "Failed to parse OHLCV for regime computation",
            error=e.message,
        )
        return None
    except Exception as e:
        logger.error(
            "Regime computation failed",
            error=str(e),
        )
        return None


def add_regime_to_metrics(
    metrics: dict,
    file_content: bytes,
    filename: str = "data.csv",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    source: str = "live",
    timeframe: Optional[str] = None,
    instrument: Optional[str] = None,
) -> dict:
    """
    Add regime snapshot to metrics dictionary.

    Computes regime for the window and embeds it in metrics["regime"].

    Args:
        metrics: Serialized metrics dict (will be modified)
        file_content: Raw CSV bytes
        filename: Original filename
        date_from: Start of window
        date_to: End of window
        source: Computation source
        timeframe: Optional timeframe hint
        instrument: Optional instrument hint

    Returns:
        Modified metrics dict with regime key added
    """
    regime = compute_regime_for_window(
        file_content=file_content,
        filename=filename,
        date_from=date_from,
        date_to=date_to,
        source=source,
        timeframe=timeframe,
        instrument=instrument,
    )

    if regime is not None:
        metrics["regime"] = regime.to_dict()
    else:
        metrics["regime"] = None

    return metrics


def detect_timeframe_from_ohlcv(
    file_content: bytes,
    filename: str = "data.csv",
) -> Optional[str]:
    """
    Detect timeframe from OHLCV data.

    Args:
        file_content: Raw CSV bytes
        filename: Original filename

    Returns:
        Timeframe string or None
    """
    try:
        result = parse_ohlcv_csv(file_content, filename)
        df = result.df

        if len(df) < 2:
            return None

        # Get median time difference
        diffs = df.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return None

        median_diff = diffs.median()
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
        ]

        for seconds, tf in timeframe_map:
            if 0.9 * seconds <= median_seconds <= 1.1 * seconds:
                return tf

        return None

    except Exception:
        return None


def extract_instrument_from_filename(filename: str) -> Optional[str]:
    """
    Extract instrument/symbol from filename.

    Args:
        filename: Original filename

    Returns:
        Instrument string or None
    """
    import re

    if not filename:
        return None

    name = filename.lower()
    name = re.sub(r"\.(csv|txt|data)$", "", name)

    # Crypto patterns
    crypto_pattern = (
        r"(btc|eth|sol|xrp|ada|doge|bnb|ltc|dot|link)[-_]?(usd[t]?|usdc|busd|eur|gbp)"
    )
    crypto_match = re.search(crypto_pattern, name, re.IGNORECASE)
    if crypto_match:
        return crypto_match.group().upper().replace("-", "").replace("_", "")

    # Stock patterns (2-5 uppercase at start)
    stock_pattern = r"^([a-z]{2,5})[-_]"
    stock_match = re.search(stock_pattern, name, re.IGNORECASE)
    if stock_match:
        return stock_match.group(1).upper()

    return None
