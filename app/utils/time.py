"""
Timezone utilities for session-aware trading logic.

All bar/snapshot timestamps are stored as UTC. Session windows
(NY AM, London, etc.) are defined in ET clock time. This module
provides the single conversion point.
"""

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc


def ensure_utc(ts: datetime) -> datetime:
    """Validate that a datetime is tz-aware and convert to UTC.

    Use at data-ingestion boundaries (market data adapters, CSV loaders,
    DB reads) to guarantee every bar timestamp entering the system is
    tz-aware UTC.

    Args:
        ts: A timezone-aware datetime (any timezone).

    Returns:
        The same instant as a UTC-aware datetime.

    Raises:
        ValueError: If ts is naive (no tzinfo).
    """
    if ts.tzinfo is None:
        raise ValueError(
            "ensure_utc requires a tz-aware datetime, got naive. "
            "Hint: use datetime(..., tzinfo=timezone.utc) for UTC timestamps."
        )
    return ts.astimezone(_UTC)


def to_eastern_time(ts: datetime) -> time:
    """Convert a tz-aware datetime to Eastern Time time-of-day.

    Accepts any tz-aware datetime and converts to America/New_York.
    Handles DST automatically via zoneinfo.

    Args:
        ts: Timezone-aware datetime (any timezone).

    Returns:
        time-of-day in ET (naive time object).

    Raises:
        ValueError: If ts is a naive (timezone-unaware) datetime.
    """
    if ts.tzinfo is None:
        raise ValueError(
            "to_eastern_time requires a tz-aware datetime, got naive. "
            "Hint: use datetime(..., tzinfo=timezone.utc) for UTC timestamps."
        )
    return ts.astimezone(_ET).time()
