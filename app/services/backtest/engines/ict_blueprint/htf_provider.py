"""HTF data provider — decouples daily state from the H1 engine loop."""

from __future__ import annotations

import logging
from bisect import bisect_right
from typing import Protocol

import numpy as np
import pandas as pd

from .htf_bias import SwingDetector, update_htf
from .types import HTFState, HTFStateSnapshot, ICTBlueprintParams

logger = logging.getLogger(__name__)


class HTFProvider(Protocol):
    """Provides causally-aligned HTF state snapshots for any H1 timestamp."""

    def get_state_at(self, h1_ts: int) -> HTFStateSnapshot: ...


class DefaultHTFProvider:
    """Builds HTF state lazily as H1 bars advance chronologically.

    daily_df must have a DatetimeIndex (or date index) and OHLCV columns.
    Timestamps are stored as int64 nanoseconds for fast bisecting.

    If the daily index is date-only (midnight timestamps), a session close
    offset is added so that an H1 bar at e.g. 10:00 on day D does NOT see
    day D's daily close.  The default offset of 16 hours corresponds to
    ES/NQ RTH close (16:00 ET).
    """

    def __init__(
        self,
        daily_df: pd.DataFrame,
        params: ICTBlueprintParams,
        session_close_hour: int = 16,
    ) -> None:
        self._params = params

        # Detect if index is date-only (no meaningful time component).
        # If so, shift each daily timestamp to session close time so that
        # intraday H1 bars don't peek into the still-forming daily bar.
        close_offset_ns = self._compute_close_offset(daily_df.index, session_close_hour)

        self._daily_close_ts: np.ndarray = (
            daily_df.index.astype("datetime64[ns]").astype(np.int64) + close_offset_ns
        )
        self._daily_bars: list[tuple[int, int, float, float, float, float]] = []
        for i, (ts_val, row) in enumerate(daily_df.iterrows()):
            ts_ns = int(pd.Timestamp(ts_val).value) + close_offset_ns
            self._daily_bars.append(
                (
                    i,
                    ts_ns,
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                )
            )

        self._state = HTFState()
        self._swing_detector = SwingDetector(lookback=params.swing_lookback)
        self._processed_up_to: int = -1  # daily index already processed
        self._bars_history: list[tuple[int, int, float, float, float, float]] = []

        # Cache snapshots at each daily index
        self._snapshots: dict[int, HTFStateSnapshot] = {}

        self._validate_daily_timestamps(daily_df.index, session_close_hour)

    @staticmethod
    def _compute_close_offset(index: pd.Index, session_close_hour: int) -> int:
        """Return ns offset to add to daily timestamps.

        If the index already carries intraday times (e.g. 16:00), returns 0.
        If dates-only (midnight), returns session_close_hour in nanoseconds.
        """
        if len(index) == 0:
            return 0

        sample_ts = pd.Timestamp(index[0])
        # Date-only indices convert to midnight; check if time is 00:00
        if sample_ts.hour == 0 and sample_ts.minute == 0 and sample_ts.second == 0:
            return int(session_close_hour * 3_600 * 1_000_000_000)
        return 0

    @staticmethod
    def _validate_daily_timestamps(index: pd.Index, session_close_hour: int) -> None:
        """Warn if daily bar timestamps look inconsistent with session_close_hour.

        Checks:
        - If index has intraday times, the hour should match session_close_hour.
        - Duplicate dates suggest non-daily granularity was passed in.
        """
        if len(index) < 2:
            return

        sample_ts = pd.Timestamp(index[0])

        # Check 1: if timestamps carry time, it should be close to session_close_hour
        if not (
            sample_ts.hour == 0 and sample_ts.minute == 0 and sample_ts.second == 0
        ):
            if sample_ts.hour != session_close_hour:
                logger.warning(
                    "Daily bar timestamp hour (%d) != session_close_hour (%d). "
                    "Verify session_type matches data (RTH vs Globex).",
                    sample_ts.hour,
                    session_close_hour,
                )

        # Check 2: duplicate calendar dates suggest sub-daily data was passed
        dates = pd.DatetimeIndex(index).normalize()
        n_unique = dates.nunique()
        if n_unique < len(index):
            logger.warning(
                "Daily DataFrame has %d rows but only %d unique dates. "
                "This likely means sub-daily data was passed as daily.",
                len(index),
                n_unique,
            )

    def get_state_at(self, h1_ts: int) -> HTFStateSnapshot:
        """Return HTF state valid for an H1 bar with timestamp *h1_ts* (ns).

        Uses bisect_right to find the last completed daily bar before h1_ts.
        """
        # Find index of last daily bar that closed before h1_ts
        # daily_close_ts[i] is the close ts of daily bar i.
        # We want the largest i such that daily_close_ts[i] < h1_ts.
        idx = bisect_right(self._daily_close_ts, h1_ts - 1) - 1

        if idx < 0:
            return HTFStateSnapshot.from_state(self._state)

        # Process daily bars we haven't handled yet
        self._advance_to(idx)

        if idx in self._snapshots:
            return self._snapshots[idx]
        return HTFStateSnapshot.from_state(self._state)

    def _advance_to(self, target_idx: int) -> None:
        """Incrementally process daily bars up to target_idx."""
        start = self._processed_up_to + 1
        for i in range(start, min(target_idx + 1, len(self._daily_bars))):
            bar = self._daily_bars[i]
            self._bars_history.append(bar)
            update_htf(
                state=self._state,
                bar_index=bar[0],
                bar=bar,
                params=self._params,
                swing_detector=self._swing_detector,
                bars_history=self._bars_history,
            )
            self._processed_up_to = i
            self._snapshots[i] = HTFStateSnapshot.from_state(self._state)
