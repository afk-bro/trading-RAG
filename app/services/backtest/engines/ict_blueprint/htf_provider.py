"""HTF data provider — decouples daily state from the H1 engine loop."""

from __future__ import annotations

from bisect import bisect_right
from typing import Protocol

import numpy as np
import pandas as pd

from .htf_bias import SwingDetector, update_htf
from .types import HTFState, HTFStateSnapshot, ICTBlueprintParams


class HTFProvider(Protocol):
    """Provides causally-aligned HTF state snapshots for any H1 timestamp."""

    def get_state_at(self, h1_ts: int) -> HTFStateSnapshot: ...


class DefaultHTFProvider:
    """Builds HTF state lazily as H1 bars advance chronologically.

    daily_df must have a DatetimeIndex and OHLCV columns (Open, High, Low, Close).
    Timestamps are stored as int64 nanoseconds for fast bisecting.
    """

    def __init__(self, daily_df: pd.DataFrame, params: ICTBlueprintParams) -> None:
        self._params = params

        # Build int64 ns array of daily close timestamps
        self._daily_close_ts: np.ndarray = daily_df.index.astype(np.int64).values
        self._daily_bars: list[tuple[int, int, float, float, float, float]] = []
        for i, (ts_val, row) in enumerate(daily_df.iterrows()):
            ts_ns = int(pd.Timestamp(ts_val).value)
            self._daily_bars.append(
                (i, ts_ns, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]))
            )

        self._state = HTFState()
        self._swing_detector = SwingDetector(lookback=params.swing_lookback)
        self._processed_up_to: int = -1  # daily index already processed
        self._bars_history: list[tuple[int, int, float, float, float, float]] = []

        # Cache snapshots at each daily index
        self._snapshots: dict[int, HTFStateSnapshot] = {}

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
