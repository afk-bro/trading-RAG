"""
Unit tests for CLI + orchestrator wiring in run_unicorn_backtest.py.

Structure/type assertions only — no backtest execution.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.strategy.models import OHLCVBar  # noqa: E402
from app.services.backtest.engines.unicorn_runner import BiasState  # noqa: E402
from app.services.strategy.strategies.unicorn_model import BiasDirection  # noqa: E402

from scripts.run_unicorn_backtest import (  # noqa: E402
    build_reference_bias_series,
    _sort_and_normalize_tz,
)


def _make_bar(ts: datetime, price: float = 100.0) -> OHLCVBar:
    """Helper: create a minimal OHLCVBar (ts must be tz-aware)."""
    return OHLCVBar(
        ts=ts,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=1000,
    )


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _utc_bar_series(n: int, base_hour: int = 9) -> list[OHLCVBar]:
    """Generate n bars spaced 15 min apart, handling hour rollover."""
    from datetime import timedelta

    base = _utc(2024, 1, 2, base_hour, 0)
    return [_make_bar(base + timedelta(minutes=i * 15)) for i in range(n)]


class TestSortAndNormalizeTz:
    """_sort_and_normalize_tz returns sorted, UTC-aware bars."""

    def test_sorts_bars_by_timestamp(self):
        b1 = _make_bar(_utc(2024, 1, 2, 10, 0))
        b2 = _make_bar(_utc(2024, 1, 2, 9, 0))
        b3 = _make_bar(_utc(2024, 1, 2, 8, 0))

        result = _sort_and_normalize_tz([b1, b2, b3])

        assert len(result) == 3
        # Sorted ascending
        assert result[0].ts <= result[1].ts <= result[2].ts
        # All UTC-aware
        for bar in result:
            assert bar.ts.tzinfo is not None


class TestBuildReferenceBiasSeries:
    """build_reference_bias_series returns correct types and respects causality."""

    def test_returns_list_of_bias_states(self):
        bars = _utc_bar_series(20)
        series = build_reference_bias_series(bars, [])

        assert isinstance(series, list)
        assert len(series) == len(bars)
        for item in series:
            assert isinstance(item, BiasState)
            assert isinstance(item.direction, BiasDirection)
            assert 0.0 <= item.confidence <= 1.0

    def test_empty_htf_returns_empty(self):
        series = build_reference_bias_series([], [])
        assert series == []

    def test_tolerates_empty_ltf(self):
        bars = _utc_bar_series(5)
        series = build_reference_bias_series(bars, [])
        assert len(series) == 5

    def test_timestamps_match_htf_bars(self):
        bars = _utc_bar_series(10)
        series = build_reference_bias_series(bars, [])
        for bar, state in zip(bars, series):
            assert state.ts == bar.ts


class TestCliRefArgs:
    """CLI argument parsing validates --ref-symbol / --ref-htf constraints."""

    def test_ref_htf_without_ref_symbol_errors(self):
        """--ref-htf without --ref-symbol should error."""
        from scripts.run_unicorn_backtest import main

        with pytest.raises(SystemExit):
            with patch(
                "sys.argv",
                [
                    "prog",
                    "--symbol",
                    "NQ",
                    "--synthetic",
                    "--ref-htf",
                    "some/path.csv",
                ],
            ):
                main()

    def test_ref_symbol_same_as_symbol_errors(self):
        """--ref-symbol matching --symbol should error."""
        from scripts.run_unicorn_backtest import main

        with pytest.raises(SystemExit):
            with patch(
                "sys.argv",
                [
                    "prog",
                    "--symbol",
                    "NQ",
                    "--synthetic",
                    "--ref-symbol",
                    "NQ",
                ],
            ):
                main()
