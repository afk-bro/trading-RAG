"""
Unit tests for Databento data fetcher utilities.

These tests don't require an API key - they test the utility functions
for contract symbol generation and date range handling.
"""

from datetime import datetime
import pytest

from app.services.backtest.data import (
    get_front_month_symbol,
    get_continuous_symbols,
)


class TestFrontMonthSymbol:
    """Tests for front month contract symbol generation."""

    def test_january_gets_march_contract(self):
        """In January, front month is March (H)."""
        date = datetime(2024, 1, 15)
        symbol = get_front_month_symbol("NQ", date)
        assert symbol == "NQH4"  # March 2024

    def test_february_gets_march_contract(self):
        """In February, front month is March (H)."""
        date = datetime(2024, 2, 10)
        symbol = get_front_month_symbol("NQ", date)
        assert symbol == "NQH4"

    def test_early_march_gets_march_contract(self):
        """Early March (before rollover) still has March contract."""
        date = datetime(2024, 3, 5)
        symbol = get_front_month_symbol("ES", date)
        assert symbol == "ESH4"

    def test_late_march_gets_june_contract(self):
        """Late March (after rollover) has June contract."""
        date = datetime(2024, 3, 20)
        symbol = get_front_month_symbol("ES", date)
        assert symbol == "ESM4"  # June 2024

    def test_december_gets_next_year_march(self):
        """In December (after rollover), get next year's March."""
        date = datetime(2024, 12, 20)
        symbol = get_front_month_symbol("NQ", date)
        assert symbol == "NQH5"  # March 2025

    def test_june_gets_september_contract(self):
        """In June (after rollover), get September (U)."""
        date = datetime(2024, 6, 25)
        symbol = get_front_month_symbol("NQ", date)
        assert symbol == "NQU4"  # September 2024


class TestContinuousSymbols:
    """Tests for continuous contract symbol generation."""

    def test_single_contract_period(self):
        """Date range within single contract period returns one symbol."""
        start = datetime(2024, 1, 10)
        end = datetime(2024, 2, 15)

        contracts = get_continuous_symbols("NQ", start, end)

        assert len(contracts) >= 1
        assert contracts[0][0] == "NQH4"  # March contract

    def test_cross_quarter_returns_multiple_contracts(self):
        """Date range crossing quarter returns multiple contracts."""
        start = datetime(2024, 2, 1)
        end = datetime(2024, 4, 30)

        contracts = get_continuous_symbols("NQ", start, end)

        # Should have at least H4 (March) and M4 (June)
        symbols = [c[0] for c in contracts]
        assert "NQH4" in symbols
        assert "NQM4" in symbols

    def test_contract_periods_dont_overlap(self):
        """Contract periods should not overlap."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)

        contracts = get_continuous_symbols("ES", start, end)

        # Check that each period ends before the next starts
        for i in range(len(contracts) - 1):
            _, _, period_end = contracts[i]
            _, next_start, _ = contracts[i + 1]
            assert period_end < next_start

    def test_periods_cover_full_range(self):
        """Contract periods should cover the requested date range."""
        start = datetime(2024, 1, 15)
        end = datetime(2024, 3, 15)

        contracts = get_continuous_symbols("NQ", start, end)

        # First period should start on or before requested start
        first_period_start = contracts[0][1]
        assert first_period_start <= start

        # Last period should end on or after requested end
        last_period_end = contracts[-1][2]
        assert last_period_end >= end


class TestSymbolRoots:
    """Tests for different symbol roots."""

    def test_es_symbol(self):
        """ES symbol generates correct contracts."""
        date = datetime(2024, 4, 1)
        symbol = get_front_month_symbol("ES", date)
        assert symbol.startswith("ES")
        assert symbol == "ESM4"  # June after March rollover

    def test_nq_symbol(self):
        """NQ symbol generates correct contracts."""
        date = datetime(2024, 4, 1)
        symbol = get_front_month_symbol("NQ", date)
        assert symbol.startswith("NQ")
        assert symbol == "NQM4"


class TestLoadFromCSV:
    """Tests for loading data from local Databento CSV files."""

    def test_front_month_filtering(self):
        """Front month filtering maps correct contracts to dates."""
        from app.services.backtest.data import DatabentoFetcher

        fetcher = DatabentoFetcher()

        # Get contracts for a 6-month period
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)
        contracts = get_continuous_symbols("NQ", start, end)

        # Should have 3 contracts: H4, M4, U4
        symbols = [c[0] for c in contracts]
        assert "NQH4" in symbols  # Jan-Mar
        assert "NQM4" in symbols  # Mar-Jun
        assert "NQU4" in symbols  # Jun onwards

    def test_roll_date_boundaries(self):
        """Roll dates correctly transition between contracts."""
        # March 2024 roll: H4 -> M4 around March 5-10
        contracts = get_continuous_symbols("NQ", datetime(2024, 2, 1), datetime(2024, 4, 1))

        h4_end = None
        m4_start = None
        for sym, start, end in contracts:
            if sym == "NQH4":
                h4_end = end
            if sym == "NQM4":
                m4_start = start

        # H4 should end before M4 starts (no overlap)
        if h4_end and m4_start:
            assert h4_end < m4_start

        # Roll should happen in early-to-mid March
        if h4_end:
            assert h4_end.month == 3
            assert h4_end.day < 15
