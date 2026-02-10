"""
Unit tests for Databento data fetcher utilities.

These tests don't require an API key - they test the utility functions
for contract symbol generation and date range handling.
"""

from datetime import datetime, timezone, timedelta
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

        DatabentoFetcher()

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
        contracts = get_continuous_symbols(
            "NQ", datetime(2024, 2, 1), datetime(2024, 4, 1)
        )

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


class TestResample4H:
    """Tests for 4-hour bar resampling."""

    def _make_1m_bars(self, start_ts: datetime, count: int, base_price: float = 100.0):
        """Generate synthetic 1-minute bars."""
        from app.services.strategy.models import OHLCVBar

        bars = []
        price = base_price
        for i in range(count):
            ts = start_ts + timedelta(minutes=i)
            bars.append(
                OHLCVBar(
                    ts=ts,
                    open=price,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price + 0.5,
                    volume=100.0,
                )
            )
            price += 0.5
        return bars

    def test_resample_4h_basic(self):
        """240 synthetic 1m bars => 1 bar with correct OHLCV."""
        from app.services.backtest.data import DatabentoFetcher

        fetcher = DatabentoFetcher()
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars_1m = self._make_1m_bars(start, 240)

        result = fetcher._resample_bars(bars_1m, "4h")

        assert len(result) == 1
        # OHLCV aggregation checks
        assert result[0].ts == bars_1m[0].ts
        assert result[0].open == bars_1m[0].open
        assert result[0].close == bars_1m[-1].close
        assert result[0].high == max(b.high for b in bars_1m)
        assert result[0].low == min(b.low for b in bars_1m)
        assert result[0].volume == sum(b.volume for b in bars_1m)

    def test_resample_4h_bucket_boundaries(self):
        """Bars at 03:59 and 04:00 fall in different 4H buckets."""
        from app.services.backtest.data import DatabentoFetcher
        from app.services.strategy.models import OHLCVBar

        fetcher = DatabentoFetcher()
        # 03:59 is in bucket [0:00-4:00), bucket_key = (date, 0)
        # 04:00 is in bucket [4:00-8:00), bucket_key = (date, 240)
        bars = [
            OHLCVBar(
                ts=datetime(2024, 1, 2, 3, 59, tzinfo=timezone.utc),
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=50,
            ),
            OHLCVBar(
                ts=datetime(2024, 1, 2, 4, 0, tzinfo=timezone.utc),
                open=101,
                high=102,
                low=100,
                close=101.5,
                volume=60,
            ),
        ]

        result = fetcher._resample_bars(bars, "4h")

        assert len(result) == 2
        assert result[0].ts == datetime(2024, 1, 2, 3, 59, tzinfo=timezone.utc)
        assert result[1].ts == datetime(2024, 1, 2, 4, 0, tzinfo=timezone.utc)

    def test_resample_4h_multiple_days(self):
        """Correct bar count across multiple days."""
        from app.services.backtest.data import DatabentoFetcher

        fetcher = DatabentoFetcher()
        # 2 full days: 24h * 60 = 1440 minutes each, expect 6 4H bars/day = 12 total
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        bars_1m = self._make_1m_bars(start, 1440 * 2)

        result = fetcher._resample_bars(bars_1m, "4h")

        assert len(result) == 12  # 6 buckets/day * 2 days


class TestBarBundle:
    """Tests for BarBundle dataclass."""

    def test_bar_bundle_creation(self):
        """BarBundle can be created with all fields including None."""
        from app.services.backtest.engines.unicorn_runner import BarBundle
        from app.services.strategy.models import OHLCVBar

        bar = OHLCVBar(
            ts=datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc),
            open=100,
            high=101,
            low=99,
            close=100.5,
            volume=100,
        )
        bundle = BarBundle(
            h4=[bar],
            h1=[bar, bar],
            m15=[bar],
            m5=None,
            m1=None,
        )
        assert len(bundle.h4) == 1
        assert len(bundle.h1) == 2
        assert len(bundle.m15) == 1
        assert bundle.m5 is None
        assert bundle.m1 is None

    def test_bar_bundle_all_none(self):
        """BarBundle with all None fields is valid."""
        from app.services.backtest.engines.unicorn_runner import BarBundle

        bundle = BarBundle()
        assert bundle.h4 is None
        assert bundle.m1 is None

    def test_bar_bundle_frozen(self):
        """BarBundle is immutable."""
        from app.services.backtest.engines.unicorn_runner import BarBundle

        bundle = BarBundle()
        with pytest.raises(AttributeError):
            bundle.h4 = []


class TestFetchMultiTF:
    """Tests for multi-TF resampling via _resample_to_bundle."""

    def _make_1m_bars(self, start_ts: datetime, count: int, base_price: float = 100.0):
        """Generate synthetic 1-minute bars."""
        from app.services.strategy.models import OHLCVBar

        bars = []
        price = base_price
        for i in range(count):
            ts = start_ts + timedelta(minutes=i)
            bars.append(
                OHLCVBar(
                    ts=ts,
                    open=price,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price + 0.5,
                    volume=100.0,
                )
            )
            price += 0.5
        return bars

    def test_fetch_multi_tf_resamples_all(self):
        """_resample_to_bundle produces all TFs from 1m data."""
        from app.services.backtest.data import DatabentoFetcher

        fetcher = DatabentoFetcher()
        # 1 full day of 1m bars = 1440
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        m1_bars = self._make_1m_bars(start, 1440)

        bundle = fetcher._resample_to_bundle(m1_bars)

        assert bundle.m1 is not None
        assert len(bundle.m1) == 1440
        assert bundle.m5 is not None
        assert len(bundle.m5) == 288  # 1440 / 5
        assert bundle.m15 is not None
        assert len(bundle.m15) == 96  # 1440 / 15
        assert bundle.h1 is not None
        assert len(bundle.h1) == 24  # 1440 / 60
        assert bundle.h4 is not None
        assert len(bundle.h4) == 6  # 1440 / 240
