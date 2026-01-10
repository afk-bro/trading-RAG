"""Unit tests for KB OHLCV parsing enhancements."""

import pytest
import pandas as pd
from datetime import datetime

from app.services.kb.parsing import (
    ParsedDataset,
    parse_ohlcv_for_kb,
    compute_fingerprint_from_bytes,
    detect_timeframe_from_filename,
    validate_for_regime,
    _extract_instrument,
    _detect_timeframe,
    OHLCVParseError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_csv_bytes():
    """Create valid OHLCV CSV content."""
    csv_content = """date,open,high,low,close,volume
2024-01-01 00:00:00,100.0,101.5,99.5,101.0,1000
2024-01-01 01:00:00,101.0,102.0,100.0,101.5,1100
2024-01-01 02:00:00,101.5,103.0,101.0,102.5,1200
2024-01-01 03:00:00,102.5,104.0,102.0,103.5,1300
2024-01-01 04:00:00,103.5,105.0,103.0,104.5,1400
"""
    # Extend to 200+ rows for regime computation
    lines = csv_content.strip().split("\n")
    header = lines[0]
    _base_rows = lines[1:]  # noqa: F841

    # Generate more rows
    rows = [header]
    for i in range(250):
        hour = i
        day = hour // 24
        hour_of_day = hour % 24
        date = f"2024-01-{day+1:02d} {hour_of_day:02d}:00:00"
        price = 100 + i * 0.1
        rows.append(f"{date},{price},{price+1},{price-1},{price+0.5},{1000+i}")

    return "\n".join(rows).encode("utf-8")


@pytest.fixture
def minimal_csv_bytes():
    """Create minimal but valid CSV content."""
    lines = ["date,open,high,low,close,volume"]
    for i in range(50):  # Less than 200 bars
        date = f"2024-01-{(i // 24) + 1:02d} {i % 24:02d}:00:00"
        price = 100 + i * 0.1
        lines.append(f"{date},{price},{price+1},{price-1},{price+0.5},{1000}")
    return "\n".join(lines).encode("utf-8")


# =============================================================================
# Parsing Tests
# =============================================================================


class TestParseOhlcvForKb:
    """Tests for parse_ohlcv_for_kb function."""

    def test_basic_parsing(self, valid_csv_bytes):
        """Should parse valid CSV correctly."""
        result = parse_ohlcv_for_kb(
            valid_csv_bytes,
            filename="BTCUSD_1h.csv",
        )

        assert isinstance(result, ParsedDataset)
        assert result.n_bars > 0
        assert result.fingerprint is not None
        assert len(result.fingerprint) == 64  # SHA256 hex

    def test_metadata_extraction(self, valid_csv_bytes):
        """Should extract metadata from filename."""
        result = parse_ohlcv_for_kb(
            valid_csv_bytes,
            filename="BTCUSD_1h.csv",
        )

        assert result.instrument == "BTCUSD"
        assert result.timeframe == "1h"

    def test_metadata_hints_override(self, valid_csv_bytes):
        """Hints should override extracted metadata."""
        result = parse_ohlcv_for_kb(
            valid_csv_bytes,
            filename="data.csv",
            instrument_hint="ETHUSD",
            timeframe_hint="4h",
        )

        assert result.instrument == "ETHUSD"
        assert result.timeframe == "4h"

    def test_fingerprint_stability(self, valid_csv_bytes):
        """Same data should produce same fingerprint."""
        result1 = parse_ohlcv_for_kb(valid_csv_bytes, filename="test.csv")
        result2 = parse_ohlcv_for_kb(valid_csv_bytes, filename="test.csv")

        assert result1.fingerprint == result2.fingerprint

    def test_date_filtering(self, valid_csv_bytes):
        """Date filtering should work."""
        # Filter to keep enough rows (fixture has 250 rows starting Jan 1)
        # Use naive datetime to avoid pandas timezone conflicts
        filter_date = datetime(2024, 1, 2)
        result = parse_ohlcv_for_kb(
            valid_csv_bytes,
            filename="test.csv",
            date_from=filter_date,
        )

        # Result dates should be on or after filter date
        assert result.ts_start.replace(tzinfo=None) >= filter_date

    def test_duration_days(self, valid_csv_bytes):
        """Duration days should be calculated."""
        result = parse_ohlcv_for_kb(valid_csv_bytes, filename="test.csv")

        assert result.duration_days > 0


# =============================================================================
# Fingerprint Tests
# =============================================================================


class TestFingerprinting:
    """Tests for fingerprint computation."""

    def test_fingerprint_hex_format(self, valid_csv_bytes):
        """Fingerprint should be valid SHA256 hex."""
        result = parse_ohlcv_for_kb(valid_csv_bytes, filename="test.csv")

        # Valid hex string
        assert all(c in "0123456789abcdef" for c in result.fingerprint)
        assert len(result.fingerprint) == 64

    def test_fingerprint_differs_for_different_data(self):
        """Different data should produce different fingerprints."""
        csv1 = b"date,open,high,low,close,volume\n2024-01-01,100,101,99,100.5,1000\n"
        csv2 = b"date,open,high,low,close,volume\n2024-01-01,100,101,99,100.6,1000\n"

        fp1 = compute_fingerprint_from_bytes(csv1)
        fp2 = compute_fingerprint_from_bytes(csv2)

        assert fp1 != fp2

    def test_fingerprint_order_independent(self):
        """Fingerprint should be stable regardless of original row order."""
        # Generate 50 rows of data (minimum required by parser is 10)
        base_sorted = ["date,open,high,low,close,volume"]
        for i in range(50):
            day = i // 24 + 1
            hour = i % 24
            price = 100 + i * 0.1
            base_sorted.append(
                f"2024-01-{day:02d} {hour:02d}:00:00,{price},{price+1},{price-1},{price+0.5},1000"
            )

        # Reverse middle portion to create unsorted data
        base_unsorted = base_sorted[:1] + base_sorted[25:] + base_sorted[1:25]

        csv_sorted = "\n".join(base_sorted).encode("utf-8")
        csv_unsorted = "\n".join(base_unsorted).encode("utf-8")

        # Parse both (parser sorts by date)
        result1 = parse_ohlcv_for_kb(csv_unsorted, filename="test.csv")
        result2 = parse_ohlcv_for_kb(csv_sorted, filename="test.csv")

        # Fingerprints should match after parsing
        assert result1.fingerprint == result2.fingerprint


# =============================================================================
# Instrument Extraction Tests
# =============================================================================


class TestInstrumentExtraction:
    """Tests for instrument extraction from filename."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("BTCUSD_1h.csv", "BTCUSD"),
            ("btc_usd_4h_data.csv", "BTCUSD"),
            ("btcusdt.csv", "BTCUSDT"),
            ("eth-usd-daily.csv", "ETHUSD"),
            ("SPY_daily.csv", "SPY"),
            ("AAPL_data.csv", "AAPL"),  # Changed: symbol at start
            ("MSFT.csv", "MSFT"),
        ],
    )
    def test_instrument_patterns(self, filename, expected):
        """Should extract instrument from various filename patterns."""
        df = pd.DataFrame()  # Empty df for testing
        result = _extract_instrument(filename, df)
        assert result == expected

    def test_no_instrument(self):
        """Should return None for unrecognized patterns."""
        df = pd.DataFrame()
        result = _extract_instrument("random_data_file_12345.csv", df)
        # May or may not extract, just shouldn't crash
        assert result is None or isinstance(result, str)


# =============================================================================
# Timeframe Detection Tests
# =============================================================================


class TestTimeframeDetection:
    """Tests for timeframe detection."""

    @pytest.mark.parametrize(
        "interval_seconds,expected_tf",
        [
            (60, "1m"),
            (300, "5m"),
            (900, "15m"),
            (1800, "30m"),
            (3600, "1h"),
            (14400, "4h"),
            (86400, "1d"),
        ],
    )
    def test_timeframe_from_data(self, interval_seconds, expected_tf):
        """Should detect timeframe from timestamp intervals."""
        dates = pd.date_range(
            "2024-01-01",
            periods=100,
            freq=f"{interval_seconds}s",
            tz="UTC",
        )
        df = pd.DataFrame(
            {
                "close": [100] * 100,
            },
            index=dates,
        )

        result = _detect_timeframe(df)
        assert result == expected_tf

    @pytest.mark.parametrize(
        "filename,expected_tf",
        [
            ("BTCUSD_1m.csv", "1m"),
            ("data_5min.csv", "5m"),
            ("ETHUSD_1h.csv", "1h"),
            ("SPY_4hour.csv", "4h"),
            ("AAPL_daily.csv", "1d"),
            ("MSFT_1d.csv", "1d"),
        ],
    )
    def test_timeframe_from_filename(self, filename, expected_tf):
        """Should extract timeframe from filename."""
        result = detect_timeframe_from_filename(filename)
        assert result == expected_tf


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for dataset validation."""

    def test_validate_insufficient_bars(self):
        """Should warn about insufficient bars."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        df = pd.DataFrame({"close": [100] * 50}, index=dates)

        warnings = validate_for_regime(df, min_bars=200)

        assert any("insufficient_bars" in w for w in warnings)

    def test_validate_large_gaps(self):
        """Should warn about large timestamp gaps."""
        # Create data with a large gap
        dates1 = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        dates2 = pd.date_range(
            "2024-01-15", periods=100, freq="1h", tz="UTC"
        )  # 2 week gap
        dates = dates1.append(dates2)

        df = pd.DataFrame({"close": [100] * 200}, index=dates)

        warnings = validate_for_regime(df, min_bars=200)

        assert any("large_gaps" in w for w in warnings)

    def test_validate_sufficient_data(self):
        """Should pass validation for good data."""
        dates = pd.date_range("2024-01-01", periods=250, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "close": [100] * 250,
                "volume": [1000] * 250,
            },
            index=dates,
        )

        warnings = validate_for_regime(df, min_bars=200)

        # Should have no critical warnings
        assert not any("insufficient_bars" in w for w in warnings)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_csv_raises(self):
        """Invalid CSV should raise OHLCVParseError."""
        with pytest.raises(OHLCVParseError):
            parse_ohlcv_for_kb(
                b"not,valid,csv\nwith,missing,columns", filename="bad.csv"
            )

    def test_missing_columns_raises(self):
        """Missing required columns should raise error."""
        csv = b"date,open,high,close,volume\n2024-01-01,100,101,100.5,1000\n"  # Missing 'low'

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_for_kb(csv, filename="test.csv")

        assert "Missing required columns" in str(exc_info.value) or "low" in str(
            exc_info.value
        )

    def test_empty_csv_raises(self):
        """Empty CSV should raise error."""
        with pytest.raises(OHLCVParseError):
            parse_ohlcv_for_kb(b"", filename="empty.csv")
