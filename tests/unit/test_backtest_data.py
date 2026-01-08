"""Unit tests for backtest OHLCV data parsing."""

import pytest
from pathlib import Path
from datetime import datetime

from app.services.backtest.data import (
    parse_ohlcv_csv,
    OHLCVParseResult,
    OHLCVParseError,
)


# Get fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestParseOhlcvCsv:
    """Tests for parse_ohlcv_csv function."""

    def test_valid_csv_parses_correctly(self):
        """Valid CSV should parse successfully."""
        csv_path = FIXTURES_DIR / "valid_ohlcv.csv"
        content = csv_path.read_bytes()

        result = parse_ohlcv_csv(content, filename="valid_ohlcv.csv")

        assert isinstance(result, OHLCVParseResult)
        assert result.row_count == 30
        assert result.df is not None
        assert len(result.df) == 30
        assert list(result.df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.warnings == []  # No warnings for valid sorted data

    def test_missing_required_column_raises_error(self):
        """CSV missing required column should raise OHLCVParseError."""
        # CSV without 'close' column
        csv_content = b"""date,open,high,low,volume
2024-01-01,100,105,99,1000000
2024-01-02,104,106,102,1100000
"""

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_csv(csv_content, filename="missing_close.csv")

        assert "Missing required columns" in str(exc_info.value)
        assert "close" in exc_info.value.details.get("missing_columns", [])

    def test_unsorted_dates_get_sorted_with_warning(self):
        """Unsorted dates should be sorted and generate a warning."""
        csv_content = b"""date,open,high,low,close,volume
2024-01-03,103,108,101,107,1200000
2024-01-01,100,105,99,104,1000000
2024-01-02,104,106,102,103,1100000
2024-01-05,108,109,100,101,1500000
2024-01-04,107,110,105,108,900000
2024-01-06,101,103,98,99,1300000
2024-01-07,99,102,97,100,1100000
2024-01-08,100,104,99,103,1000000
2024-01-09,103,107,102,106,1200000
2024-01-10,106,108,104,105,900000
"""

        result = parse_ohlcv_csv(csv_content, filename="unsorted.csv")

        # Data should be sorted
        dates = result.df.index.tolist()
        assert dates == sorted(dates), "Data should be sorted by date"

        # Should have a warning about sorting
        assert any("sorted" in w.lower() for w in result.warnings)

    def test_column_aliases_are_mapped(self):
        """Column aliases like 'timestamp' should map to 'date'."""
        csv_content = b"""timestamp,open,high,low,close,vol
2024-01-01,100,105,99,104,1000000
2024-01-02,104,106,102,103,1100000
2024-01-03,103,108,101,107,1200000
2024-01-04,107,110,105,108,900000
2024-01-05,108,109,100,101,1500000
2024-01-06,101,103,98,99,1300000
2024-01-07,99,102,97,100,1100000
2024-01-08,100,104,99,103,1000000
2024-01-09,103,107,102,106,1200000
2024-01-10,106,108,104,105,900000
"""

        result = parse_ohlcv_csv(csv_content, filename="aliased.csv")

        # Should have warnings about mapping
        assert any("timestamp" in w.lower() for w in result.warnings)
        assert any("vol" in w.lower() for w in result.warnings)
        assert result.row_count == 10

    def test_nan_values_are_dropped_with_warning(self):
        """Rows with NaN values should be dropped with warning."""
        csv_content = b"""date,open,high,low,close,volume
2024-01-01,100,105,99,104,1000000
2024-01-02,104,106,102,,1100000
2024-01-03,103,108,101,107,1200000
2024-01-04,107,110,105,108,
2024-01-05,108,109,100,101,1500000
2024-01-06,101,103,98,99,1300000
2024-01-07,99,102,97,100,1100000
2024-01-08,100,104,99,103,1000000
2024-01-09,103,107,102,106,1200000
2024-01-10,106,108,104,105,900000
2024-01-11,105,106,100,101,1400000
2024-01-12,101,104,99,103,1100000
"""

        result = parse_ohlcv_csv(csv_content, filename="with_nans.csv")

        # NaN rows should be dropped
        assert result.row_count == 10  # 12 - 2 with NaN

        # Should have warning about dropped rows
        assert any("dropped" in w.lower() and "nan" in w.lower() for w in result.warnings)

    def test_duplicate_dates_are_removed_with_warning(self):
        """Duplicate dates should be removed with warning."""
        csv_content = b"""date,open,high,low,close,volume
2024-01-01,100,105,99,104,1000000
2024-01-01,101,106,100,105,1100000
2024-01-02,104,106,102,103,1100000
2024-01-03,103,108,101,107,1200000
2024-01-04,107,110,105,108,900000
2024-01-05,108,109,100,101,1500000
2024-01-06,101,103,98,99,1300000
2024-01-07,99,102,97,100,1100000
2024-01-08,100,104,99,103,1000000
2024-01-09,103,107,102,106,1200000
2024-01-10,106,108,104,105,900000
"""

        result = parse_ohlcv_csv(csv_content, filename="with_dupes.csv")

        # Should keep last duplicate
        assert result.row_count == 10  # 11 - 1 duplicate

        # Should have warning about duplicates
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_negative_prices_raise_error(self):
        """Negative prices should raise OHLCVParseError."""
        csv_content = b"""date,open,high,low,close,volume
2024-01-01,100,105,99,104,1000000
2024-01-02,-10,106,102,103,1100000
2024-01-03,103,108,101,107,1200000
2024-01-04,107,110,105,108,900000
2024-01-05,108,109,100,101,1500000
2024-01-06,101,103,98,99,1300000
2024-01-07,99,102,97,100,1100000
2024-01-08,100,104,99,103,1000000
2024-01-09,103,107,102,106,1200000
2024-01-10,106,108,104,105,900000
"""

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_csv(csv_content, filename="negative.csv")

        assert "negative" in str(exc_info.value).lower()

    def test_high_below_low_raises_error(self):
        """High < Low should raise OHLCVParseError."""
        csv_content = b"""date,open,high,low,close,volume
2024-01-01,100,105,99,104,1000000
2024-01-02,104,102,106,103,1100000
2024-01-03,103,108,101,107,1200000
2024-01-04,107,110,105,108,900000
2024-01-05,108,109,100,101,1500000
2024-01-06,101,103,98,99,1300000
2024-01-07,99,102,97,100,1100000
2024-01-08,100,104,99,103,1000000
2024-01-09,103,107,102,106,1200000
2024-01-10,106,108,104,105,900000
"""

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_csv(csv_content, filename="invalid_ohlc.csv")

        assert "high < low" in str(exc_info.value).lower()

    def test_empty_csv_raises_error(self):
        """Empty CSV should raise OHLCVParseError."""
        csv_content = b"""date,open,high,low,close,volume
"""

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_csv(csv_content, filename="empty.csv")

        assert "no data" in str(exc_info.value).lower() or "no rows" in str(exc_info.value).lower()

    def test_insufficient_rows_raises_error(self):
        """Less than 10 rows should raise OHLCVParseError."""
        csv_content = b"""date,open,high,low,close,volume
2024-01-01,100,105,99,104,1000000
2024-01-02,104,106,102,103,1100000
2024-01-03,103,108,101,107,1200000
"""

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_csv(csv_content, filename="too_few.csv")

        assert "insufficient" in str(exc_info.value).lower()

    def test_date_filter_from(self):
        """date_from filter should exclude earlier dates."""
        csv_path = FIXTURES_DIR / "valid_ohlcv.csv"
        content = csv_path.read_bytes()

        result = parse_ohlcv_csv(
            content,
            filename="valid_ohlcv.csv",
            date_from=datetime(2024, 1, 15),
        )

        # Should have rows from Jan 15 onwards
        assert result.row_count == 16  # Jan 15-30

        # Should have filter warning
        assert any("filtered" in w.lower() for w in result.warnings)

    def test_date_filter_to(self):
        """date_to filter should exclude later dates."""
        csv_path = FIXTURES_DIR / "valid_ohlcv.csv"
        content = csv_path.read_bytes()

        result = parse_ohlcv_csv(
            content,
            filename="valid_ohlcv.csv",
            date_to=datetime(2024, 1, 15),
        )

        # Should have rows up to Jan 15
        assert result.row_count == 15  # Jan 1-15

        # Should have filter warning
        assert any("filtered" in w.lower() for w in result.warnings)

    def test_file_too_large_raises_error(self):
        """File exceeding max size should raise OHLCVParseError."""
        # Create content larger than 1KB limit
        csv_content = b"date,open,high,low,close,volume\n" + b"2024-01-01,100,105,99,104,1000000\n" * 1000

        with pytest.raises(OHLCVParseError) as exc_info:
            parse_ohlcv_csv(csv_content, filename="large.csv", max_file_size_mb=0.001)

        assert "too large" in str(exc_info.value).lower()
