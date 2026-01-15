"""
Unit tests for Pine Script ingestion service.

Tests content formatting and ingest logic without database/vector dependencies.
"""

import pytest

from app.services.pine.ingest import (
    extract_symbols_from_script,
    format_script_content,
    PineIngestResult,
    ScriptIngestResult,
)
from app.services.pine.models import (
    InputType,
    LintSummary,
    PineImport,
    PineInput,
    PineScriptEntry,
    PineVersion,
    ScriptType,
)


class TestFormatScriptContent:
    """Tests for format_script_content."""

    def test_format_basic_indicator(self):
        """Formats a basic indicator script."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test Indicator",
        )

        content = format_script_content(
            entry, source_content=None, include_source=False
        )

        assert "# Test Indicator" in content
        assert "**Type**: indicator" in content
        assert "**Pine Version**: 5" in content
        assert "**File**: test.pine" in content

    def test_format_with_inputs(self):
        """Includes input parameters in formatted content."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
            inputs=[
                PineInput(
                    name="Length",
                    type=InputType.INT,
                    default=14,
                    line=3,
                ),
                PineInput(
                    name="Source",
                    type=InputType.SOURCE,
                    default_expr="close",
                    tooltip="Price source to use",
                    line=4,
                ),
            ],
        )

        content = format_script_content(entry, include_source=False)

        assert "## Inputs" in content
        assert "**Length** (int)" in content
        assert "(default: 14)" in content
        assert "**Source** (source)" in content
        assert "(default: close)" in content
        assert "Price source to use" in content

    def test_format_with_imports(self):
        """Includes library imports in formatted content."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
            imports=[
                PineImport(path="TradingView/ta", alias="ta"),
                PineImport(path="MyLib/utils", alias="utils"),
            ],
        )

        content = format_script_content(entry, include_source=False)

        assert "## Imports" in content
        assert "TradingView/ta" in content
        assert "as ta" in content
        assert "MyLib/utils" in content

    def test_format_with_features(self):
        """Includes detected features in formatted content."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.STRATEGY,
            title="Test Strategy",
            features={
                "uses_request_security": True,
                "uses_arrays": True,
                "uses_alert": False,
                "is_library": False,
            },
        )

        content = format_script_content(entry, include_source=False)

        assert "## Features" in content
        assert "Uses arrays" in content
        assert "Uses request security" in content
        assert "Uses alert" not in content  # False features excluded

    def test_format_with_lint_summary(self):
        """Includes lint summary when present."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
            lint=LintSummary(error_count=2, warning_count=1, info_count=0),
        )

        content = format_script_content(entry, include_source=False)

        assert "## Lint Summary" in content
        assert "Errors: 2" in content
        assert "Warnings: 1" in content
        assert "Info:" not in content  # 0 count excluded

    def test_format_with_source_code(self):
        """Includes source code when provided."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
        )

        source = "//@version=5\nindicator('Test')\nplot(close)"
        content = format_script_content(
            entry, source_content=source, include_source=True
        )

        assert "## Source Code" in content
        assert "```pine" in content
        assert "//@version=5" in content
        assert "plot(close)" in content
        assert "```" in content

    def test_format_truncates_long_source(self):
        """Truncates source code exceeding max lines."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
        )

        # Create source with 150 lines
        source = "\n".join([f"line_{i}" for i in range(150)])
        content = format_script_content(
            entry, source_content=source, include_source=True, max_source_lines=50
        )

        assert "line_0" in content
        assert "line_49" in content
        assert "line_50" not in content
        assert "(100 more lines)" in content

    def test_format_uses_rel_path_as_title_fallback(self):
        """Uses rel_path as title when title is None."""
        entry = PineScriptEntry(
            rel_path="scripts/my_indicator.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title=None,
        )

        content = format_script_content(entry, include_source=False)

        assert "# scripts/my_indicator.pine" in content


class TestExtractSymbolsFromScript:
    """Tests for extract_symbols_from_script."""

    def test_extract_ticker_strings(self):
        """Extracts ticker symbols from quoted strings."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
//@version=5
indicator("Test")
spy = request.security("SPY", "D", close)
aapl = request.security("AAPL", "D", close)
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "SPY" in symbols
        assert "AAPL" in symbols

    def test_extract_index_symbols(self):
        """Extracts common index symbols."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
//@version=5
indicator("Market Dashboard")
// Compare SPX, QQQ, and VIX
plot(close)
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "SPX" in symbols
        assert "QQQ" in symbols
        assert "VIX" in symbols

    def test_extract_crypto_pairs(self):
        """Extracts cryptocurrency pairs."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
btc = request.security("BTCUSD", "D", close)
eth = request.security("ETHUSDT", "D", close)
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "BTCUSD" in symbols
        assert "ETHUSDT" in symbols

    def test_filters_common_non_tickers(self):
        """Filters out common non-ticker strings."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
if condition == "TRUE"
    label.set_text(lbl, "NA")
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "TRUE" not in symbols
        assert "NA" not in symbols

    def test_limits_symbol_count(self):
        """Limits extracted symbols to 10."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        # Create content with many tickers
        tickers = " ".join([f'"TICK{i}"' for i in range(20)])
        content = f"//@version=5\n{tickers}"

        symbols = extract_symbols_from_script(entry, content)

        assert len(symbols) <= 10


class TestPineIngestResultProperties:
    """Tests for PineIngestResult properties."""

    def test_empty_result(self):
        """Empty result has default values."""
        result = PineIngestResult()

        assert result.scripts_processed == 0
        assert result.scripts_indexed == 0
        assert result.scripts_skipped == 0
        assert result.scripts_failed == 0
        assert result.total_chunks == 0
        assert result.success is True  # No failures = success

    def test_success_false_when_failures(self):
        """success is False when there are failures."""
        result = PineIngestResult(
            scripts_processed=3,
            scripts_indexed=2,
            scripts_failed=1,
        )

        assert result.success is False

    def test_success_true_when_no_failures(self):
        """success is True when no failures."""
        result = PineIngestResult(
            scripts_processed=3,
            scripts_indexed=2,
            scripts_skipped=1,
            scripts_failed=0,
        )

        assert result.success is True


class TestScriptIngestResult:
    """Tests for ScriptIngestResult."""

    def test_default_values(self):
        """Default values are set correctly."""
        result = ScriptIngestResult(rel_path="test.pine", success=True)

        assert result.rel_path == "test.pine"
        assert result.success is True
        assert result.doc_id is None
        assert result.chunks_created == 0
        assert result.status == "pending"
        assert result.error is None

    def test_failed_result(self):
        """Failed result captures error."""
        result = ScriptIngestResult(
            rel_path="bad.pine",
            success=False,
            status="failed",
            error="Parse error",
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error == "Parse error"
