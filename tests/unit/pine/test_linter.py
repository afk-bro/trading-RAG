"""
Unit tests for Pine Script linter.

Tests lint rules for Pine Script static analysis.
"""

from app.services.pine.constants import (
    LINT_E001_NO_VERSION,
    LINT_E002_INVALID_VERSION,
    LINT_E003_NO_DECLARATION,
    LINT_I001_CONSIDER_LIBRARY,
    LINT_I002_LARGE_SCRIPT,
    LINT_W002_LOOKAHEAD_ON,
    LINT_W003_DEPRECATED_SECURITY,
)
from app.services.pine.linter import (
    LinterConfig,
    LintResult,
    check_declaration,
    check_deprecated_security,
    check_large_script,
    check_library_candidate,
    check_lookahead_on,
    check_version_directive,
    lint_content,
    lint_pine,
)
from app.services.pine.models import (
    LintSeverity,
    PineVersion,
    ScriptType,
    SourceFile,
)
from app.services.pine.parser import ParseResult, parse_pine


class TestCheckVersionDirective:
    """Tests for E001/E002 version checks."""

    def test_e001_missing_version(self):
        """E001: Missing version directive."""
        content = "indicator('Test')\nplot(close)"
        parse_result = ParseResult(pine_version=PineVersion.UNKNOWN)

        findings = check_version_directive(parse_result, content)

        assert len(findings) == 1
        assert findings[0].code == LINT_E001_NO_VERSION
        assert findings[0].severity == LintSeverity.ERROR
        assert findings[0].line == 1

    def test_e002_invalid_version(self):
        """E002: Invalid version number."""
        content = "//@version=99\nindicator('Test')"
        parse_result = ParseResult(pine_version=PineVersion.UNKNOWN)

        findings = check_version_directive(parse_result, content)

        assert len(findings) == 1
        assert findings[0].code == LINT_E002_INVALID_VERSION
        assert findings[0].severity == LintSeverity.ERROR
        assert "99" in findings[0].message

    def test_e002_invalid_version_line_number(self):
        """E002: Reports correct line number for invalid version."""
        content = "// comment\n// another\n//@version=abc\nindicator('Test')"
        parse_result = ParseResult(pine_version=PineVersion.UNKNOWN)

        findings = check_version_directive(parse_result, content)

        assert len(findings) == 1
        assert findings[0].line == 3

    def test_valid_version_no_findings(self):
        """Valid version produces no findings."""
        content = "//@version=5\nindicator('Test')"
        parse_result = ParseResult(pine_version=PineVersion.V5, version_line=1)

        findings = check_version_directive(parse_result, content)

        assert len(findings) == 0


class TestCheckDeclaration:
    """Tests for E003 declaration check."""

    def test_e003_missing_declaration(self):
        """E003: Missing declaration."""
        content = "//@version=5\nplot(close)"
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.UNKNOWN,
        )

        findings = check_declaration(parse_result, content)

        assert len(findings) == 1
        assert findings[0].code == LINT_E003_NO_DECLARATION
        assert findings[0].severity == LintSeverity.ERROR

    def test_valid_declaration_no_findings(self):
        """Valid declaration produces no findings."""
        content = "//@version=5\nindicator('Test')"
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        findings = check_declaration(parse_result, content)

        assert len(findings) == 0


class TestCheckLookaheadOn:
    """Tests for W002 lookahead check."""

    def test_w002_lookahead_on_detected(self):
        """W002: Detects lookahead_on usage."""
        content = """//@version=5
indicator('Test')
htf = request.security(syminfo.tickerid, "D", close, lookahead=barmerge.lookahead_on)
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            features={"uses_lookahead_on": True},
        )

        findings = check_lookahead_on(parse_result, content)

        assert len(findings) == 1
        assert findings[0].code == LINT_W002_LOOKAHEAD_ON
        assert findings[0].severity == LintSeverity.WARNING
        assert findings[0].line == 3

    def test_w002_multiple_occurrences(self):
        """W002: Reports multiple lookahead_on occurrences."""
        content = """//@version=5
indicator('Test')
a = request.security(s, "D", close, lookahead=barmerge.lookahead_on)
b = request.security(s, "W", close, lookahead=barmerge.lookahead_on)
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            features={"uses_lookahead_on": True},
        )

        findings = check_lookahead_on(parse_result, content)

        assert len(findings) == 2
        assert findings[0].line == 3
        assert findings[1].line == 4

    def test_lookahead_off_no_findings(self):
        """lookahead_off produces no findings."""
        content = """//@version=5
indicator('Test')
htf = request.security(syminfo.tickerid, "D", close, lookahead=barmerge.lookahead_off)
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            features={"uses_lookahead_on": False},
        )

        findings = check_lookahead_on(parse_result, content)

        assert len(findings) == 0


class TestCheckDeprecatedSecurity:
    """Tests for W003 deprecated security check."""

    def test_w003_deprecated_security(self):
        """W003: Detects deprecated security() function."""
        content = """//@version=5
indicator('Test')
htf = security(syminfo.tickerid, "D", close)
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            features={"uses_security": True},
        )

        findings = check_deprecated_security(parse_result, content)

        assert len(findings) == 1
        assert findings[0].code == LINT_W003_DEPRECATED_SECURITY
        assert findings[0].severity == LintSeverity.WARNING
        assert findings[0].line == 3

    def test_request_security_no_findings(self):
        """request.security() produces no findings."""
        content = """//@version=5
indicator('Test')
htf = request.security(syminfo.tickerid, "D", close)
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            features={"uses_security": False, "uses_request_security": True},
        )

        findings = check_deprecated_security(parse_result, content)

        assert len(findings) == 0


class TestCheckLargeScript:
    """Tests for I002 large script check."""

    def test_i002_large_script(self):
        """I002: Detects scripts exceeding line limit."""
        # Create a script with 600 lines
        lines = ["//@version=5", "indicator('Test')"] + ["plot(close)"] * 598
        content = "\n".join(lines)

        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        findings = check_large_script(parse_result, content, max_lines=500)

        assert len(findings) == 1
        assert findings[0].code == LINT_I002_LARGE_SCRIPT
        assert findings[0].severity == LintSeverity.INFO
        assert "600" in findings[0].message

    def test_small_script_no_findings(self):
        """Small script produces no findings."""
        content = "//@version=5\nindicator('Test')\nplot(close)"
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        findings = check_large_script(parse_result, content, max_lines=500)

        assert len(findings) == 0


class TestCheckLibraryCandidate:
    """Tests for I001 library candidate check."""

    def test_i001_export_in_indicator(self):
        """I001: Detects export in non-library script."""
        content = """//@version=5
indicator('Utils')
export myFunc() => 42
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        findings = check_library_candidate(parse_result, content)

        assert len(findings) == 1
        assert findings[0].code == LINT_I001_CONSIDER_LIBRARY
        assert findings[0].severity == LintSeverity.INFO

    def test_library_no_findings(self):
        """Library script produces no findings."""
        content = """//@version=5
library('Utils')
export myFunc() => 42
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.LIBRARY,
        )

        findings = check_library_candidate(parse_result, content)

        assert len(findings) == 0

    def test_no_exports_no_findings(self):
        """Script without exports produces no findings."""
        content = """//@version=5
indicator('Test')
plot(close)
"""
        parse_result = ParseResult(
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        findings = check_library_candidate(parse_result, content)

        assert len(findings) == 0


class TestLintPine:
    """Tests for lint_pine() main function."""

    def test_lint_complete_script(self):
        """Lints a complete valid script."""
        content = """//@version=5
indicator("Test", overlay=true)
length = input.int(14, title="Length")
plot(ta.sma(close, length))
"""
        source = SourceFile(rel_path="test.pine", content=content)
        parse_result = parse_pine(source)

        result = lint_pine(parse_result, content)

        assert isinstance(result, LintResult)
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_lint_script_with_errors(self):
        """Lints script with multiple issues."""
        content = """// no version
// no declaration
htf = security(sym, "D", close, lookahead=barmerge.lookahead_on)
"""
        source = SourceFile(rel_path="test.pine", content=content)
        parse_result = parse_pine(source)

        result = lint_pine(parse_result, content)

        assert result.error_count >= 2  # E001, E003
        # Note: W003 not triggered for v4 scripts (unknown version)

    def test_lint_findings_sorted_by_severity(self):
        """Findings are sorted by severity (errors first)."""
        content = """//@version=5
indicator('Test')
htf = security(sym, "D", close)
"""
        source = SourceFile(rel_path="test.pine", content=content)
        parse_result = parse_pine(source)

        result = lint_pine(parse_result, content)

        if len(result.findings) > 1:
            severities = [f.severity for f in result.findings]
            # Check errors come before warnings
            error_indices = [
                i for i, s in enumerate(severities) if s == LintSeverity.ERROR
            ]
            warning_indices = [
                i for i, s in enumerate(severities) if s == LintSeverity.WARNING
            ]
            if error_indices and warning_indices:
                assert max(error_indices) < min(warning_indices)

    def test_lint_with_config(self):
        """Linter respects configuration."""
        content = "plot(close)"  # Missing version and declaration

        source = SourceFile(rel_path="test.pine", content=content)
        parse_result = parse_pine(source)

        # Default config
        result1 = lint_pine(parse_result, content)
        assert result1.error_count >= 2

        # Disable version check
        config = LinterConfig(check_version=False)
        result2 = lint_pine(parse_result, content, config)
        assert result2.error_count < result1.error_count

    def test_lint_to_summary(self):
        """LintResult converts to LintSummary."""
        content = """//@version=5
indicator('Test')
htf = security(sym, "D", close)
"""
        source = SourceFile(rel_path="test.pine", content=content)
        parse_result = parse_pine(source)

        result = lint_pine(parse_result, content)
        summary = result.to_summary()

        assert summary.error_count == result.error_count
        assert summary.warning_count == result.warning_count
        assert summary.info_count == result.info_count


class TestLintContent:
    """Tests for lint_content() convenience function."""

    def test_lint_content_valid(self):
        """Lints valid content directly."""
        content = """//@version=5
indicator("Test")
plot(close)
"""
        result = lint_content(content)

        assert isinstance(result, LintResult)
        assert result.error_count == 0

    def test_lint_content_with_issues(self):
        """Lints content with issues."""
        content = "plot(close)"  # Missing everything

        result = lint_content(content)

        assert result.error_count >= 2  # E001, E003


class TestLintResultProperties:
    """Tests for LintResult properties."""

    def test_empty_result(self):
        """Empty result has zero counts."""
        result = LintResult()

        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.info_count == 0
        assert result.has_errors is False

    def test_has_errors(self):
        """has_errors returns True when errors present."""
        from app.services.pine.models import LintFinding, LintSeverity

        result = LintResult(
            findings=[
                LintFinding(
                    severity=LintSeverity.ERROR,
                    code="E001",
                    message="Test error",
                )
            ]
        )

        assert result.has_errors is True
        assert result.error_count == 1

    def test_mixed_severities(self):
        """Correctly counts mixed severities."""
        from app.services.pine.models import LintFinding, LintSeverity

        result = LintResult(
            findings=[
                LintFinding(severity=LintSeverity.ERROR, code="E001", message="e1"),
                LintFinding(severity=LintSeverity.ERROR, code="E002", message="e2"),
                LintFinding(severity=LintSeverity.WARNING, code="W001", message="w1"),
                LintFinding(severity=LintSeverity.INFO, code="I001", message="i1"),
                LintFinding(severity=LintSeverity.INFO, code="I002", message="i2"),
            ]
        )

        assert result.error_count == 2
        assert result.warning_count == 1
        assert result.info_count == 2
