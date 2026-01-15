"""
Pine Script linter.

Static analysis rules for Pine Script files.
Uses ParseResult from parser and content for pattern-based checks.

Lint Codes:
- E001: Missing //@version directive
- E002: Invalid/unrecognized version number
- E003: Missing indicator/strategy/library declaration
- W001: Unused variable (not implemented - requires deeper analysis)
- W002: lookahead=barmerge.lookahead_on detected
- W003: Deprecated security() instead of request.security()
- W004: Potential repainting pattern (not implemented - requires deeper analysis)
- I001: Could be extracted to library
- I002: Large script exceeds recommended size
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.services.pine.constants import (
    FEATURE_USES_LOOKAHEAD_ON,
    FEATURE_USES_SECURITY,
    LINT_E001_NO_VERSION,
    LINT_E002_INVALID_VERSION,
    LINT_E003_NO_DECLARATION,
    LINT_I001_CONSIDER_LIBRARY,
    LINT_I002_LARGE_SCRIPT,
    LINT_W002_LOOKAHEAD_ON,
    LINT_W003_DEPRECATED_SECURITY,
)
from app.services.pine.models import (
    LintFinding,
    LintSeverity,
    LintSummary,
    PineVersion,
    ScriptType,
)
from app.services.pine.parser import ParseResult


# =============================================================================
# Configuration
# =============================================================================

# Thresholds
DEFAULT_MAX_LINES = 500  # I002 threshold
DEFAULT_MAX_INPUTS = 50  # Informational, not a lint rule yet


# =============================================================================
# Lint Result
# =============================================================================


@dataclass
class LintResult:
    """Result of linting a Pine Script file."""

    findings: list[LintFinding] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == LintSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == LintSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == LintSeverity.INFO)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_summary(self) -> LintSummary:
        """Convert to LintSummary for registry embedding."""
        return LintSummary.from_findings(self.findings)


# =============================================================================
# Individual Lint Rules
# =============================================================================


def check_version_directive(
    parse_result: ParseResult,
    content: str,
) -> list[LintFinding]:
    """
    E001: Check for missing //@version directive.
    E002: Check for invalid/unrecognized version number.
    """
    findings = []

    if parse_result.pine_version == PineVersion.UNKNOWN:
        # Check if there's a version line at all (malformed vs missing)
        version_pattern = re.compile(r"//\s*@version\s*=\s*(\S+)", re.MULTILINE)
        match = version_pattern.search(content)

        if match:
            # Version line exists but value is invalid
            version_value = match.group(1)
            # Find line number
            line_num = content[: match.start()].count("\n") + 1
            findings.append(
                LintFinding(
                    severity=LintSeverity.ERROR,
                    code=LINT_E002_INVALID_VERSION,
                    message=f"Invalid Pine Script version: '{version_value}'. "
                    f"Expected 4, 5, or 6.",
                    line=line_num,
                )
            )
        else:
            # No version directive at all
            findings.append(
                LintFinding(
                    severity=LintSeverity.ERROR,
                    code=LINT_E001_NO_VERSION,
                    message="Missing //@version directive. "
                    "Add '//@version=5' at the start of your script.",
                    line=1,
                )
            )

    return findings


def check_declaration(
    parse_result: ParseResult,
    content: str,
) -> list[LintFinding]:
    """
    E003: Check for missing indicator/strategy/library declaration.
    """
    findings = []

    if parse_result.script_type == ScriptType.UNKNOWN:
        findings.append(
            LintFinding(
                severity=LintSeverity.ERROR,
                code=LINT_E003_NO_DECLARATION,
                message="Missing script declaration. "
                "Add indicator(), strategy(), or library() call.",
                line=None,
            )
        )

    return findings


def check_lookahead_on(
    parse_result: ParseResult,
    content: str,
) -> list[LintFinding]:
    """
    W002: Check for lookahead=barmerge.lookahead_on usage.

    This can cause future data leakage in backtests.
    """
    findings = []

    if parse_result.features.get(FEATURE_USES_LOOKAHEAD_ON, False):
        # Find all occurrences
        pattern = re.compile(r"lookahead\s*=\s*barmerge\.lookahead_on")
        for match in pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            findings.append(
                LintFinding(
                    severity=LintSeverity.WARNING,
                    code=LINT_W002_LOOKAHEAD_ON,
                    message="lookahead=barmerge.lookahead_on can cause future data leakage. "
                    "Consider using barmerge.lookahead_off unless intentional.",
                    line=line_num,
                )
            )

    return findings


def check_deprecated_security(
    parse_result: ParseResult,
    content: str,
) -> list[LintFinding]:
    """
    W003: Check for deprecated security() function.

    Should use request.security() in v5+.
    """
    findings = []

    # Only warn if using deprecated security() AND not request.security()
    uses_deprecated = parse_result.features.get(FEATURE_USES_SECURITY, False)

    if uses_deprecated:
        # Find security( but not request.security(
        # Negative lookbehind for request.
        pattern = re.compile(r"(?<!request\.)\bsecurity\s*\(")
        for match in pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            findings.append(
                LintFinding(
                    severity=LintSeverity.WARNING,
                    code=LINT_W003_DEPRECATED_SECURITY,
                    message="security() is deprecated. Use request.security() instead.",
                    line=line_num,
                )
            )

    return findings


def check_large_script(
    parse_result: ParseResult,
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[LintFinding]:
    """
    I002: Check if script exceeds recommended size.
    """
    findings = []

    line_count = content.count("\n") + 1
    if line_count > max_lines:
        findings.append(
            LintFinding(
                severity=LintSeverity.INFO,
                code=LINT_I002_LARGE_SCRIPT,
                message=f"Script has {line_count} lines (recommended max: {max_lines}). "
                "Consider splitting into multiple files or extracting to a library.",
                line=None,
            )
        )

    return findings


def check_library_candidate(
    parse_result: ParseResult,
    content: str,
) -> list[LintFinding]:
    """
    I001: Check if script could be extracted to a library.

    Suggests library extraction if:
    - Script is an indicator/strategy (not already a library)
    - Has export statements
    """
    findings: list[LintFinding] = []

    # Skip if already a library
    if parse_result.script_type == ScriptType.LIBRARY:
        return findings

    # Check for export statements
    export_pattern = re.compile(r"^\s*export\s+", re.MULTILINE)
    if export_pattern.search(content):
        findings.append(
            LintFinding(
                severity=LintSeverity.INFO,
                code=LINT_I001_CONSIDER_LIBRARY,
                message="Script contains export statements but is not a library. "
                "Consider extracting reusable functions to a separate library.",
                line=None,
            )
        )

    return findings


# =============================================================================
# Main Linter
# =============================================================================


@dataclass
class LinterConfig:
    """Configuration for the linter."""

    # Enable/disable specific rules
    check_version: bool = True
    check_declaration: bool = True
    check_lookahead: bool = True
    check_deprecated_security: bool = True
    check_large_script: bool = True
    check_library_candidate: bool = True

    # Thresholds
    max_lines: int = DEFAULT_MAX_LINES

    # Minimum version for certain checks
    min_version_for_request_security: int = 5


def lint_pine(
    parse_result: ParseResult,
    content: str,
    config: Optional[LinterConfig] = None,
) -> LintResult:
    """
    Run all lint checks on a Pine Script.

    Args:
        parse_result: ParseResult from parser
        content: Raw script content
        config: Optional linter configuration

    Returns:
        LintResult with all findings
    """
    if config is None:
        config = LinterConfig()

    findings: list[LintFinding] = []

    # E001/E002: Version checks
    if config.check_version:
        findings.extend(check_version_directive(parse_result, content))

    # E003: Declaration check
    if config.check_declaration:
        findings.extend(check_declaration(parse_result, content))

    # W002: Lookahead check
    if config.check_lookahead:
        findings.extend(check_lookahead_on(parse_result, content))

    # W003: Deprecated security check (only for v5+)
    if config.check_deprecated_security:
        version_num = _version_to_int(parse_result.pine_version)
        if version_num >= config.min_version_for_request_security:
            findings.extend(check_deprecated_security(parse_result, content))

    # I002: Large script check
    if config.check_large_script:
        findings.extend(check_large_script(parse_result, content, config.max_lines))

    # I001: Library candidate check
    if config.check_library_candidate:
        findings.extend(check_library_candidate(parse_result, content))

    # Sort findings by severity (errors first), then by line number
    findings.sort(key=lambda f: (_severity_order(f.severity), f.line or 0))

    return LintResult(findings=findings)


def _version_to_int(version: PineVersion) -> int:
    """Convert PineVersion to integer for comparison."""
    version_map = {
        PineVersion.V4: 4,
        PineVersion.V5: 5,
        PineVersion.V6: 6,
        PineVersion.UNKNOWN: 0,
    }
    return version_map.get(version, 0)


def _severity_order(severity: LintSeverity) -> int:
    """Get sort order for severity (errors first)."""
    order = {
        LintSeverity.ERROR: 0,
        LintSeverity.WARNING: 1,
        LintSeverity.INFO: 2,
    }
    return order.get(severity, 99)


# =============================================================================
# Convenience Functions
# =============================================================================


def lint_content(content: str, config: Optional[LinterConfig] = None) -> LintResult:
    """
    Lint Pine Script content directly.

    Convenience function that parses and lints in one call.
    """
    from app.services.pine.models import SourceFile
    from app.services.pine.parser import parse_pine

    source = SourceFile(rel_path="<inline>", content=content)
    parse_result = parse_pine(source)
    return lint_pine(parse_result, content, config)
