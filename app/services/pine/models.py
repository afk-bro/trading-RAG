"""
Pine Script Registry models.

Design decisions:
- Dataclasses (not Pydantic) - explicit serialization, frozen for immutables
- Timestamps always UTC with +00:00 (fromisoformat compatible)
- Defaults normalized: primitives in `default`, expressions in `default_expr`
- Lint findings in separate report; registry has summary only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional


# =============================================================================
# Enums
# =============================================================================


class PineVersion(str, Enum):
    """Pine Script version (numeric string for comparison)."""

    V4 = "4"
    V5 = "5"
    V6 = "6"
    UNKNOWN = "unknown"


class ScriptType(str, Enum):
    """Pine Script declaration type."""

    INDICATOR = "indicator"
    STRATEGY = "strategy"
    LIBRARY = "library"
    UNKNOWN = "unknown"


class InputType(str, Enum):
    """Pine input.* types."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    COLOR = "color"
    SOURCE = "source"
    TIMEFRAME = "timeframe"
    SESSION = "session"
    SYMBOL = "symbol"
    UNKNOWN = "unknown"


class LintSeverity(str, Enum):
    """Lint finding severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# Filesystem Adapter Output (frozen - immutable source data)
# =============================================================================


@dataclass(frozen=True)
class SourceFile:
    """
    Adapter output for a single file.

    GitHub parity: rel_path is string, source_id for blob SHA.
    abs_path is filesystem-only and never serialized to JSON.
    """

    rel_path: str  # "strategies/breakout.pine"
    content: str
    abs_path: Optional[str] = None  # filesystem only, never in JSON
    source_id: Optional[str] = None  # GitHub: blob SHA or commit:path
    mtime: Optional[datetime] = None  # UTC, for staleness detection


# =============================================================================
# Import Reference (frozen)
# =============================================================================


@dataclass(frozen=True)
class PineImport:
    """Import statement reference."""

    path: str  # "lib/utils.pine" or "TradingView/ta"
    alias: str  # "utils" or "ta"
    line: Optional[int] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"path": self.path, "alias": self.alias}
        if self.line is not None:
            d["line"] = self.line
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PineImport:
        return cls(
            path=data["path"],
            alias=data["alias"],
            line=data.get("line"),
        )


# =============================================================================
# Input Specification
# =============================================================================


@dataclass
class PineInput:
    """
    Pine Script input.* definition.

    Defaults normalized:
    - `default`: primitive value (int/float/bool/str/None)
    - `default_expr`: string for complex expressions (color.new, close, etc.)

    Note: Normalization logic lives in parser.py, not here.
    """

    name: str
    type: InputType

    # Normalized default
    default: Optional[int | float | bool | str] = None
    default_expr: Optional[str] = None  # "color.new(#FF0000, 50)", "close"

    # UI grouping
    group: Optional[str] = None
    tooltip: Optional[str] = None
    inline: Optional[str] = None

    # Bounds (numeric types)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

    # Choices (normalized: primitives or stringified)
    options: Optional[list[Any]] = None

    # Source position
    line: Optional[int] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type.value,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.default_expr is not None:
            d["default_expr"] = self.default_expr
        if self.group is not None:
            d["group"] = self.group
        if self.tooltip is not None:
            d["tooltip"] = self.tooltip
        if self.inline is not None:
            d["inline"] = self.inline
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.step is not None:
            d["step"] = self.step
        if self.options is not None:
            d["options"] = self.options
        if self.line is not None:
            d["line"] = self.line
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PineInput:
        return cls(
            name=data["name"],
            type=InputType(data.get("type", "unknown")),
            default=data.get("default"),
            default_expr=data.get("default_expr"),
            group=data.get("group"),
            tooltip=data.get("tooltip"),
            inline=data.get("inline"),
            min_value=data.get("min"),
            max_value=data.get("max"),
            step=data.get("step"),
            options=data.get("options"),
            line=data.get("line"),
        )


# =============================================================================
# Lint Finding (frozen - immutable diagnostic)
# =============================================================================


@dataclass(frozen=True)
class LintFinding:
    """Single lint finding."""

    severity: LintSeverity
    code: str  # "W001", "E100"
    message: str
    line: Optional[int] = None
    column: Optional[int] = None

    def to_dict(self, max_message_len: int = 500) -> dict:
        """Convert to dict with optional message truncation."""
        msg = self.message
        if len(msg) > max_message_len:
            msg = msg[: max_message_len - 3] + "..."

        d: dict[str, Any] = {
            "severity": self.severity.value,
            "code": self.code,
            "message": msg,
        }
        if self.line is not None:
            d["line"] = self.line
        if self.column is not None:
            d["column"] = self.column
        return d

    @classmethod
    def from_dict(cls, data: dict) -> LintFinding:
        return cls(
            severity=LintSeverity(data["severity"]),
            code=data["code"],
            message=data["message"],
            line=data.get("line"),
            column=data.get("column"),
        )


# =============================================================================
# Lint Summary (for registry embedding)
# =============================================================================


@dataclass
class LintSummary:
    """Lint summary for a single script (embedded in registry)."""

    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        return self.warning_count > 0

    def to_dict(self) -> dict:
        return {
            "errors": self.error_count,
            "warnings": self.warning_count,
            "info": self.info_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LintSummary:
        return cls(
            error_count=data.get("errors", 0),
            warning_count=data.get("warnings", 0),
            info_count=data.get("info", 0),
        )

    @classmethod
    def from_findings(cls, findings: list[LintFinding]) -> LintSummary:
        return cls(
            error_count=sum(1 for f in findings if f.severity == LintSeverity.ERROR),
            warning_count=sum(
                1 for f in findings if f.severity == LintSeverity.WARNING
            ),
            info_count=sum(1 for f in findings if f.severity == LintSeverity.INFO),
        )


# =============================================================================
# Script Entry (per-file in registry)
# =============================================================================


@dataclass
class PineScriptEntry:
    """
    Complete metadata for a single Pine Script file.

    Lint findings are NOT embedded here - only summary.
    Full findings go in pine_lint_report.json.
    """

    # Identity & source tracking
    rel_path: str
    sha256: str
    source_id: Optional[str] = None

    # Script metadata
    pine_version: PineVersion = PineVersion.UNKNOWN
    script_type: ScriptType = ScriptType.UNKNOWN
    title: Optional[str] = None
    short_title: Optional[str] = None
    overlay: Optional[bool] = None

    # Imports
    imports: list[PineImport] = field(default_factory=list)

    # Inputs
    inputs: list[PineInput] = field(default_factory=list)

    # Features (cheap signals, queryable without parsing findings)
    features: dict[str, bool] = field(default_factory=dict)
    # e.g. {"uses_request_security": True, "uses_lookahead_on": False}

    # Lint summary only (full findings in lint report)
    lint: LintSummary = field(default_factory=LintSummary)

    # Timestamps (UTC)
    parsed_at: Optional[datetime] = None
    source_mtime: Optional[datetime] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "rel_path": self.rel_path,
            "sha256": self.sha256,
        }
        if self.source_id is not None:
            d["source_id"] = self.source_id

        d["pine_version"] = self.pine_version.value
        d["script_type"] = self.script_type.value

        if self.title is not None:
            d["title"] = self.title
        if self.short_title is not None:
            d["short_title"] = self.short_title
        if self.overlay is not None:
            d["overlay"] = self.overlay

        if self.imports:
            d["imports"] = [imp.to_dict() for imp in self.imports]
        if self.inputs:
            d["inputs"] = [inp.to_dict() for inp in self.inputs]
        if self.features:
            d["features"] = self.features

        d["lint"] = self.lint.to_dict()

        if self.parsed_at is not None:
            d["parsed_at"] = self.parsed_at.isoformat()
        if self.source_mtime is not None:
            d["source_mtime"] = self.source_mtime.isoformat()

        return d

    @classmethod
    def from_dict(cls, data: dict) -> PineScriptEntry:
        return cls(
            rel_path=data["rel_path"],
            sha256=data["sha256"],
            source_id=data.get("source_id"),
            pine_version=PineVersion(data.get("pine_version", "unknown")),
            script_type=ScriptType(data.get("script_type", "unknown")),
            title=data.get("title"),
            short_title=data.get("short_title"),
            overlay=data.get("overlay"),
            imports=[PineImport.from_dict(i) for i in data.get("imports", [])],
            inputs=[PineInput.from_dict(i) for i in data.get("inputs", [])],
            features=data.get("features", {}),
            lint=LintSummary.from_dict(data.get("lint", {})),
            parsed_at=_parse_iso(data.get("parsed_at")),
            source_mtime=_parse_iso(data.get("source_mtime")),
        )


# =============================================================================
# Registry (top-level artifact)
# =============================================================================


@dataclass
class PineRegistry:
    """
    Top-level registry - the pine_registry.json root.

    Contains script metadata with lint summaries.
    Full lint findings are in separate pine_lint_report.json.
    """

    # Header
    schema_version: str = "pine_registry_v1"
    generated_at: Optional[datetime] = None
    root: Optional[str] = None
    root_kind: Literal["filesystem", "github", "unknown"] = "unknown"

    # Tool versions
    parser_version: Optional[str] = None
    linter_version: Optional[str] = None

    # Scripts by rel_path
    scripts: dict[str, PineScriptEntry] = field(default_factory=dict)

    @property
    def script_count(self) -> int:
        return len(self.scripts)

    @property
    def total_errors(self) -> int:
        return sum(s.lint.error_count for s in self.scripts.values())

    @property
    def total_warnings(self) -> int:
        return sum(s.lint.warning_count for s in self.scripts.values())

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
        }
        if self.generated_at is not None:
            d["generated_at"] = self.generated_at.isoformat()
        if self.root is not None:
            d["root"] = self.root
        d["root_kind"] = self.root_kind
        if self.parser_version is not None:
            d["parser_version"] = self.parser_version
        if self.linter_version is not None:
            d["linter_version"] = self.linter_version

        d["scripts"] = {
            path: entry.to_dict() for path, entry in sorted(self.scripts.items())
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PineRegistry:
        return cls(
            schema_version=data.get("schema_version", "pine_registry_v1"),
            generated_at=_parse_iso(data.get("generated_at")),
            root=data.get("root"),
            root_kind=data.get("root_kind", "unknown"),
            parser_version=data.get("parser_version"),
            linter_version=data.get("linter_version"),
            scripts={
                path: PineScriptEntry.from_dict(entry)
                for path, entry in data.get("scripts", {}).items()
            },
        )


# =============================================================================
# Lint Report (separate artifact)
# =============================================================================


@dataclass
class ScriptLintResult:
    """Lint results for a single script (in lint report)."""

    rel_path: str
    sha256: str
    findings: list[LintFinding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rel_path": self.rel_path,
            "sha256": self.sha256,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScriptLintResult:
        return cls(
            rel_path=data["rel_path"],
            sha256=data["sha256"],
            findings=[LintFinding.from_dict(f) for f in data.get("findings", [])],
        )


@dataclass
class PineLintReport:
    """
    Lint report artifact - pine_lint_report.json.

    Contains full findings per script.
    Registry has summaries only.
    """

    schema_version: str = "pine_lint_v1"
    generated_at: Optional[datetime] = None
    linter_version: Optional[str] = None

    # Results by rel_path
    results: dict[str, ScriptLintResult] = field(default_factory=dict)

    @property
    def total_errors(self) -> int:
        return sum(
            1
            for r in self.results.values()
            for f in r.findings
            if f.severity == LintSeverity.ERROR
        )

    @property
    def total_warnings(self) -> int:
        return sum(
            1
            for r in self.results.values()
            for f in r.findings
            if f.severity == LintSeverity.WARNING
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
        }
        if self.generated_at is not None:
            d["generated_at"] = self.generated_at.isoformat()
        if self.linter_version is not None:
            d["linter_version"] = self.linter_version

        d["results"] = {
            path: result.to_dict() for path, result in sorted(self.results.items())
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PineLintReport:
        return cls(
            schema_version=data.get("schema_version", "pine_lint_v1"),
            generated_at=_parse_iso(data.get("generated_at")),
            linter_version=data.get("linter_version"),
            results={
                path: ScriptLintResult.from_dict(r)
                for path, r in data.get("results", {}).items()
            },
        )


# =============================================================================
# Helpers
# =============================================================================


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp, handling None, Z suffix, and naive datetimes."""
    if value is None:
        return None
    # Handle Z suffix (fromisoformat doesn't accept it)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    # Coerce naive → UTC (defensive, shouldn't happen if we control writers)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)
