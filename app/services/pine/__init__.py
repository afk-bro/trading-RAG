"""
Pine Script Registry module.

Provides parsing, linting, and registry generation for Pine Script files.

Public API:
- Models: PineRegistry, PineScriptEntry, PineInput, PineLintReport, etc.
- Constants: Schema versions, lint codes, feature flags
- Adapters: SourceFile for filesystem/GitHub adapters

Example usage:
    from app.services.pine import (
        PineRegistry,
        PineScriptEntry,
        SourceFile,
        REGISTRY_SCHEMA_VERSION,
    )
"""

from app.services.pine.constants import (
    # Schema versions
    LINT_REPORT_SCHEMA_VERSION,
    LINTER_VERSION,
    PARSER_VERSION,
    REGISTRY_SCHEMA_VERSION,
    # File handling
    PINE_EXTENSIONS,
    # Feature flags
    FEATURE_IS_LIBRARY,
    FEATURE_USES_ALERT,
    FEATURE_USES_ARRAYS,
    FEATURE_USES_LOOKAHEAD_ON,
    FEATURE_USES_LOG,
    FEATURE_USES_MAPS,
    FEATURE_USES_MATRICES,
    FEATURE_USES_METHODS,
    FEATURE_USES_REQUEST_SECURITY,
    FEATURE_USES_SECURITY,
    FEATURE_USES_STRATEGY_FUNCTIONS,
    FEATURE_USES_UDT,
    FEATURE_USES_VARIP,
    KNOWN_FEATURES,
    # Lint codes
    LINT_E001_NO_VERSION,
    LINT_E002_INVALID_VERSION,
    LINT_E003_NO_DECLARATION,
    LINT_I001_CONSIDER_LIBRARY,
    LINT_I002_LARGE_SCRIPT,
    LINT_W001_UNUSED_VAR,
    LINT_W002_LOOKAHEAD_ON,
    LINT_W003_DEPRECATED_SECURITY,
    LINT_W004_REPAINTING_RISK,
)
from app.services.pine.models import (
    # Enums
    InputType,
    LintSeverity,
    PineVersion,
    ScriptType,
    # Adapter output
    SourceFile,
    # Core models
    LintFinding,
    LintSummary,
    PineImport,
    PineInput,
    PineLintReport,
    PineRegistry,
    PineScriptEntry,
    ScriptLintResult,
    # Helpers
    utc_now,
)
from app.services.pine.parser import (
    ParseResult,
    parse_pine,
    parse_to_entry,
    # Lower-level parsing functions
    parse_version,
    parse_declaration,
    parse_imports,
    parse_inputs,
    detect_features,
    # Normalization helpers
    normalize_value,
    normalize_list,
)
from app.services.pine.linter import (
    LinterConfig,
    LintResult,
    lint_pine,
    lint_content,
)
from app.services.pine.registry import (
    RegistryConfig,
    RegistryBuildResult,
    build_registry,
    build_and_write_registry,
    load_registry,
    load_lint_report,
)
from app.services.pine.ingest import (
    PineIngestService,
    PineIngestResult,
    ScriptIngestResult,
    ingest_pine_registry,
    format_script_content,
)

__all__ = [
    # Enums
    "PineVersion",
    "ScriptType",
    "InputType",
    "LintSeverity",
    # Adapter
    "SourceFile",
    # Models
    "PineImport",
    "PineInput",
    "LintFinding",
    "LintSummary",
    "PineScriptEntry",
    "PineRegistry",
    "ScriptLintResult",
    "PineLintReport",
    # Helpers
    "utc_now",
    # Constants - schema versions
    "REGISTRY_SCHEMA_VERSION",
    "LINT_REPORT_SCHEMA_VERSION",
    "PARSER_VERSION",
    "LINTER_VERSION",
    # Constants - file handling
    "PINE_EXTENSIONS",
    # Constants - feature flags
    "FEATURE_USES_REQUEST_SECURITY",
    "FEATURE_USES_SECURITY",
    "FEATURE_USES_LOOKAHEAD_ON",
    "FEATURE_USES_VARIP",
    "FEATURE_USES_ARRAYS",
    "FEATURE_USES_MATRICES",
    "FEATURE_USES_MAPS",
    "FEATURE_USES_METHODS",
    "FEATURE_USES_UDT",
    "FEATURE_USES_STRATEGY_FUNCTIONS",
    "FEATURE_USES_ALERT",
    "FEATURE_USES_LOG",
    "FEATURE_IS_LIBRARY",
    "KNOWN_FEATURES",
    # Constants - lint codes
    "LINT_E001_NO_VERSION",
    "LINT_E002_INVALID_VERSION",
    "LINT_E003_NO_DECLARATION",
    "LINT_W001_UNUSED_VAR",
    "LINT_W002_LOOKAHEAD_ON",
    "LINT_W003_DEPRECATED_SECURITY",
    "LINT_W004_REPAINTING_RISK",
    "LINT_I001_CONSIDER_LIBRARY",
    "LINT_I002_LARGE_SCRIPT",
    # Parser
    "ParseResult",
    "parse_pine",
    "parse_to_entry",
    "parse_version",
    "parse_declaration",
    "parse_imports",
    "parse_inputs",
    "detect_features",
    "normalize_value",
    "normalize_list",
    # Linter
    "LinterConfig",
    "LintResult",
    "lint_pine",
    "lint_content",
    # Registry
    "RegistryConfig",
    "RegistryBuildResult",
    "build_registry",
    "build_and_write_registry",
    "load_registry",
    "load_lint_report",
    # Ingest
    "PineIngestService",
    "PineIngestResult",
    "ScriptIngestResult",
    "ingest_pine_registry",
    "format_script_content",
]
