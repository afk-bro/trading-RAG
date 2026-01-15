"""
Pine Script Registry constants.

This module imports nothing - it's the dependency root.
Other modules import from here; this never imports from them.
"""

# =============================================================================
# Schema Versions
# =============================================================================

# Bump these when making breaking changes to JSON structure
REGISTRY_SCHEMA_VERSION = "pine_registry_v1"
LINT_REPORT_SCHEMA_VERSION = "pine_lint_v1"

# =============================================================================
# Tool Versions
# =============================================================================

# Parser version - bump when parsing logic changes
PARSER_VERSION = "0.1.0"

# Linter version - bump when lint rules change
LINTER_VERSION = "0.1.0"

# =============================================================================
# File Extensions
# =============================================================================

PINE_EXTENSIONS = (".pine", ".pinescript")

# =============================================================================
# Feature Flags (for features dict in PineScriptEntry)
# =============================================================================

# These are the canonical feature keys detected by the parser/linter
FEATURE_USES_REQUEST_SECURITY = "uses_request_security"
FEATURE_USES_SECURITY = "uses_security"  # deprecated alias
FEATURE_USES_LOOKAHEAD_ON = "uses_lookahead_on"
FEATURE_USES_BARMERGE_GAPS = "uses_barmerge_gaps"
FEATURE_USES_VARIP = "uses_varip"
FEATURE_USES_ARRAYS = "uses_arrays"
FEATURE_USES_MATRICES = "uses_matrices"
FEATURE_USES_MAPS = "uses_maps"
FEATURE_USES_METHODS = "uses_methods"
FEATURE_USES_UDT = "uses_udt"  # user-defined types
FEATURE_USES_STRATEGY_FUNCTIONS = "uses_strategy_functions"
FEATURE_USES_ALERT = "uses_alert"
FEATURE_USES_LOG = "uses_log"
FEATURE_IS_LIBRARY = "is_library"

# All known feature keys (for validation)
KNOWN_FEATURES = frozenset(
    [
        FEATURE_USES_REQUEST_SECURITY,
        FEATURE_USES_SECURITY,
        FEATURE_USES_LOOKAHEAD_ON,
        FEATURE_USES_BARMERGE_GAPS,
        FEATURE_USES_VARIP,
        FEATURE_USES_ARRAYS,
        FEATURE_USES_MATRICES,
        FEATURE_USES_MAPS,
        FEATURE_USES_METHODS,
        FEATURE_USES_UDT,
        FEATURE_USES_STRATEGY_FUNCTIONS,
        FEATURE_USES_ALERT,
        FEATURE_USES_LOG,
        FEATURE_IS_LIBRARY,
    ]
)

# =============================================================================
# Lint Codes
# =============================================================================

# Error codes (E-prefix) - likely bugs or invalid code
LINT_E001_NO_VERSION = "E001"  # Missing //@version directive
LINT_E002_INVALID_VERSION = "E002"  # Unrecognized version number
LINT_E003_NO_DECLARATION = "E003"  # Missing indicator/strategy/library

# Warning codes (W-prefix) - potential issues
LINT_W001_UNUSED_VAR = "W001"  # Unused variable
LINT_W002_LOOKAHEAD_ON = "W002"  # lookahead=barmerge.lookahead_on
LINT_W003_DEPRECATED_SECURITY = "W003"  # security() instead of request.security()
LINT_W004_REPAINTING_RISK = "W004"  # Potential repainting pattern

# Info codes (I-prefix) - suggestions
LINT_I001_CONSIDER_LIBRARY = "I001"  # Could be extracted to library
LINT_I002_LARGE_SCRIPT = "I002"  # Script exceeds recommended size
