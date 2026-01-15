"""
Shallow Pine Script parser.

Extracts metadata from Pine Script files using regex patterns.
Not a full AST parser - designed for registry/linting purposes.

Extracts:
- Version directive (//@version=N)
- Script declaration (indicator/strategy/library)
- Input definitions (input.int, input.float, etc.)
- Import statements
- Feature detection (request.security, lookahead, etc.)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.services.pine.constants import (
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
)
from app.services.pine.models import (
    InputType,
    PineImport,
    PineInput,
    PineScriptEntry,
    PineVersion,
    ScriptType,
    SourceFile,
    utc_now,
)


# =============================================================================
# Value Normalization
# =============================================================================


def normalize_value(
    value: Any,
) -> tuple[Optional[int | float | bool | str], Optional[str]]:
    """
    Normalize a parsed default value.

    Returns:
        (primitive_or_none, expr_string_or_none)

    Primitives (int, float, bool, str) are returned in first position.
    Complex expressions (color.new, close, etc.) are stringified in second position.
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        # Check bool before int (bool is subclass of int)
        return value, None
    if isinstance(value, (int, float, str)):
        return value, None
    # Non-primitive: store as expression string
    return None, str(value)


def normalize_list(values: list[Any]) -> list[int | float | bool | str]:
    """
    Normalize a list of option values.

    Primitives stay as-is, complex values are stringified.
    """
    result: list[int | float | bool | str] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            result.append(v)
        elif isinstance(v, (int, float, str)):
            result.append(v)
        else:
            result.append(str(v))
    return result


# =============================================================================
# Parse Result
# =============================================================================


@dataclass
class ParseResult:
    """Result of parsing a Pine Script file."""

    # Version
    pine_version: PineVersion = PineVersion.UNKNOWN
    version_line: Optional[int] = None

    # Declaration
    script_type: ScriptType = ScriptType.UNKNOWN
    title: Optional[str] = None
    short_title: Optional[str] = None
    overlay: Optional[bool] = None
    declaration_line: Optional[int] = None

    # Imports
    imports: list[PineImport] = field(default_factory=list)

    # Inputs
    inputs: list[PineInput] = field(default_factory=list)

    # Features
    features: dict[str, bool] = field(default_factory=dict)

    # Parse metadata
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Version Parsing
# =============================================================================

# Matches //@version=5 or // @version = 5 (flexible whitespace)
VERSION_PATTERN = re.compile(
    r"^\s*//\s*@version\s*=\s*(\d+)\s*$",
    re.MULTILINE,
)


def parse_version(content: str) -> tuple[PineVersion, Optional[int]]:
    """
    Extract Pine Script version from content.

    Returns:
        (version, line_number) - line is 1-indexed, None if not found
    """
    for i, line in enumerate(content.splitlines(), start=1):
        match = VERSION_PATTERN.match(line)
        if match:
            version_num = match.group(1)
            version_map = {
                "4": PineVersion.V4,
                "5": PineVersion.V5,
                "6": PineVersion.V6,
            }
            return version_map.get(version_num, PineVersion.UNKNOWN), i
    return PineVersion.UNKNOWN, None


# =============================================================================
# Declaration Parsing
# =============================================================================

# Matches indicator(...), strategy(...), library(...)
DECLARATION_PATTERN = re.compile(
    r"^\s*(indicator|strategy|library)\s*\(",
    re.MULTILINE,
)

# Extract title from first string argument
TITLE_PATTERN = re.compile(
    r'^\s*(?:indicator|strategy|library)\s*\(\s*["\']([^"\']+)["\']',
    re.MULTILINE,
)

# Extract shorttitle parameter
SHORTTITLE_PATTERN = re.compile(
    r'shorttitle\s*=\s*["\']([^"\']+)["\']',
)

# Extract overlay parameter
OVERLAY_PATTERN = re.compile(
    r"overlay\s*=\s*(true|false)",
    re.IGNORECASE,
)


def parse_declaration(
    content: str,
) -> tuple[ScriptType, Optional[str], Optional[str], Optional[bool], Optional[int]]:
    """
    Extract script declaration (indicator/strategy/library).

    Returns:
        (script_type, title, short_title, overlay, line_number)
    """
    script_type = ScriptType.UNKNOWN
    title = None
    short_title = None
    overlay = None
    line_num = None

    # Find declaration type and line
    for i, line in enumerate(content.splitlines(), start=1):
        match = DECLARATION_PATTERN.match(line)
        if match:
            decl_type = match.group(1).lower()
            type_map = {
                "indicator": ScriptType.INDICATOR,
                "strategy": ScriptType.STRATEGY,
                "library": ScriptType.LIBRARY,
            }
            script_type = type_map.get(decl_type, ScriptType.UNKNOWN)
            line_num = i
            break

    # Extract title (first string argument)
    title_match = TITLE_PATTERN.search(content)
    if title_match:
        title = title_match.group(1)

    # Extract shorttitle
    shorttitle_match = SHORTTITLE_PATTERN.search(content)
    if shorttitle_match:
        short_title = shorttitle_match.group(1)

    # Extract overlay
    overlay_match = OVERLAY_PATTERN.search(content)
    if overlay_match:
        overlay = overlay_match.group(1).lower() == "true"

    return script_type, title, short_title, overlay, line_num


# =============================================================================
# Import Parsing
# =============================================================================

# Matches: import user/repo/script/version as alias
# Or: import path as alias
IMPORT_PATTERN = re.compile(
    r"^\s*import\s+([^\s]+)\s+as\s+(\w+)",
    re.MULTILINE,
)


def parse_imports(content: str) -> list[PineImport]:
    """
    Extract import statements.

    Returns list of PineImport objects.
    """
    imports = []
    for i, line in enumerate(content.splitlines(), start=1):
        match = IMPORT_PATTERN.match(line)
        if match:
            path = match.group(1)
            alias = match.group(2)
            imports.append(PineImport(path=path, alias=alias, line=i))
    return imports


# =============================================================================
# Input Parsing
# =============================================================================

# Map input function names to InputType
INPUT_TYPE_MAP = {
    "input.int": InputType.INT,
    "input.float": InputType.FLOAT,
    "input.bool": InputType.BOOL,
    "input.string": InputType.STRING,
    "input.color": InputType.COLOR,
    "input.source": InputType.SOURCE,
    "input.timeframe": InputType.TIMEFRAME,
    "input.session": InputType.SESSION,
    "input.symbol": InputType.SYMBOL,
    "input": InputType.UNKNOWN,  # Legacy input() without type
}

# Pattern to find input assignments: varname = input.type(...)
# Captures: variable name, input function, arguments
INPUT_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(\w+)\s*=\s*(input(?:\.\w+)?)\s*\(([^)]*)\)",
    re.MULTILINE,
)

# Patterns for extracting named parameters
DEFVAL_PATTERN = re.compile(r"defval\s*=\s*([^,)]+)")
TITLE_PARAM_PATTERN = re.compile(r'title\s*=\s*["\']([^"\']+)["\']')
GROUP_PATTERN = re.compile(r'group\s*=\s*["\']([^"\']+)["\']')
TOOLTIP_PATTERN = re.compile(r'tooltip\s*=\s*["\']([^"\']+)["\']')
INLINE_PATTERN = re.compile(r'inline\s*=\s*["\']([^"\']+)["\']')
MINVAL_PATTERN = re.compile(r"minval\s*=\s*([+-]?\d+(?:\.\d+)?)")
MAXVAL_PATTERN = re.compile(r"maxval\s*=\s*([+-]?\d+(?:\.\d+)?)")
STEP_PATTERN = re.compile(r"step\s*=\s*([+-]?\d+(?:\.\d+)?)")
OPTIONS_PATTERN = re.compile(r"options\s*=\s*\[([^\]]+)\]")


class _Identifier:
    """
    Marker class for Pine Script identifiers (not string literals).

    Used to distinguish `close` (identifier) from `"close"` (string).
    normalize_value() treats these as expressions.
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"_Identifier({self.name!r})"


def _parse_literal(value: str) -> Any:
    """
    Parse a literal value from Pine Script.

    Handles: numbers, booleans, strings, identifiers.
    Identifiers are wrapped in _Identifier to distinguish from string literals.
    """
    value = value.strip()

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Quoted string - return the inner string (primitive)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    # Identifier or expression - wrap in _Identifier (non-primitive)
    return _Identifier(value)


def _parse_options(options_str: str) -> list[Any]:
    """Parse options array content."""
    options = []
    # Split by comma, but be careful with nested structures
    # Simple approach: split and parse each
    for item in options_str.split(","):
        item = item.strip()
        if item:
            options.append(_parse_literal(item))
    return options


def parse_inputs(content: str) -> list[PineInput]:
    """
    Extract input definitions from Pine Script.

    Returns list of PineInput objects.
    """
    inputs = []

    for i, line in enumerate(content.splitlines(), start=1):
        match = INPUT_ASSIGNMENT_PATTERN.match(line)
        if not match:
            continue

        var_name = match.group(1)
        input_func = match.group(2)
        args_str = match.group(3)

        # Determine input type
        input_type = INPUT_TYPE_MAP.get(input_func, InputType.UNKNOWN)

        # Extract parameters
        # First positional argument is often defval (for input.int, etc.)
        # or title (for legacy input())
        default_val = None
        default_expr = None
        title = None
        group = None
        tooltip = None
        inline = None
        min_value = None
        max_value = None
        step = None
        options = None

        # Check for defval parameter
        defval_match = DEFVAL_PATTERN.search(args_str)
        if defval_match:
            raw_default = _parse_literal(defval_match.group(1))
            default_val, default_expr = normalize_value(raw_default)
        else:
            # First positional might be the default
            # For typed inputs like input.int(10, ...), first arg is defval
            args_parts = args_str.split(",")
            if args_parts:
                first_arg = args_parts[0].strip()
                # Skip if it's a named param
                if "=" not in first_arg and first_arg:
                    raw_default = _parse_literal(first_arg)
                    default_val, default_expr = normalize_value(raw_default)

        # Extract title
        title_match = TITLE_PARAM_PATTERN.search(args_str)
        if title_match:
            title = title_match.group(1)

        # Use variable name as fallback title
        if not title:
            title = var_name

        # Extract group
        group_match = GROUP_PATTERN.search(args_str)
        if group_match:
            group = group_match.group(1)

        # Extract tooltip
        tooltip_match = TOOLTIP_PATTERN.search(args_str)
        if tooltip_match:
            tooltip = tooltip_match.group(1)

        # Extract inline
        inline_match = INLINE_PATTERN.search(args_str)
        if inline_match:
            inline = inline_match.group(1)

        # Extract numeric bounds
        minval_match = MINVAL_PATTERN.search(args_str)
        if minval_match:
            min_value = float(minval_match.group(1))

        maxval_match = MAXVAL_PATTERN.search(args_str)
        if maxval_match:
            max_value = float(maxval_match.group(1))

        step_match = STEP_PATTERN.search(args_str)
        if step_match:
            step = float(step_match.group(1))

        # Extract options
        options_match = OPTIONS_PATTERN.search(args_str)
        if options_match:
            raw_options = _parse_options(options_match.group(1))
            options = normalize_list(raw_options)

        inputs.append(
            PineInput(
                name=title or var_name,
                type=input_type,
                default=default_val,
                default_expr=default_expr,
                group=group,
                tooltip=tooltip,
                inline=inline,
                min_value=min_value,
                max_value=max_value,
                step=step,
                options=options,
                line=i,
            )
        )

    return inputs


# =============================================================================
# Feature Detection
# =============================================================================

# Feature detection patterns
FEATURE_PATTERNS: dict[str, re.Pattern] = {
    FEATURE_USES_REQUEST_SECURITY: re.compile(r"\brequest\.security\s*\("),
    FEATURE_USES_SECURITY: re.compile(r"\bsecurity\s*\("),  # deprecated
    FEATURE_USES_LOOKAHEAD_ON: re.compile(
        r"\blookahead\s*=\s*barmerge\.lookahead_on\b"
    ),
    FEATURE_USES_VARIP: re.compile(r"\bvarip\b"),
    FEATURE_USES_ARRAYS: re.compile(r"\barray\.(new_\w+|from|get|set|push|pop|size)\b"),
    FEATURE_USES_MATRICES: re.compile(r"\bmatrix\.(new_\w+|get|set|rows|columns)\b"),
    FEATURE_USES_MAPS: re.compile(r"\bmap\.(new|get|put|keys|values)\b"),
    FEATURE_USES_METHODS: re.compile(r"\bmethod\s+\w+\s*\("),
    FEATURE_USES_UDT: re.compile(r"\btype\s+\w+\s*\n"),
    FEATURE_USES_STRATEGY_FUNCTIONS: re.compile(
        r"\bstrategy\.(entry|exit|close|order|cancel)\s*\("
    ),
    FEATURE_USES_ALERT: re.compile(r"\balert(?:condition)?\s*\("),
    FEATURE_USES_LOG: re.compile(r"\blog\.(info|warning|error)\s*\("),
}


def detect_features(content: str, script_type: ScriptType) -> dict[str, bool]:
    """
    Detect features used in Pine Script content.

    Returns dict of feature flags.
    """
    features = {}

    for feature_name, pattern in FEATURE_PATTERNS.items():
        features[feature_name] = bool(pattern.search(content))

    # Add is_library based on script type
    features[FEATURE_IS_LIBRARY] = script_type == ScriptType.LIBRARY

    return features


# =============================================================================
# Main Parser
# =============================================================================


def parse_pine(source: SourceFile) -> ParseResult:
    """
    Parse a Pine Script file.

    Args:
        source: SourceFile from adapter

    Returns:
        ParseResult with extracted metadata
    """
    content = source.content
    result = ParseResult()

    # Parse version
    result.pine_version, result.version_line = parse_version(content)

    # Parse declaration
    (
        result.script_type,
        result.title,
        result.short_title,
        result.overlay,
        result.declaration_line,
    ) = parse_declaration(content)

    # Parse imports
    result.imports = parse_imports(content)

    # Parse inputs
    result.inputs = parse_inputs(content)

    # Detect features
    result.features = detect_features(content, result.script_type)

    # Add warnings for missing version/declaration
    if result.pine_version == PineVersion.UNKNOWN:
        result.warnings.append("No //@version directive found")
    if result.script_type == ScriptType.UNKNOWN:
        result.warnings.append("No indicator/strategy/library declaration found")

    return result


def parse_to_entry(
    source: SourceFile,
    parse_result: Optional[ParseResult] = None,
) -> PineScriptEntry:
    """
    Parse a SourceFile and create a PineScriptEntry.

    Args:
        source: SourceFile from adapter
        parse_result: Optional pre-computed ParseResult

    Returns:
        PineScriptEntry ready for registry
    """
    if parse_result is None:
        parse_result = parse_pine(source)

    # Compute content hash
    sha256 = hashlib.sha256(source.content.encode("utf-8")).hexdigest()

    return PineScriptEntry(
        rel_path=source.rel_path,
        sha256=sha256,
        source_id=source.source_id,
        pine_version=parse_result.pine_version,
        script_type=parse_result.script_type,
        title=parse_result.title,
        short_title=parse_result.short_title,
        overlay=parse_result.overlay,
        imports=list(parse_result.imports),
        inputs=list(parse_result.inputs),
        features=dict(parse_result.features),
        parsed_at=utc_now(),
        source_mtime=source.mtime,
    )
