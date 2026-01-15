"""
Unit tests for Pine Script parser.

Tests shallow parsing of Pine Script files for registry purposes.
"""

from app.services.pine.constants import (
    FEATURE_IS_LIBRARY,
    FEATURE_USES_ALERT,
    FEATURE_USES_ARRAYS,
    FEATURE_USES_LOOKAHEAD_ON,
    FEATURE_USES_REQUEST_SECURITY,
    FEATURE_USES_SECURITY,
    FEATURE_USES_STRATEGY_FUNCTIONS,
    FEATURE_USES_VARIP,
)
from app.services.pine.models import (
    InputType,
    PineVersion,
    ScriptType,
    SourceFile,
)
from app.services.pine.parser import (
    detect_features,
    normalize_list,
    normalize_value,
    parse_declaration,
    parse_imports,
    parse_inputs,
    parse_pine,
    parse_to_entry,
    parse_version,
)


class TestNormalizeValue:
    """Tests for normalize_value()."""

    def test_normalize_none(self):
        """None returns (None, None)."""
        assert normalize_value(None) == (None, None)

    def test_normalize_int(self):
        """Integers are primitives."""
        assert normalize_value(42) == (42, None)
        assert normalize_value(-10) == (-10, None)
        assert normalize_value(0) == (0, None)

    def test_normalize_float(self):
        """Floats are primitives."""
        assert normalize_value(3.14) == (3.14, None)
        assert normalize_value(-0.5) == (-0.5, None)

    def test_normalize_bool(self):
        """Booleans are primitives (checked before int)."""
        assert normalize_value(True) == (True, None)
        assert normalize_value(False) == (False, None)

    def test_normalize_str(self):
        """Strings are primitives."""
        assert normalize_value("hello") == ("hello", None)
        assert normalize_value("") == ("", None)

    def test_normalize_complex(self):
        """Non-primitives become expression strings."""
        assert normalize_value({"key": "value"}) == (None, "{'key': 'value'}")
        assert normalize_value([1, 2, 3]) == (None, "[1, 2, 3]")


class TestNormalizeList:
    """Tests for normalize_list()."""

    def test_normalize_primitives(self):
        """Primitives stay as-is."""
        assert normalize_list([1, 2, 3]) == [1, 2, 3]
        assert normalize_list(["a", "b"]) == ["a", "b"]
        assert normalize_list([True, False]) == [True, False]
        assert normalize_list([1.5, 2.5]) == [1.5, 2.5]

    def test_normalize_mixed(self):
        """Mixed types work."""
        assert normalize_list([1, "two", 3.0, True]) == [1, "two", 3.0, True]

    def test_normalize_skips_none(self):
        """None values are skipped."""
        assert normalize_list([1, None, 2]) == [1, 2]

    def test_normalize_stringifies_complex(self):
        """Complex values are stringified."""
        result = normalize_list([1, {"x": 1}, 2])
        assert result == [1, "{'x': 1}", 2]


class TestParseVersion:
    """Tests for parse_version()."""

    def test_parse_version_5(self):
        """Parses //@version=5."""
        content = "//@version=5\nindicator('Test')"
        version, line = parse_version(content)
        assert version == PineVersion.V5
        assert line == 1

    def test_parse_version_4(self):
        """Parses //@version=4."""
        content = "//@version=4\n"
        version, line = parse_version(content)
        assert version == PineVersion.V4

    def test_parse_version_6(self):
        """Parses //@version=6."""
        content = "//@version=6"
        version, line = parse_version(content)
        assert version == PineVersion.V6

    def test_parse_version_with_spaces(self):
        """Handles flexible whitespace."""
        content = "// @version = 5\n"
        version, line = parse_version(content)
        assert version == PineVersion.V5

    def test_parse_version_not_first_line(self):
        """Version can be on any line."""
        content = "// comment\n// another\n//@version=5\n"
        version, line = parse_version(content)
        assert version == PineVersion.V5
        assert line == 3

    def test_parse_version_missing(self):
        """Missing version returns UNKNOWN."""
        content = "indicator('Test')"
        version, line = parse_version(content)
        assert version == PineVersion.UNKNOWN
        assert line is None

    def test_parse_version_unknown_number(self):
        """Unknown version number returns UNKNOWN."""
        content = "//@version=99"
        version, line = parse_version(content)
        assert version == PineVersion.UNKNOWN


class TestParseDeclaration:
    """Tests for parse_declaration()."""

    def test_parse_indicator(self):
        """Parses indicator declaration."""
        content = '//@version=5\nindicator("My Indicator", overlay=true)'
        script_type, title, short_title, overlay, line = parse_declaration(content)
        assert script_type == ScriptType.INDICATOR
        assert title == "My Indicator"
        assert overlay is True
        assert line == 2

    def test_parse_strategy(self):
        """Parses strategy declaration."""
        content = """//@version=5
strategy("Test Strategy", shorttitle="TS", overlay=false)
"""
        script_type, title, short_title, overlay, line = parse_declaration(content)
        assert script_type == ScriptType.STRATEGY
        assert title == "Test Strategy"
        assert short_title == "TS"
        assert overlay is False

    def test_parse_library(self):
        """Parses library declaration."""
        content = '//@version=5\nlibrary("Utils")'
        script_type, title, short_title, overlay, line = parse_declaration(content)
        assert script_type == ScriptType.LIBRARY
        assert title == "Utils"

    def test_parse_single_quotes(self):
        """Handles single-quoted titles."""
        content = "//@version=5\nindicator('Single Quotes')"
        _, title, _, _, _ = parse_declaration(content)
        assert title == "Single Quotes"

    def test_parse_no_declaration(self):
        """Missing declaration returns UNKNOWN."""
        content = "//@version=5\nplot(close)"
        script_type, title, _, _, line = parse_declaration(content)
        assert script_type == ScriptType.UNKNOWN
        assert line is None


class TestParseImports:
    """Tests for parse_imports()."""

    def test_parse_simple_import(self):
        """Parses simple import statement."""
        content = "import TradingView/ta/1 as ta\n"
        imports = parse_imports(content)
        assert len(imports) == 1
        assert imports[0].path == "TradingView/ta/1"
        assert imports[0].alias == "ta"
        assert imports[0].line == 1

    def test_parse_multiple_imports(self):
        """Parses multiple imports."""
        content = """import user/lib1/1 as lib1
import user/lib2/2 as lib2
"""
        imports = parse_imports(content)
        assert len(imports) == 2
        assert imports[0].alias == "lib1"
        assert imports[1].alias == "lib2"

    def test_parse_no_imports(self):
        """No imports returns empty list."""
        content = "//@version=5\nindicator('Test')"
        imports = parse_imports(content)
        assert imports == []


class TestParseInputs:
    """Tests for parse_inputs()."""

    def test_parse_input_int(self):
        """Parses input.int()."""
        content = 'length = input.int(14, title="Length", minval=1, maxval=100)'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.name == "Length"
        assert inp.type == InputType.INT
        assert inp.default == 14
        assert inp.min_value == 1
        assert inp.max_value == 100

    def test_parse_input_float(self):
        """Parses input.float()."""
        content = 'mult = input.float(2.0, title="Multiplier", step=0.5)'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.name == "Multiplier"
        assert inp.type == InputType.FLOAT
        assert inp.default == 2.0
        assert inp.step == 0.5

    def test_parse_input_bool(self):
        """Parses input.bool()."""
        content = 'showMA = input.bool(true, title="Show MA")'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.name == "Show MA"
        assert inp.type == InputType.BOOL
        assert inp.default is True

    def test_parse_input_string_with_options(self):
        """Parses input.string() with options."""
        content = 'maType = input.string("SMA", title="MA Type", options=["SMA", "EMA", "WMA"])'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.name == "MA Type"
        assert inp.type == InputType.STRING
        assert inp.default == "SMA"
        assert inp.options == ["SMA", "EMA", "WMA"]

    def test_parse_input_source(self):
        """Parses input.source()."""
        content = 'src = input.source(close, title="Source")'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        inp = inputs[0]
        assert inp.name == "Source"
        assert inp.type == InputType.SOURCE
        # 'close' is an identifier, stored as expression
        assert inp.default_expr == "close"

    def test_parse_input_with_group(self):
        """Parses input with group parameter."""
        content = 'len = input.int(20, title="Length", group="Settings")'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        assert inputs[0].group == "Settings"

    def test_parse_input_with_tooltip(self):
        """Parses input with tooltip parameter."""
        content = 'len = input.int(20, title="Length", tooltip="The lookback period")'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        assert inputs[0].tooltip == "The lookback period"

    def test_parse_input_with_defval(self):
        """Parses input with explicit defval parameter."""
        content = 'len = input.int(defval=20, title="Length")'
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        assert inputs[0].default == 20

    def test_parse_input_fallback_to_varname(self):
        """Uses variable name as fallback title."""
        content = "myLength = input.int(14)"
        inputs = parse_inputs(content)
        assert len(inputs) == 1
        assert inputs[0].name == "myLength"

    def test_parse_multiple_inputs(self):
        """Parses multiple inputs."""
        content = """length = input.int(14, title="Length")
mult = input.float(2.0, title="Mult")
show = input.bool(true, title="Show")
"""
        inputs = parse_inputs(content)
        assert len(inputs) == 3
        assert inputs[0].name == "Length"
        assert inputs[1].name == "Mult"
        assert inputs[2].name == "Show"

    def test_parse_input_line_numbers(self):
        """Captures line numbers for inputs."""
        content = """// comment
length = input.int(14, title="Length")
// another comment
mult = input.float(2.0, title="Mult")
"""
        inputs = parse_inputs(content)
        assert len(inputs) == 2
        assert inputs[0].line == 2
        assert inputs[1].line == 4


class TestDetectFeatures:
    """Tests for detect_features()."""

    def test_detect_request_security(self):
        """Detects request.security()."""
        content = 'htfClose = request.security(syminfo.tickerid, "D", close)'
        features = detect_features(content, ScriptType.INDICATOR)
        assert features[FEATURE_USES_REQUEST_SECURITY] is True

    def test_detect_legacy_security(self):
        """Detects deprecated security()."""
        content = 'htfClose = security(syminfo.tickerid, "D", close)'
        features = detect_features(content, ScriptType.INDICATOR)
        assert features[FEATURE_USES_SECURITY] is True

    def test_detect_lookahead_on(self):
        """Detects lookahead=barmerge.lookahead_on."""
        content = "htfClose = request.security(sym, tf, close, lookahead=barmerge.lookahead_on)"
        features = detect_features(content, ScriptType.INDICATOR)
        assert features[FEATURE_USES_LOOKAHEAD_ON] is True

    def test_detect_varip(self):
        """Detects varip keyword."""
        content = "varip int counter = 0"
        features = detect_features(content, ScriptType.INDICATOR)
        assert features[FEATURE_USES_VARIP] is True

    def test_detect_arrays(self):
        """Detects array functions."""
        content = "arr = array.new_float(10)"
        features = detect_features(content, ScriptType.INDICATOR)
        assert features[FEATURE_USES_ARRAYS] is True

    def test_detect_strategy_functions(self):
        """Detects strategy.entry() etc."""
        content = 'strategy.entry("Long", strategy.long)'
        features = detect_features(content, ScriptType.STRATEGY)
        assert features[FEATURE_USES_STRATEGY_FUNCTIONS] is True

    def test_detect_alert(self):
        """Detects alert() and alertcondition()."""
        content = 'alert("Signal!")'
        features = detect_features(content, ScriptType.INDICATOR)
        assert features[FEATURE_USES_ALERT] is True

        content2 = "alertcondition(crossover(fast, slow))"
        features2 = detect_features(content2, ScriptType.INDICATOR)
        assert features2[FEATURE_USES_ALERT] is True

    def test_detect_is_library(self):
        """Sets is_library based on script type."""
        content = "export func() => 1"
        features = detect_features(content, ScriptType.LIBRARY)
        assert features[FEATURE_IS_LIBRARY] is True

        features2 = detect_features(content, ScriptType.INDICATOR)
        assert features2[FEATURE_IS_LIBRARY] is False


class TestParsePine:
    """Tests for parse_pine() main function."""

    def test_parse_complete_script(self):
        """Parses a complete Pine Script."""
        content = """//@version=5
indicator("Test Indicator", overlay=true)

import TradingView/ta/1 as ta

length = input.int(14, title="Length", minval=1)
mult = input.float(2.0, title="Multiplier")

basis = ta.sma(close, length)
plot(basis)
"""
        source = SourceFile(rel_path="test.pine", content=content)
        result = parse_pine(source)

        assert result.pine_version == PineVersion.V5
        assert result.script_type == ScriptType.INDICATOR
        assert result.title == "Test Indicator"
        assert result.overlay is True
        assert len(result.imports) == 1
        assert len(result.inputs) == 2
        assert result.warnings == []

    def test_parse_missing_version_warning(self):
        """Warns on missing version directive."""
        content = "indicator('Test')\nplot(close)"
        source = SourceFile(rel_path="test.pine", content=content)
        result = parse_pine(source)

        assert result.pine_version == PineVersion.UNKNOWN
        assert "No //@version directive found" in result.warnings

    def test_parse_missing_declaration_warning(self):
        """Warns on missing declaration."""
        content = "//@version=5\nplot(close)"
        source = SourceFile(rel_path="test.pine", content=content)
        result = parse_pine(source)

        assert result.script_type == ScriptType.UNKNOWN
        assert "No indicator/strategy/library declaration found" in result.warnings


class TestParseToEntry:
    """Tests for parse_to_entry()."""

    def test_creates_entry_with_hash(self):
        """Creates entry with correct SHA256 hash."""
        content = "//@version=5\nindicator('Test')"
        source = SourceFile(rel_path="test.pine", content=content)
        entry = parse_to_entry(source)

        assert entry.rel_path == "test.pine"
        assert len(entry.sha256) == 64  # SHA256 hex length
        assert entry.pine_version == PineVersion.V5
        assert entry.script_type == ScriptType.INDICATOR

    def test_creates_entry_with_timestamps(self):
        """Entry has parsed_at timestamp."""
        content = "//@version=5\nindicator('Test')"
        source = SourceFile(rel_path="test.pine", content=content)
        entry = parse_to_entry(source)

        assert entry.parsed_at is not None
        assert entry.parsed_at.tzinfo is not None  # timezone-aware

    def test_preserves_source_id(self):
        """Preserves source_id from SourceFile."""
        content = "//@version=5\nindicator('Test')"
        source = SourceFile(
            rel_path="test.pine",
            content=content,
            source_id="abc123",
        )
        entry = parse_to_entry(source)

        assert entry.source_id == "abc123"

    def test_accepts_precomputed_parse_result(self):
        """Can use pre-computed ParseResult."""
        content = "//@version=5\nindicator('Test')"
        source = SourceFile(rel_path="test.pine", content=content)

        # Pre-compute
        parse_result = parse_pine(source)
        entry = parse_to_entry(source, parse_result)

        assert entry.pine_version == PineVersion.V5
