"""Unit tests for Pine Script strategy spec generator."""

from app.services.pine.models import InputType, PineInput
from app.services.pine.spec_generator import (
    ParamSpec,
    _compute_priority,
    _is_sweepable,
    _normalize_name,
    pine_input_to_param_spec,
)


class TestNormalizeName:
    """Tests for name normalization."""

    def test_space_to_snake(self):
        """Spaces should become underscores."""
        assert _normalize_name("RSI Length") == "rsi_length"

    def test_camel_to_snake(self):
        """CamelCase should become snake_case."""
        assert _normalize_name("fastEMA") == "fast_ema"

    def test_special_chars(self):
        """Special characters should become underscores."""
        assert _normalize_name("ATR-Period") == "atr_period"

    def test_consecutive_underscores(self):
        """Consecutive underscores should collapse."""
        assert _normalize_name("BB  Width") == "bb_width"

    def test_strip_edges(self):
        """Leading/trailing underscores should be stripped."""
        assert _normalize_name("_test_") == "test"


class TestIsSweepable:
    """Tests for sweepable detection."""

    def test_bool_always_sweepable(self):
        """Bool inputs are always sweepable."""
        input = PineInput(name="show_signal", type=InputType.BOOL, default=True)
        assert _is_sweepable(input) is True

    def test_int_with_bounds_sweepable(self):
        """Int with min/max is sweepable."""
        input = PineInput(
            name="length",
            type=InputType.INT,
            default=14,
            min_value=1,
            max_value=100,
        )
        assert _is_sweepable(input) is True

    def test_int_without_bounds_not_sweepable(self):
        """Int without bounds is not sweepable."""
        input = PineInput(name="length", type=InputType.INT, default=14)
        assert _is_sweepable(input) is False

    def test_float_with_bounds_sweepable(self):
        """Float with min/max is sweepable."""
        input = PineInput(
            name="multiplier",
            type=InputType.FLOAT,
            default=2.0,
            min_value=0.1,
            max_value=5.0,
        )
        assert _is_sweepable(input) is True

    def test_with_options_sweepable(self):
        """Input with options array is sweepable."""
        input = PineInput(
            name="ma_type",
            type=InputType.STRING,
            default="SMA",
            options=["SMA", "EMA", "WMA"],
        )
        assert _is_sweepable(input) is True

    def test_single_option_not_sweepable(self):
        """Single option is not sweepable."""
        input = PineInput(
            name="ma_type",
            type=InputType.STRING,
            default="SMA",
            options=["SMA"],
        )
        assert _is_sweepable(input) is False

    def test_source_not_sweepable(self):
        """Source type without options is not sweepable."""
        input = PineInput(name="src", type=InputType.SOURCE, default_expr="close")
        assert _is_sweepable(input) is False


class TestComputePriority:
    """Tests for priority computation."""

    def test_int_type_gets_base_priority(self):
        """Int type adds base priority."""
        input = PineInput(name="value", type=InputType.INT, default=10)
        assert _compute_priority(input) >= 10

    def test_bounds_add_priority(self):
        """Min/max bounds add significant priority."""
        without_bounds = PineInput(name="length", type=InputType.INT, default=14)
        with_bounds = PineInput(
            name="length",
            type=InputType.INT,
            default=14,
            min_value=1,
            max_value=100,
        )
        assert _compute_priority(with_bounds) > _compute_priority(without_bounds)

    def test_keyword_adds_priority(self):
        """High-priority keywords add priority."""
        generic = PineInput(name="value", type=InputType.INT, default=10)
        with_keyword = PineInput(name="rsi_length", type=InputType.INT, default=14)
        assert _compute_priority(with_keyword) > _compute_priority(generic)

    def test_color_keyword_reduces_priority(self):
        """Visual keywords reduce priority."""
        generic = PineInput(name="value", type=InputType.INT, default=10)
        color_input = PineInput(name="line_color", type=InputType.COLOR, default=None)
        assert _compute_priority(color_input) < _compute_priority(generic)


class TestPineInputToParamSpec:
    """Tests for conversion to ParamSpec."""

    def test_basic_conversion(self):
        """Should convert basic input correctly."""
        input = PineInput(
            name="RSI Length",
            type=InputType.INT,
            default=14,
            min_value=1,
            max_value=100,
            step=1,
            tooltip="RSI calculation period",
        )

        spec = pine_input_to_param_spec(input)

        assert spec.name == "rsi_length"
        assert spec.display_name == "RSI Length"
        assert spec.type == "int"
        assert spec.default == 14
        assert spec.min_value == 1
        assert spec.max_value == 100
        assert spec.step == 1
        assert spec.tooltip == "RSI calculation period"
        assert spec.sweepable is True

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        spec = ParamSpec(
            name="rsi_length",
            display_name="RSI Length",
            type="int",
            default=14,
            min_value=1,
            max_value=100,
            sweepable=True,
            priority=25,
        )

        d = spec.to_dict()

        assert d["name"] == "rsi_length"
        assert d["type"] == "int"
        assert d["default"] == 14
        assert d["min"] == 1
        assert d["max"] == 100
        assert d["sweepable"] is True
        # Optional fields not in dict if None
        assert "step" not in d
        assert "options" not in d

    def test_with_options(self):
        """Should handle options array."""
        input = PineInput(
            name="MA Type",
            type=InputType.STRING,
            default="SMA",
            options=["SMA", "EMA", "WMA", "VWMA"],
        )

        spec = pine_input_to_param_spec(input)

        assert spec.options == ["SMA", "EMA", "WMA", "VWMA"]
        assert spec.sweepable is True

    def test_with_expression_default(self):
        """Should handle expression defaults."""
        input = PineInput(
            name="Source",
            type=InputType.SOURCE,
            default=None,
            default_expr="close",
        )

        spec = pine_input_to_param_spec(input)

        assert spec.default is None
        assert spec.default_expr == "close"
