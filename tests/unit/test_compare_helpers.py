"""Unit tests for tune compare helper functions."""

from app.admin.router import _normalize_compare_value, _values_differ, _overfit_class


class TestNormalizeCompareValue:
    """Tests for value normalization in compare view."""

    def test_none_returns_dash(self):
        assert _normalize_compare_value(None) == "—"
        assert _normalize_compare_value(None, "pct") == "—"
        assert _normalize_compare_value(None, "float2") == "—"

    def test_pct_format_with_sign(self):
        assert _normalize_compare_value(12.345, "pct") == "+12.35%"
        assert _normalize_compare_value(-5.5, "pct") == "-5.50%"
        assert _normalize_compare_value(0, "pct") == "+0.00%"

    def test_pct_neg_format_always_negative(self):
        # Drawdown should always show as negative
        assert _normalize_compare_value(15.5, "pct_neg") == "-15.5%"
        assert _normalize_compare_value(-15.5, "pct_neg") == "-15.5%"

    def test_float2_format(self):
        assert _normalize_compare_value(1.2345, "float2") == "1.23"
        assert _normalize_compare_value(0.1, "float2") == "0.10"

    def test_float4_format(self):
        assert _normalize_compare_value(1.23456789, "float4") == "1.2346"

    def test_int_format(self):
        assert _normalize_compare_value(42, "int") == "42"
        assert _normalize_compare_value(42.7, "int") == "42"

    def test_pct_ratio_format(self):
        assert _normalize_compare_value(0.2, "pct_ratio") == "20%"
        assert _normalize_compare_value(0.15, "pct_ratio") == "15%"

    def test_default_format_stringifies(self):
        assert _normalize_compare_value("hello", "default") == "hello"
        assert _normalize_compare_value(123, "default") == "123"


class TestValuesDiffer:
    """Tests for difference detection in compare view."""

    def test_identical_values_do_not_differ(self):
        assert not _values_differ(["1.23", "1.23"])
        assert not _values_differ(["hello", "hello", "hello"])

    def test_different_values_differ(self):
        assert _values_differ(["1.23", "4.56"])
        assert _values_differ(["sharpe", "sharpe_dd_penalty"])

    def test_one_missing_differs(self):
        # If one has value and other is missing, they differ
        assert _values_differ(["1.23", "—"])
        assert _values_differ(["—", "hello"])

    def test_all_missing_does_not_differ(self):
        # Both missing = same (no difference to highlight)
        assert not _values_differ(["—", "—"])
        assert not _values_differ(["—", "—", "—"])

    def test_three_way_all_same(self):
        assert not _values_differ(["a", "a", "a"])

    def test_three_way_one_different(self):
        assert _values_differ(["a", "a", "b"])
        assert _values_differ(["a", "b", "a"])

    def test_three_way_all_different(self):
        assert _values_differ(["a", "b", "c"])


class TestOverfitClass:
    """Tests for overfit gap CSS class selection."""

    def test_none_returns_empty(self):
        assert _overfit_class(None) == ""

    def test_negative_gap_is_good(self):
        # OOS better than IS (rare but good)
        assert _overfit_class(-0.1) == "overfit-good"
        assert _overfit_class(-0.5) == "overfit-good"

    def test_small_gap_is_normal(self):
        assert _overfit_class(0) == ""
        assert _overfit_class(0.1) == ""
        assert _overfit_class(0.29) == ""

    def test_medium_gap_is_warning(self):
        assert _overfit_class(0.31) == "overfit-warning"
        assert _overfit_class(0.4) == "overfit-warning"
        assert _overfit_class(0.5) == "overfit-warning"

    def test_large_gap_is_danger(self):
        assert _overfit_class(0.51) == "overfit-danger"
        assert _overfit_class(1.0) == "overfit-danger"
        assert _overfit_class(2.0) == "overfit-danger"
