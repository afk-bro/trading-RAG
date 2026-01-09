"""Tests for regime key canonicalization."""

import pytest
from app.services.kb.regime_key import (
    RegimeDims,
    canonicalize_regime_key,
    parse_regime_key,
    extract_marginal_keys,
    VALID_TREND_VALUES,
    VALID_VOL_VALUES,
)


class TestCanonicalizeRegimeKey:
    """Tests for canonicalize_regime_key()."""

    def test_basic_canonicalization(self):
        """Canonical key is sorted alphabetically by dimension."""
        dims = RegimeDims(trend="uptrend", vol="high_vol")
        assert canonicalize_regime_key(dims) == "trend=uptrend|vol=high_vol"

    def test_order_independent(self):
        """Same dims produce same key regardless of creation order."""
        dims1 = RegimeDims(trend="flat", vol="low_vol")
        dims2 = RegimeDims(vol="low_vol", trend="flat")
        assert canonicalize_regime_key(dims1) == canonicalize_regime_key(dims2)

    def test_all_combinations_valid(self):
        """All 9 combinations produce valid keys."""
        keys = set()
        for trend in VALID_TREND_VALUES:
            for vol in VALID_VOL_VALUES:
                dims = RegimeDims(trend=trend, vol=vol)
                key = canonicalize_regime_key(dims)
                assert key not in keys, f"Duplicate key: {key}"
                keys.add(key)
        assert len(keys) == 9

    def test_invalid_trend_raises(self):
        """Invalid trend value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid trend"):
            RegimeDims(trend="bullish", vol="high_vol")

    def test_invalid_vol_raises(self):
        """Invalid vol value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid vol"):
            RegimeDims(trend="uptrend", vol="extreme_vol")


class TestParseRegimeKey:
    """Tests for parse_regime_key()."""

    def test_roundtrip(self):
        """Parse inverts canonicalize."""
        dims = RegimeDims(trend="downtrend", vol="mid_vol")
        key = canonicalize_regime_key(dims)
        parsed = parse_regime_key(key)
        assert parsed == dims

    def test_invalid_format_raises(self):
        """Malformed key raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regime key format"):
            parse_regime_key("uptrend|high_vol")  # Missing dimension names


class TestExtractMarginalKeys:
    """Tests for extract_marginal_keys()."""

    def test_extracts_both_marginals(self):
        """Returns both single-dimension marginal keys."""
        key = "trend=uptrend|vol=high_vol"
        marginals = extract_marginal_keys(key)
        assert set(marginals) == {"trend=uptrend", "vol=high_vol"}

    def test_marginals_are_valid_keys(self):
        """Marginal keys can be parsed (for backoff queries)."""
        key = "trend=flat|vol=low_vol"
        for marginal in extract_marginal_keys(key):
            # Should not raise
            assert "=" in marginal
