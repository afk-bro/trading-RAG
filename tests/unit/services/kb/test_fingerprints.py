"""Unit tests for regime fingerprint service."""

import hashlib

import pytest

from app.services.kb.fingerprints import RegimeVector


class TestRegimeVector:
    """Tests for RegimeVector dataclass."""

    def test_to_array(self):
        """Should convert to array in correct order."""
        vec = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )

        arr = vec.to_array()

        assert arr == [0.0145, 45.23, 0.0234, 0.6543, 0.7821, -0.5234]

    def test_from_array(self):
        """Should reconstruct from array."""
        arr = [0.0145, 45.23, 0.0234, 0.6543, 0.7821, -0.5234]

        vec = RegimeVector.from_array(arr)

        assert vec.atr_norm == 0.0145
        assert vec.rsi == 45.23
        assert vec.bb_width == 0.0234
        assert vec.efficiency == 0.6543
        assert vec.trend_strength == 0.7821
        assert vec.zscore == -0.5234

    def test_from_array_wrong_length_raises(self):
        """Should raise on wrong array length."""
        with pytest.raises(ValueError, match="Expected 6 elements"):
            RegimeVector.from_array([1.0, 2.0, 3.0])

    def test_compute_fingerprint_deterministic(self):
        """Same vector should produce same fingerprint."""
        vec1 = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )
        vec2 = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )

        fp1 = vec1.compute_fingerprint()
        fp2 = vec2.compute_fingerprint()

        assert fp1 == fp2
        assert len(fp1) == 32  # SHA256 is 32 bytes

    def test_compute_fingerprint_canonical(self):
        """Fingerprint should use canonical format."""
        vec = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )

        # Expected canonical format
        expected_canonical = "0.0145|45.2300|0.0234|0.6543|0.7821|-0.5234"
        expected_hash = hashlib.sha256(expected_canonical.encode("utf-8")).digest()

        fp = vec.compute_fingerprint()

        assert fp == expected_hash

    def test_compute_fingerprint_rounding(self):
        """Small floating point differences should produce same fingerprint."""
        vec1 = RegimeVector(
            atr_norm=0.01450001,
            rsi=45.22999999,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )
        vec2 = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )

        fp1 = vec1.compute_fingerprint()
        fp2 = vec2.compute_fingerprint()

        assert fp1 == fp2

    def test_compute_fingerprint_different_values(self):
        """Different vectors should produce different fingerprints."""
        vec1 = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )
        vec2 = RegimeVector(
            atr_norm=0.0200,  # Different
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )

        fp1 = vec1.compute_fingerprint()
        fp2 = vec2.compute_fingerprint()

        assert fp1 != fp2

    def test_roundtrip(self):
        """Array conversion should roundtrip correctly."""
        original = RegimeVector(
            atr_norm=0.0145,
            rsi=45.23,
            bb_width=0.0234,
            efficiency=0.6543,
            trend_strength=0.7821,
            zscore=-0.5234,
        )

        arr = original.to_array()
        restored = RegimeVector.from_array(arr)

        assert restored.atr_norm == original.atr_norm
        assert restored.rsi == original.rsi
        assert restored.bb_width == original.bb_width
        assert restored.efficiency == original.efficiency
        assert restored.trend_strength == original.trend_strength
        assert restored.zscore == original.zscore
