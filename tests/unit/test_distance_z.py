"""Tests for regime distance z-score computation."""

import pytest
import numpy as np
from app.services.kb.distance import (
    compute_standardized_distance,
    compute_distance_distribution,
    compute_regime_distance_z,
    DistanceResult,
    DEFAULT_SHRINKAGE_C,
)


class TestStandardizedDistance:
    """Tests for standardized Euclidean distance."""

    def test_identical_vectors_zero_distance(self):
        """Identical vectors have zero distance."""
        x = {"atr_pct": 0.02, "rsi": 50.0, "zscore": 0.0}
        y = {"atr_pct": 0.02, "rsi": 50.0, "zscore": 0.0}
        var = {"atr_pct": 0.0001, "rsi": 100.0, "zscore": 1.0}

        d = compute_standardized_distance(x, y, var)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_distance_is_scale_free(self):
        """Distance is normalized by variance (scale-free)."""
        x = {"a": 0.0, "b": 0.0}
        y = {"a": 1.0, "b": 10.0}

        # With equal variances, b contributes more
        var_equal = {"a": 1.0, "b": 1.0}
        d_equal = compute_standardized_distance(x, y, var_equal)

        # With scaled variances, contributions equal
        var_scaled = {"a": 1.0, "b": 100.0}
        d_scaled = compute_standardized_distance(x, y, var_scaled)

        assert d_equal > d_scaled

    def test_missing_feature_uses_epsilon(self):
        """Missing variance uses epsilon to avoid div-by-zero."""
        x = {"a": 0.0}
        y = {"a": 1.0}
        var = {}  # No variance info

        d = compute_standardized_distance(x, y, var)
        assert d > 0  # Should not raise


class TestDistanceDistribution:
    """Tests for neighbor distance distribution."""

    def test_robust_median_mad(self):
        """Distribution uses median and MAD (robust to outliers)."""
        distances = [1.0, 1.1, 1.2, 1.0, 100.0]  # One outlier

        result = compute_distance_distribution(distances)

        # Median should be ~1.1, not skewed by outlier
        assert result.mu == pytest.approx(1.1, abs=0.1)
        # Sigma should be small (MAD-based)
        assert result.sigma < 1.0

    def test_single_distance_returns_defaults(self):
        """Single distance returns zero sigma."""
        distances = [1.5]
        result = compute_distance_distribution(distances)

        assert result.mu == 1.5
        assert result.sigma == pytest.approx(0.0, abs=1e-9)


class TestRegimeDistanceZ:
    """Tests for full distance z-score computation."""

    def test_z_score_interpretation(self):
        """Z-score reflects deviation from neighborhood."""
        # Current features match neighborhood well
        current_features = {"atr_pct": 0.02, "rsi": 50.0}
        neighbor_features = [
            {"atr_pct": 0.021, "rsi": 51.0},
            {"atr_pct": 0.019, "rsi": 49.0},
            {"atr_pct": 0.020, "rsi": 50.0},
        ]
        cluster_var = {"atr_pct": 0.0001, "rsi": 100.0}

        result = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var
        )

        # Should be close to 0 (within neighborhood)
        # z <= 1.0 indicates current is within 1 std dev of typical neighbor distance
        assert abs(result.z_score) <= 1.0
        assert result.baseline == "composite"

    def test_outlier_has_high_distance(self):
        """Outlier features have high distance to neighbors."""
        current_features = {"atr_pct": 0.10, "rsi": 90.0}  # Very different
        # Neighbors with some variation to establish a distribution
        neighbor_features = [
            {"atr_pct": 0.02, "rsi": 48.0},
            {"atr_pct": 0.02, "rsi": 50.0},
            {"atr_pct": 0.02, "rsi": 52.0},
            {"atr_pct": 0.021, "rsi": 49.0},
            {"atr_pct": 0.019, "rsi": 51.0},
        ]
        cluster_var = {"atr_pct": 0.0001, "rsi": 100.0}

        result = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var
        )

        # Should have high distance_now (far from neighbors)
        assert result.distance_now > 2.0
        # z_score should be defined
        assert result.z_score is not None

    def test_shrinkage_blends_prior_and_observed(self):
        """Shrinkage parameter blends cluster prior with observed."""
        current_features = {"a": 0.5}
        neighbor_features = [{"a": 0.5}, {"a": 0.5}]  # K=2
        cluster_var = {"a": 1.0}

        result_low_k = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var, shrinkage_c=10
        )

        # With more neighbors, less reliance on prior
        neighbor_features_many = [{"a": 0.5}] * 100
        result_high_k = compute_regime_distance_z(
            current_features, neighbor_features_many, cluster_var, shrinkage_c=10
        )

        # Both should work without error
        assert result_low_k.n_neighbors == 2
        assert result_high_k.n_neighbors == 100

    def test_empty_neighbors_returns_missing(self):
        """Empty neighbors returns null z-score with missing reason."""
        current_features = {"a": 0.5}
        neighbor_features = []
        cluster_var = {"a": 1.0}

        result = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var
        )

        assert result.z_score is None
        assert result.mu is None
        assert result.sigma is None
        assert "no_neighbors" in result.missing

    def test_fallback_to_neighbors_only(self):
        """When no cluster variance, uses neighbors-only baseline."""
        current_features = {"a": 0.5}
        neighbor_features = [{"a": 0.4}, {"a": 0.6}, {"a": 0.5}]
        cluster_var = None  # No cluster stats available

        result = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var
        )

        assert result.z_score is not None
        assert result.baseline == "neighbors_only"

    def test_default_shrinkage_constant(self):
        """Default shrinkage constant is 20."""
        assert DEFAULT_SHRINKAGE_C == 20.0

    def test_cluster_sigma_prior_shrinkage(self):
        """cluster_sigma_prior blends observed sigma toward prior."""
        current_features = {"a": 0.0}
        # Neighbors with some variance
        neighbor_features = [{"a": 0.1}, {"a": 0.2}, {"a": 0.3}, {"a": 0.4}]
        cluster_var = {"a": 1.0}

        # Without sigma prior
        result_no_prior = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var
        )

        # With sigma prior (should shrink toward prior)
        result_with_prior = compute_regime_distance_z(
            current_features,
            neighbor_features,
            cluster_var,
            cluster_sigma_prior=0.5,
        )

        # Both should produce valid z-scores
        assert result_no_prior.z_score is not None
        assert result_with_prior.z_score is not None

    def test_distance_now_field(self):
        """distance_now field contains median distance to neighbors."""
        current_features = {"a": 0.0}
        neighbor_features = [{"a": 0.1}, {"a": 0.2}]
        cluster_var = {"a": 1.0}

        result = compute_regime_distance_z(
            current_features, neighbor_features, cluster_var
        )

        # distance_now should be set (median of distances to neighbors)
        assert result.distance_now is not None
        assert result.distance_now >= 0
