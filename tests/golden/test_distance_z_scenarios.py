"""
Golden tests for distance z-score computation.

These tests use JSON fixtures that define:
1. Input data (current features, neighbor features, cluster variance)
2. Expected outputs (z_score, distance_now, mu, sigma, baseline)

Golden tests ensure the distance z-score computation produces consistent,
documented behavior across all scenarios including:
- Identical features (z-score should be 0)
- Outlier current regime (high z-score)
- Tight neighborhood (MAD stability)
- Sparse neighbors (edge case handling with 2-3 neighbors)
- Backoff to neighbors_only baseline (when cluster_var=None)
"""

import json
import pytest
from pathlib import Path

from app.services.kb.distance import compute_regime_distance_z


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_scenario(name: str) -> dict:
    """Load golden scenario from JSON fixture file."""
    path = FIXTURES_DIR / f"distance_{name}.json"
    with open(path) as f:
        return json.load(f)


class TestDistanceZGoldenScenarios:
    """Golden tests for distance z-score computation using recorded scenarios."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "identical_features",
            "outlier_current",
            "tight_neighborhood",
            "sparse_neighbors",
            "backoff_marginal",
        ],
    )
    def test_scenario(self, scenario_name: str):
        """
        Run golden scenario and verify outputs match expected values.

        Each scenario defines input features, cluster variance, and expected
        z-score computation results with tolerances for floating-point comparison.
        """
        scenario = load_scenario(scenario_name)

        result = compute_regime_distance_z(
            current_features=scenario["current_features"],
            neighbor_features=scenario["neighbor_features"],
            cluster_var=scenario.get("cluster_var"),
            shrinkage_c=scenario.get("shrinkage_c", 20.0),
            cluster_sigma_prior=scenario.get("cluster_sigma_prior"),
        )

        expected = scenario["expected"]
        tol = scenario.get("tolerance", {})

        # Check z_score
        if expected["z_score"] is not None:
            assert result.z_score == pytest.approx(
                expected["z_score"],
                abs=tol.get("z_score", 0.01),
            ), (
                f"z_score mismatch in {scenario_name}: "
                f"got {result.z_score}, expected {expected['z_score']}"
            )
        else:
            assert result.z_score is None, (
                f"z_score should be None in {scenario_name}, got {result.z_score}"
            )

        # Check baseline type
        assert result.baseline == expected["baseline"], (
            f"baseline mismatch in {scenario_name}: "
            f"got '{result.baseline}', expected '{expected['baseline']}'"
        )

        # Check n_neighbors
        assert result.n_neighbors == expected["n_neighbors"], (
            f"n_neighbors mismatch in {scenario_name}: "
            f"got {result.n_neighbors}, expected {expected['n_neighbors']}"
        )

        # Check distance_now
        if expected.get("distance_now") is not None:
            assert result.distance_now == pytest.approx(
                expected["distance_now"],
                abs=tol.get("distance_now", 0.01),
            ), (
                f"distance_now mismatch in {scenario_name}: "
                f"got {result.distance_now}, expected {expected['distance_now']}"
            )

        # Check mu (median distance)
        if expected.get("mu") is not None:
            assert result.mu == pytest.approx(
                expected["mu"],
                abs=tol.get("mu", 0.01),
            ), (
                f"mu mismatch in {scenario_name}: "
                f"got {result.mu}, expected {expected['mu']}"
            )

        # Check sigma (dispersion)
        if expected.get("sigma") is not None:
            assert result.sigma == pytest.approx(
                expected["sigma"],
                abs=tol.get("sigma", 0.01),
            ), (
                f"sigma mismatch in {scenario_name}: "
                f"got {result.sigma}, expected {expected['sigma']}"
            )


class TestDistanceZGoldenFixtureIntegrity:
    """Tests to verify golden fixtures are well-formed."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "identical_features",
            "outlier_current",
            "tight_neighborhood",
            "sparse_neighbors",
            "backoff_marginal",
        ],
    )
    def test_fixture_has_required_fields(self, scenario_name: str):
        """Verify each fixture has all required fields."""
        scenario = load_scenario(scenario_name)

        # Top-level required fields
        assert "name" in scenario, "Missing 'name' field"
        assert "description" in scenario, "Missing 'description' field"
        assert "current_features" in scenario, "Missing 'current_features' field"
        assert "neighbor_features" in scenario, "Missing 'neighbor_features' field"
        assert "expected" in scenario, "Missing 'expected' field"

        # Expected required fields
        expected = scenario["expected"]
        assert "z_score" in expected, "Missing 'z_score' in expected"
        assert "baseline" in expected, "Missing 'baseline' in expected"
        assert "n_neighbors" in expected, "Missing 'n_neighbors' in expected"

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "identical_features",
            "outlier_current",
            "tight_neighborhood",
            "sparse_neighbors",
            "backoff_marginal",
        ],
    )
    def test_fixture_neighbor_count_matches(self, scenario_name: str):
        """Verify n_neighbors matches actual neighbor list length."""
        scenario = load_scenario(scenario_name)

        actual_count = len(scenario["neighbor_features"])
        expected_count = scenario["expected"]["n_neighbors"]

        assert actual_count == expected_count, (
            f"n_neighbors mismatch: fixture has {actual_count} neighbors "
            f"but expected specifies {expected_count}"
        )

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "identical_features",
            "outlier_current",
            "tight_neighborhood",
            "sparse_neighbors",
            "backoff_marginal",
        ],
    )
    def test_fixture_feature_keys_consistent(self, scenario_name: str):
        """Verify feature keys are consistent across current and neighbors."""
        scenario = load_scenario(scenario_name)

        current_keys = set(scenario["current_features"].keys())

        for i, neighbor in enumerate(scenario["neighbor_features"]):
            neighbor_keys = set(neighbor.keys())
            assert current_keys == neighbor_keys, (
                f"Neighbor {i} has different keys: "
                f"current={current_keys}, neighbor={neighbor_keys}"
            )


class TestDistanceZGoldenScenarioDocumentation:
    """Tests that verify each scenario tests what it claims to test."""

    def test_identical_features_has_zero_z_score(self):
        """Verify identical_features scenario has z=0."""
        scenario = load_scenario("identical_features")

        assert scenario["expected"]["z_score"] == pytest.approx(0.0, abs=0.01), (
            "identical_features scenario should have z_score of 0"
        )

        # Verify all neighbors match current
        current = scenario["current_features"]
        for neighbor in scenario["neighbor_features"]:
            for key in current:
                assert current[key] == neighbor[key], (
                    f"identical_features: neighbor should match current for {key}"
                )

    def test_outlier_current_has_large_distance(self):
        """Verify outlier_current scenario has large distance_now."""
        scenario = load_scenario("outlier_current")

        # The outlier should have a large distance from neighbors
        assert scenario["expected"]["distance_now"] is not None
        assert scenario["expected"]["distance_now"] > 2.0, (
            "outlier_current scenario should have distance_now > 2.0"
        )

    def test_tight_neighborhood_has_small_sigma(self):
        """Verify tight_neighborhood tests MAD stability with small sigma."""
        scenario = load_scenario("tight_neighborhood")

        # Sigma should be small due to tight clustering
        assert scenario["expected"]["sigma"] is not None
        assert scenario["expected"]["sigma"] < 0.5, (
            "tight_neighborhood should have small sigma"
        )

    def test_sparse_neighbors_handles_few_points(self):
        """Verify sparse_neighbors tests edge case with few neighbors."""
        scenario = load_scenario("sparse_neighbors")

        n_neighbors = scenario["expected"]["n_neighbors"]
        assert 2 <= n_neighbors <= 3, (
            f"sparse_neighbors should have 2-3 neighbors, got {n_neighbors}"
        )

    def test_backoff_marginal_uses_neighbors_only(self):
        """Verify backoff_marginal tests neighbors_only baseline."""
        scenario = load_scenario("backoff_marginal")

        assert scenario.get("cluster_var") is None, (
            "backoff_marginal should have cluster_var=None"
        )
        assert scenario["expected"]["baseline"] == "neighbors_only", (
            "backoff_marginal should use 'neighbors_only' baseline"
        )
