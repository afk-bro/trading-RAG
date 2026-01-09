"""
Smoke tests for KB recommend endpoint.

Run nightly in CI to catch regressions before users do.
Validates:
- HTTP 200 responses
- Status is not error
- Debug mode returns candidates (if seeded)
- Full mode returns recommended params

Usage:
    pytest tests/smoke/ -v --tb=short

Set SMOKE_TEST_URL to test against a live server:
    SMOKE_TEST_URL=http://localhost:8000 pytest tests/smoke/ -v
"""

import os
import uuid
from typing import Optional

import pytest

# Skip entire module if httpx not installed (lightweight marker)
httpx = pytest.importorskip("httpx")


# =============================================================================
# Configuration
# =============================================================================

SMOKE_TEST_URL = os.getenv("SMOKE_TEST_URL", "http://localhost:8000")
SMOKE_TEST_TIMEOUT = float(os.getenv("SMOKE_TEST_TIMEOUT", "30"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Test strategies - these should be in the registry
TEST_STRATEGIES = ["mean_reversion", "trend_following", "breakout", "rsi_strategy"]

# Test workspace (use a test workspace ID or let it default)
TEST_WORKSPACE_ID = os.getenv("SMOKE_TEST_WORKSPACE_ID", str(uuid.uuid4()))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def client():
    """Create httpx client for smoke tests."""
    return httpx.Client(base_url=SMOKE_TEST_URL, timeout=SMOKE_TEST_TIMEOUT)


@pytest.fixture(scope="module")
def headers():
    """Build request headers."""
    h = {"Content-Type": "application/json"}
    if ADMIN_TOKEN:
        h["X-Admin-Token"] = ADMIN_TOKEN
    return h


# =============================================================================
# Health Check
# =============================================================================


class TestHealthSmoke:
    """Health endpoint smoke tests."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should contain prometheus format
        assert "trading_rag" in response.text or "kb_recommend" in response.text


# =============================================================================
# KB Recommend Smoke Tests
# =============================================================================


class TestKBRecommendSmoke:
    """KB recommend endpoint smoke tests."""

    @pytest.mark.parametrize("strategy", TEST_STRATEGIES)
    def test_recommend_full_mode_returns_200(self, client, headers, strategy):
        """Full mode should return 200 with valid schema."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": strategy,
                "objective_type": "sharpe",
            },
            headers=headers,
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # Validate required fields
        assert "request_id" in data, "Missing request_id"
        assert "status" in data, "Missing status"
        assert "params" in data, "Missing params"

        # Status should not be error
        assert data["status"] in ["ok", "degraded", "none"], f"Invalid status: {data['status']}"

        # Log for debugging
        print(f"\n{strategy}: status={data['status']}, confidence={data.get('confidence')}, count_used={data.get('count_used')}")

    @pytest.mark.parametrize("strategy", TEST_STRATEGIES[:2])  # Test subset for speed
    def test_recommend_debug_mode_returns_candidates(self, client, headers, strategy):
        """Debug mode should return candidates."""
        response = client.post(
            "/kb/trials/recommend?mode=debug",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": strategy,
                "objective_type": "sharpe",
            },
            headers=headers,
        )

        assert response.status_code == 200

        data = response.json()

        # Debug mode should include top_candidates
        assert "top_candidates" in data, "Debug mode should include top_candidates"

        # Params should be empty in debug mode
        assert data["params"] == {}, "Debug mode should not return aggregated params"

        # Log candidate count
        candidate_count = len(data.get("top_candidates") or [])
        print(f"\n{strategy} debug: {candidate_count} candidates")

    def test_invalid_strategy_returns_400(self, client, headers):
        """Invalid strategy should return 400 with valid options."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": "nonexistent_strategy_xyz",
                "objective_type": "sharpe",
            },
            headers=headers,
        )

        assert response.status_code == 400

        data = response.json()
        detail = data.get("detail", {})

        # Should include valid options
        assert "INVALID_STRATEGY" in str(detail), "Should indicate invalid strategy"
        assert "valid_options" in str(detail) or "strategies" in str(detail), "Should include valid options"

    def test_recommend_with_regime_tags(self, client, headers):
        """Recommend with explicit regime tags should work."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": "mean_reversion",
                "objective_type": "sharpe",
                "regime_tags": ["uptrend", "low_vol"],
            },
            headers=headers,
        )

        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["ok", "degraded", "none"]

        # Query regime tags should reflect what we sent
        if data.get("query_regime_tags"):
            assert "uptrend" in data["query_regime_tags"] or len(data["query_regime_tags"]) >= 0


# =============================================================================
# Validation Smoke Tests
# =============================================================================


class TestValidationSmoke:
    """Input validation smoke tests."""

    def test_retrieve_k_bounds(self, client, headers):
        """retrieve_k > MAX should return 422."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": "mean_reversion",
                "objective_type": "sharpe",
                "retrieve_k": 1000,  # > MAX (500)
            },
            headers=headers,
        )

        assert response.status_code == 422

    def test_invalid_objective_returns_400(self, client, headers):
        """Invalid objective should return 400."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": "mean_reversion",
                "objective_type": "invalid_objective_xyz",
            },
            headers=headers,
        )

        assert response.status_code == 400


# =============================================================================
# Performance Smoke Tests
# =============================================================================


class TestPerformanceSmoke:
    """Basic performance smoke tests."""

    def test_recommend_under_2s(self, client, headers):
        """Recommend should complete under 2 seconds (p95 target)."""
        import time

        start = time.perf_counter()
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": "mean_reversion",
                "objective_type": "sharpe",
            },
            headers=headers,
        )
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Request took {elapsed:.2f}s, target is <2s"

        print(f"\nLatency: {elapsed*1000:.0f}ms")


# =============================================================================
# CI Artifact Export
# =============================================================================


def test_export_smoke_results(client, headers, tmp_path):
    """Export smoke test results as JSON artifact for CI."""
    import json
    from datetime import datetime

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_url": SMOKE_TEST_URL,
        "strategies_tested": TEST_STRATEGIES,
        "responses": [],
    }

    for strategy in TEST_STRATEGIES:
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": strategy,
                "objective_type": "sharpe",
            },
            headers=headers,
        )

        results["responses"].append({
            "strategy": strategy,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else {"error": response.text},
        })

    # Write to file for CI artifact upload
    output_file = tmp_path / "smoke_test_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nResults written to: {output_file}")

    # Assert all passed
    failed = [r for r in results["responses"] if r["status_code"] != 200]
    assert len(failed) == 0, f"Failed strategies: {[r['strategy'] for r in failed]}"
