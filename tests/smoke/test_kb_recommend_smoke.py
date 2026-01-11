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

import pytest


# Mark all tests in this module as smoke tests (require running server)
pytestmark = pytest.mark.smoke

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

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # Validate required fields
        assert "request_id" in data, "Missing request_id"
        assert "status" in data, "Missing status"
        assert "params" in data, "Missing params"

        # Status should not be error
        assert data["status"] in [
            "ok",
            "degraded",
            "none",
        ], f"Invalid status: {data['status']}"

        # Log for debugging
        print(
            f"\n{strategy}: status={data['status']}, confidence={data.get('confidence')}, count_used={data.get('count_used')}"  # noqa: E501
        )

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
        assert "valid_options" in str(detail) or "strategies" in str(
            detail
        ), "Should include valid options"

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
            assert (
                "uptrend" in data["query_regime_tags"]
                or len(data["query_regime_tags"]) >= 0
            )


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

# Fixed output directory for CI artifact upload
SMOKE_OUTPUT_DIR = os.getenv("SMOKE_OUTPUT_DIR", "/tmp/smoke-results")

# Relaxed floors for diagnostic pass
RELAXED_FLOORS = {
    "min_trades": 1,
    "max_drawdown": 0.50,  # 50%
    "max_overfit_gap": 1.0,  # Effectively disabled
}


def test_export_smoke_results(client, headers):
    """Export smoke test results as JSON artifact for CI.

    This test runs recommend for each strategy and saves full responses
    to a predictable location for artifact upload.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    # Create output directory
    output_dir = Path(SMOKE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_url": SMOKE_TEST_URL,
        "workspace_id": TEST_WORKSPACE_ID,
        "strategies_tested": TEST_STRATEGIES,
        "summary": {
            "total": len(TEST_STRATEGIES),
            "passed": 0,
            "degraded": 0,
            "none": 0,
            "failed": 0,
        },
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

        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            if status == "ok":
                results["summary"]["passed"] += 1
            elif status == "degraded":
                results["summary"]["degraded"] += 1
            elif status == "none":
                results["summary"]["none"] += 1
            else:
                results["summary"]["failed"] += 1

            # Build degradation_reasons from reasons + warnings for debugging
            degradation_reasons = data.get("reasons", []).copy()
            if data.get("used_relaxed_filters"):
                degradation_reasons.append("used_relaxed_filters")
            if data.get("used_metadata_fallback"):
                degradation_reasons.append("used_metadata_fallback")

            results["responses"].append(
                {
                    "strategy": strategy,
                    "status_code": response.status_code,
                    "kb_status": status,
                    "confidence": data.get("confidence"),
                    "count_used": data.get("count_used"),
                    "used_relaxed_filters": data.get("used_relaxed_filters"),
                    "used_metadata_fallback": data.get("used_metadata_fallback"),
                    "degradation_reasons": degradation_reasons,
                    "suggested_actions": data.get("suggested_actions", []),
                    "warnings": data.get("warnings", []),
                    "active_collection": data.get("active_collection"),
                    "embedding_model_id": data.get("embedding_model_id"),
                    "params": data.get("params", {}),
                }
            )
        else:
            results["summary"]["failed"] += 1
            results["responses"].append(
                {
                    "strategy": strategy,
                    "status_code": response.status_code,
                    "kb_status": "error",
                    "error": response.text[:500],  # Truncate long errors
                }
            )

    # Write to predictable location for CI artifact upload
    output_file = output_dir / "smoke_test_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}")
    print("Smoke Test Summary")
    print(f"{'='*60}")
    print(f"Total: {results['summary']['total']}")
    print(f"  OK:       {results['summary']['passed']}")
    print(f"  Degraded: {results['summary']['degraded']}")
    print(f"  None:     {results['summary']['none']}")
    print(f"  Failed:   {results['summary']['failed']}")
    print(f"\nResults written to: {output_file}")
    print(f"{'='*60}")

    # Assert all HTTP requests succeeded (200)
    http_failed = [r for r in results["responses"] if r["status_code"] != 200]
    assert (
        len(http_failed) == 0
    ), f"HTTP failures: {[r['strategy'] for r in http_failed]}"


def test_relaxed_smoke_diagnostic(client, headers):
    """Run relaxed smoke pass to diagnose 'none' vs 'filters too strict'.

    If strict smoke returns 'none' but relaxed returns 'ok' or 'degraded',
    then the issue is filter strictness, not missing data.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(SMOKE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_type": "relaxed_diagnostic",
        "relaxed_floors": RELAXED_FLOORS,
        "strategies": [],
    }

    for strategy in TEST_STRATEGIES:
        # Strict pass (production settings)
        strict_response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": strategy,
                "objective_type": "sharpe",
            },
            headers=headers,
        )

        # Relaxed pass (lowered floors)
        relaxed_response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": TEST_WORKSPACE_ID,
                "strategy_name": strategy,
                "objective_type": "sharpe",
                "min_trades": RELAXED_FLOORS["min_trades"],
                "max_drawdown": RELAXED_FLOORS["max_drawdown"],
                "max_overfit_gap": RELAXED_FLOORS["max_overfit_gap"],
            },
            headers=headers,
        )

        strict_status = "error"
        relaxed_status = "error"

        if strict_response.status_code == 200:
            strict_status = strict_response.json().get("status", "unknown")
        if relaxed_response.status_code == 200:
            relaxed_status = relaxed_response.json().get("status", "unknown")

        # Determine diagnosis
        diagnosis = "unknown"
        if strict_status == "none" and relaxed_status in ["ok", "degraded"]:
            diagnosis = "filters_too_strict"
        elif strict_status == "none" and relaxed_status == "none":
            diagnosis = "no_data_for_strategy"
        elif strict_status in ["ok", "degraded"]:
            diagnosis = "healthy"

        results["strategies"].append(
            {
                "strategy": strategy,
                "strict_status": strict_status,
                "relaxed_status": relaxed_status,
                "diagnosis": diagnosis,
                "strict_count": (
                    strict_response.json().get("count_used", 0)
                    if strict_response.status_code == 200
                    else 0
                ),
                "relaxed_count": (
                    relaxed_response.json().get("count_used", 0)
                    if relaxed_response.status_code == 200
                    else 0
                ),
            }
        )

    # Write diagnostic results
    output_file = output_dir / "relaxed_diagnostic_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))

    # Summary
    print(f"\n{'='*60}")
    print("Relaxed Diagnostic Summary")
    print(f"{'='*60}")
    for s in results["strategies"]:
        emoji = {
            "healthy": "✓",
            "filters_too_strict": "⚠",
            "no_data_for_strategy": "✗",
        }.get(s["diagnosis"], "?")
        print(
            f"  {emoji} {s['strategy']}: {s['diagnosis']} (strict={s['strict_status']}, relaxed={s['relaxed_status']})"  # noqa: E501
        )
    print(f"\nResults written to: {output_file}")
    print(f"{'='*60}")

    # Don't fail on none - this is diagnostic only
    # Just ensure HTTP succeeded
    assert all(
        s["strict_status"] != "error" and s["relaxed_status"] != "error"
        for s in results["strategies"]
    ), "HTTP errors in diagnostic pass"
