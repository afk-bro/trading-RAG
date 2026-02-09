"""
E2E smoke tests for the Coaching API endpoints.

Verifies:
 - coaching endpoint doesn't crash (no 500s) with real or missing data
 - response shape matches the CoachingData contract
 - lineage endpoint returns correct structure
 - backward compat: include_coaching=false omits coaching key

Run with:
    pytest tests/e2e/test_coaching_api.py --base-url http://localhost:8000
"""

import pytest

pytestmark = pytest.mark.e2e

FAKE_WS = "00000000-0000-0000-0000-000000000001"
FAKE_RUN = "00000000-0000-0000-0000-000000000002"


class TestCoachingEndpointContract:
    """Verify coaching API returns correct shape or graceful errors."""

    def test_coaching_no_500_on_missing_run(self, api_request):
        """Coaching endpoint returns 404, not 500, for missing run."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}"
            f"?include_coaching=true"
        )
        assert resp.status in (200, 404), f"Unexpected {resp.status}: {resp.text()}"

    def test_coaching_false_omits_key(self, api_request):
        """include_coaching=false should NOT add coaching key."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}"
            f"?include_coaching=false"
        )
        if resp.status == 200:
            body = resp.json()
            assert "coaching" not in body, "coaching key present when include_coaching=false"

    def test_coaching_response_shape(self, api_request):
        """When coaching is present, verify it has the expected keys."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}"
            f"?include_coaching=true"
        )
        if resp.status != 200:
            pytest.skip("No completed run available to test coaching shape")

        body = resp.json()
        if "coaching" not in body:
            pytest.skip("Run not completed, coaching not returned")

        coaching = body["coaching"]

        # Lineage is always present
        assert "lineage" in coaching
        lineage = coaching["lineage"]
        assert "previous_run_id" in lineage
        assert "deltas" in lineage
        assert "params_changed" in lineage
        assert "param_diffs" in lineage
        assert "comparison_warnings" in lineage

        # Process score
        assert "process_score" in coaching
        ps = coaching["process_score"]
        assert "total" in ps
        assert "grade" in ps
        assert "components" in ps

        # Loss attribution
        assert "loss_attribution" in coaching

        # coaching_partial flag
        assert "coaching_partial" in coaching
        assert isinstance(coaching["coaching_partial"], bool)

    def test_coaching_delta_shape(self, api_request):
        """Each delta has metric, current, previous, delta, improved, higher_is_better."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}"
            f"?include_coaching=true"
        )
        if resp.status != 200:
            pytest.skip("No run available")

        body = resp.json()
        if "coaching" not in body:
            pytest.skip("No coaching data")

        for d in body["coaching"]["lineage"]["deltas"]:
            assert "metric" in d
            assert "current" in d
            assert "previous" in d
            assert "delta" in d
            assert "improved" in d
            assert "higher_is_better" in d
            assert isinstance(d["higher_is_better"], bool)

    def test_trajectory_shape(self, api_request):
        """Trajectory data returns runs list when coaching enabled."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}"
            f"?include_coaching=true"
        )
        if resp.status != 200:
            pytest.skip("No run available")

        body = resp.json()
        if "trajectory" not in body:
            pytest.skip("No trajectory data")

        assert "runs" in body["trajectory"]
        assert isinstance(body["trajectory"]["runs"], list)


class TestLineageEndpoint:
    """Verify lineage candidates endpoint."""

    def test_lineage_no_500(self, api_request):
        """Lineage endpoint returns 404 or valid data, not 500."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}/lineage"
        )
        assert resp.status in (200, 404), f"Unexpected {resp.status}: {resp.text()}"

    def test_lineage_response_shape(self, api_request):
        """Lineage candidates have expected structure."""
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}/lineage"
        )
        if resp.status != 200:
            pytest.skip("No lineage data available")

        body = resp.json()
        assert "candidates" in body
        assert isinstance(body["candidates"], list)

        for c in body["candidates"]:
            assert "run_id" in c
            assert "completed_at" in c
            assert "is_auto_baseline" in c


class TestBaselineOverride:
    """Verify baseline_run_id override param."""

    def test_baseline_override_no_500(self, api_request):
        """Baseline override param doesn't crash the endpoint."""
        fake_baseline = "00000000-0000-0000-0000-000000000003"
        resp = api_request.get(
            f"/dashboards/{FAKE_WS}/backtests/{FAKE_RUN}"
            f"?include_coaching=true&baseline_run_id={fake_baseline}"
        )
        assert resp.status in (200, 404), f"Unexpected {resp.status}: {resp.text()}"
