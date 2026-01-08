"""Happy path integration tests for backtest parameter tuning.

Uses synthetic OHLCV fixture that reliably generates trades across strategy types.
Tests the full tuning pipeline: tune creation, trial execution, results.

Run with: pytest tests/integration/test_backtest_tuner_happy_path.py -v
Requires: Real database connection with test strategy entity
"""

import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Fixture path
SYNTH_FIXTURE = Path(__file__).parent.parent / "unit" / "fixtures" / "ohlcv_synth_trendy_range.csv"

# Test strategy with tunable params (period: integer 1-200)
# This must exist in your test database
TEST_STRATEGY_ENTITY_ID = "8fd7589a-97c6-49bf-a65e-357fb063fe33"
TEST_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


# Skip entire module if DATABASE_URL not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DATABASE_URL") and not os.getenv("SUPABASE_URL"),
        reason="Requires DATABASE_URL or SUPABASE_URL"
    ),
]


@pytest.mark.requires_db
class TestTunerHappyPath:
    """Integration tests for tuner with synthetic data that generates trades."""

    @pytest.fixture
    def client(self):
        """Create test client with real database connection."""
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    @pytest.fixture
    def synth_csv(self):
        """Load synthetic OHLCV fixture."""
        assert SYNTH_FIXTURE.exists(), f"Fixture not found: {SYNTH_FIXTURE}"
        return SYNTH_FIXTURE

    def test_tune_produces_completed_trials(self, client, synth_csv):
        """Tuning with synthetic data should produce completed trials with scores."""
        # Arrange
        with open(synth_csv, "rb") as f:
            resp = client.post(
                "/backtests/tune",
                files={"file": ("ohlcv_synth.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": TEST_STRATEGY_ENTITY_ID,
                    "workspace_id": TEST_WORKSPACE_ID,
                    "n_trials": 5,
                    "min_trades": 1,
                },
            )

        # Act
        assert resp.status_code == 200, f"Tune failed: {resp.text}"
        result = resp.json()

        # Assert: trials completed
        assert result["status"] == "completed"
        assert result["trials_completed"] >= 1, "Expected at least 1 completed trial"

        # Assert: best results populated
        assert result["best_score"] is not None, "Expected best_score when trials complete"
        assert result["best_params"] is not None, "Expected best_params when trials complete"
        assert result["best_run_id"] is not None, "Expected best_run_id when trials complete"

        # Assert: leaderboard has entries
        assert len(result["leaderboard"]) >= 1, "Expected leaderboard entries"

    def test_tune_detail_has_status_counts(self, client, synth_csv):
        """GET /tunes/{id} should include status counts."""
        # Create a tune
        with open(synth_csv, "rb") as f:
            create_resp = client.post(
                "/backtests/tune",
                files={"file": ("ohlcv_synth.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": TEST_STRATEGY_ENTITY_ID,
                    "workspace_id": TEST_WORKSPACE_ID,
                    "n_trials": 3,
                    "min_trades": 1,
                },
            )

        assert create_resp.status_code == 200
        tune_id = create_resp.json()["tune_id"]

        # Fetch details
        detail_resp = client.get(f"/backtests/tunes/{tune_id}")
        assert detail_resp.status_code == 200

        detail = detail_resp.json()

        # Assert: counts present and valid
        counts = detail.get("counts")
        assert counts is not None, "Expected counts in tune detail"
        assert "completed" in counts
        assert "skipped" in counts
        assert "failed" in counts

        # At least some trials should be completed (not all skipped)
        assert counts["completed"] >= 1, f"Expected completed trials, got: {counts}"

    def test_tune_runs_have_scores(self, client, synth_csv):
        """GET /tunes/{id}/runs should show completed trials with scores."""
        # Create tune
        with open(synth_csv, "rb") as f:
            create_resp = client.post(
                "/backtests/tune",
                files={"file": ("ohlcv_synth.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": TEST_STRATEGY_ENTITY_ID,
                    "workspace_id": TEST_WORKSPACE_ID,
                    "n_trials": 5,
                    "min_trades": 1,
                },
            )

        assert create_resp.status_code == 200
        tune_id = create_resp.json()["tune_id"]

        # Fetch runs
        runs_resp = client.get(f"/backtests/tunes/{tune_id}/runs")
        assert runs_resp.status_code == 200

        runs = runs_resp.json()
        items = runs["items"]

        # Assert: runs returned
        assert len(items) >= 1, "Expected tune runs"

        # Assert: at least one completed run with score
        completed_runs = [r for r in items if r["status"] == "completed"]
        assert len(completed_runs) >= 1, "Expected at least 1 completed run"

        # Completed runs should have scores
        for run in completed_runs:
            assert run["score"] is not None, f"Completed run missing score: {run}"
            assert run["run_id"] is not None, f"Completed run missing run_id: {run}"

    def test_skipped_runs_have_skip_reason(self, client):
        """Skipped trials should have skip_reason populated."""
        # Use the tiny fixture that doesn't generate trades
        tiny_fixture = Path(__file__).parent.parent / "unit" / "fixtures" / "valid_ohlcv.csv"

        with open(tiny_fixture, "rb") as f:
            create_resp = client.post(
                "/backtests/tune",
                files={"file": ("tiny.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": TEST_STRATEGY_ENTITY_ID,
                    "workspace_id": TEST_WORKSPACE_ID,
                    "n_trials": 3,
                    "min_trades": 5,  # High threshold to force skips
                },
            )

        assert create_resp.status_code == 200
        tune_id = create_resp.json()["tune_id"]

        # Fetch runs
        runs_resp = client.get(f"/backtests/tunes/{tune_id}/runs?status=skipped")
        runs = runs_resp.json()

        if runs["total"] > 0:
            # Skipped runs should have skip_reason
            for run in runs["items"]:
                assert run["status"] == "skipped"
                assert run["skip_reason"] is not None, f"Skipped run missing skip_reason: {run}"
                assert "min_trades" in run["skip_reason"].lower(), f"Expected min_trades in reason: {run}"

    def test_persistence_integrity(self, client, synth_csv):
        """Verify data integrity: completed trials have run_id, FK valid."""
        # Create tune
        with open(synth_csv, "rb") as f:
            create_resp = client.post(
                "/backtests/tune",
                files={"file": ("ohlcv_synth.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": TEST_STRATEGY_ENTITY_ID,
                    "workspace_id": TEST_WORKSPACE_ID,
                    "n_trials": 5,
                    "min_trades": 1,
                },
            )

        assert create_resp.status_code == 200
        tune_id = create_resp.json()["tune_id"]

        # Fetch all runs
        runs_resp = client.get(f"/backtests/tunes/{tune_id}/runs?limit=100")
        assert runs_resp.status_code == 200
        runs = runs_resp.json()["items"]

        # Track seen trial indices for uniqueness check
        seen_indices = set()

        for run in runs:
            trial_index = run["trial_index"]

            # Check: no duplicate trial_index (PK integrity)
            assert trial_index not in seen_indices, f"Duplicate trial_index: {trial_index}"
            seen_indices.add(trial_index)

            # Check: completed trials MUST have run_id
            if run["status"] == "completed":
                assert run["run_id"] is not None, f"Completed trial {trial_index} missing run_id"

                # Verify run_id exists in backtest_runs (FK integrity)
                run_resp = client.get(f"/backtests/{run['run_id']}")
                assert run_resp.status_code == 200, f"run_id {run['run_id']} not found in backtest_runs"

        # Verify tune has best_* persisted (not just derived)
        detail_resp = client.get(f"/backtests/tunes/{tune_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()

        # If any completed trials, best_* should be set
        if detail["counts"]["completed"] > 0:
            assert detail["best_score"] is not None, "best_score not persisted"
            assert detail["best_params"] is not None, "best_params not persisted"
            assert detail["best_run_id"] is not None, "best_run_id not persisted"
