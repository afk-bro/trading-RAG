"""Unit tests for strategy intel API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps.security import require_admin_token
from app.repositories.strategy_intel import IntelSnapshot
from app.repositories.strategy_versions import StrategyVersion


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return MagicMock()


@pytest.fixture
def sample_strategy():
    """Sample strategy dict as returned by repository."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "name": "Test Strategy",
        "slug": "test-strategy",
        "description": "A test strategy",
        "engine": "pine",
        "source_ref": {},
        "status": "draft",
        "review_status": "unreviewed",
        "risk_level": None,
        "tags": {},
        "backtest_summary": None,
        "strategy_entity_id": uuid4(),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_version(sample_strategy):
    """Sample StrategyVersion dataclass."""
    return StrategyVersion(
        id=uuid4(),
        strategy_id=sample_strategy["id"],
        strategy_entity_id=sample_strategy["strategy_entity_id"],
        version_number=1,
        version_tag="v1.0",
        config_snapshot={"param": 10},
        config_hash="a" * 64,
        state="active",
        regime_awareness={},
        created_at=datetime.now(timezone.utc),
        created_by="admin:test",
        activated_at=datetime.now(timezone.utc),
        paused_at=None,
        retired_at=None,
        kb_strategy_spec_id=None,
    )


@pytest.fixture
def sample_snapshot(sample_strategy, sample_version):
    """Sample IntelSnapshot dataclass."""
    return IntelSnapshot(
        id=uuid4(),
        workspace_id=sample_strategy["workspace_id"],
        strategy_version_id=sample_version.id,
        as_of_ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        computed_at=datetime(2024, 1, 15, 12, 0, 5, tzinfo=timezone.utc),
        regime="trend-up|volatility-normal",
        confidence_score=0.75,
        confidence_components={
            "performance": 0.8,
            "drawdown": 0.7,
            "stability": 0.85,
            "data_freshness": 0.6,
            "regime_fit": 0.8,
        },
        features={
            "atr_percentile": 0.45,
            "trend_strength": 0.7,
        },
        explain={
            "regime_reason": "ADX=32 > 25, trending",
            "confidence_reason": "Strong backtest performance",
        },
        engine_version="intel_runner_v0.1",
        inputs_hash="b" * 64,
        run_id=None,
    )


def override_admin_token():
    """Override for require_admin_token dependency."""
    return True


@pytest.fixture
def client(mock_db_pool):
    """Create test client with mocked dependencies."""
    from app.routers import strategies

    app = FastAPI()
    app.include_router(strategies.router, prefix="/strategies")

    # Override the admin token dependency
    app.dependency_overrides[require_admin_token] = override_admin_token

    # Patch the pool
    with patch.object(strategies, "_get_pool", return_value=mock_db_pool):
        yield TestClient(app)

    # Clean up
    app.dependency_overrides.clear()


# =============================================================================
# Get Latest Intel Tests
# =============================================================================


class TestGetLatestIntel:
    """Tests for GET /strategies/{id}/versions/{vid}/intel/latest."""

    def test_get_latest_intel_success(
        self, client, mock_db_pool, sample_strategy, sample_version, sample_snapshot
    ):
        """Should return the most recent intel snapshot."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_intel_repo = AsyncMock()
        mock_intel_repo.get_latest_snapshot.return_value = sample_snapshot

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.StrategyIntelRepository",
                    return_value=mock_intel_repo,
                ):
                    response = client.get(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel/latest",
                        params={"workspace_id": str(sample_strategy["workspace_id"])},
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["regime"] == "trend-up|volatility-normal"
        assert data["confidence_score"] == 0.75
        assert "confidence_components" in data
        assert data["confidence_components"]["performance"] == 0.8
        assert data["engine_version"] == "intel_runner_v0.1"

    def test_get_latest_intel_strategy_not_found(self, client, mock_db_pool, sample_strategy):
        """Should return 404 if strategy not found."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.get(
                f"/strategies/{uuid4()}/versions/{uuid4()}/intel/latest",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert "strategy" in response.json()["detail"].lower()

    def test_get_latest_intel_version_not_found(
        self, client, mock_db_pool, sample_strategy
    ):
        """Should return 404 if version not found."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.get(
                    f"/strategies/{sample_strategy['id']}/versions/{uuid4()}/intel/latest",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 404
        assert "version" in response.json()["detail"].lower()

    def test_get_latest_intel_no_snapshots(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return 404 if no intel snapshots exist."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_intel_repo = AsyncMock()
        mock_intel_repo.get_latest_snapshot.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.StrategyIntelRepository",
                    return_value=mock_intel_repo,
                ):
                    response = client.get(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel/latest",
                        params={"workspace_id": str(sample_strategy["workspace_id"])},
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 404
        assert "intel snapshot" in response.json()["detail"].lower()


# =============================================================================
# List Intel Snapshots Tests
# =============================================================================


class TestListIntelSnapshots:
    """Tests for GET /strategies/{id}/versions/{vid}/intel."""

    def test_list_intel_snapshots_success(
        self, client, mock_db_pool, sample_strategy, sample_version, sample_snapshot
    ):
        """Should return paginated list of intel snapshots."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_intel_repo = AsyncMock()
        mock_intel_repo.list_snapshots.return_value = [sample_snapshot]

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.StrategyIntelRepository",
                    return_value=mock_intel_repo,
                ):
                    response = client.get(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel",
                        params={"workspace_id": str(sample_strategy["workspace_id"])},
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["regime"] == "trend-up|volatility-normal"
        assert data["items"][0]["confidence_score"] == 0.75
        assert data["next_cursor"] is None  # No more pages

    def test_list_intel_snapshots_with_pagination(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should handle pagination correctly."""
        # Create multiple snapshots
        snapshots = []
        for i in range(3):
            snapshots.append(
                IntelSnapshot(
                    id=uuid4(),
                    workspace_id=sample_strategy["workspace_id"],
                    strategy_version_id=sample_version.id,
                    as_of_ts=datetime(2024, 1, 15, 12, i, 0, tzinfo=timezone.utc),
                    computed_at=datetime(2024, 1, 15, 12, i, 5, tzinfo=timezone.utc),
                    regime=f"regime-{i}",
                    confidence_score=0.7 + i * 0.1,
                    confidence_components={},
                    features={},
                    explain={},
                    engine_version="intel_runner_v0.1",
                    inputs_hash="c" * 64,
                    run_id=None,
                )
            )

        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_intel_repo = AsyncMock()
        # Return 3 items when requesting 2+1 (to check for more)
        mock_intel_repo.list_snapshots.return_value = snapshots

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.StrategyIntelRepository",
                    return_value=mock_intel_repo,
                ):
                    response = client.get(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel",
                        params={
                            "workspace_id": str(sample_strategy["workspace_id"]),
                            "limit": 2,
                        },
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2  # Trimmed to limit
        assert data["next_cursor"] is not None  # More pages available

    def test_list_intel_snapshots_empty(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return empty list when no snapshots exist."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_intel_repo = AsyncMock()
        mock_intel_repo.list_snapshots.return_value = []

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.StrategyIntelRepository",
                    return_value=mock_intel_repo,
                ):
                    response = client.get(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel",
                        params={"workspace_id": str(sample_strategy["workspace_id"])},
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0
        assert data["next_cursor"] is None


# =============================================================================
# Recompute Intel Tests
# =============================================================================


class TestRecomputeIntel:
    """Tests for POST /strategies/{id}/versions/{vid}/intel/recompute."""

    def test_recompute_intel_success(
        self, client, mock_db_pool, sample_strategy, sample_version, sample_snapshot
    ):
        """Should trigger recomputation and return new snapshot."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_runner = AsyncMock()
        mock_runner.run_for_version.return_value = sample_snapshot

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.IntelRunner",
                    return_value=mock_runner,
                ):
                    response = client.post(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel/recompute",
                        params={"workspace_id": str(sample_strategy["workspace_id"])},
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["regime"] == "trend-up|volatility-normal"
        assert data["confidence_score"] == 0.75

        # Verify runner was called with correct args
        mock_runner.run_for_version.assert_called_once()
        call_kwargs = mock_runner.run_for_version.call_args.kwargs
        assert call_kwargs["version_id"] == sample_version.id
        assert call_kwargs["workspace_id"] == sample_strategy["workspace_id"]
        assert call_kwargs["force"] is False

    def test_recompute_intel_with_force(
        self, client, mock_db_pool, sample_strategy, sample_version, sample_snapshot
    ):
        """Should pass force flag to runner."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_runner = AsyncMock()
        mock_runner.run_for_version.return_value = sample_snapshot

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.IntelRunner",
                    return_value=mock_runner,
                ):
                    response = client.post(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel/recompute",
                        params={
                            "workspace_id": str(sample_strategy["workspace_id"]),
                            "force": True,
                        },
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 200

        # Verify force was passed
        call_kwargs = mock_runner.run_for_version.call_args.kwargs
        assert call_kwargs["force"] is True

    def test_recompute_intel_dedupe_returns_existing(
        self, client, mock_db_pool, sample_strategy, sample_version, sample_snapshot
    ):
        """Should return existing snapshot when dedupe kicks in."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_runner = AsyncMock()
        mock_runner.run_for_version.return_value = None  # Dedupe

        mock_intel_repo = AsyncMock()
        mock_intel_repo.get_latest_snapshot.return_value = sample_snapshot

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.IntelRunner",
                    return_value=mock_runner,
                ):
                    with patch(
                        "app.routers.strategies.StrategyIntelRepository",
                        return_value=mock_intel_repo,
                    ):
                        response = client.post(
                            f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel/recompute",
                            params={"workspace_id": str(sample_strategy["workspace_id"])},
                            headers={"X-Admin-Token": "test-token"},
                        )

        assert response.status_code == 200
        data = response.json()
        assert data["regime"] == "trend-up|volatility-normal"

    def test_recompute_intel_runner_error(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return 500 on runner error."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        mock_runner = AsyncMock()
        mock_runner.run_for_version.side_effect = Exception("DB connection failed")

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                with patch(
                    "app.routers.strategies.IntelRunner",
                    return_value=mock_runner,
                ):
                    response = client.post(
                        f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/intel/recompute",
                        params={"workspace_id": str(sample_strategy["workspace_id"])},
                        headers={"X-Admin-Token": "test-token"},
                    )

        assert response.status_code == 500
        assert "computation failed" in response.json()["detail"].lower()

    def test_recompute_intel_strategy_not_found(self, client, mock_db_pool, sample_strategy):
        """Should return 404 if strategy not found."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.post(
                f"/strategies/{uuid4()}/versions/{uuid4()}/intel/recompute",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404

    def test_recompute_intel_version_not_found(
        self, client, mock_db_pool, sample_strategy
    ):
        """Should return 404 if version not found."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions/{uuid4()}/intel/recompute",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 404
