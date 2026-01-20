"""Unit tests for strategy version API endpoints."""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps.security import require_admin_token
from app.repositories.strategy_versions import StrategyVersion, VersionTransition


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
        state="draft",
        regime_awareness={},
        created_at=datetime.now(timezone.utc),
        created_by="admin:test",
        activated_at=None,
        paused_at=None,
        retired_at=None,
        kb_strategy_spec_id=None,
    )


@pytest.fixture
def sample_transition(sample_version):
    """Sample VersionTransition dataclass."""
    return VersionTransition(
        id=uuid4(),
        version_id=sample_version.id,
        from_state=None,
        to_state="draft",
        triggered_by="admin:test",
        triggered_at=datetime.now(timezone.utc),
        reason="Initial creation",
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
# Create Version Tests
# =============================================================================


class TestCreateVersion:
    """Tests for POST /strategies/{id}/versions."""

    def test_create_version_success(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should create version and return 201."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.create_version.return_value = sample_version

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    json={
                        "config_snapshot": {"param": 10},
                        "version_tag": "v1.0",
                        "created_by": "admin:test",
                    },
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 201
        data = response.json()
        assert data["version_number"] == 1
        assert data["state"] == "draft"
        assert data["config_snapshot"] == {"param": 10}

    def test_create_version_strategy_not_found(
        self, client, mock_db_pool, sample_strategy
    ):
        """Should return 404 if strategy not found."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.post(
                f"/strategies/{uuid4()}/versions",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                json={"config_snapshot": {"param": 10}},
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_version_no_entity_id(self, client, mock_db_pool, sample_strategy):
        """Should return 400 if strategy has no entity_id mapping."""
        strategy_without_entity = {**sample_strategy, "strategy_entity_id": None}
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = strategy_without_entity

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.post(
                f"/strategies/{sample_strategy['id']}/versions",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                json={"config_snapshot": {"param": 10}},
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 400
        assert "no entity_id mapping" in response.json()["detail"].lower()


# =============================================================================
# List Versions Tests
# =============================================================================


class TestListVersions:
    """Tests for GET /strategies/{id}/versions."""

    def test_list_versions_success(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should list versions with pagination."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.list_versions.return_value = ([sample_version], 1)

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.get(
                    f"/strategies/{sample_strategy['id']}/versions",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["version_number"] == 1
        # Config hash should be truncated to 16 chars in list view
        assert len(data["items"][0]["config_hash"]) == 16

    def test_list_versions_strategy_not_found(
        self, client, mock_db_pool, sample_strategy
    ):
        """Should return 404 if strategy not found."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = None

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.get(
                f"/strategies/{uuid4()}/versions",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404


# =============================================================================
# Get Version Tests
# =============================================================================


class TestGetVersion:
    """Tests for GET /strategies/{id}/versions/{vid}."""

    def test_get_version_success(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return version details."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.get(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["version_number"] == 1
        assert data["config_snapshot"] == {"param": 10}
        # Full config hash in detail view
        assert len(data["config_hash"]) == 64

    def test_get_version_not_found(self, client, mock_db_pool, sample_strategy):
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
                    f"/strategies/{sample_strategy['id']}/versions/{uuid4()}",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 404


# =============================================================================
# Activate Version Tests
# =============================================================================


class TestActivateVersion:
    """Tests for POST /strategies/{id}/versions/{vid}/activate."""

    def test_activate_version_success(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should activate version and return updated state."""
        activated_version = StrategyVersion(
            id=sample_version.id,
            strategy_id=sample_version.strategy_id,
            strategy_entity_id=sample_version.strategy_entity_id,
            version_number=sample_version.version_number,
            version_tag=sample_version.version_tag,
            config_snapshot=sample_version.config_snapshot,
            config_hash=sample_version.config_hash,
            state="active",
            regime_awareness=sample_version.regime_awareness,
            created_at=sample_version.created_at,
            created_by=sample_version.created_by,
            activated_at=datetime.now(timezone.utc),
            paused_at=None,
            retired_at=None,
            kb_strategy_spec_id=None,
        )

        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.activate.return_value = activated_version

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/activate",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    json={
                        "triggered_by": "admin:deploy",
                        "reason": "Initial deployment",
                    },
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "active"
        assert data["activated_at"] is not None

    def test_activate_version_invalid_state(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return 400 for invalid state transition."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.activate.side_effect = ValueError(
            "Cannot activate from state 'retired'"
        )

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/activate",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    json={"triggered_by": "admin:deploy"},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 400
        assert "retired" in response.json()["detail"].lower()


# =============================================================================
# Pause Version Tests
# =============================================================================


class TestPauseVersion:
    """Tests for POST /strategies/{id}/versions/{vid}/pause."""

    def test_pause_version_success(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should pause version and return updated state."""
        paused_version = StrategyVersion(
            id=sample_version.id,
            strategy_id=sample_version.strategy_id,
            strategy_entity_id=sample_version.strategy_entity_id,
            version_number=sample_version.version_number,
            version_tag=sample_version.version_tag,
            config_snapshot=sample_version.config_snapshot,
            config_hash=sample_version.config_hash,
            state="paused",
            regime_awareness=sample_version.regime_awareness,
            created_at=sample_version.created_at,
            created_by=sample_version.created_by,
            activated_at=sample_version.created_at,
            paused_at=datetime.now(timezone.utc),
            retired_at=None,
            kb_strategy_spec_id=None,
        )

        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.pause.return_value = paused_version

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/pause",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    json={
                        "triggered_by": "admin:emergency",
                        "reason": "Market volatility",
                    },
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "paused"
        assert data["paused_at"] is not None

    def test_pause_version_invalid_state(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return 400 for invalid state transition."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.pause.side_effect = ValueError(
            "Cannot pause from state 'draft'"
        )

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/pause",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    json={"triggered_by": "admin:emergency"},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 400


# =============================================================================
# Retire Version Tests
# =============================================================================


class TestRetireVersion:
    """Tests for POST /strategies/{id}/versions/{vid}/retire."""

    def test_retire_version_success(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should retire version and return updated state."""
        retired_version = StrategyVersion(
            id=sample_version.id,
            strategy_id=sample_version.strategy_id,
            strategy_entity_id=sample_version.strategy_entity_id,
            version_number=sample_version.version_number,
            version_tag=sample_version.version_tag,
            config_snapshot=sample_version.config_snapshot,
            config_hash=sample_version.config_hash,
            state="retired",
            regime_awareness=sample_version.regime_awareness,
            created_at=sample_version.created_at,
            created_by=sample_version.created_by,
            activated_at=None,
            paused_at=None,
            retired_at=datetime.now(timezone.utc),
            kb_strategy_spec_id=None,
        )

        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.retire.return_value = retired_version

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.post(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/retire",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    json={"triggered_by": "admin:cleanup", "reason": "Obsolete config"},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "retired"
        assert data["retired_at"] is not None


# =============================================================================
# Get Transitions Tests
# =============================================================================


class TestGetTransitions:
    """Tests for GET /strategies/{id}/versions/{vid}/transitions."""

    def test_get_transitions_success(
        self, client, mock_db_pool, sample_strategy, sample_version, sample_transition
    ):
        """Should return transition history."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        mock_version_repo = AsyncMock()
        mock_version_repo.get_version.return_value = sample_version
        mock_version_repo.get_transitions.return_value = [sample_transition]

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            with patch(
                "app.routers.strategies.StrategyVersionsRepository",
                return_value=mock_version_repo,
            ):
                response = client.get(
                    f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/transitions",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["to_state"] == "draft"
        assert data[0]["triggered_by"] == "admin:test"

    def test_get_transitions_version_not_found(
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
                    f"/strategies/{sample_strategy['id']}/versions/{uuid4()}/transitions",
                    params={"workspace_id": str(sample_strategy["workspace_id"])},
                    headers={"X-Admin-Token": "test-token"},
                )

        assert response.status_code == 404


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemaValidation:
    """Tests for request schema validation."""

    def test_create_version_missing_config(self, client, mock_db_pool, sample_strategy):
        """Should return 422 if config_snapshot is missing."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.post(
                f"/strategies/{sample_strategy['id']}/versions",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                json={},  # Missing config_snapshot
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 422

    def test_transition_missing_triggered_by(
        self, client, mock_db_pool, sample_strategy, sample_version
    ):
        """Should return 422 if triggered_by is missing."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.post(
                f"/strategies/{sample_strategy['id']}/versions/{sample_version.id}/activate",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                json={},  # Missing triggered_by
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 422

    def test_version_tag_too_long(self, client, mock_db_pool, sample_strategy):
        """Should return 422 if version_tag exceeds max length."""
        mock_strategy_repo = AsyncMock()
        mock_strategy_repo.get_by_id.return_value = sample_strategy

        with patch(
            "app.routers.strategies.StrategyRepository",
            return_value=mock_strategy_repo,
        ):
            response = client.post(
                f"/strategies/{sample_strategy['id']}/versions",
                params={"workspace_id": str(sample_strategy["workspace_id"])},
                json={
                    "config_snapshot": {"param": 10},
                    "version_tag": "x" * 51,  # > 50 chars
                },
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 422
