"""Unit tests for KB trials API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return MagicMock()


@pytest.fixture
def mock_recommender():
    """Create mock recommender with default response."""
    from app.services.kb.recommend import RecommendResponse, TrialSummary

    mock = AsyncMock()
    mock.recommend.return_value = RecommendResponse(
        params={"period": 20, "threshold": 2.0},
        status="ok",
        confidence=0.85,
        top_trials=[
            TrialSummary(
                point_id="test_1",
                strategy_name="mean_reversion",  # Valid strategy from registry
                objective_score=1.5,
                similarity_score=0.9,
                jaccard_score=0.8,
                rerank_score=0.85,
                params={"period": 20, "threshold": 2.0},
            )
        ],
        count_used=15,
        warnings=[],
        reasons=[],
        suggested_actions=[],
        retrieval_strict_count=15,
        retrieval_relaxed_count=0,
        used_relaxed_filters=False,
        used_metadata_fallback=False,
        query_regime_tags=["uptrend", "low_vol"],
        collection_name="trading_kb_trials__nomic-embed-text__768",
        embedding_model="nomic-embed-text",
    )
    return mock


@pytest.fixture
def client(mock_db_pool):
    """Create test client with mocked dependencies."""
    from app.routers import kb_trials
    from fastapi import FastAPI

    # Set up mock db pool
    kb_trials.set_db_pool(mock_db_pool)

    # Create test app with just this router
    app = FastAPI()
    app.include_router(kb_trials.router)

    return TestClient(app)


# =============================================================================
# Validation Tests
# =============================================================================


class TestRecommendValidation:
    """Tests for request validation."""

    def test_invalid_strategy_returns_400(self, client):
        """Should return 400 with valid strategy list for unknown strategy."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "nonexistent_strategy",
                "objective_type": "sharpe",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "INVALID_STRATEGY" in str(data)
        assert "valid_options" in str(data) or "strategies" in str(data)

    def test_invalid_objective_returns_400(self, client, mock_recommender):
        """Should return 400 for invalid objective type."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy from registry
                    "objective_type": "invalid_objective",
                },
            )

            assert response.status_code == 400
            data = response.json()
            assert "INVALID_OBJECTIVE" in str(data)

    def test_both_ohlcv_source_and_regime_tags_allowed(self, client, mock_recommender):
        """Should allow dataset_id with regime_tags (they serve different purposes)."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            # dataset_id references historical data
            # regime_tags override computed regime
            # Both can be provided together
            response = client.post(
                "/kb/trials/recommend",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                    "dataset_id": "test_dataset",
                    "regime_tags": ["uptrend", "high_vol"],
                },
            )

            # Should succeed (or at least not fail validation)
            assert response.status_code == 200

    def test_retrieve_k_too_large_returns_422(self, client):
        """Should return 422 for retrieve_k > MAX_RETRIEVE_K."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "mean_reversion",  # Valid strategy
                "objective_type": "sharpe",
                "retrieve_k": 1000,  # > MAX_RETRIEVE_K (500)
            },
        )

        assert response.status_code == 422

    def test_rerank_keep_too_large_returns_422(self, client):
        """Should return 422 for rerank_keep > MAX_RERANK_KEEP."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "mean_reversion",  # Valid strategy
                "objective_type": "sharpe",
                "rerank_keep": 500,  # > MAX_RERANK_KEEP (200)
            },
        )

        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self, client):
        """Should return 422 for top_k > MAX_TOP_K."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "mean_reversion",  # Valid strategy
                "objective_type": "sharpe",
                "top_k": 100,  # > MAX_TOP_K (50)
            },
        )

        assert response.status_code == 422


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestRecommendHappyPath:
    """Tests for successful recommendation requests."""

    def test_recommend_returns_200_with_valid_schema(self, client, mock_recommender):
        """Should return 200 with valid response schema."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Check required fields
            assert "request_id" in data
            assert "params" in data
            assert "status" in data
            assert "active_collection" in data
            assert "embedding_model_id" in data

            # Check status is valid enum
            assert data["status"] in ["ok", "degraded", "none"]

    def test_recommend_includes_request_id(self, client, mock_recommender):
        """Should include unique request_id in response."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                },
            )

            data = response.json()
            assert "request_id" in data
            # Should be valid UUID
            import uuid
            uuid.UUID(data["request_id"])


# =============================================================================
# Debug Mode Tests
# =============================================================================


class TestDebugMode:
    """Tests for debug mode."""

    def test_debug_mode_returns_candidates(self, client, mock_recommender):
        """Debug mode should return top candidates."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend?mode=debug",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Debug mode should include candidates
            assert "top_candidates" in data
            assert data["top_candidates"] is not None

    def test_debug_mode_no_params(self, client, mock_recommender):
        """Debug mode should not include aggregated params."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend?mode=debug",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                },
            )

            data = response.json()
            # In debug mode, params should be empty
            assert data["params"] == {}

    def test_include_candidates_flag(self, client, mock_recommender):
        """include_candidates=true should include candidates in full mode."""
        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                    "include_candidates": True,
                },
            )

            data = response.json()
            assert "top_candidates" in data
            assert data["top_candidates"] is not None
            # But params should still be present
            assert "params" in data


# =============================================================================
# Status None Tests
# =============================================================================


class TestStatusNone:
    """Tests for status=none response."""

    def test_no_candidates_returns_200_with_status_none(self, client, mock_recommender):
        """No candidates should return 200 with status=none, not 503."""
        from app.services.kb.recommend import RecommendResponse

        # Mock empty response
        mock_recommender.recommend.return_value = RecommendResponse(
            params={},
            status="none",
            confidence=None,
            top_trials=[],
            count_used=0,
            warnings=["no_candidates_found"],
            reasons=["no_candidates_found"],
            suggested_actions=["ingest_more_trials"],
            retrieval_strict_count=0,
            retrieval_relaxed_count=0,
            used_relaxed_filters=False,
            used_metadata_fallback=False,
            query_regime_tags=[],
            collection_name="trading_kb_trials__nomic-embed-text__768",
            embedding_model="nomic-embed-text",
        )

        with patch(
            "app.routers.kb_trials._get_recommender",
            return_value=mock_recommender,
        ):
            response = client.post(
                "/kb/trials/recommend",
                json={
                    "workspace_id": str(uuid4()),
                    "strategy_name": "mean_reversion",  # Valid strategy
                    "objective_type": "sharpe",
                },
            )

            # Should be 200, not 503
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "none"
            assert data["params"] == {}


# =============================================================================
# Filter Override Tests
# =============================================================================


class TestFilterOverrides:
    """Tests for filter override bounds."""

    def test_max_overfit_gap_bounded(self, client):
        """max_overfit_gap should be bounded 0-1."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "mean_reversion",  # Valid strategy
                "objective_type": "sharpe",
                "max_overfit_gap": 1.5,  # > 1
            },
        )

        assert response.status_code == 422

    def test_max_drawdown_bounded(self, client):
        """max_drawdown should be bounded 0-1."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "mean_reversion",  # Valid strategy
                "objective_type": "sharpe",
                "max_drawdown": -0.1,  # < 0
            },
        )

        assert response.status_code == 422

    def test_min_trades_bounded(self, client):
        """min_trades should be bounded >= 1."""
        response = client.post(
            "/kb/trials/recommend",
            json={
                "workspace_id": str(uuid4()),
                "strategy_name": "mean_reversion",  # Valid strategy
                "objective_type": "sharpe",
                "min_trades": 0,  # < 1
            },
        )

        assert response.status_code == 422
