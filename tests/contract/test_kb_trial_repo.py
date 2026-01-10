"""Contract tests for KB Trial Repository.

Tests verify the repository contract with Qdrant without requiring
a live Qdrant instance. Uses mocking to verify correct API usage.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.repositories.kb_trials import KBTrialRepository
from app.services.kb.constants import (
    KB_TRIALS_COLLECTION_NAME,
    KB_TRIALS_DOC_TYPE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant async client."""
    client = AsyncMock()

    # Default: collection doesn't exist
    collections_response = MagicMock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response

    return client


@pytest.fixture
def repository(mock_qdrant_client):
    """Create repository with mock client."""
    return KBTrialRepository(client=mock_qdrant_client)


@pytest.fixture
def sample_vector():
    """Sample embedding vector."""
    return [0.1] * 768  # Standard embedding dimension


@pytest.fixture
def sample_payload():
    """Sample trial payload."""
    return {
        "doc_type": KB_TRIALS_DOC_TYPE,
        "workspace_id": str(uuid4()),
        "tune_id": str(uuid4()),
        "tune_run_id": str(uuid4()),
        "strategy_name": "mean_reversion",
        "objective_type": "sharpe",
        "instrument": "BTCUSD",
        "timeframe": "1h",
        "is_valid": True,
        "has_oos": True,
        "sharpe_oos": 1.5,
        "return_frac_oos": 0.25,
        "max_dd_frac_oos": 0.08,
        "n_trades_oos": 42,
        "overfit_gap": 0.15,
        "objective_score": 1.35,
        "regime_tags": ["uptrend", "low_vol"],
    }


# =============================================================================
# Collection Management Tests
# =============================================================================


class TestCollectionManagement:
    """Tests for collection creation and validation."""

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_new(self, repository, mock_qdrant_client):
        """Should create collection when it doesn't exist."""
        await repository.ensure_collection(dimension=768)

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == KB_TRIALS_COLLECTION_NAME

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_indexes(
        self, repository, mock_qdrant_client
    ):
        """Should create payload indexes on new collection."""
        await repository.ensure_collection(dimension=768)

        # Should create multiple indexes
        assert mock_qdrant_client.create_payload_index.call_count >= 10

        # Check key indexes were created
        index_calls = mock_qdrant_client.create_payload_index.call_args_list
        indexed_fields = [call.kwargs["field_name"] for call in index_calls]

        assert "doc_type" in indexed_fields
        assert "workspace_id" in indexed_fields
        assert "strategy_name" in indexed_fields
        assert "regime_tags" in indexed_fields
        assert "sharpe_oos" in indexed_fields

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_existing(
        self, repository, mock_qdrant_client
    ):
        """Should skip creation for existing collection with matching dimension."""
        # Setup: collection exists with correct dimension
        collections_response = MagicMock()
        collection_mock = MagicMock()
        collection_mock.name = KB_TRIALS_COLLECTION_NAME
        collections_response.collections = [collection_mock]
        mock_qdrant_client.get_collections.return_value = collections_response

        collection_info = MagicMock()
        collection_info.config.params.vectors.size = 768
        mock_qdrant_client.get_collection.return_value = collection_info

        await repository.ensure_collection(dimension=768)

        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_recreates_on_dimension_mismatch(
        self, repository, mock_qdrant_client
    ):
        """Should recreate collection if dimension mismatches."""
        # Setup: collection exists with wrong dimension
        collections_response = MagicMock()
        collection_mock = MagicMock()
        collection_mock.name = KB_TRIALS_COLLECTION_NAME
        collections_response.collections = [collection_mock]
        mock_qdrant_client.get_collections.return_value = collections_response

        collection_info = MagicMock()
        collection_info.config.params.vectors.size = 384  # Wrong dimension
        mock_qdrant_client.get_collection.return_value = collection_info

        await repository.ensure_collection(dimension=768)

        mock_qdrant_client.delete_collection.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_with_recreate_flag(
        self, repository, mock_qdrant_client
    ):
        """Should recreate collection when recreate=True."""
        # Setup: collection exists
        collections_response = MagicMock()
        collection_mock = MagicMock()
        collection_mock.name = KB_TRIALS_COLLECTION_NAME
        collections_response.collections = [collection_mock]
        mock_qdrant_client.get_collections.return_value = collections_response

        await repository.ensure_collection(dimension=768, recreate=True)

        mock_qdrant_client.delete_collection.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_collection_exists_true(self, repository, mock_qdrant_client):
        """Should return True when collection exists."""
        collections_response = MagicMock()
        collection_mock = MagicMock()
        collection_mock.name = KB_TRIALS_COLLECTION_NAME
        collections_response.collections = [collection_mock]
        mock_qdrant_client.get_collections.return_value = collections_response

        result = await repository.collection_exists()

        assert result is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self, repository, mock_qdrant_client):
        """Should return False when collection doesn't exist."""
        result = await repository.collection_exists()

        assert result is False


# =============================================================================
# Upsert Tests
# =============================================================================


class TestUpsert:
    """Tests for trial upsert operations."""

    @pytest.mark.asyncio
    async def test_upsert_trial_single(
        self, repository, mock_qdrant_client, sample_vector, sample_payload
    ):
        """Should upsert single trial document."""
        point_id = uuid4()

        await repository.upsert_trial(
            point_id=point_id,
            vector=sample_vector,
            payload=sample_payload,
        )

        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == KB_TRIALS_COLLECTION_NAME
        assert len(call_kwargs["points"]) == 1
        assert call_kwargs["points"][0].id == str(point_id)

    @pytest.mark.asyncio
    async def test_upsert_trial_string_id(
        self, repository, mock_qdrant_client, sample_vector, sample_payload
    ):
        """Should handle string point ID."""
        point_id = "custom-string-id"

        await repository.upsert_trial(
            point_id=point_id,
            vector=sample_vector,
            payload=sample_payload,
        )

        call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
        assert call_kwargs["points"][0].id == point_id

    @pytest.mark.asyncio
    async def test_upsert_batch(
        self, repository, mock_qdrant_client, sample_vector, sample_payload
    ):
        """Should upsert multiple trials in batch."""
        points = [
            {"id": uuid4(), "vector": sample_vector, "payload": sample_payload}
            for _ in range(5)
        ]

        count = await repository.upsert_batch(points)

        assert count == 5
        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
        assert len(call_kwargs["points"]) == 5

    @pytest.mark.asyncio
    async def test_upsert_batch_empty(self, repository, mock_qdrant_client):
        """Should handle empty batch gracefully."""
        count = await repository.upsert_batch([])

        assert count == 0
        mock_qdrant_client.upsert.assert_not_called()


# =============================================================================
# Delete Tests
# =============================================================================


class TestDelete:
    """Tests for trial deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_trial(self, repository, mock_qdrant_client):
        """Should delete single trial by ID."""
        point_id = uuid4()

        await repository.delete_trial(point_id)

        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == KB_TRIALS_COLLECTION_NAME

    @pytest.mark.asyncio
    async def test_delete_by_tune_id(self, repository, mock_qdrant_client):
        """Should delete all trials for a tune."""
        tune_id = uuid4()

        # Setup count response
        count_response = MagicMock()
        count_response.count = 10
        mock_qdrant_client.count.return_value = count_response

        count = await repository.delete_by_tune_id(tune_id)

        assert count == 10
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_tune_id_none_found(self, repository, mock_qdrant_client):
        """Should handle case with no matching trials."""
        tune_id = uuid4()

        count_response = MagicMock()
        count_response.count = 0
        mock_qdrant_client.count.return_value = count_response

        count = await repository.delete_by_tune_id(tune_id)

        assert count == 0
        mock_qdrant_client.delete.assert_not_called()


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for vector similarity search."""

    @pytest.mark.asyncio
    async def test_search_basic(self, repository, mock_qdrant_client, sample_vector):
        """Should perform vector similarity search."""
        workspace_id = uuid4()

        # Setup search response
        result_mock = MagicMock()
        result_mock.id = str(uuid4())
        result_mock.score = 0.95
        result_mock.payload = {"strategy_name": "mean_reversion"}
        mock_qdrant_client.search.return_value = [result_mock]

        results = await repository.search(
            vector=sample_vector,
            workspace_id=workspace_id,
            strategy_name="mean_reversion",
            objective_type="sharpe",
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["strategy_name"] == "mean_reversion"

        # Verify search was called with correct filter
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert call_kwargs["collection_name"] == KB_TRIALS_COLLECTION_NAME
        assert call_kwargs["limit"] == 100
        assert call_kwargs["with_payload"] is True

    @pytest.mark.asyncio
    async def test_search_with_limit(
        self, repository, mock_qdrant_client, sample_vector
    ):
        """Should respect limit parameter."""
        workspace_id = uuid4()
        mock_qdrant_client.search.return_value = []

        await repository.search(
            vector=sample_vector,
            workspace_id=workspace_id,
            strategy_name="mean_reversion",
            objective_type="sharpe",
            limit=20,
        )

        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 20

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self, repository, mock_qdrant_client, sample_vector
    ):
        """Should apply additional filters."""
        workspace_id = uuid4()
        mock_qdrant_client.search.return_value = []

        filters = {
            "require_oos": True,
            "min_sharpe": 1.0,
            "min_trades": 10,
            "max_drawdown": 0.15,
            "max_overfit_gap": 0.3,
            "regime_tags": ["uptrend", "low_vol"],
        }

        await repository.search(
            vector=sample_vector,
            workspace_id=workspace_id,
            strategy_name="mean_reversion",
            objective_type="sharpe",
            filters=filters,
        )

        # Verify filter was constructed
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        query_filter = call_kwargs["query_filter"]

        # Should have base conditions + filter conditions
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_by_filters_only(self, repository, mock_qdrant_client):
        """Should search by filters only (no vector)."""
        workspace_id = uuid4()

        result_mock = MagicMock()
        result_mock.id = str(uuid4())
        result_mock.payload = {"strategy_name": "mean_reversion"}
        mock_qdrant_client.scroll.return_value = ([result_mock], None)

        results = await repository.search_by_filters(
            workspace_id=workspace_id,
            strategy_name="mean_reversion",
            objective_type="sharpe",
        )

        assert len(results) == 1
        mock_qdrant_client.scroll.assert_called_once()
        call_kwargs = mock_qdrant_client.scroll.call_args.kwargs
        assert call_kwargs["with_vectors"] is False


# =============================================================================
# Filter Building Tests
# =============================================================================


class TestFilterBuilding:
    """Tests for filter condition construction."""

    def test_build_filter_require_oos(self, repository):
        """Should build has_oos filter."""
        conditions = repository._build_filter_conditions({"require_oos": True})

        assert len(conditions) == 1
        assert conditions[0].key == "has_oos"

    def test_build_filter_min_sharpe(self, repository):
        """Should build min sharpe range filter."""
        conditions = repository._build_filter_conditions({"min_sharpe": 1.5})

        assert len(conditions) == 1
        assert conditions[0].key == "sharpe_oos"
        assert conditions[0].range.gte == 1.5

    def test_build_filter_min_trades(self, repository):
        """Should build min trades range filter."""
        conditions = repository._build_filter_conditions({"min_trades": 10})

        assert len(conditions) == 1
        assert conditions[0].key == "n_trades_oos"

    def test_build_filter_max_drawdown(self, repository):
        """Should build max drawdown range filter."""
        conditions = repository._build_filter_conditions({"max_drawdown": 0.15})

        assert len(conditions) == 1
        assert conditions[0].key == "max_dd_frac_oos"
        assert conditions[0].range.lte == 0.15

    def test_build_filter_max_overfit_gap(self, repository):
        """Should build max overfit gap range filter."""
        conditions = repository._build_filter_conditions({"max_overfit_gap": 0.3})

        assert len(conditions) == 1
        assert conditions[0].key == "overfit_gap"

    def test_build_filter_regime_tags(self, repository):
        """Should build regime tags match-any filter."""
        conditions = repository._build_filter_conditions(
            {"regime_tags": ["uptrend", "low_vol"]}
        )

        assert len(conditions) == 1
        assert conditions[0].key == "regime_tags"

    def test_build_filter_multiple(self, repository):
        """Should combine multiple filter conditions."""
        conditions = repository._build_filter_conditions(
            {
                "require_oos": True,
                "min_sharpe": 1.0,
                "max_drawdown": 0.2,
            }
        )

        assert len(conditions) == 3

    def test_build_filter_ignores_none(self, repository):
        """Should ignore None values."""
        conditions = repository._build_filter_conditions(
            {
                "min_sharpe": None,
                "min_trades": 10,
            }
        )

        assert len(conditions) == 1
        assert conditions[0].key == "n_trades_oos"

    def test_build_filter_empty(self, repository):
        """Should return empty list for empty filters."""
        conditions = repository._build_filter_conditions({})

        assert len(conditions) == 0


# =============================================================================
# Count and Info Tests
# =============================================================================


class TestCountAndInfo:
    """Tests for count and collection info."""

    @pytest.mark.asyncio
    async def test_count_all(self, repository, mock_qdrant_client):
        """Should count all trials."""
        count_response = MagicMock()
        count_response.count = 100
        mock_qdrant_client.count.return_value = count_response

        count = await repository.count()

        assert count == 100

    @pytest.mark.asyncio
    async def test_count_by_workspace(self, repository, mock_qdrant_client):
        """Should count trials by workspace."""
        workspace_id = uuid4()
        count_response = MagicMock()
        count_response.count = 50
        mock_qdrant_client.count.return_value = count_response

        count = await repository.count(workspace_id=workspace_id)

        assert count == 50
        call_kwargs = mock_qdrant_client.count.call_args.kwargs
        count_filter = call_kwargs["count_filter"]
        assert count_filter is not None

    @pytest.mark.asyncio
    async def test_count_by_strategy(self, repository, mock_qdrant_client):
        """Should count trials by strategy."""
        count_response = MagicMock()
        count_response.count = 25
        mock_qdrant_client.count.return_value = count_response

        count = await repository.count(strategy_name="mean_reversion")

        assert count == 25

    @pytest.mark.asyncio
    async def test_get_collection_info(self, repository, mock_qdrant_client):
        """Should return collection info."""
        collection_info = MagicMock()
        collection_info.vectors_count = 1000
        collection_info.points_count = 1000
        collection_info.status.value = "green"
        mock_qdrant_client.get_collection.return_value = collection_info

        info = await repository.get_collection_info()

        assert info["name"] == KB_TRIALS_COLLECTION_NAME
        assert info["vectors_count"] == 1000
        assert info["status"] == "green"

    @pytest.mark.asyncio
    async def test_get_collection_info_error(self, repository, mock_qdrant_client):
        """Should handle collection info error gracefully."""
        mock_qdrant_client.get_collection.side_effect = Exception("Not found")

        info = await repository.get_collection_info()

        assert info["name"] == KB_TRIALS_COLLECTION_NAME
        assert "error" in info
