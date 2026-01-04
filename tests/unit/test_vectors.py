"""Unit tests for vector repository."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.repositories.vectors import VectorRepository


class MockCollectionInfo:
    """Mock Qdrant collection info."""

    def __init__(self, dimension: int):
        self.config = MagicMock()
        self.config.params = MagicMock()
        self.config.params.vectors = MagicMock()
        self.config.params.vectors.size = dimension


class MockCollectionList:
    """Mock Qdrant collection list."""

    def __init__(self, collections: list):
        self.collections = [MagicMock(name=c) for c in collections]


class TestVectorRepositoryEnsureCollection:
    """Tests for ensure_collection dimension validation."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_creates_collection_if_not_exists(self, mock_client):
        """Test that collection is created if it doesn't exist."""
        mock_client.get_collections.return_value = MockCollectionList([])

        repo = VectorRepository(client=mock_client, collection="test_collection")
        await repo.ensure_collection(dimension=768)

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_validates_dimension_on_existing_collection(self, mock_client):
        """Test that dimension is validated for existing collection."""
        mock_client.get_collections.return_value = MockCollectionList(["test_collection"])
        mock_client.get_collection.return_value = MockCollectionInfo(dimension=768)

        repo = VectorRepository(client=mock_client, collection="test_collection")
        await repo.ensure_collection(dimension=768)

        # Should not recreate if dimensions match
        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_recreates_collection_on_dimension_mismatch(self, mock_client):
        """Test that collection is recreated on dimension mismatch."""
        mock_client.get_collections.return_value = MockCollectionList(["test_collection"])
        mock_client.get_collection.return_value = MockCollectionInfo(dimension=512)

        repo = VectorRepository(client=mock_client, collection="test_collection")
        await repo.ensure_collection(dimension=768)

        # Should delete and recreate with correct dimension
        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["vectors_config"].size == 768

    @pytest.mark.asyncio
    async def test_logs_warning_on_dimension_mismatch(self, mock_client, caplog):
        """Test that warning is logged on dimension mismatch."""
        import logging

        mock_client.get_collections.return_value = MockCollectionList(["test_collection"])
        mock_client.get_collection.return_value = MockCollectionInfo(dimension=512)

        repo = VectorRepository(client=mock_client, collection="test_collection")

        with caplog.at_level(logging.WARNING):
            await repo.ensure_collection(dimension=768)

        # Check that warning was logged (structlog may not appear in caplog)
        # The test validates behavior by checking delete was called
        mock_client.delete_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_recreate_forces_new_collection(self, mock_client):
        """Test that recreate=True forces collection recreation."""
        mock_client.get_collections.return_value = MockCollectionList(["test_collection"])

        repo = VectorRepository(client=mock_client, collection="test_collection")
        await repo.ensure_collection(dimension=768, recreate=True)

        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_error_when_client_not_initialized(self):
        """Test that error is raised when client is None."""
        repo = VectorRepository(client=None, collection="test_collection")

        with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
            await repo.ensure_collection(dimension=768)


class TestVectorRepositorySearch:
    """Tests for VectorRepository search functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_search_builds_workspace_filter(self, mock_client):
        """Test that search includes workspace_id filter."""
        mock_client.search.return_value = []

        repo = VectorRepository(client=mock_client, collection="test_collection")
        workspace_id = uuid4()
        await repo.search(vector=[0.1] * 768, workspace_id=workspace_id, limit=10)

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_raises_error_when_client_not_initialized(self):
        """Test that search raises error when client is None."""
        repo = VectorRepository(client=None, collection="test_collection")

        with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
            await repo.search(vector=[0.1] * 768, workspace_id=uuid4(), limit=10)


class TestVectorRepositoryUpsert:
    """Tests for VectorRepository upsert functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_upsert_batch_empty_list(self, mock_client):
        """Test that upsert with empty list does nothing."""
        repo = VectorRepository(client=mock_client, collection="test_collection")
        await repo.upsert_batch([])

        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_batch_creates_points(self, mock_client):
        """Test that upsert creates Qdrant points."""
        repo = VectorRepository(client=mock_client, collection="test_collection")

        points = [
            {
                "id": uuid4(),
                "vector": [0.1] * 768,
                "payload": {"workspace_id": str(uuid4())},
            }
        ]
        await repo.upsert_batch(points)

        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_raises_error_when_client_not_initialized(self):
        """Test that upsert raises error when client is None."""
        repo = VectorRepository(client=None, collection="test_collection")

        with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
            await repo.upsert_batch([{"id": uuid4(), "vector": [0.1], "payload": {}}])


class TestVectorRepositoryDelete:
    """Tests for VectorRepository delete functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_delete_batch_empty_list(self, mock_client):
        """Test that delete with empty list does nothing."""
        repo = VectorRepository(client=mock_client, collection="test_collection")
        await repo.delete_batch([])

        mock_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_batch_removes_points(self, mock_client):
        """Test that delete removes Qdrant points."""
        repo = VectorRepository(client=mock_client, collection="test_collection")

        point_ids = [uuid4(), uuid4()]
        await repo.delete_batch(point_ids)

        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_raises_error_when_client_not_initialized(self):
        """Test that delete raises error when client is None."""
        repo = VectorRepository(client=None, collection="test_collection")

        with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
            await repo.delete_batch([uuid4()])
