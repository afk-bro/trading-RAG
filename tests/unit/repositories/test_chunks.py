"""Tests for chunk repository."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.chunks import ChunkRepository, NeighborChunk


class TestChunkRepository:
    """Test basic repository functionality."""

    def test_repository_creation(self):
        """Test creating repository instance."""
        mock_pool = MagicMock()
        repo = ChunkRepository(mock_pool)
        assert repo.pool == mock_pool


class TestCreateBatch:
    """Test batch chunk creation."""

    @pytest.mark.asyncio
    async def test_create_batch_empty_list(self):
        """Test creating batch with empty list returns empty."""
        mock_pool = MagicMock()
        repo = ChunkRepository(mock_pool)

        result = await repo.create_batch(
            doc_id=uuid4(),
            workspace_id=uuid4(),
            chunks=[],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_create_batch_single_chunk(self):
        """Test creating batch with single chunk."""
        doc_id = uuid4()
        workspace_id = uuid4()
        chunk_id = uuid4()

        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[{"id": chunk_id}])
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)

        chunks = [
            {
                "content": "Test content",
                "chunk_index": 0,
                "token_count": 10,
            }
        ]

        result = await repo.create_batch(doc_id, workspace_id, chunks)

        assert len(result) == 1
        assert result[0] == chunk_id

    @pytest.mark.asyncio
    async def test_create_batch_multiple_chunks(self):
        """Test creating batch with multiple chunks."""
        doc_id = uuid4()
        workspace_id = uuid4()
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()

        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[{"id": chunk_id1}, {"id": chunk_id2}])
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)

        chunks = [
            {"content": "Chunk 1", "chunk_index": 0},
            {"content": "Chunk 2", "chunk_index": 1},
        ]

        result = await repo.create_batch(doc_id, workspace_id, chunks)

        assert len(result) == 2
        assert result == [chunk_id1, chunk_id2]


class TestGetById:
    """Test getting chunk by ID."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        """Test getting a chunk that exists."""
        chunk_id = uuid4()
        mock_row = {
            "id": chunk_id,
            "content": "Test content",
            "chunk_index": 0,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_id(chunk_id)

        assert result == mock_row

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        """Test getting a chunk that doesn't exist."""
        chunk_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_id(chunk_id)

        assert result is None


class TestGetByDocId:
    """Test getting chunks by document ID."""

    @pytest.mark.asyncio
    async def test_get_by_doc_id(self):
        """Test getting all chunks for a document."""
        doc_id = uuid4()
        mock_rows = [
            {"id": uuid4(), "chunk_index": 0, "content": "Chunk 1"},
            {"id": uuid4(), "chunk_index": 1, "content": "Chunk 2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_doc_id(doc_id)

        assert len(result) == 2
        assert result == mock_rows


class TestGetByIds:
    """Test getting chunks by multiple IDs."""

    @pytest.mark.asyncio
    async def test_get_by_ids_empty_list(self):
        """Test getting chunks with empty ID list."""
        mock_pool = MagicMock()
        repo = ChunkRepository(mock_pool)

        result = await repo.get_by_ids([])

        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_ids_preserve_order(self):
        """Test getting chunks preserving input order."""
        chunk_ids = [uuid4(), uuid4()]
        mock_rows = [
            {"id": chunk_ids[0], "content": "Chunk 1"},
            {"id": chunk_ids[1], "content": "Chunk 2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_ids(chunk_ids, preserve_order=True)

        assert len(result) == 2
        assert result == mock_rows

    @pytest.mark.asyncio
    async def test_get_by_ids_no_order_preservation(self):
        """Test getting chunks without order preservation."""
        chunk_ids = [uuid4(), uuid4()]
        mock_rows = [
            {"id": chunk_ids[0], "content": "Chunk 1"},
            {"id": chunk_ids[1], "content": "Chunk 2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_ids(chunk_ids, preserve_order=False)

        assert len(result) == 2


class TestGetByWorkspace:
    """Test getting chunks by workspace."""

    @pytest.mark.asyncio
    async def test_get_by_workspace_without_doc_filter(self):
        """Test getting chunks for workspace without doc filter."""
        workspace_id = uuid4()
        mock_rows = [
            {"id": uuid4(), "content": "Chunk 1"},
            {"id": uuid4(), "content": "Chunk 2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_workspace(workspace_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_by_workspace_with_doc_filter(self):
        """Test getting chunks for workspace with doc ID filter."""
        workspace_id = uuid4()
        doc_ids = [uuid4(), uuid4()]
        mock_rows = [
            {"id": uuid4(), "doc_id": doc_ids[0], "content": "Chunk 1"},
            {"id": uuid4(), "doc_id": doc_ids[1], "content": "Chunk 2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_workspace(workspace_id, doc_ids=doc_ids)

        assert len(result) == 2


class TestDeleteByDocId:
    """Test deleting chunks by document ID."""

    @pytest.mark.asyncio
    async def test_delete_by_doc_id(self):
        """Test deleting all chunks for a document."""
        doc_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 5")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        count = await repo.delete_by_doc_id(doc_id)

        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_by_doc_id_none_found(self):
        """Test deleting when no chunks exist."""
        doc_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        count = await repo.delete_by_doc_id(doc_id)

        assert count == 0


class TestCountByWorkspace:
    """Test counting chunks in workspace."""

    @pytest.mark.asyncio
    async def test_count_by_workspace(self):
        """Test counting chunks in a workspace."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=42)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        count = await repo.count_by_workspace(workspace_id)

        assert count == 42


class TestGetByIdsMap:
    """Test getting chunks as a dict."""

    @pytest.mark.asyncio
    async def test_get_by_ids_map_empty(self):
        """Test getting chunks map with empty list."""
        mock_pool = MagicMock()
        repo = ChunkRepository(mock_pool)

        result = await repo.get_by_ids_map([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_by_ids_map(self):
        """Test getting chunks as a dict keyed by ID."""
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()
        chunk_ids = [str(chunk_id1), str(chunk_id2)]

        mock_rows = [
            {"id": chunk_id1, "content": "Chunk 1"},
            {"id": chunk_id2, "content": "Chunk 2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_by_ids_map(chunk_ids)

        assert len(result) == 2
        assert str(chunk_id1) in result
        assert str(chunk_id2) in result
        assert result[str(chunk_id1)]["content"] == "Chunk 1"


class TestGetNeighborsByDocIndices:
    """Test fetching neighbor chunks."""

    @pytest.mark.asyncio
    async def test_get_neighbors_empty_requests(self):
        """Test getting neighbors with empty request list."""
        mock_pool = MagicMock()
        repo = ChunkRepository(mock_pool)

        result = await repo.get_neighbors_by_doc_indices([])

        assert result == []

    @pytest.mark.asyncio
    async def test_get_neighbors_by_doc_indices(self):
        """Test fetching neighbor chunks by document and index."""
        doc_id = uuid4()
        seed_chunk_id = uuid4()
        chunk_id = uuid4()

        requests = [(str(doc_id), 5, str(seed_chunk_id))]

        mock_rows = [
            {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": 5,
                "text": "Neighbor chunk content",
                "source_type": "youtube",
                "seed_chunk_id": seed_chunk_id,
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = ChunkRepository(mock_pool)
        result = await repo.get_neighbors_by_doc_indices(requests)

        assert len(result) == 1
        assert isinstance(result[0], NeighborChunk)
        assert result[0].chunk_id == str(chunk_id)
        assert result[0].chunk_index == 5
        assert result[0].source_type == "youtube"
        assert result[0].seed_chunk_id == str(seed_chunk_id)


class TestNeighborChunk:
    """Test NeighborChunk dataclass."""

    def test_neighbor_chunk_creation(self):
        """Test creating NeighborChunk instance."""
        chunk_id = str(uuid4())
        document_id = str(uuid4())
        seed_chunk_id = str(uuid4())

        neighbor = NeighborChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=5,
            text="Test content",
            source_type="youtube",
            seed_chunk_id=seed_chunk_id,
        )

        assert neighbor.chunk_id == chunk_id
        assert neighbor.document_id == document_id
        assert neighbor.chunk_index == 5
        assert neighbor.text == "Test content"
        assert neighbor.source_type == "youtube"
        assert neighbor.seed_chunk_id == seed_chunk_id
