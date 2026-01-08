"""Unit tests for neighbor expansion service."""

import pytest
from unittest.mock import Mock, AsyncMock

from app.services.neighbor_expansion import expand_neighbors, ExpandedChunk
from app.services.reranker import RerankResult
from app.repositories.chunks import NeighborChunk


def make_seed(
    chunk_id: str = "seed1",
    document_id: str = "doc1",
    chunk_index: int = 5,
    rerank_rank: int = 0,
    rerank_score: float = 0.9,
    vector_score: float = 0.8,
    source_type: str | None = None,
) -> RerankResult:
    """Factory function to create test seeds."""
    return RerankResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        rerank_score=rerank_score,
        rerank_rank=rerank_rank,
        vector_score=vector_score,
        source_type=source_type,
    )


def make_neighbor(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    seed_chunk_id: str,
    text: str = "Neighbor text with enough characters to pass min_chars filter",
    source_type: str | None = None,
) -> NeighborChunk:
    """Factory function to create test neighbor chunks."""
    return NeighborChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text * 5,  # Ensure enough text to pass min_chars
        source_type=source_type,
        seed_chunk_id=seed_chunk_id,
    )


@pytest.fixture
def mock_chunk_repo():
    """Create a mock chunk repository."""
    repo = Mock()
    repo.get_neighbors_by_doc_indices = AsyncMock(return_value=[])
    return repo


class TestExpandNeighbors:
    """Tests for expand_neighbors function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_seeds_only(self, mock_chunk_repo):
        """When enabled=False, returns only seeds with no neighbors."""
        seeds = [make_seed("s1", chunk_index=5)]
        config = {"enabled": False}

        expanded, new_ids = await expand_neighbors(seeds, mock_chunk_repo, config)

        assert len(expanded) == 1
        assert not expanded[0].is_neighbor
        assert expanded[0].chunk_id == "s1"
        assert new_ids == []
        mock_chunk_repo.get_neighbors_by_doc_indices.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self, mock_chunk_repo):
        """Empty seeds returns empty list."""
        config = {"enabled": True}

        expanded, new_ids = await expand_neighbors([], mock_chunk_repo, config)

        assert expanded == []
        assert new_ids == []

    @pytest.mark.asyncio
    async def test_fetches_neighbors_within_window(self, mock_chunk_repo):
        """Fetches neighbors within the configured window."""
        seeds = [make_seed("s1", document_id="doc1", chunk_index=5, rerank_rank=0)]
        config = {"enabled": True, "window": 1}

        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = [
            make_neighbor("n4", "doc1", 4, "s1"),
            make_neighbor("n6", "doc1", 6, "s1"),
        ]

        expanded, new_ids = await expand_neighbors(seeds, mock_chunk_repo, config)

        # Should request indices 4 and 6 (window=1 around index 5)
        call_args = mock_chunk_repo.get_neighbors_by_doc_indices.call_args[0][0]
        requested_indices = {(r[0], r[1]) for r in call_args}
        assert ("doc1", 4) in requested_indices
        assert ("doc1", 6) in requested_indices

    @pytest.mark.asyncio
    async def test_pdf_uses_larger_window(self, mock_chunk_repo):
        """PDF source type uses pdf_window instead of window."""
        seeds = [make_seed("s1", chunk_index=5, source_type="pdf")]
        config = {"enabled": True, "window": 1, "pdf_window": 2}

        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = []

        await expand_neighbors(seeds, mock_chunk_repo, config)

        # Should request indices 3, 4, 6, 7 (pdf_window=2)
        call_args = mock_chunk_repo.get_neighbors_by_doc_indices.call_args[0][0]
        requested_indices = {r[1] for r in call_args}
        assert 3 in requested_indices
        assert 4 in requested_indices
        assert 6 in requested_indices
        assert 7 in requested_indices

    @pytest.mark.asyncio
    async def test_best_seed_wins_attribution(self, mock_chunk_repo):
        """When multiple seeds share a neighbor, best rank wins attribution."""
        seeds = [
            make_seed("s1", document_id="doc1", chunk_index=5, rerank_rank=1),
            make_seed("s2", document_id="doc1", chunk_index=7, rerank_rank=0),
        ]
        config = {"enabled": True, "window": 1}

        # Chunk 6 is neighbor to both seeds
        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = [
            make_neighbor("n6", "doc1", 6, "s2"),  # Will be attributed to s2
        ]

        expanded, _ = await expand_neighbors(seeds, mock_chunk_repo, config)

        neighbor_6 = next((e for e in expanded if e.chunk_index == 6), None)
        assert neighbor_6 is not None
        assert neighbor_6.neighbor_of == "s2"  # s2 has better rank (0)

    @pytest.mark.asyncio
    async def test_min_chars_filter(self, mock_chunk_repo):
        """Neighbors with too few characters are filtered out."""
        seeds = [make_seed("s1", chunk_index=5)]
        config = {"enabled": True, "window": 1, "min_chars": 200}

        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = [
            NeighborChunk(
                chunk_id="short",
                document_id="doc1",
                chunk_index=4,
                text="Short",  # Less than min_chars
                source_type=None,
                seed_chunk_id="s1",
            ),
            make_neighbor("long", "doc1", 6, "s1"),  # Has enough chars
        ]

        expanded, new_ids = await expand_neighbors(seeds, mock_chunk_repo, config)

        # Only 'long' should be included as neighbor
        neighbor_ids = [e.chunk_id for e in expanded if e.is_neighbor]
        assert "short" not in neighbor_ids
        assert "long" in neighbor_ids

    @pytest.mark.asyncio
    async def test_soft_cap_preserves_all_seeds(self, mock_chunk_repo):
        """max_total soft cap preserves all seeds even if exceeded."""
        seeds = [make_seed(f"s{i}", chunk_index=i * 10, rerank_rank=i) for i in range(5)]
        config = {"enabled": True, "window": 1, "max_total": 3}

        # Return many neighbors
        neighbors = [
            make_neighbor(f"n{i}", "doc1", s.chunk_index + 1, s.chunk_id)
            for i, s in enumerate(seeds)
        ]
        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = neighbors

        expanded, _ = await expand_neighbors(seeds, mock_chunk_repo, config)

        # All 5 seeds should be preserved
        seed_count = sum(1 for e in expanded if not e.is_neighbor)
        assert seed_count == 5

    @pytest.mark.asyncio
    async def test_deduplication(self, mock_chunk_repo):
        """Duplicate chunk IDs are filtered out."""
        seeds = [make_seed("s1", chunk_index=5)]
        config = {"enabled": True, "window": 1}

        # Repo returns same neighbor twice (shouldn't happen in practice)
        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = [
            make_neighbor("n4", "doc1", 4, "s1"),
            make_neighbor("n4", "doc1", 4, "s1"),  # Duplicate
        ]

        expanded, _ = await expand_neighbors(seeds, mock_chunk_repo, config)

        chunk_ids = [e.chunk_id for e in expanded]
        assert chunk_ids.count("n4") == 1

    @pytest.mark.asyncio
    async def test_does_not_duplicate_seeds(self, mock_chunk_repo):
        """Seeds are not duplicated if returned as neighbors."""
        seeds = [
            make_seed("s1", document_id="doc1", chunk_index=5),
            make_seed("s2", document_id="doc1", chunk_index=6),
        ]
        config = {"enabled": True, "window": 1}

        # Neighbor fetch might return s2's position as neighbor of s1
        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = []

        expanded, _ = await expand_neighbors(seeds, mock_chunk_repo, config)

        chunk_ids = [e.chunk_id for e in expanded]
        assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    @pytest.mark.asyncio
    async def test_ordering_by_doc_rank_then_chunk_index(self, mock_chunk_repo):
        """Results ordered by best doc rank, then chunk_index within doc."""
        seeds = [
            make_seed("s1", document_id="doc1", chunk_index=5, rerank_rank=1),
            make_seed("s2", document_id="doc2", chunk_index=10, rerank_rank=0),
        ]
        config = {"enabled": True, "window": 1}

        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = [
            make_neighbor("n4", "doc1", 4, "s1"),
            make_neighbor("n9", "doc2", 9, "s2"),
        ]

        expanded, _ = await expand_neighbors(seeds, mock_chunk_repo, config)

        # doc2 should come first (rank 0), then doc1 (rank 1)
        doc_order = [e.document_id for e in expanded]
        doc2_idx = next(i for i, d in enumerate(doc_order) if d == "doc2")
        doc1_idx = next(i for i, d in enumerate(doc_order) if d == "doc1")
        assert doc2_idx < doc1_idx

    @pytest.mark.asyncio
    async def test_already_have_ids_excluded(self, mock_chunk_repo):
        """Chunks in already_have_ids are excluded from new_ids."""
        seeds = [make_seed("s1", chunk_index=5)]
        config = {"enabled": True, "window": 1}

        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = [
            make_neighbor("n4", "doc1", 4, "s1"),
        ]

        _, new_ids = await expand_neighbors(
            seeds, mock_chunk_repo, config, already_have_ids={"n4"}
        )

        # n4 is already had, so shouldn't be in new_ids
        assert "n4" not in new_ids

    @pytest.mark.asyncio
    async def test_negative_chunk_indices_skipped(self, mock_chunk_repo):
        """Negative chunk indices are not requested."""
        seeds = [make_seed("s1", chunk_index=0, rerank_rank=0)]  # At beginning
        config = {"enabled": True, "window": 2}

        mock_chunk_repo.get_neighbors_by_doc_indices.return_value = []

        await expand_neighbors(seeds, mock_chunk_repo, config)

        call_args = mock_chunk_repo.get_neighbors_by_doc_indices.call_args[0][0]
        requested_indices = [r[1] for r in call_args]
        assert all(idx >= 0 for idx in requested_indices)

    @pytest.mark.asyncio
    async def test_no_neighbors_when_window_zero(self, mock_chunk_repo):
        """Window=0 means no neighbors are fetched."""
        seeds = [make_seed("s1", chunk_index=5)]
        config = {"enabled": True, "window": 0}

        await expand_neighbors(seeds, mock_chunk_repo, config)

        # No neighbor requests should be made
        if mock_chunk_repo.get_neighbors_by_doc_indices.called:
            call_args = mock_chunk_repo.get_neighbors_by_doc_indices.call_args[0][0]
            assert len(call_args) == 0


class TestExpandedChunk:
    """Tests for ExpandedChunk dataclass."""

    def test_seed_creation(self):
        """ExpandedChunk for a seed has is_neighbor=False."""
        chunk = ExpandedChunk(
            chunk_id="c1",
            document_id="doc1",
            chunk_index=5,
            rerank_score=0.9,
            rerank_rank=0,
            vector_score=0.8,
            source_type="pdf",
            is_neighbor=False,
            neighbor_of=None,
        )

        assert chunk.is_neighbor is False
        assert chunk.neighbor_of is None

    def test_neighbor_creation(self):
        """ExpandedChunk for a neighbor has is_neighbor=True."""
        chunk = ExpandedChunk(
            chunk_id="n1",
            document_id="doc1",
            chunk_index=6,
            rerank_score=0.0,
            rerank_rank=-1,
            vector_score=0.0,
            source_type="pdf",
            is_neighbor=True,
            neighbor_of="seed1",
        )

        assert chunk.is_neighbor is True
        assert chunk.neighbor_of == "seed1"
