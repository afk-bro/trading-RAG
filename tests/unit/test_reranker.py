"""Unit tests for cross-encoder and LLM reranker services."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.services.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    RerankCandidate,
    RerankResult,
    get_reranker,
    reset_rerankers,
)


def make_candidate(
    chunk_id: str = "chunk1",
    document_id: str = "doc1",
    chunk_index: int = 0,
    text: str = "Sample text",
    vector_score: float = 0.9,
    workspace_id: str = "ws1",
    source_type: str | None = None,
) -> RerankCandidate:
    """Factory function to create test candidates."""
    return RerankCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        vector_score=vector_score,
        workspace_id=workspace_id,
        source_type=source_type,
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset reranker singletons before each test."""
    reset_rerankers()
    yield
    reset_rerankers()


class TestGetReranker:
    """Tests for get_reranker factory function."""

    def test_disabled_returns_none(self):
        """When enabled=False, returns None."""
        config = {"enabled": False}
        assert get_reranker(config) is None

    def test_empty_config_returns_none(self):
        """Empty config defaults to disabled."""
        assert get_reranker({}) is None

    @patch("app.services.reranker.CrossEncoderReranker")
    def test_returns_cross_encoder_by_default(self, mock_class):
        """Default method is cross_encoder."""
        config = {"enabled": True}
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = get_reranker(config)

        assert result is mock_instance
        mock_class.assert_called_once()

    @patch("app.services.reranker.CrossEncoderReranker")
    def test_singleton_returns_same_instance(self, mock_class):
        """Subsequent calls return the same instance."""
        config = {"enabled": True, "method": "cross_encoder"}
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        r1 = get_reranker(config)
        r2 = get_reranker(config)

        assert r1 is r2
        mock_class.assert_called_once()

    @patch("app.services.reranker.LLMReranker")
    def test_returns_llm_reranker(self, mock_class):
        """method=llm returns LLMReranker."""
        config = {"enabled": True, "method": "llm"}
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = get_reranker(config)

        assert result is mock_instance
        mock_class.assert_called_once()

    @patch("app.services.reranker.CrossEncoderReranker")
    def test_unknown_method_falls_back_to_cross_encoder(self, mock_class):
        """Unknown method falls back to cross_encoder."""
        config = {"enabled": True, "method": "unknown"}
        mock_instance = Mock()
        mock_class.return_value = mock_instance

        result = get_reranker(config)

        assert result is mock_instance


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock the CrossEncoder class."""
        with patch(
            "app.services.reranker.CrossEncoderReranker._load_model"
        ) as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [0.9, 0.7, 0.8, 0.6, 0.5]
            mock_load.return_value = mock_model
            yield mock_model

    @pytest.fixture
    def candidates(self) -> list[RerankCandidate]:
        """Create test candidates."""
        return [
            make_candidate("c1", text="Python is a language", vector_score=0.9),
            make_candidate("c2", text="Java is a language", vector_score=0.8),
            make_candidate("c3", text="Weather is nice", vector_score=0.7),
            make_candidate("c4", text="Random text", vector_score=0.6),
            make_candidate("c5", text="More text", vector_score=0.5),
        ]

    @pytest.mark.asyncio
    async def test_rerank_returns_top_k(self, mock_cross_encoder, candidates):
        """Rerank returns top_k results."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        results = await reranker.rerank("test query", candidates, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        reranker.close()

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self, mock_cross_encoder):
        """Empty candidates returns empty list."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        results = await reranker.rerank("test query", [], top_k=5)

        assert results == []
        reranker.close()

    @pytest.mark.asyncio
    async def test_rerank_fewer_than_top_k(self, mock_cross_encoder):
        """When candidates < top_k, returns all candidates."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)
        candidates = [make_candidate("c1"), make_candidate("c2")]
        mock_cross_encoder.predict.return_value = [0.9, 0.7]

        results = await reranker.rerank("test query", candidates, top_k=5)

        assert len(results) == 2
        reranker.close()

    @pytest.mark.asyncio
    async def test_truncates_to_max_candidates(self, mock_cross_encoder):
        """Candidates exceeding MAX_CANDIDATES are truncated."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)
        # Create more than MAX_CANDIDATES
        candidates = [make_candidate(f"c{i}", vector_score=i / 300) for i in range(300)]
        # Mock predict to return scores for truncated list
        mock_cross_encoder.predict.return_value = [0.5] * 200

        await reranker.rerank("test", candidates, top_k=10)

        # Should only score MAX_CANDIDATES (200)
        call_args = mock_cross_encoder.predict.call_args
        assert len(call_args[0][0]) <= 200
        reranker.close()

    @pytest.mark.asyncio
    async def test_deterministic_ordering_same_scores(self, mock_cross_encoder):
        """With same rerank scores, order by vector_score then chunk_id."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)
        mock_cross_encoder.predict.return_value = [0.5, 0.5, 0.5]
        candidates = [
            make_candidate("c", vector_score=0.8),
            make_candidate("a", vector_score=0.8),
            make_candidate("b", vector_score=0.9),
        ]

        results = await reranker.rerank("test", candidates, top_k=3)

        # b first (higher vector), then c before a (reverse sort, higher chunk_id first)
        assert [r.chunk_id for r in results] == ["b", "c", "a"]
        reranker.close()

    @pytest.mark.asyncio
    async def test_rank_assigned_correctly(self, mock_cross_encoder, candidates):
        """Rerank_rank is assigned 0, 1, 2... in order."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        results = await reranker.rerank("test", candidates, top_k=5)

        assert [r.rerank_rank for r in results] == [0, 1, 2, 3, 4]
        reranker.close()

    @pytest.mark.asyncio
    async def test_preserves_vector_score(self, mock_cross_encoder, candidates):
        """Original vector_score is preserved in results."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        results = await reranker.rerank("test", candidates, top_k=3)

        # Check that vector scores are preserved (may be reordered)
        _result_vector_scores = {  # noqa: F841
            r.chunk_id: r.vector_score for r in results
        }  # noqa: F841
        for r in results:
            original = next(c for c in candidates if c.chunk_id == r.chunk_id)
            assert r.vector_score == original.vector_score
        reranker.close()

    def test_method_property(self):
        """method property returns 'cross_encoder'."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        assert reranker.method == "cross_encoder"
        reranker.close()

    def test_model_id_property(self):
        """model_id returns the configured model."""
        config = {"model": "test-model", "device": "cpu"}
        reranker = CrossEncoderReranker(config)

        assert reranker.model_id == "test-model"
        reranker.close()

    def test_default_model(self):
        """Default model is BGE reranker."""
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        assert "bge-reranker" in reranker.model_id.lower()
        reranker.close()


class TestLLMReranker:
    """Tests for LLMReranker."""

    @pytest.mark.asyncio
    async def test_no_llm_returns_vector_order(self):
        """Without LLM, returns candidates in vector score order."""
        with patch("app.services.llm_factory.get_llm", return_value=None):
            reranker = LLMReranker({})
            candidates = [
                make_candidate("c1", vector_score=0.9),
                make_candidate("c2", vector_score=0.8),
                make_candidate("c3", vector_score=0.7),
            ]

            results = await reranker.rerank("test", candidates, top_k=2)

            assert len(results) == 2
            assert results[0].chunk_id == "c1"
            assert results[1].chunk_id == "c2"
            assert results[0].rerank_score == 0.0

    @pytest.mark.asyncio
    async def test_with_llm_calls_rerank(self):
        """With LLM available, calls llm.rerank()."""
        mock_llm = Mock()
        mock_llm.rerank = AsyncMock(
            return_value=[
                Mock(chunk={"chunk_id": "c2"}, score=0.95),
                Mock(chunk={"chunk_id": "c1"}, score=0.85),
            ]
        )

        with patch("app.services.llm_factory.get_llm", return_value=mock_llm):
            reranker = LLMReranker({})
            candidates = [
                make_candidate("c1", vector_score=0.9),
                make_candidate("c2", vector_score=0.8),
            ]

            results = await reranker.rerank("test query", candidates, top_k=2)

            mock_llm.rerank.assert_called_once()
            assert len(results) == 2
            assert results[0].chunk_id == "c2"  # LLM ranked it higher
            assert results[0].rerank_score == 0.95

    def test_method_property(self):
        """method property returns 'llm'."""
        reranker = LLMReranker({})
        assert reranker.method == "llm"

    def test_model_id_returns_llm_model(self):
        """model_id returns effective rerank model from LLM."""
        mock_llm = Mock()
        mock_llm.effective_rerank_model = "test-llm-model"

        with patch("app.services.llm_factory.get_llm", return_value=mock_llm):
            reranker = LLMReranker({})
            assert reranker.model_id == "test-llm-model"

    def test_model_id_none_when_no_llm(self):
        """model_id returns None when no LLM configured."""
        with patch("app.services.llm_factory.get_llm", return_value=None):
            reranker = LLMReranker({})
            assert reranker.model_id is None


class TestRerankCandidate:
    """Tests for RerankCandidate dataclass."""

    def test_creation(self):
        """Can create RerankCandidate with required fields."""
        candidate = RerankCandidate(
            chunk_id="test",
            document_id="doc1",
            chunk_index=0,
            text="Sample text",
            vector_score=0.9,
            workspace_id="ws1",
        )

        assert candidate.chunk_id == "test"
        assert candidate.document_id == "doc1"
        assert candidate.source_type is None
        assert candidate.metadata == {}

    def test_optional_fields(self):
        """Optional fields can be set."""
        candidate = RerankCandidate(
            chunk_id="test",
            document_id="doc1",
            chunk_index=0,
            text="Sample text",
            vector_score=0.9,
            workspace_id="ws1",
            source_type="pdf",
            metadata={"key": "value"},
        )

        assert candidate.source_type == "pdf"
        assert candidate.metadata == {"key": "value"}


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_creation(self):
        """Can create RerankResult with all fields."""
        result = RerankResult(
            chunk_id="test",
            document_id="doc1",
            chunk_index=0,
            rerank_score=0.95,
            rerank_rank=0,
            vector_score=0.9,
            source_type="youtube",
        )

        assert result.chunk_id == "test"
        assert result.rerank_score == 0.95
        assert result.rerank_rank == 0
        assert result.vector_score == 0.9
        assert result.source_type == "youtube"
