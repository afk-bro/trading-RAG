"""Unit tests for KB retrieval module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.services.kb.retrieval import (
    build_filters,
    RetrievalRequest,
    RetrievalCandidate,
    KBRetriever,
    MIN_CANDIDATES_THRESHOLD,
)
from app.services.kb.types import RegimeSnapshot


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repository():
    """Create mock KB repository."""
    repo = MagicMock()
    repo.collection = "trading_kb_trials__nomic-embed-text__768"
    repo.search = AsyncMock(return_value=[])
    repo.search_by_filters = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    embedder = MagicMock()
    embedder.model_id = "nomic-embed-text"
    embedder.embed_single = AsyncMock(return_value=[0.1] * 768)
    return embedder


@pytest.fixture
def basic_request():
    """Create basic retrieval request."""
    return RetrievalRequest(
        workspace_id=uuid4(),
        strategy_name="ema_crossover",
        objective_type="sharpe",
    )


# =============================================================================
# Filter Building Tests
# =============================================================================


class TestBuildFilters:
    """Tests for build_filters function."""

    def test_strict_filters_default(self, basic_request):
        """Should use strict filters by default."""
        filters = build_filters(basic_request, strict=True)

        assert filters["require_oos"] is True
        assert filters["max_overfit_gap"] == 0.3
        assert filters["min_trades"] == 5
        assert filters["max_drawdown"] == 0.25

    def test_relaxed_filters(self, basic_request):
        """Should use relaxed filters when strict=False.

        Relaxed only loosens overfit_gap - keeps other quality gates.
        """
        filters = build_filters(basic_request, strict=False)

        # Only overfit_gap is relaxed
        assert filters["max_overfit_gap"] is None

        # Other quality gates remain strict
        assert filters["require_oos"] is True
        assert filters["min_trades"] == 5
        assert filters["max_drawdown"] == 0.25

    def test_request_overrides(self, basic_request):
        """Should apply request overrides."""
        basic_request.require_oos = False
        basic_request.max_overfit_gap = 0.5
        basic_request.min_trades = 10

        filters = build_filters(basic_request, strict=True)

        assert filters["require_oos"] is False
        assert filters["max_overfit_gap"] == 0.5
        assert filters["min_trades"] == 10

    def test_regime_tags_included(self, basic_request):
        """Should include regime tags in filter."""
        basic_request.regime_tags = ["high_vol", "uptrend"]

        filters = build_filters(basic_request, strict=True)

        assert filters["regime_tags"] == ["high_vol", "uptrend"]


# =============================================================================
# Retriever Tests
# =============================================================================


class TestKBRetriever:
    """Tests for KBRetriever class."""

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_repository, mock_embedder, basic_request):
        """Should handle empty results gracefully."""
        retriever = KBRetriever(repository=mock_repository, embedder=mock_embedder)
        basic_request.query_regime = RegimeSnapshot(regime_tags=["high_vol"])

        result = await retriever.retrieve(basic_request)

        assert result.candidates == []
        assert result.stats.strict_count == 0
        assert result.stats.total_returned == 0

    @pytest.mark.asyncio
    async def test_strict_search_sufficient(
        self, mock_repository, mock_embedder, basic_request
    ):
        """Should not use relaxed filters when strict has enough candidates."""
        # Return enough candidates
        mock_repository.search.return_value = [
            {"id": f"id_{i}", "score": 0.9, "payload": {"objective_score": 1.0}}
            for i in range(20)
        ]

        retriever = KBRetriever(repository=mock_repository, embedder=mock_embedder)
        basic_request.query_regime = RegimeSnapshot(regime_tags=["high_vol"])

        result = await retriever.retrieve(basic_request)

        assert result.stats.strict_count == 20
        assert result.stats.relaxed_count == 0
        assert result.stats.used_relaxed_filters is False
        assert "used_relaxed_filters" not in result.warnings

    @pytest.mark.asyncio
    async def test_relaxed_fallback(
        self, mock_repository, mock_embedder, basic_request
    ):
        """Should fall back to relaxed when strict has too few."""
        # First call (strict) returns few
        # Second call (relaxed) returns more
        mock_repository.search.side_effect = [
            [{"id": f"strict_{i}", "score": 0.9, "payload": {}} for i in range(3)],
            [{"id": f"relaxed_{i}", "score": 0.8, "payload": {}} for i in range(15)],
        ]

        retriever = KBRetriever(repository=mock_repository, embedder=mock_embedder)
        basic_request.query_regime = RegimeSnapshot(regime_tags=["high_vol"])
        basic_request.min_candidates = MIN_CANDIDATES_THRESHOLD

        result = await retriever.retrieve(basic_request)

        assert result.stats.strict_count == 3
        assert result.stats.relaxed_count == 15  # All 15 are new
        assert result.stats.used_relaxed_filters is True
        assert "used_relaxed_filters" in result.warnings

    @pytest.mark.asyncio
    async def test_relaxed_dedupes(self, mock_repository, mock_embedder, basic_request):
        """Should not duplicate candidates from strict in relaxed."""
        # Both return same ID
        mock_repository.search.side_effect = [
            [{"id": "same_id", "score": 0.9, "payload": {}}],
            [{"id": "same_id", "score": 0.8, "payload": {}}],
        ]

        retriever = KBRetriever(repository=mock_repository, embedder=mock_embedder)
        basic_request.query_regime = RegimeSnapshot(regime_tags=["high_vol"])

        result = await retriever.retrieve(basic_request)

        assert len(result.candidates) == 1  # Not duplicated
        assert result.stats.strict_count == 1
        assert result.stats.relaxed_count == 0  # Already in strict

    @pytest.mark.asyncio
    async def test_metadata_fallback_on_embed_failure(
        self, mock_repository, mock_embedder, basic_request
    ):
        """Should use metadata-only fallback when embedding fails."""
        from app.services.kb.embed import EmbeddingError

        mock_embedder.embed_single.side_effect = EmbeddingError("timeout")
        mock_repository.search_by_filters.return_value = [
            {"id": f"meta_{i}", "payload": {"objective_score": 1.0}} for i in range(10)
        ]

        retriever = KBRetriever(repository=mock_repository, embedder=mock_embedder)
        basic_request.query_regime = RegimeSnapshot(regime_tags=["high_vol"])

        result = await retriever.retrieve(basic_request)

        assert result.stats.used_metadata_fallback is True
        assert "metadata_only_fallback" in result.warnings
        assert "embedding_failed" in result.warnings
        assert len(result.candidates) == 10

    @pytest.mark.asyncio
    async def test_candidates_tagged_relaxed(
        self, mock_repository, mock_embedder, basic_request
    ):
        """Should tag relaxed candidates correctly."""
        mock_repository.search.side_effect = [
            [{"id": "strict_0", "score": 0.9, "payload": {}}],
            [{"id": "relaxed_0", "score": 0.8, "payload": {}}],
        ]

        retriever = KBRetriever(repository=mock_repository, embedder=mock_embedder)
        basic_request.query_regime = RegimeSnapshot(regime_tags=["high_vol"])

        result = await retriever.retrieve(basic_request)

        strict_cand = [c for c in result.candidates if c.point_id == "strict_0"][0]
        relaxed_cand = [c for c in result.candidates if c.point_id == "relaxed_0"][0]

        assert strict_cand._relaxed is False
        assert relaxed_cand._relaxed is True


# =============================================================================
# RetrievalCandidate Tests
# =============================================================================


class TestRetrievalCandidate:
    """Tests for RetrievalCandidate dataclass."""

    def test_default_values(self):
        """Should have correct defaults."""
        candidate = RetrievalCandidate(
            point_id="test",
            payload={},
        )

        assert candidate.similarity_score == 0.0
        assert candidate._relaxed is False
        assert candidate._metadata_only is False

    def test_all_fields(self):
        """Should store all fields."""
        candidate = RetrievalCandidate(
            point_id="test",
            payload={"key": "value"},
            similarity_score=0.95,
            _relaxed=True,
            _metadata_only=True,
        )

        assert candidate.point_id == "test"
        assert candidate.payload == {"key": "value"}
        assert candidate.similarity_score == 0.95
        assert candidate._relaxed is True
        assert candidate._metadata_only is True
