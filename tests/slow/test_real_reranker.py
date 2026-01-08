"""Slow tests for real cross-encoder model inference.

These tests actually load the BGE reranker model and run inference.
They verify semantic ordering works correctly.

Run with: pytest tests/slow/ -v -m slow
Skip with: pytest -m "not slow"

Requirements:
- ~2GB disk space for model download on first run
- ~2GB RAM (or VRAM if using CUDA)
- First run takes 15-30s for model download

Note: These tests use CPU by default to avoid CUDA dependency in CI.
For GPU testing, set RERANKER_DEVICE=cuda environment variable.
"""

import os
import pytest

from app.services.reranker import (
    CrossEncoderReranker,
    RerankCandidate,
    reset_rerankers,
)


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


def make_candidate(
    chunk_id: str,
    text: str,
    vector_score: float = 0.5,
    document_id: str = "doc1",
    chunk_index: int = 0,
) -> RerankCandidate:
    """Create a test candidate."""
    return RerankCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        vector_score=vector_score,
        workspace_id="test",
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset reranker singletons before each test."""
    reset_rerankers()
    yield
    reset_rerankers()


@pytest.fixture
def device():
    """Get device from environment or default to CPU."""
    return os.environ.get("RERANKER_DEVICE", "cpu")


class TestRealCrossEncoderInference:
    """Tests that load the real BGE model and run inference."""

    @pytest.mark.asyncio
    async def test_python_query_ranks_python_content_higher(self, device):
        """Verify model ranks Python content higher for Python query.

        Robust to model variance: asserts Python content is in top 2,
        not strictly first. Different model versions may have slight
        ordering differences while still providing useful reranking.
        """
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate("weather", "The weather today is sunny and warm with clear skies."),
                make_candidate("python", "Python is a high-level programming language used for web development and data science."),
                make_candidate("random", "The cat sat on the mat and looked out the window."),
            ]

            results = await reranker.rerank("What is Python?", candidates, top_k=3)

            # Python content should be in top 2 (robust to model variance)
            assert len(results) == 3
            top_2_ids = {results[0].chunk_id, results[1].chunk_id}
            assert "python" in top_2_ids, f"Expected 'python' in top 2, got {top_2_ids}"

            # Python should have higher rerank score than clearly irrelevant content
            python_result = next(r for r in results if r.chunk_id == "python")
            random_result = next(r for r in results if r.chunk_id == "random")
            assert python_result.rerank_score > random_result.rerank_score

        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_weather_query_ranks_weather_content_higher(self, device):
        """Verify model ranks weather content higher for weather query.

        Robust to model variance: asserts weather content is in top 2,
        not strictly first.
        """
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate("python", "Python is a programming language for building applications."),
                make_candidate("weather", "Tomorrow's weather forecast shows rain and thunderstorms expected."),
                make_candidate("random", "The stock market closed up 2% today on earnings news."),
            ]

            results = await reranker.rerank("What is the weather forecast?", candidates, top_k=3)

            # Weather content should be in top 2 (robust to model variance)
            top_2_ids = {results[0].chunk_id, results[1].chunk_id}
            assert "weather" in top_2_ids, f"Expected 'weather' in top 2, got {top_2_ids}"

        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_scores(self, device):
        """Verify original vector scores are preserved in results."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate("c1", "Text about machine learning algorithms.", vector_score=0.95),
                make_candidate("c2", "Text about cooking recipes.", vector_score=0.85),
            ]

            results = await reranker.rerank("machine learning", candidates, top_k=2)

            # Find results and verify vector_score preserved
            c1_result = next(r for r in results if r.chunk_id == "c1")
            c2_result = next(r for r in results if r.chunk_id == "c2")

            assert c1_result.vector_score == 0.95
            assert c2_result.vector_score == 0.85

        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_rerank_score_in_valid_range(self, device):
        """Verify rerank scores are in valid range."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate("c1", "Relevant content about the query topic."),
                make_candidate("c2", "Completely unrelated random text here."),
            ]

            results = await reranker.rerank("query topic", candidates, top_k=2)

            # Scores should be reasonable (model outputs sigmoid in 0-1 range typically)
            for r in results:
                # BGE reranker outputs can be outside 0-1, but should be finite
                assert -100 < r.rerank_score < 100, f"Score {r.rerank_score} out of expected range"

        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self, device):
        """Empty candidates returns empty list."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            results = await reranker.rerank("test query", [], top_k=5)
            assert results == []
        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, device):
        """Verify top_k limits number of results."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate(f"c{i}", f"Content number {i} about various topics.")
                for i in range(10)
            ]

            results = await reranker.rerank("topics", candidates, top_k=3)

            assert len(results) == 3

        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_rank_field_is_sequential(self, device):
        """Verify rerank_rank is assigned 0, 1, 2... in order."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate(f"c{i}", f"Content {i}")
                for i in range(5)
            ]

            results = await reranker.rerank("content", candidates, top_k=5)

            ranks = [r.rerank_rank for r in results]
            assert ranks == [0, 1, 2, 3, 4]

        finally:
            reranker.close()


class TestModelLoadingBehavior:
    """Tests for model loading and singleton behavior."""

    @pytest.mark.asyncio
    async def test_model_loads_on_first_rerank(self, device):
        """Verify model loads on first rerank call."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            # Model should not be loaded yet
            assert reranker._model is None

            candidates = [make_candidate("c1", "Test content")]
            await reranker.rerank("test", candidates, top_k=1)

            # Model should now be loaded
            assert reranker._model is not None

        finally:
            reranker.close()

    @pytest.mark.asyncio
    async def test_close_releases_model(self, device):
        """Verify close() releases model resources."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [make_candidate("c1", "Test content")]
            await reranker.rerank("test", candidates, top_k=1)
            assert reranker._model is not None

            reranker.close()
            assert reranker._model is None

        finally:
            # Ensure cleanup even if test fails
            if reranker._model is not None:
                reranker.close()


class TestModelIdProperty:
    """Tests for model_id property."""

    def test_default_model_is_bge_reranker(self, device):
        """Default model should be BGE reranker."""
        config = {"device": device}
        reranker = CrossEncoderReranker(config)

        try:
            assert "bge-reranker" in reranker.model_id.lower()
        finally:
            reranker.close()

    def test_custom_model_id_returned(self, device):
        """Custom model ID should be returned."""
        config = {"device": device, "model": "custom/model-name"}
        reranker = CrossEncoderReranker(config)

        try:
            assert reranker.model_id == "custom/model-name"
        finally:
            reranker.close()


class TestConcurrentReranking:
    """Tests for concurrent rerank calls."""

    @pytest.mark.asyncio
    async def test_concurrent_reranks_dont_crash(self, device):
        """Verify concurrent rerank calls work without CUDA errors.

        Verifies:
        - All concurrent calls complete successfully
        - Each result respects top_k limit
        - Ranks are sequential 0..N-1
        - No exceptions from thread pool contention
        """
        import asyncio

        config = {"device": device, "max_concurrent": 2}
        reranker = CrossEncoderReranker(config)

        try:
            candidates = [
                make_candidate(f"c{i}", f"Content about topic {i}")
                for i in range(5)
            ]

            top_k = 3

            # Run 5 concurrent rerank calls
            tasks = [
                reranker.rerank(f"query {i}", candidates, top_k=top_k)
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should complete successfully
            assert len(results) == 5

            for i, r in enumerate(results):
                # Result count should respect top_k
                assert len(r) <= top_k, f"Result {i} exceeded top_k: {len(r)} > {top_k}"
                assert len(r) == top_k, f"Result {i} has fewer than top_k: {len(r)} < {top_k}"

                # Ranks should be sequential 0..N-1
                ranks = [item.rerank_rank for item in r]
                expected_ranks = list(range(len(r)))
                assert ranks == expected_ranks, f"Result {i} has non-sequential ranks: {ranks}"

                # Scores should be finite numbers
                for item in r:
                    assert item.rerank_score is not None
                    assert -100 < item.rerank_score < 100, f"Score out of range: {item.rerank_score}"

        finally:
            reranker.close()
