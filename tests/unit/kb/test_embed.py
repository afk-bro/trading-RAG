"""Unit tests for KB embedding adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.services.kb.embed import (
    KBEmbeddingAdapter,
    EmbeddingResult,
    EmbeddingError,
    EmbeddingTimeoutError,
    EmbeddingBatchError,
    EmbeddingServiceError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Create mock OllamaEmbedder."""
    embedder = MagicMock()
    embedder.model = "nomic-embed-text"
    embedder.base_url = "http://localhost:11434"
    embedder.get_dimension = AsyncMock(return_value=768)
    embedder.health_check = AsyncMock(return_value=True)
    return embedder


@pytest.fixture
def adapter(mock_embedder):
    """Create KB embedding adapter with mock embedder."""
    return KBEmbeddingAdapter(
        embedder=mock_embedder,
        batch_size=4,
        timeout=10,
        max_retries=2,
        min_batch_size=1,
    )


# =============================================================================
# Basic Tests
# =============================================================================


class TestBasicFunctionality:
    """Tests for basic embedding functionality."""

    def test_model_id(self, adapter, mock_embedder):
        """Should return model ID from embedder."""
        assert adapter.model_id == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_get_vector_dim(self, adapter, mock_embedder):
        """Should return vector dimension from embedder."""
        dim = await adapter.get_vector_dim()
        assert dim == 768
        mock_embedder.get_dimension.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_vector_dim_cached(self, adapter, mock_embedder):
        """Should cache vector dimension."""
        await adapter.get_vector_dim()
        await adapter.get_vector_dim()
        # Only called once due to caching
        mock_embedder.get_dimension.assert_called_once()

    def test_collection_name(self, adapter):
        """Should generate versioned collection name."""
        adapter._vector_dim = 768
        name = adapter.get_collection_name()
        assert name == "trading_kb_trials__nomic-embed-text__768"

    def test_collection_name_custom_base(self, adapter):
        """Should use custom base name."""
        adapter._vector_dim = 768
        name = adapter.get_collection_name("my_collection")
        assert name == "my_collection__nomic-embed-text__768"

    @pytest.mark.asyncio
    async def test_health_check(self, adapter, mock_embedder):
        """Should delegate health check to embedder."""
        result = await adapter.health_check()
        assert result is True
        mock_embedder.health_check.assert_called_once()


# =============================================================================
# Embedding Tests
# =============================================================================


class TestEmbedding:
    """Tests for text embedding."""

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self, adapter):
        """Should handle empty input."""
        result = await adapter.embed_texts([])
        assert result.vectors == []
        assert result.failed_indices == []

    @pytest.mark.asyncio
    async def test_embed_texts_success(self, adapter):
        """Should embed texts successfully."""
        texts = ["hello world", "test text"]

        with patch.object(adapter, "_embed_batch_raw") as mock_embed:
            mock_embed.return_value = [[0.1] * 768, [0.2] * 768]

            result = await adapter.embed_texts(texts)

            assert len(result.vectors) == 2
            assert result.model_id == "nomic-embed-text"
            assert result.vector_dim == 768
            assert result.failed_indices == []

    @pytest.mark.asyncio
    async def test_embed_texts_batching(self, adapter):
        """Should process texts in batches."""
        texts = ["text" + str(i) for i in range(10)]

        async def mock_embed(batch):
            # Return correct number of vectors for actual batch size
            return [[0.1] * 768] * len(batch)

        with patch.object(adapter, "_embed_batch_raw", side_effect=mock_embed):
            result = await adapter.embed_texts(texts)

            # Should call _embed_batch_raw 3 times (4+4+2)
            assert len(result.vectors) == 10

    @pytest.mark.asyncio
    async def test_embed_single(self, adapter):
        """Should embed single text."""
        with patch.object(adapter, "_embed_batch_raw") as mock_embed:
            mock_embed.return_value = [[0.1] * 768]

            vector = await adapter.embed_single("test")

            assert len(vector) == 768


# =============================================================================
# Retry Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry and failure handling."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, adapter):
        """Should retry on timeout."""
        call_count = 0

        async def mock_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("timeout")
            return [[0.1] * 768] * len(texts)

        with patch.object(adapter, "_embed_batch_raw", side_effect=mock_embed):
            result = await adapter.embed_texts(["test"])

            assert call_count == 2
            assert len(result.vectors) == 1

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, adapter):
        """Should retry on server error (5xx)."""
        call_count = 0

        async def mock_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                response = MagicMock()
                response.status_code = 500
                raise httpx.HTTPStatusError(
                    "error", request=MagicMock(), response=response
                )
            return [[0.1] * 768] * len(texts)

        with patch.object(adapter, "_embed_batch_raw", side_effect=mock_embed):
            result = await adapter.embed_texts(["test"])

            assert call_count == 2
            assert len(result.vectors) == 1

    @pytest.mark.asyncio
    async def test_binary_split_on_client_error(self, adapter):
        """Should binary split on client error (4xx)."""
        texts = ["good1", "bad", "good2"]
        call_count = 0

        async def mock_embed(batch):
            nonlocal call_count
            call_count += 1
            if len(batch) == 3:
                # Full batch fails
                response = MagicMock()
                response.status_code = 400
                raise httpx.HTTPStatusError(
                    "error", request=MagicMock(), response=response
                )
            if "bad" in batch:
                # Batch with bad item fails
                response = MagicMock()
                response.status_code = 400
                raise httpx.HTTPStatusError(
                    "error", request=MagicMock(), response=response
                )
            return [[0.1] * 768] * len(batch)

        with patch.object(adapter, "_embed_batch_raw", side_effect=mock_embed):
            result = await adapter.embed_texts(texts, skip_failures=True)

            # Some should succeed, some should fail
            assert call_count >= 3

    @pytest.mark.asyncio
    async def test_skip_failures_mode(self, adapter):
        """Should skip failures when skip_failures=True."""

        async def mock_embed(texts):
            raise httpx.TimeoutException("timeout")

        with patch.object(adapter, "_embed_batch_raw", side_effect=mock_embed):
            result = await adapter.embed_texts(["test1", "test2"], skip_failures=True)

            # All should be marked as failed
            assert len(result.failed_indices) == 2
            assert result.vectors == [[], []]

    @pytest.mark.asyncio
    async def test_raise_on_failure_mode(self, adapter):
        """Should raise when skip_failures=False."""

        async def mock_embed(texts):
            raise httpx.TimeoutException("timeout")

        with patch.object(adapter, "_embed_batch_raw", side_effect=mock_embed):
            with pytest.raises(EmbeddingTimeoutError):
                await adapter.embed_texts(["test"], skip_failures=False)


# =============================================================================
# Error Types Tests
# =============================================================================


class TestErrorTypes:
    """Tests for error type handling."""

    def test_embedding_error_base(self):
        """EmbeddingError should have message and retryable flag."""
        error = EmbeddingError("test error", retryable=True)
        assert error.message == "test error"
        assert error.retryable is True

    def test_embedding_timeout_error(self):
        """EmbeddingTimeoutError should be retryable."""
        error = EmbeddingTimeoutError("timeout")
        assert error.retryable is True

    def test_embedding_batch_error(self):
        """EmbeddingBatchError should track failed indices."""
        error = EmbeddingBatchError("batch failed", failed_indices=[1, 3, 5])
        assert error.failed_indices == [1, 3, 5]
        assert error.retryable is True

    def test_embedding_service_error(self):
        """EmbeddingServiceError should be retryable."""
        error = EmbeddingServiceError("service down")
        assert error.retryable is True


# =============================================================================
# EmbeddingResult Tests
# =============================================================================


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_embedding_result_fields(self):
        """Should have all required fields."""
        result = EmbeddingResult(
            vectors=[[0.1, 0.2]],
            model_id="test-model",
            vector_dim=2,
            failed_indices=[],
        )
        assert result.vectors == [[0.1, 0.2]]
        assert result.model_id == "test-model"
        assert result.vector_dim == 2
        assert result.failed_indices == []

    def test_embedding_result_with_failures(self):
        """Should track failed indices."""
        result = EmbeddingResult(
            vectors=[[0.1], [], [0.3]],
            model_id="test",
            vector_dim=1,
            failed_indices=[1],
        )
        assert result.failed_indices == [1]
        assert result.vectors[1] == []
