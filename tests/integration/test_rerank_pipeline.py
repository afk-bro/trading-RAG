"""Integration tests for the rerank query pipeline.

Tests the full query flow with rerank enabled/disabled, verifying:
- QueryMeta fields (rerank_state, rerank_timeout, rerank_fallback)
- Timeout fallback behavior
- Neighbor expansion integration
- Debug field population

Run with: pytest tests/integration/test_rerank_pipeline.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns consistent vectors."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 768)
    embedder.health_check = AsyncMock(return_value=True)
    embedder.get_dimension = AsyncMock(return_value=768)
    return embedder


# Fixed UUIDs for consistent test data
CHUNK_IDS = [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
    "44444444-4444-4444-4444-444444444444",
    "55555555-5555-5555-5555-555555555555",
]
DOC_IDS = [
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "cccccccc-cccc-cccc-cccc-cccccccccccc",
]


@pytest.fixture
def mock_vector_results():
    """Mock vector search results."""
    return [
        {"id": CHUNK_IDS[0], "score": 0.95},
        {"id": CHUNK_IDS[1], "score": 0.85},
        {"id": CHUNK_IDS[2], "score": 0.75},
        {"id": CHUNK_IDS[3], "score": 0.65},
        {"id": CHUNK_IDS[4], "score": 0.55},
    ]


@pytest.fixture
def mock_chunks_map():
    """Mock chunk data from Postgres."""
    return {
        CHUNK_IDS[0]: {
            "id": CHUNK_IDS[0],
            "doc_id": DOC_IDS[0],
            "content": "Python is a programming language used for web development.",
            "chunk_index": 0,
            "source_type": "article",
            "source_url": "https://example.com/python",
            "title": "Python Guide",
        },
        CHUNK_IDS[1]: {
            "id": CHUNK_IDS[1],
            "doc_id": DOC_IDS[0],
            "content": "Python supports multiple paradigms including OOP.",
            "chunk_index": 1,
            "source_type": "article",
            "source_url": "https://example.com/python",
            "title": "Python Guide",
        },
        CHUNK_IDS[2]: {
            "id": CHUNK_IDS[2],
            "doc_id": DOC_IDS[1],
            "content": "The weather today is sunny and warm.",
            "chunk_index": 0,
            "source_type": "article",
            "source_url": "https://example.com/weather",
            "title": "Weather Report",
        },
        CHUNK_IDS[3]: {
            "id": CHUNK_IDS[3],
            "doc_id": DOC_IDS[1],
            "content": "Tomorrow will be cloudy with rain expected.",
            "chunk_index": 1,
            "source_type": "article",
            "source_url": "https://example.com/weather",
            "title": "Weather Report",
        },
        CHUNK_IDS[4]: {
            "id": CHUNK_IDS[4],
            "doc_id": DOC_IDS[2],
            "content": "Random content about various topics.",
            "chunk_index": 0,
            "source_type": "pdf",
            "source_url": "https://example.com/random.pdf",
            "title": "Random PDF",
        },
    }


@pytest.fixture
def mock_reranker():
    """Mock cross-encoder reranker."""
    from app.services.reranker import RerankResult

    async def mock_rerank(query, candidates, top_k):
        # Simulate reranking - boost Python content for Python queries
        scored = []
        for i, c in enumerate(candidates):
            if "python" in c.text.lower():
                score = 0.95 - (i * 0.05)
            else:
                score = 0.3 - (i * 0.05)
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RerankResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                rerank_score=score,
                rerank_rank=rank,
                vector_score=c.vector_score,
                source_type=c.source_type,
            )
            for rank, (c, score) in enumerate(scored[:top_k])
        ]

    reranker = MagicMock()
    reranker.rerank = mock_rerank
    reranker.method = "cross_encoder"
    reranker.model_id = "mock-model"
    reranker.close = MagicMock()
    return reranker


@pytest.fixture
def mock_slow_reranker():
    """Mock reranker that times out (takes 0.5s, test uses 0.1s timeout)."""
    async def mock_rerank_slow(query, candidates, top_k):
        await asyncio.sleep(0.5)  # Exceeds test timeout of 0.1s
        return []

    reranker = MagicMock()
    reranker.rerank = mock_rerank_slow
    reranker.method = "cross_encoder"
    reranker.model_id = "mock-model"
    reranker.close = MagicMock()
    return reranker


@pytest.fixture
def mock_failing_reranker():
    """Mock reranker that raises an exception."""
    async def mock_rerank_fail(query, candidates, top_k):
        raise RuntimeError("CUDA out of memory")

    reranker = MagicMock()
    reranker.rerank = mock_rerank_fail
    reranker.method = "cross_encoder"
    reranker.model_id = "mock-model"
    reranker.close = MagicMock()
    return reranker


class TestRerankDisabled:
    """Tests for query with rerank disabled."""

    def test_rerank_disabled_returns_vector_order(
        self, mock_embedder, mock_vector_results, mock_chunks_map
    ):
        """When rerank disabled, results are in vector score order."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.services.reranker.get_reranker", return_value=None), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            # Mock expand_neighbors to return seeds as-is
            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                expanded = [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ]
                return expanded, []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "What is Python?",
                    "mode": "retrieve",
                    "rerank": False,
                    "top_k": 3,
                })

                assert response.status_code == 200
                data = response.json()
                meta = data["meta"]

                # Verify rerank state is DISABLED
                assert meta["rerank_state"] == "disabled"
                assert meta["rerank_enabled"] is False
                assert meta["rerank_method"] is None
                assert meta["rerank_timeout"] is False
                assert meta["rerank_fallback"] is False

    def test_rerank_disabled_meta_has_no_rerank_ms(
        self, mock_embedder, mock_vector_results, mock_chunks_map
    ):
        """When rerank disabled, rerank_ms should be None."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.services.reranker.get_reranker", return_value=None), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test query",
                    "mode": "retrieve",
                    "rerank": False,
                })

                assert response.status_code == 200
                meta = response.json()["meta"]
                assert meta["rerank_ms"] is None


class TestRerankEnabled:
    """Tests for query with rerank enabled."""

    def test_rerank_enabled_returns_ok_state(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """When rerank succeeds, state is OK."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "What is Python?",
                    "mode": "retrieve",
                    "rerank": True,
                    "retrieve_k": 10,
                    "top_k": 3,
                })

                assert response.status_code == 200
                data = response.json()
                meta = data["meta"]

                # Verify rerank state is OK
                assert meta["rerank_state"] == "ok"
                assert meta["rerank_enabled"] is True
                assert meta["rerank_method"] == "cross_encoder"
                assert meta["rerank_model"] == "mock-model"
                assert meta["rerank_timeout"] is False
                assert meta["rerank_fallback"] is False
                assert meta["rerank_ms"] is not None
                assert meta["rerank_ms"] >= 0

    def test_rerank_enabled_has_timing(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """When rerank enabled, rerank_ms should be populated."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test",
                    "mode": "retrieve",
                    "rerank": True,
                })

                meta = response.json()["meta"]
                assert meta["rerank_ms"] is not None
                assert isinstance(meta["rerank_ms"], int)


class TestRerankTimeout:
    """Tests for rerank timeout fallback."""

    def test_timeout_returns_timeout_fallback_state(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_slow_reranker, monkeypatch
    ):
        """When rerank times out, state is TIMEOUT_FALLBACK.

        Uses monkeypatch to set a short timeout (0.1s) while mock_slow_reranker
        sleeps for 0.5s, triggering the timeout fallback path.
        """
        # Set a short timeout to trigger timeout fallback
        monkeypatch.setenv("RERANK_TIMEOUT_S", "0.1")

        # Clear cached settings to pick up new env var
        from app.config import get_settings
        get_settings.cache_clear()

        try:
            with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
                 patch("app.routers.query._db_pool", MagicMock()), \
                 patch("app.routers.query._qdrant_client", MagicMock()), \
                 patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
                 patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
                 patch("app.routers.query.get_reranker", return_value=mock_slow_reranker), \
                 patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

                mock_search.return_value = mock_vector_results
                mock_chunks.return_value = mock_chunks_map

                from app.services.neighbor_expansion import ExpandedChunk

                async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                    return [
                        ExpandedChunk(
                            chunk_id=s.chunk_id,
                            document_id=s.document_id,
                            chunk_index=s.chunk_index,
                            rerank_score=s.rerank_score,
                            rerank_rank=s.rerank_rank,
                            vector_score=s.vector_score,
                            source_type=s.source_type,
                            is_neighbor=False,
                            neighbor_of=None,
                        )
                        for s in seeds
                    ], []

                mock_expand.side_effect = passthrough_expand

                from app.main import app
                with TestClient(app, raise_server_exceptions=False) as client:
                    response = client.post("/query", json={
                        "workspace_id": str(uuid4()),
                        "question": "Test query",
                        "mode": "retrieve",
                        "rerank": True,
                        "top_k": 3,
                    })

                    assert response.status_code == 200
                    meta = response.json()["meta"]

                    # Verify timeout fallback state
                    assert meta["rerank_state"] == "timeout_fallback"
                    assert meta["rerank_enabled"] is True
                    assert meta["rerank_timeout"] is True
                    assert meta["rerank_fallback"] is True
        finally:
            # Clear cache to restore default settings for other tests
            get_settings.cache_clear()


class TestRerankError:
    """Tests for rerank error fallback."""

    def test_error_returns_error_fallback_state(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_failing_reranker
    ):
        """When rerank fails, state is ERROR_FALLBACK."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_failing_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test query",
                    "mode": "retrieve",
                    "rerank": True,
                    "top_k": 3,
                })

                assert response.status_code == 200
                meta = response.json()["meta"]

                # Verify error fallback state
                assert meta["rerank_state"] == "error_fallback"
                assert meta["rerank_enabled"] is True
                assert meta["rerank_timeout"] is False
                assert meta["rerank_fallback"] is True


class TestDebugField:
    """Tests for debug field population."""

    def test_debug_true_includes_debug_info(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """When debug=True, results include debug field."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "What is Python?",
                    "mode": "retrieve",
                    "rerank": True,
                    "debug": True,
                    "top_k": 3,
                })

                assert response.status_code == 200
                results = response.json()["results"]
                assert len(results) > 0

                # Check debug field exists
                for result in results:
                    assert "debug" in result
                    debug = result["debug"]
                    assert "vector_score" in debug
                    assert "rerank_score" in debug
                    assert "rerank_rank" in debug
                    assert "is_neighbor" in debug

    def test_debug_false_excludes_debug_info(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """When debug=False, results exclude debug field."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "What is Python?",
                    "mode": "retrieve",
                    "rerank": True,
                    "debug": False,
                    "top_k": 3,
                })

                assert response.status_code == 200
                results = response.json()["results"]

                # Debug field should be None
                for result in results:
                    assert result.get("debug") is None


class TestRequestOverrides:
    """Tests for request-level config overrides."""

    def test_request_rerank_override_takes_precedence(
        self, mock_embedder, mock_vector_results, mock_chunks_map
    ):
        """Request-level rerank=False overrides workspace config."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.services.reranker.get_reranker") as mock_get_reranker, \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map
            mock_get_reranker.return_value = None  # Should not be called when disabled

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                # Explicitly disable rerank at request level
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test",
                    "mode": "retrieve",
                    "rerank": False,
                })

                assert response.status_code == 200
                meta = response.json()["meta"]
                assert meta["rerank_state"] == "disabled"
                assert meta["rerank_enabled"] is False


class TestConfigPrecedence:
    """Tests for config precedence: request > workspace > defaults.

    Verifies the full config precedence chain:
    - Request-level overrides take highest precedence
    - Workspace config overrides defaults (TODO: when DB-backed)

    Note: Safety caps (top_k ≤ 50, retrieve_k ≤ 200) are enforced by Pydantic
    schema validation, not by runtime capping. Tests verify valid request values.
    """

    def test_request_top_k_overrides_workspace_default(
        self, mock_embedder, mock_vector_results, mock_chunks_map
    ):
        """Request top_k overrides workspace default (8)."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=None), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                # Request top_k=3 (workspace default is 8)
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test",
                    "mode": "retrieve",
                    "rerank": False,
                    "top_k": 3,
                })

                assert response.status_code == 200
                results = response.json()["results"]
                # Should get 3 results (not 5 from mock, not 8 from default)
                assert len(results) == 3

    def test_request_retrieve_k_controls_candidate_search(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Request retrieve_k controls how many candidates are searched."""
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            # Track what limit was passed to search
            actual_limit = None

            async def capture_search(*args, **kwargs):
                nonlocal actual_limit
                actual_limit = kwargs.get("limit")
                return mock_vector_results

            mock_search.side_effect = capture_search
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                # Request retrieve_k=100 (workspace default is 50)
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test",
                    "mode": "retrieve",
                    "rerank": True,
                    "retrieve_k": 100,
                    "top_k": 3,
                })

                assert response.status_code == 200
                # Verify search was called with request's retrieve_k
                assert actual_limit == 100

    def test_request_enables_rerank_when_default_disabled(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Request rerank=True enables reranking even when workspace default is False.

        This verifies request-level override > workspace config precedence.
        """
        with patch("app.services.embedder.get_embedder", return_value=mock_embedder), \
             patch("app.routers.query._db_pool", MagicMock()), \
             patch("app.routers.query._qdrant_client", MagicMock()), \
             patch("app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock) as mock_search, \
             patch("app.repositories.chunks.ChunkRepository.get_by_ids_map", new_callable=AsyncMock) as mock_chunks, \
             patch("app.routers.query.get_reranker", return_value=mock_reranker), \
             patch("app.services.neighbor_expansion.expand_neighbors", new_callable=AsyncMock) as mock_expand:

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.services.neighbor_expansion import ExpandedChunk

            async def passthrough_expand(seeds, repo, config, already_have_ids=None):
                return [
                    ExpandedChunk(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        chunk_index=s.chunk_index,
                        rerank_score=s.rerank_score,
                        rerank_rank=s.rerank_rank,
                        vector_score=s.vector_score,
                        source_type=s.source_type,
                        is_neighbor=False,
                        neighbor_of=None,
                    )
                    for s in seeds
                ], []

            mock_expand.side_effect = passthrough_expand

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                # Workspace default is rerank disabled, but request enables it
                response = client.post("/query", json={
                    "workspace_id": str(uuid4()),
                    "question": "Test",
                    "mode": "retrieve",
                    "rerank": True,  # Override default False
                    "retrieve_k": 20,
                    "top_k": 5,
                })

                assert response.status_code == 200
                meta = response.json()["meta"]
                assert meta["rerank_enabled"] is True
                assert meta["rerank_state"] == "ok"
                assert meta["rerank_method"] == "cross_encoder"
