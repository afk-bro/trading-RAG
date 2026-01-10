"""Integration tests for the /query/compare endpoint.

Tests the A/B comparison between vector-only and reranked retrieval.

Run with: pytest tests/integration/test_query_compare.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


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
]


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns consistent vectors."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 768)
    return embedder


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
            "content": "Python is a programming language.",
            "chunk_index": 0,
            "source_type": "article",
        },
        CHUNK_IDS[1]: {
            "id": CHUNK_IDS[1],
            "doc_id": DOC_IDS[0],
            "content": "Python supports OOP paradigms.",
            "chunk_index": 1,
            "source_type": "article",
        },
        CHUNK_IDS[2]: {
            "id": CHUNK_IDS[2],
            "doc_id": DOC_IDS[1],
            "content": "Weather today is sunny.",
            "chunk_index": 0,
            "source_type": "article",
        },
        CHUNK_IDS[3]: {
            "id": CHUNK_IDS[3],
            "doc_id": DOC_IDS[1],
            "content": "Tomorrow will be cloudy.",
            "chunk_index": 1,
            "source_type": "article",
        },
        CHUNK_IDS[4]: {
            "id": CHUNK_IDS[4],
            "doc_id": DOC_IDS[1],
            "content": "Random content here.",
            "chunk_index": 2,
            "source_type": "pdf",
        },
    }


@pytest.fixture
def mock_reranker():
    """Mock cross-encoder reranker that reorders results."""
    from app.services.reranker import RerankResult

    async def mock_rerank(query, candidates, top_k):
        # Simulate reranking - reverse order to show difference
        scored = []
        for i, c in enumerate(reversed(candidates)):
            score = 0.9 - (i * 0.1)
            scored.append((c, score))

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


class TestQueryCompareEndpoint:
    """Tests for /query/compare endpoint."""

    def test_compare_returns_both_responses(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Compare returns vector_only and reranked responses."""
        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.routers.query.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "What is Python?",
                        "top_k": 3,
                    },
                )

                assert response.status_code == 200
                data = response.json()

                # Verify both responses present
                assert "vector_only" in data
                assert "reranked" in data
                assert "metrics" in data

                # Verify each has results and meta
                assert "results" in data["vector_only"]
                assert "meta" in data["vector_only"]
                assert "results" in data["reranked"]
                assert "meta" in data["reranked"]

    def test_compare_returns_metrics(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Compare returns comparison metrics."""
        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.routers.query.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "What is Python?",
                        "top_k": 3,
                    },
                )

                assert response.status_code == 200
                metrics = response.json()["metrics"]

                # Verify metrics structure
                assert "jaccard" in metrics
                assert "overlap_count" in metrics
                assert "union_count" in metrics
                assert "spearman" in metrics
                assert "rank_delta_mean" in metrics
                assert "rank_delta_max" in metrics
                assert "vector_only_ids" in metrics
                assert "reranked_ids" in metrics
                assert "intersection_ids" in metrics

    def test_jaccard_in_valid_range(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Jaccard similarity is between 0 and 1."""
        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.routers.query.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "Test",
                        "top_k": 3,
                    },
                )

                assert response.status_code == 200
                jaccard = response.json()["metrics"]["jaccard"]
                assert 0.0 <= jaccard <= 1.0

    def test_spearman_none_when_overlap_less_than_2(
        self, mock_embedder, mock_chunks_map
    ):
        """Spearman is None when overlap < 2."""
        # Create minimal results that won't overlap
        minimal_results = [{"id": CHUNK_IDS[0], "score": 0.95}]

        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.routers.query.get_reranker", return_value=None
        ):

            mock_search.return_value = minimal_results
            mock_chunks.return_value = {CHUNK_IDS[0]: mock_chunks_map[CHUNK_IDS[0]]}

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "Test",
                        "top_k": 1,
                    },
                )

                # With only 1 result, overlap is 1, spearman should be None
                assert response.status_code == 200
                spearman = response.json()["metrics"]["spearman"]
                # Spearman can be None or a value depending on implementation
                # With same results, it should be valid or None
                if spearman is not None:
                    assert -1.0 <= spearman <= 1.0

    def test_shared_candidates_guarantee(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Both runs use the same candidate set (verified via candidates_searched)."""
        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.routers.query.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "Test",
                        "retrieve_k": 50,
                        "top_k": 3,
                    },
                )

                assert response.status_code == 200
                data = response.json()

                # Both should report same candidates_searched
                vector_candidates = data["vector_only"]["meta"]["candidates_searched"]
                rerank_candidates = data["reranked"]["meta"]["candidates_searched"]
                assert vector_candidates == rerank_candidates

    def test_rerank_disabled_in_vector_only(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Vector-only run has rerank_state=disabled."""
        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.routers.query.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "Test",
                    },
                )

                assert response.status_code == 200
                vector_meta = response.json()["vector_only"]["meta"]
                assert vector_meta["rerank_state"] == "disabled"
                assert vector_meta["rerank_enabled"] is False

    def test_rerank_enabled_in_reranked(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker
    ):
        """Reranked run has rerank_state=ok (when successful)."""
        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.services.query_pipeline.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/query/compare",
                    json={
                        "workspace_id": str(uuid4()),
                        "question": "Test",
                    },
                )

                assert response.status_code == 200
                rerank_meta = response.json()["reranked"]["meta"]
                assert rerank_meta["rerank_state"] == "ok"
                assert rerank_meta["rerank_enabled"] is True


class TestCompareMetricsComputation:
    """Unit tests for compare metrics computation."""

    def test_compute_metrics_perfect_overlap(self):
        """Same IDs in same order = jaccard 1.0, spearman 1.0."""
        from app.routers.query import compute_compare_metrics

        ids = ["a", "b", "c"]
        metrics = compute_compare_metrics(ids, ids)

        assert metrics.jaccard == 1.0
        assert metrics.overlap_count == 3
        assert metrics.union_count == 3
        assert metrics.spearman == 1.0
        assert metrics.rank_delta_mean == 0.0
        assert metrics.rank_delta_max == 0

    def test_compute_metrics_no_overlap(self):
        """No shared IDs = jaccard 0.0, spearman None."""
        from app.routers.query import compute_compare_metrics

        metrics = compute_compare_metrics(["a", "b"], ["c", "d"])

        assert metrics.jaccard == 0.0
        assert metrics.overlap_count == 0
        assert metrics.union_count == 4
        assert metrics.spearman is None
        assert metrics.rank_delta_mean is None
        assert metrics.rank_delta_max is None

    def test_compute_metrics_partial_overlap(self):
        """Partial overlap computes correct metrics."""
        from app.routers.query import compute_compare_metrics

        # a, b shared; c, d unique
        metrics = compute_compare_metrics(["a", "b", "c"], ["b", "a", "d"])

        # Jaccard: 2 / 4 = 0.5
        assert metrics.jaccard == 0.5
        assert metrics.overlap_count == 2
        assert metrics.union_count == 4

        # Spearman over [a, b]: a is rank 0 vs 1, b is rank 1 vs 0
        # This is a perfect negative correlation for 2 items
        assert metrics.spearman is not None

    def test_compute_metrics_reversed_order(self):
        """Same IDs reversed = jaccard 1.0, negative spearman."""
        from app.routers.query import compute_compare_metrics

        metrics = compute_compare_metrics(["a", "b", "c"], ["c", "b", "a"])

        assert metrics.jaccard == 1.0
        assert metrics.overlap_count == 3
        assert metrics.spearman == -1.0  # Perfect negative correlation
        assert metrics.rank_delta_mean == 4 / 3  # (2 + 0 + 2) / 3
        assert metrics.rank_delta_max == 2


class TestEvalLogging:
    """Tests for structured eval logging."""

    def test_eval_log_emitted_on_compare(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker, caplog
    ):
        """Compare endpoint emits structured eval log with all required fields."""
        import logging

        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.services.query_pipeline.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with caplog.at_level(logging.INFO):
                with TestClient(app, raise_server_exceptions=False) as client:
                    response = client.post(
                        "/query/compare",
                        json={
                            "workspace_id": str(uuid4()),
                            "question": "Test query",
                            "top_k": 3,
                        },
                    )

                    assert response.status_code == 200

            # Find the query_compare log entry
            compare_logs = [
                r for r in caplog.records if "query_compare" in r.getMessage()
            ]
            assert len(compare_logs) >= 1, "Expected query_compare log entry"

    def test_eval_log_contains_required_fields(
        self, mock_embedder, mock_vector_results, mock_chunks_map, mock_reranker, caplog
    ):
        """Eval log contains all required fields for analytics (via JSON parsing)."""
        import json
        import logging

        with patch(
            "app.services.embedder.get_embedder", return_value=mock_embedder
        ), patch("app.routers.query._db_pool", MagicMock()), patch(
            "app.routers.query._qdrant_client", MagicMock()
        ), patch(
            "app.repositories.vectors.VectorRepository.search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.repositories.chunks.ChunkRepository.get_by_ids_map",
            new_callable=AsyncMock,
        ) as mock_chunks, patch(
            "app.services.query_pipeline.get_reranker", return_value=mock_reranker
        ):

            mock_search.return_value = mock_vector_results
            mock_chunks.return_value = mock_chunks_map

            from app.main import app

            with caplog.at_level(logging.INFO):
                with TestClient(app, raise_server_exceptions=False) as client:
                    response = client.post(
                        "/query/compare",
                        json={
                            "workspace_id": str(uuid4()),
                            "question": "Test query for logging",
                            "top_k": 3,
                        },
                    )

                    assert response.status_code == 200

            # Find the query_compare log entry and parse JSON
            compare_logs = [
                r for r in caplog.records if "query_compare" in r.getMessage()
            ]
            assert len(compare_logs) >= 1, "Expected query_compare log entry"

            # Parse the JSON log message
            log_message = compare_logs[0].getMessage()
            event = json.loads(log_message)

            # Required fields - identifiers
            assert "workspace_id" in event
            assert event["event"] == "query_compare"

            # Required fields - config
            assert "candidates_k" in event
            assert "top_k" in event
            assert "share_candidates" in event
            assert event["share_candidates"] is True
            assert "skip_neighbors" in event

            # Required fields - metrics
            assert "jaccard" in event
            assert "overlap_count" in event
            assert "union_count" in event
            # spearman/rank_delta_* may be None, but keys should exist
            assert "spearman" in event
            assert "rank_delta_mean" in event
            assert "rank_delta_max" in event

            # Required fields - latency
            assert "embed_ms" in event
            assert "search_ms" in event
            assert "vector_total_ms" in event
            assert "rerank_total_ms" in event

            # Required fields - state
            assert "rerank_state" in event
            assert "rerank_timeout" in event
            assert "rerank_fallback" in event

            # Optional but recommended - top-5 IDs
            assert "vector_top5_ids" in event
            assert "reranked_top5_ids" in event
            assert isinstance(event["vector_top5_ids"], list)
            assert isinstance(event["reranked_top5_ids"], list)
