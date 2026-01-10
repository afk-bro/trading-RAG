"""Unit tests for KB ingestion pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime

from app.services.kb.ingestion import (
    KBIngestionPipeline,
    IngestionStats,
    IngestionReport,
)
from app.services.kb.embed import EmbeddingResult, EmbeddingError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Create mock KB embedding adapter."""
    embedder = MagicMock()
    embedder.model_id = "nomic-embed-text"
    embedder.get_vector_dim = AsyncMock(return_value=768)
    embedder.get_collection_name = MagicMock(
        return_value="trading_kb_trials__nomic-embed-text__768"
    )
    embedder.embed_texts = AsyncMock(
        return_value=EmbeddingResult(
            vectors=[[0.1] * 768],
            model_id="nomic-embed-text",
            vector_dim=768,
            failed_indices=[],
        )
    )
    embedder.embed_single = AsyncMock(return_value=[0.1] * 768)
    return embedder


@pytest.fixture
def mock_repository():
    """Create mock KB trial repository."""
    repo = MagicMock()
    repo.ensure_collection = AsyncMock()
    repo.upsert_batch = AsyncMock(return_value=1)
    repo.upsert_trial = AsyncMock()
    return repo


@pytest.fixture
def mock_tune_repo():
    """Create mock backtest tune repository."""
    repo = MagicMock()
    repo.list_tune_runs_for_kb = AsyncMock(return_value=[])
    repo.mark_kb_ingested = AsyncMock()
    # Advisory lock support
    repo.try_advisory_lock = AsyncMock(return_value=True)
    repo.release_advisory_lock = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def pipeline(mock_embedder, mock_repository, mock_tune_repo):
    """Create ingestion pipeline with mocks."""
    return KBIngestionPipeline(
        embedder=mock_embedder,
        repository=mock_repository,
        tune_repo=mock_tune_repo,
        batch_size=10,
    )


def make_tune_run(run_id=None, status="completed"):
    """Create a mock tune_run dict."""
    return {
        "tune_run": {
            "id": str(run_id or uuid4()),
            "tune_id": str(uuid4()),
            "status": status,
            "params_json": {"ema_fast": 12, "ema_slow": 26},
            "is_return": 10.5,
            "is_sharpe": 1.2,
            "is_max_drawdown_pct": -15.0,
            "is_total_trades": 50,
            "oos_return": 8.0,
            "oos_sharpe": 0.9,
            "oos_max_drawdown_pct": -18.0,
            "oos_total_trades": 25,
            "created_at": "2026-01-01T00:00:00Z",
        },
        "tune": {
            "id": str(uuid4()),
            "strategy_name": "ema_crossover",
            "objective_type": "sharpe",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "oos_enabled": True,
            "split_pct": 0.7,
        },
    }


# =============================================================================
# Basic Tests
# =============================================================================


class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_creates_with_dependencies(
        self, mock_embedder, mock_repository, mock_tune_repo
    ):
        """Should initialize with provided dependencies."""
        pipeline = KBIngestionPipeline(
            embedder=mock_embedder,
            repository=mock_repository,
            tune_repo=mock_tune_repo,
        )
        assert pipeline._embedder is mock_embedder
        assert pipeline._repository is mock_repository
        assert pipeline._tune_repo is mock_tune_repo

    def test_default_batch_size(self, mock_embedder):
        """Should use default batch size."""
        pipeline = KBIngestionPipeline(embedder=mock_embedder)
        assert pipeline.batch_size == 50  # Default from config


# =============================================================================
# Ingest Tune Runs Tests
# =============================================================================


class TestIngestTuneRuns:
    """Tests for ingest_tune_runs method."""

    @pytest.mark.asyncio
    async def test_empty_runs(self, pipeline, mock_tune_repo):
        """Should handle empty run list."""
        mock_tune_repo.list_tune_runs_for_kb.return_value = []
        workspace_id = uuid4()

        report = await pipeline.ingest_tune_runs(workspace_id=workspace_id)

        assert report.stats.total_fetched == 0
        assert report.stats.total_embedded == 0
        assert report.dry_run is False

    @pytest.mark.asyncio
    async def test_single_run_success(
        self, pipeline, mock_embedder, mock_tune_repo, mock_repository
    ):
        """Should ingest a single run successfully."""
        run = make_tune_run()
        mock_tune_repo.list_tune_runs_for_kb.return_value = [run]
        workspace_id = uuid4()

        report = await pipeline.ingest_tune_runs(workspace_id=workspace_id)

        assert report.stats.total_fetched == 1
        assert report.stats.total_failed == 0
        assert report.model_id == "nomic-embed-text"
        assert report.vector_dim == 768
        mock_repository.ensure_collection.assert_called_once()
        mock_repository.upsert_batch.assert_called()

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, pipeline, mock_tune_repo, mock_repository):
        """Should not upsert in dry run mode."""
        run = make_tune_run()
        mock_tune_repo.list_tune_runs_for_kb.return_value = [run]
        workspace_id = uuid4()

        report = await pipeline.ingest_tune_runs(
            workspace_id=workspace_id,
            dry_run=True,
        )

        assert report.dry_run is True
        assert report.stats.total_upserted == 1  # Would have upserted
        mock_repository.ensure_collection.assert_not_called()
        mock_repository.upsert_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_limit(self, pipeline, mock_tune_repo):
        """Should pass limit to repository."""
        mock_tune_repo.list_tune_runs_for_kb.return_value = []
        workspace_id = uuid4()

        await pipeline.ingest_tune_runs(
            workspace_id=workspace_id,
            limit=100,
        )

        mock_tune_repo.list_tune_runs_for_kb.assert_called_once()
        call_kwargs = mock_tune_repo.list_tune_runs_for_kb.call_args.kwargs
        assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_respects_since(self, pipeline, mock_tune_repo):
        """Should pass since filter to repository."""
        mock_tune_repo.list_tune_runs_for_kb.return_value = []
        workspace_id = uuid4()
        since = datetime(2026, 1, 1)

        await pipeline.ingest_tune_runs(
            workspace_id=workspace_id,
            since=since,
        )

        call_kwargs = mock_tune_repo.list_tune_runs_for_kb.call_args.kwargs
        assert call_kwargs["since"] == since

    @pytest.mark.asyncio
    async def test_only_missing_default(self, pipeline, mock_tune_repo):
        """Should default to only missing vectors."""
        mock_tune_repo.list_tune_runs_for_kb.return_value = []
        workspace_id = uuid4()

        await pipeline.ingest_tune_runs(workspace_id=workspace_id)

        call_kwargs = mock_tune_repo.list_tune_runs_for_kb.call_args.kwargs
        assert call_kwargs["only_missing_kb"] is True

    @pytest.mark.asyncio
    async def test_reembed_overrides_only_missing(self, pipeline, mock_tune_repo):
        """Should fetch all when reembed=True."""
        mock_tune_repo.list_tune_runs_for_kb.return_value = []
        workspace_id = uuid4()

        await pipeline.ingest_tune_runs(
            workspace_id=workspace_id,
            reembed=True,
        )

        call_kwargs = mock_tune_repo.list_tune_runs_for_kb.call_args.kwargs
        assert call_kwargs["only_missing_kb"] is False


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Tests for batch processing logic."""

    @pytest.mark.asyncio
    async def test_batches_correctly(
        self, mock_embedder, mock_repository, mock_tune_repo
    ):
        """Should process runs in batches."""
        pipeline = KBIngestionPipeline(
            embedder=mock_embedder,
            repository=mock_repository,
            tune_repo=mock_tune_repo,
            batch_size=2,
        )

        runs = [make_tune_run() for _ in range(5)]
        mock_tune_repo.list_tune_runs_for_kb.return_value = runs

        # Configure embedder to return vectors for each batch
        async def mock_embed(texts, skip_failures=True):
            return EmbeddingResult(
                vectors=[[0.1] * 768] * len(texts),
                model_id="nomic-embed-text",
                vector_dim=768,
                failed_indices=[],
            )

        mock_embedder.embed_texts = mock_embed

        report = await pipeline.ingest_tune_runs(workspace_id=uuid4())

        assert report.stats.total_fetched == 5
        assert report.stats.total_embedded == 5

    @pytest.mark.asyncio
    async def test_skips_non_completed(self, pipeline, mock_embedder, mock_tune_repo):
        """Should skip runs that aren't completed."""
        runs = [
            make_tune_run(status="completed"),
            make_tune_run(status="running"),  # Should be skipped
            make_tune_run(status="completed"),
        ]
        mock_tune_repo.list_tune_runs_for_kb.return_value = runs

        # Track embed_texts calls
        embed_call_count = 0

        async def mock_embed(texts, skip_failures=True):
            nonlocal embed_call_count
            embed_call_count += 1
            return EmbeddingResult(
                vectors=[[0.1] * 768] * len(texts),
                model_id="nomic-embed-text",
                vector_dim=768,
                failed_indices=[],
            )

        mock_embedder.embed_texts = mock_embed

        report = await pipeline.ingest_tune_runs(workspace_id=uuid4())

        # Only 2 completed runs should be embedded
        assert report.stats.total_skipped == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_embedding_failure_tracked(
        self, pipeline, mock_embedder, mock_tune_repo
    ):
        """Should track embedding failures."""
        run = make_tune_run()
        mock_tune_repo.list_tune_runs_for_kb.return_value = [run]

        # Simulate embedding failure
        mock_embedder.embed_texts.return_value = EmbeddingResult(
            vectors=[[]],  # Empty vector = failed
            model_id="nomic-embed-text",
            vector_dim=768,
            failed_indices=[0],
        )

        report = await pipeline.ingest_tune_runs(workspace_id=uuid4())

        assert report.stats.total_failed == 1
        assert len(report.stats.failed_tune_run_ids) == 1

    @pytest.mark.asyncio
    async def test_upsert_failure_tracked(
        self, pipeline, mock_embedder, mock_tune_repo, mock_repository
    ):
        """Should track upsert failures."""
        run = make_tune_run()
        mock_tune_repo.list_tune_runs_for_kb.return_value = [run]

        # Simulate upsert failure
        mock_repository.upsert_batch.side_effect = Exception("Qdrant error")

        report = await pipeline.ingest_tune_runs(workspace_id=uuid4())

        assert report.stats.upsert_failures == 1
        assert report.stats.total_failed == 1

    @pytest.mark.asyncio
    async def test_no_tune_repo(self, mock_embedder, mock_repository):
        """Should handle missing tune repository gracefully."""
        pipeline = KBIngestionPipeline(
            embedder=mock_embedder,
            repository=mock_repository,
            tune_repo=None,
        )

        report = await pipeline.ingest_tune_runs(workspace_id=uuid4())

        assert report.stats.total_fetched == 0


# =============================================================================
# Ingest Single Tests
# =============================================================================


class TestIngestSingle:
    """Tests for ingest_single method."""

    @pytest.mark.asyncio
    async def test_single_success(
        self, pipeline, mock_embedder, mock_repository, mock_tune_repo
    ):
        """Should ingest single run successfully."""
        run_data = make_tune_run()
        workspace_id = uuid4()

        result = await pipeline.ingest_single(
            tune_run=run_data["tune_run"],
            tune=run_data["tune"],
            workspace_id=workspace_id,
        )

        assert result is True
        mock_embedder.embed_single.assert_called_once()
        mock_repository.upsert_trial.assert_called_once()
        mock_tune_repo.mark_kb_ingested.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_dry_run(
        self, pipeline, mock_embedder, mock_repository, mock_tune_repo
    ):
        """Should not upsert in dry run mode."""
        run_data = make_tune_run()
        workspace_id = uuid4()

        result = await pipeline.ingest_single(
            tune_run=run_data["tune_run"],
            tune=run_data["tune"],
            workspace_id=workspace_id,
            dry_run=True,
        )

        assert result is True
        mock_embedder.embed_single.assert_called_once()
        mock_repository.upsert_trial.assert_not_called()
        mock_tune_repo.mark_kb_ingested.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_skips_non_completed(self, pipeline, mock_embedder):
        """Should skip non-completed runs."""
        run_data = make_tune_run(status="running")
        workspace_id = uuid4()

        result = await pipeline.ingest_single(
            tune_run=run_data["tune_run"],
            tune=run_data["tune"],
            workspace_id=workspace_id,
        )

        assert result is False
        mock_embedder.embed_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_embedding_failure(self, pipeline, mock_embedder):
        """Should return False on embedding failure."""
        run_data = make_tune_run()
        workspace_id = uuid4()

        mock_embedder.embed_single.side_effect = EmbeddingError("timeout")

        result = await pipeline.ingest_single(
            tune_run=run_data["tune_run"],
            tune=run_data["tune"],
            workspace_id=workspace_id,
        )

        assert result is False


# =============================================================================
# Stats and Report Tests
# =============================================================================


class TestIngestionStats:
    """Tests for IngestionStats dataclass."""

    def test_stats_defaults(self):
        """Should have sensible defaults."""
        stats = IngestionStats()
        assert stats.total_fetched == 0
        assert stats.total_embedded == 0
        assert stats.total_failed == 0
        assert stats.failed_tune_run_ids == []

    def test_stats_fields(self):
        """Should track all fields."""
        stats = IngestionStats(
            total_fetched=100,
            total_skipped=10,
            total_embedded=80,
            total_upserted=80,
            total_failed=10,
            failed_tune_run_ids=["id1", "id2"],
            embedding_failures=5,
            upsert_failures=5,
        )
        assert stats.total_fetched == 100
        assert stats.total_failed == 10


class TestIngestionReport:
    """Tests for IngestionReport dataclass."""

    def test_report_fields(self):
        """Should have all required fields."""
        stats = IngestionStats()
        report = IngestionReport(
            workspace_id="test-workspace",
            collection_name="test_collection",
            model_id="nomic-embed-text",
            vector_dim=768,
            stats=stats,
            dry_run=False,
        )
        assert report.workspace_id == "test-workspace"
        assert report.model_id == "nomic-embed-text"
        assert report.dry_run is False
