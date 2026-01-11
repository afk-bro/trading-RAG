"""Unit tests for KB trial ingestion service."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.services.kb.idempotency import (
    BatchIngestResult,
    IndexEntry,
    IngestAction,
    IngestResult,
    compute_point_id,
)
from app.services.kb.ingest import (
    IngestConfig,
    KBTrialIngester,
)
from app.services.kb.trial_doc import build_trial_doc_from_eligible_row


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def workspace_id():
    """Fixed workspace ID for testing."""
    return UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def sample_eligible_row(workspace_id):
    """Sample row from kb_eligible_trials view."""
    return {
        "source_type": "tune_run",
        "experiment_type": "tune",
        "source_id": UUID("22222222-2222-2222-2222-222222222222"),
        "group_id": UUID("33333333-3333-3333-3333-333333333333"),
        "workspace_id": workspace_id,
        "strategy_entity_id": UUID("44444444-4444-4444-4444-444444444444"),
        "strategy_name": "breakout_52w_high",
        "params": {"lookback": 52, "stop_loss": 0.02},
        "trial_status": "completed",
        "regime_is": None,
        "regime_oos": {
            "schema_version": "regime_v1",
            "atr_pct": 0.02,
            "trend_strength": 0.6,
            "trend_dir": 1,
            "rsi": 55.0,
            "efficiency_ratio": 0.65,
            "regime_tags": ["trending", "low_vol"],
        },
        "regime_schema_version": "regime_v1",
        "sharpe_oos": 1.5,
        "return_frac_oos": 0.15,
        "max_dd_frac_oos": 0.08,
        "n_trades_oos": 25,
        "sharpe_is": 1.8,
        "kb_status": "promoted",
        "kb_promoted_at": datetime.now(timezone.utc),
        "kb_status_changed_at": datetime.now(timezone.utc),
        "objective_type": "sharpe",
        "objective_score": 1.5,
        "created_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_index_repo():
    """Mock KB trial index repository."""
    repo = MagicMock()
    repo.get_index_entry = AsyncMock(return_value=None)
    repo.insert_index_entry = AsyncMock(return_value=uuid4())
    repo.update_index_hash = AsyncMock()
    repo.unarchive_entry = AsyncMock()
    repo.archive_entry = AsyncMock()
    return repo


@pytest.fixture
def mock_eligible_repo(sample_eligible_row):
    """Mock eligible trials repository."""
    repo = MagicMock()
    repo.get_eligible_trials = AsyncMock(return_value=[sample_eligible_row])
    return repo


@pytest.fixture
def mock_embedder():
    """Mock embedding adapter."""
    embedder = MagicMock()
    embed_result = MagicMock()
    embed_result.vectors = [[0.1] * 768]
    embedder.embed = AsyncMock(return_value=embed_result)
    return embedder


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant adapter."""
    qdrant = MagicMock()
    qdrant.upsert_point = AsyncMock()
    qdrant.delete_point = AsyncMock()
    return qdrant


@pytest.fixture
def ingester(mock_index_repo, mock_eligible_repo, mock_embedder, mock_qdrant):
    """Create ingester with all mocked dependencies."""
    return KBTrialIngester(
        index_repo=mock_index_repo,
        eligible_repo=mock_eligible_repo,
        embedder=mock_embedder,
        qdrant=mock_qdrant,
        config=IngestConfig(),
    )


# =============================================================================
# Test build_trial_doc_from_eligible_row
# =============================================================================


class TestBuildTrialDocFromEligibleRow:
    """Tests for building TrialDoc from eligible view rows."""

    def test_builds_from_tune_run(self, sample_eligible_row):
        """Builds TrialDoc from tune_run row."""
        trial = build_trial_doc_from_eligible_row(sample_eligible_row)

        assert trial is not None
        assert trial.strategy_name == "breakout_52w_high"
        assert trial.params == {"lookback": 52, "stop_loss": 0.02}
        assert trial.sharpe_oos == 1.5
        assert trial.sharpe_is == 1.8
        assert trial.return_frac_oos == 0.15
        assert trial.max_dd_frac_oos == 0.08
        assert trial.n_trades_oos == 25
        assert trial.has_oos is True

    def test_builds_from_test_variant(self, sample_eligible_row):
        """Builds TrialDoc from test_variant row."""
        row = sample_eligible_row.copy()
        row["source_type"] = "test_variant"
        row["experiment_type"] = "sweep"

        trial = build_trial_doc_from_eligible_row(row)

        assert trial is not None
        assert trial.strategy_name == "breakout_52w_high"

    def test_extracts_regime_oos(self, sample_eligible_row):
        """Extracts regime snapshot from row."""
        trial = build_trial_doc_from_eligible_row(sample_eligible_row)

        assert trial.regime_oos is not None
        assert trial.regime_oos.atr_pct == 0.02
        assert trial.regime_oos.trend_strength == 0.6
        assert trial.regime_oos.regime_tags == ["trending", "low_vol"]

    def test_handles_missing_regime(self, sample_eligible_row):
        """Handles rows without regime data."""
        row = sample_eligible_row.copy()
        row["regime_is"] = None
        row["regime_oos"] = None

        trial = build_trial_doc_from_eligible_row(row)

        assert trial is not None
        assert trial.regime_is is None
        assert trial.regime_oos is None
        assert "missing_regime_oos" in trial.warnings

    def test_computes_overfit_gap(self, sample_eligible_row):
        """Computes overfit gap from IS/OOS sharpe."""
        trial = build_trial_doc_from_eligible_row(sample_eligible_row)

        # sharpe_is=1.8, sharpe_oos=1.5 → gap = 0.3
        assert trial.overfit_gap == pytest.approx(0.3)

    def test_returns_none_for_unknown_source_type(self, sample_eligible_row):
        """Returns None for unknown source types."""
        row = sample_eligible_row.copy()
        row["source_type"] = "unknown"

        trial = build_trial_doc_from_eligible_row(row)

        assert trial is None

    def test_handles_null_params(self, sample_eligible_row):
        """Handles null params gracefully."""
        row = sample_eligible_row.copy()
        row["params"] = None

        trial = build_trial_doc_from_eligible_row(row)

        assert trial is not None
        assert trial.params == {}


# =============================================================================
# Test KBTrialIngester
# =============================================================================


class TestKBTrialIngesterIngestWorkspace:
    """Tests for workspace ingestion."""

    @pytest.mark.asyncio
    async def test_ingests_new_trial(
        self, ingester, mock_index_repo, mock_qdrant, workspace_id
    ):
        """Ingests a new trial that doesn't exist in index."""
        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 1
        assert result.inserted == 1
        assert result.updated == 0
        assert result.skipped == 0
        assert result.errors == 0

        # Verify Qdrant upsert called
        mock_qdrant.upsert_point.assert_called_once()

        # Verify index insert called
        mock_index_repo.insert_index_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_unchanged_trial(
        self, ingester, mock_index_repo, sample_eligible_row, workspace_id
    ):
        """Skips trial when content hash matches."""
        from app.services.kb.constants import KB_TRIALS_COLLECTION_NAME

        # Setup existing entry with matching hash
        point_id = compute_point_id(
            workspace_id,
            sample_eligible_row["source_type"],
            sample_eligible_row["source_id"],
        )

        # Make the hash computation match by using same data
        from app.services.kb.idempotency import compute_content_hash_from_trial
        from app.services.kb.trial_doc import trial_to_text

        trial = build_trial_doc_from_eligible_row(sample_eligible_row)
        embed_text = trial_to_text(trial)
        # Use the same collection name as the ingester config
        content_hash = compute_content_hash_from_trial(
            trial=trial,
            collection_name=KB_TRIALS_COLLECTION_NAME,
            embed_text=embed_text,
            experiment_type=sample_eligible_row["experiment_type"],
            kb_status=sample_eligible_row["kb_status"],
        )

        existing = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=sample_eligible_row["source_id"],
            qdrant_point_id=point_id,
            content_hash=content_hash,
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name=KB_TRIALS_COLLECTION_NAME,
            ingested_at=datetime.now(timezone.utc),
        )

        mock_index_repo.get_index_entry = AsyncMock(return_value=existing)

        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 1
        assert result.inserted == 0
        assert result.skipped == 1

    @pytest.mark.asyncio
    async def test_updates_changed_trial(
        self, ingester, mock_index_repo, mock_qdrant, sample_eligible_row, workspace_id
    ):
        """Updates trial when content hash differs."""
        point_id = compute_point_id(
            workspace_id,
            sample_eligible_row["source_type"],
            sample_eligible_row["source_id"],
        )
        existing = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=sample_eligible_row["source_id"],
            qdrant_point_id=point_id,
            content_hash="old_hash_that_does_not_match",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name="kb_trials",
            ingested_at=datetime.now(timezone.utc),
        )
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing)

        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 1
        assert result.updated == 1
        assert result.inserted == 0

        # Verify update called
        mock_index_repo.update_index_hash.assert_called_once()
        mock_qdrant.upsert_point.assert_called_once()

    @pytest.mark.asyncio
    async def test_unarchives_archived_trial(
        self, ingester, mock_index_repo, mock_qdrant, sample_eligible_row, workspace_id
    ):
        """Unarchives a previously archived trial."""
        point_id = compute_point_id(
            workspace_id,
            sample_eligible_row["source_type"],
            sample_eligible_row["source_id"],
        )
        existing = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=sample_eligible_row["source_id"],
            qdrant_point_id=point_id,
            content_hash="old_hash",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name="kb_trials",
            ingested_at=datetime.now(timezone.utc),
            archived_at=datetime.now(timezone.utc),  # Was archived
            archived_reason="rejected",
            archived_by="admin",
        )
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing)

        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 1
        assert result.unarchived == 1

        # Verify unarchive called
        mock_index_repo.unarchive_entry.assert_called_once()
        mock_qdrant.upsert_point.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_empty_results(
        self, ingester, mock_eligible_repo, workspace_id
    ):
        """Handles case when no eligible trials found."""
        mock_eligible_repo.get_eligible_trials = AsyncMock(return_value=[])

        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 0
        assert result.inserted == 0

    @pytest.mark.asyncio
    async def test_handles_embedding_error(
        self, ingester, mock_embedder, workspace_id
    ):
        """Handles embedding failures gracefully."""
        embed_result = MagicMock()
        embed_result.vectors = []  # No vectors returned
        mock_embedder.embed = AsyncMock(return_value=embed_result)

        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 1
        # Embedding errors are counted as errors, not skipped
        assert result.errors == 1
        assert "no vectors" in result.error_details[0].lower()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_persist(
        self, mock_index_repo, mock_eligible_repo, mock_embedder, mock_qdrant, workspace_id
    ):
        """Dry run mode doesn't call Qdrant or index."""
        config = IngestConfig(dry_run=True)
        ingester = KBTrialIngester(
            index_repo=mock_index_repo,
            eligible_repo=mock_eligible_repo,
            embedder=mock_embedder,
            qdrant=mock_qdrant,
            config=config,
        )

        result = await ingester.ingest_workspace(workspace_id)

        assert result.total == 1
        assert result.inserted == 1

        # Should NOT call embedding, qdrant, or index
        mock_embedder.embed.assert_not_called()
        mock_qdrant.upsert_point.assert_not_called()
        mock_index_repo.insert_index_entry.assert_not_called()


class TestKBTrialIngesterArchive:
    """Tests for archive functionality."""

    @pytest.mark.asyncio
    async def test_archives_existing_trial(
        self, ingester, mock_index_repo, mock_qdrant, workspace_id
    ):
        """Archives an existing trial."""
        source_id = uuid4()
        point_id = uuid4()
        existing = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=source_id,
            qdrant_point_id=point_id,
            content_hash="somehash",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name="kb_trials",
            ingested_at=datetime.now(timezone.utc),
        )
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing)

        result = await ingester.archive_trial(
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=source_id,
            reason="rejected",
            actor="admin:test",
        )

        assert result is True
        mock_qdrant.delete_point.assert_called_once_with(
            collection_name="kb_trials",
            point_id=point_id,
        )
        mock_index_repo.archive_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_not_found_returns_false(
        self, ingester, mock_index_repo, workspace_id
    ):
        """Returns False when trial not in index."""
        mock_index_repo.get_index_entry = AsyncMock(return_value=None)

        result = await ingester.archive_trial(
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=uuid4(),
            reason="rejected",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_archive_already_archived_returns_true(
        self, ingester, mock_index_repo, workspace_id
    ):
        """Returns True for already archived trial."""
        source_id = uuid4()
        existing = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=source_id,
            qdrant_point_id=uuid4(),
            content_hash="somehash",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name="kb_trials",
            ingested_at=datetime.now(timezone.utc),
            archived_at=datetime.now(timezone.utc),  # Already archived
        )
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing)

        result = await ingester.archive_trial(
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=source_id,
            reason="rejected",
        )

        assert result is True


class TestIngestConfig:
    """Tests for IngestConfig defaults."""

    def test_default_values(self):
        """Default config has expected values."""
        from app.services.kb.constants import KB_TRIALS_COLLECTION_NAME

        config = IngestConfig()

        assert config.collection_name == KB_TRIALS_COLLECTION_NAME
        assert config.embed_model == "nomic-embed-text"
        assert config.content_hash_algo == "sha256_v1"
        assert config.batch_size == 50
        assert config.dry_run is False

    def test_custom_values(self):
        """Can customize config values."""
        config = IngestConfig(
            collection_name="custom_collection",
            embed_model="custom-model",
            dry_run=True,
        )

        assert config.collection_name == "custom_collection"
        assert config.embed_model == "custom-model"
        assert config.dry_run is True
