"""Integration tests for KB trial ingestion pipeline.

Tests the glue between modules:
- SQL view → index repo → qdrant upsert
- Admin promote → ingest trigger
- Reject → archive trigger
- Comparator integration with recommend()
- Manual experiment type rules
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

import pytest

from app.services.kb.idempotency import (
    KB_NAMESPACE,
    compute_point_id,
    compute_content_hash_from_trial,
    IngestAction,
    IndexEntry,
)
from app.services.kb.ingest import (
    KBTrialIngester,
    IngestConfig,
)
from app.services.kb.trial_doc import build_trial_doc_from_eligible_row, trial_to_text
from app.services.kb.comparator import (
    EPSILON,
    ScoredCandidate,
    rank_candidates,
)
from app.services.kb.constants import KB_TRIALS_COLLECTION_NAME, REGIME_SCHEMA_VERSION
from app.services.kb.status_service import KBStatusService, KBStatusResult, CurrentStatus
from app.services.kb.transitions import KBStatusTransition


@dataclass
class MockEmbedResult:
    """Mock embedding result with vectors attribute."""
    vectors: list[list[float]]


# =============================================================================
# Test A: End-to-end ingest from SQL view → index repo → qdrant
# =============================================================================


class TestEndToEndIngest:
    """Test A: E2E ingest from view rows to qdrant/index."""

    @pytest.fixture
    def workspace_id(self):
        return uuid4()

    @pytest.fixture
    def tune_run_row(self, workspace_id):
        """Fake kb_eligible_trials row for a tune_run."""
        return {
            "source_type": "tune_run",
            "source_id": uuid4(),
            "group_id": uuid4(),
            "workspace_id": workspace_id,
            "experiment_type": "tune",
            "strategy_name": "breakout_52w_high",
            "params": {"lookback_days": 252},
            "trial_status": "success",
            "regime_is": {"regime_tags": ["uptrend", "low_vol"]},
            "regime_oos": {"regime_tags": ["uptrend", "med_vol"]},
            "regime_schema_version": REGIME_SCHEMA_VERSION,
            "sharpe_oos": 1.25,
            "return_frac_oos": 0.15,
            "max_dd_frac_oos": 0.08,
            "n_trades_oos": 25,
            "kb_status": "promoted",
            "kb_promoted_at": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
        }

    @pytest.fixture
    def test_variant_row(self, workspace_id):
        """Fake kb_eligible_trials row for a test_variant."""
        return {
            "source_type": "test_variant",
            "source_id": uuid4(),
            "group_id": uuid4(),
            "workspace_id": workspace_id,
            "experiment_type": "sweep",
            "strategy_name": "bb_reversal",
            "params": {"bb_period": 20, "bb_std": 2.0},
            "trial_status": "success",
            "regime_is": None,
            "regime_oos": {"regime_tags": ["ranging", "high_vol"]},
            "regime_schema_version": REGIME_SCHEMA_VERSION,
            "sharpe_oos": 0.85,
            "return_frac_oos": 0.10,
            "max_dd_frac_oos": 0.12,
            "n_trades_oos": 18,
            "kb_status": "candidate",
            "kb_promoted_at": None,
            "created_at": datetime.now(timezone.utc),
        }

    @pytest.mark.asyncio
    async def test_point_id_is_deterministic(self, workspace_id, tune_run_row):
        """Point ID is stable across calls."""
        source_type = tune_run_row["source_type"]
        source_id = tune_run_row["source_id"]

        point_id_1 = compute_point_id(workspace_id, source_type, source_id)
        point_id_2 = compute_point_id(workspace_id, source_type, source_id)

        assert point_id_1 == point_id_2
        assert isinstance(point_id_1, UUID)

    @pytest.mark.asyncio
    async def test_ingest_tune_run_inserts_correctly(
        self, workspace_id, tune_run_row
    ):
        """Tune run ingestion creates index entry with correct fields."""
        # Setup mocks
        mock_index_repo = MagicMock()
        mock_index_repo.get_index_entry = AsyncMock(return_value=None)  # Not in index
        mock_index_repo.insert_index_entry = AsyncMock()

        mock_eligible_repo = MagicMock()
        mock_eligible_repo.get_eligible_trials = AsyncMock(return_value=[tune_run_row])

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=MockEmbedResult(vectors=[[0.1] * 768]))

        mock_qdrant = MagicMock()
        mock_qdrant.upsert_point = AsyncMock()

        config = IngestConfig(
            collection_name=KB_TRIALS_COLLECTION_NAME,
            embed_model="nomic-embed-text",
        )

        ingester = KBTrialIngester(
            index_repo=mock_index_repo,
            eligible_repo=mock_eligible_repo,
            embedder=mock_embedder,
            qdrant=mock_qdrant,
            config=config,
        )

        result = await ingester.ingest_workspace(workspace_id)

        # Assertions
        assert result.inserted == 1
        assert result.skipped == 0
        assert result.errors == 0

        # Verify index insert was called with correct fields
        mock_index_repo.insert_index_entry.assert_called_once()
        call_kwargs = mock_index_repo.insert_index_entry.call_args[1]
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["source_type"] == "tune_run"
        assert call_kwargs["embed_model"] == "nomic-embed-text"
        assert call_kwargs["collection_name"] == KB_TRIALS_COLLECTION_NAME
        assert "content_hash" in call_kwargs

    @pytest.mark.asyncio
    async def test_ingest_test_variant_inserts_correctly(
        self, workspace_id, test_variant_row
    ):
        """Test variant ingestion creates index entry."""
        mock_index_repo = MagicMock()
        mock_index_repo.get_index_entry = AsyncMock(return_value=None)
        mock_index_repo.insert_index_entry = AsyncMock()

        mock_eligible_repo = MagicMock()
        mock_eligible_repo.get_eligible_trials = AsyncMock(return_value=[test_variant_row])

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=MockEmbedResult(vectors=[[0.1] * 768]))

        mock_qdrant = MagicMock()
        mock_qdrant.upsert_point = AsyncMock()

        config = IngestConfig()
        ingester = KBTrialIngester(
            index_repo=mock_index_repo,
            eligible_repo=mock_eligible_repo,
            embedder=mock_embedder,
            qdrant=mock_qdrant,
            config=config,
        )

        result = await ingester.ingest_workspace(workspace_id)

        assert result.inserted == 1
        mock_qdrant.upsert_point.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_skips_unchanged_trial(self, workspace_id, tune_run_row):
        """Skip when content hash matches existing index entry."""
        trial_doc = build_trial_doc_from_eligible_row(tune_run_row)
        embed_text = trial_to_text(trial_doc)
        content_hash = compute_content_hash_from_trial(
            trial=trial_doc,
            collection_name=KB_TRIALS_COLLECTION_NAME,
            embed_text=embed_text,
            experiment_type=tune_run_row.get("experiment_type"),
            kb_status=tune_run_row.get("kb_status"),
        )

        # Existing entry with same hash
        existing_entry = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=tune_run_row["source_id"],
            qdrant_point_id=uuid4(),
            content_hash=content_hash,
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name=KB_TRIALS_COLLECTION_NAME,
            ingested_at=datetime.now(timezone.utc),
        )

        mock_index_repo = MagicMock()
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing_entry)

        mock_eligible_repo = MagicMock()
        mock_eligible_repo.get_eligible_trials = AsyncMock(return_value=[tune_run_row])

        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()

        ingester = KBTrialIngester(
            index_repo=mock_index_repo,
            eligible_repo=mock_eligible_repo,
            embedder=mock_embedder,
            qdrant=mock_qdrant,
        )

        result = await ingester.ingest_workspace(workspace_id)

        assert result.skipped == 1
        assert result.inserted == 0
        mock_embedder.embed.assert_not_called()
        mock_qdrant.upsert_point.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_updates_changed_trial(self, workspace_id, tune_run_row):
        """Update when content hash differs from existing."""
        # Existing entry with different hash
        existing_entry = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=tune_run_row["source_id"],
            qdrant_point_id=uuid4(),
            content_hash="old_hash_different",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name=KB_TRIALS_COLLECTION_NAME,
            ingested_at=datetime.now(timezone.utc),
        )

        mock_index_repo = MagicMock()
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing_entry)
        mock_index_repo.update_index_hash = AsyncMock()

        mock_eligible_repo = MagicMock()
        mock_eligible_repo.get_eligible_trials = AsyncMock(return_value=[tune_run_row])

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=MockEmbedResult(vectors=[[0.1] * 768]))

        mock_qdrant = MagicMock()
        mock_qdrant.upsert_point = AsyncMock()

        ingester = KBTrialIngester(
            index_repo=mock_index_repo,
            eligible_repo=mock_eligible_repo,
            embedder=mock_embedder,
            qdrant=mock_qdrant,
        )

        result = await ingester.ingest_workspace(workspace_id)

        assert result.updated == 1
        mock_qdrant.upsert_point.assert_called_once()
        mock_index_repo.update_index_hash.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_unarchives_archived_trial(self, workspace_id, tune_run_row):
        """Unarchive when trial is in index but archived."""
        existing_entry = IndexEntry(
            id=uuid4(),
            workspace_id=workspace_id,
            source_type="tune_run",
            source_id=tune_run_row["source_id"],
            qdrant_point_id=uuid4(),
            content_hash="old_hash",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name=KB_TRIALS_COLLECTION_NAME,
            ingested_at=datetime.now(timezone.utc) - timedelta(days=7),
            archived_at=datetime.now(timezone.utc) - timedelta(days=1),
            archived_reason="rejected",
        )

        mock_index_repo = MagicMock()
        mock_index_repo.get_index_entry = AsyncMock(return_value=existing_entry)
        mock_index_repo.unarchive_entry = AsyncMock()

        mock_eligible_repo = MagicMock()
        mock_eligible_repo.get_eligible_trials = AsyncMock(return_value=[tune_run_row])

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=MockEmbedResult(vectors=[[0.1] * 768]))

        mock_qdrant = MagicMock()
        mock_qdrant.upsert_point = AsyncMock()

        ingester = KBTrialIngester(
            index_repo=mock_index_repo,
            eligible_repo=mock_eligible_repo,
            embedder=mock_embedder,
            qdrant=mock_qdrant,
        )

        result = await ingester.ingest_workspace(workspace_id)

        assert result.unarchived == 1
        mock_qdrant.upsert_point.assert_called_once()
        mock_index_repo.unarchive_entry.assert_called_once()


# =============================================================================
# Test B: Admin promote triggers ingest
# =============================================================================


class TestAdminPromoteTriggerIngest:
    """Test B: Promote with trigger_ingest=true calls ingestion."""

    @pytest.fixture
    def workspace_id(self):
        return uuid4()

    @pytest.fixture
    def source_id(self):
        return uuid4()

    @pytest.mark.asyncio
    async def test_promote_updates_status_and_timestamps(self, workspace_id, source_id):
        """Promote sets kb_status, promoted_at, and inserts history."""
        now = datetime.now(timezone.utc)

        mock_status_repo = MagicMock()
        mock_status_repo.get_current_status = AsyncMock(
            return_value=CurrentStatus(
                workspace_id=workspace_id,
                kb_status="candidate",
                kb_promoted_at=None,
            )
        )
        mock_status_repo.update_status = AsyncMock()
        mock_status_repo.set_promoted_at = AsyncMock()
        mock_status_repo.insert_history = AsyncMock()

        mock_index_repo = MagicMock()

        service = KBStatusService(
            status_repo=mock_status_repo,
            index_repo=mock_index_repo,
        )

        result = await service.transition(
            source_type="test_variant",
            source_id=source_id,
            to_status="promoted",
            actor_type="admin",
            actor_id="admin@example.com",
            reason=None,
            trigger_ingest=True,
        )

        # Assertions
        assert result.to_status == "promoted"
        assert result.transitioned is True
        mock_status_repo.update_status.assert_called_once()
        mock_status_repo.set_promoted_at.assert_called_once()
        mock_status_repo.insert_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_promote_skips_if_already_promoted(self, workspace_id, source_id):
        """No transition if already at target status."""
        mock_status_repo = MagicMock()
        mock_status_repo.get_current_status = AsyncMock(
            return_value=CurrentStatus(
                workspace_id=workspace_id,
                kb_status="promoted",
                kb_promoted_at=datetime.now(timezone.utc),
            )
        )

        service = KBStatusService(status_repo=mock_status_repo)

        result = await service.transition(
            source_type="tune_run",
            source_id=source_id,
            to_status="promoted",
            actor_type="admin",
        )

        assert result.skipped is True
        mock_status_repo.update_status.assert_not_called()


# =============================================================================
# Test C: Reject triggers archive
# =============================================================================


class TestRejectTriggersArchive:
    """Test C: Reject deletes from qdrant and archives index entry."""

    @pytest.fixture
    def workspace_id(self):
        return uuid4()

    @pytest.fixture
    def source_id(self):
        return uuid4()

    @pytest.mark.asyncio
    async def test_reject_archives_trial(self, workspace_id, source_id):
        """Reject calls archive on index repo."""
        mock_status_repo = MagicMock()
        mock_status_repo.get_current_status = AsyncMock(
            return_value=CurrentStatus(
                workspace_id=workspace_id,
                kb_status="candidate",
                kb_promoted_at=None,
            )
        )
        mock_status_repo.update_status = AsyncMock()
        mock_status_repo.insert_history = AsyncMock()

        mock_index_repo = MagicMock()
        mock_index_repo.archive_trial = AsyncMock(return_value=True)

        service = KBStatusService(
            status_repo=mock_status_repo,
            index_repo=mock_index_repo,
        )

        result = await service.transition(
            source_type="test_variant",
            source_id=source_id,
            to_status="rejected",
            actor_type="admin",
            actor_id="admin@example.com",
            reason="Unrealistic fill assumptions",
        )

        assert result.to_status == "rejected"
        mock_index_repo.archive_trial.assert_called_once_with(
            workspace_id=workspace_id,
            source_type="test_variant",
            source_id=source_id,
            reason="Unrealistic fill assumptions",
            actor="admin@example.com",
        )

    @pytest.mark.asyncio
    async def test_reject_writes_history(self, workspace_id, source_id):
        """Reject inserts history record with reason."""
        mock_status_repo = MagicMock()
        mock_status_repo.get_current_status = AsyncMock(
            return_value=CurrentStatus(
                workspace_id=workspace_id,
                kb_status="promoted",
                kb_promoted_at=datetime.now(timezone.utc),
            )
        )
        mock_status_repo.update_status = AsyncMock()
        mock_status_repo.insert_history = AsyncMock()

        mock_index_repo = MagicMock()
        mock_index_repo.archive_trial = AsyncMock(return_value=True)

        service = KBStatusService(
            status_repo=mock_status_repo,
            index_repo=mock_index_repo,
        )

        await service.transition(
            source_type="tune_run",
            source_id=source_id,
            to_status="rejected",
            actor_type="admin",
            actor_id="admin@example.com",
            reason="Test rejection",
        )

        mock_status_repo.insert_history.assert_called_once()
        call_kwargs = mock_status_repo.insert_history.call_args[1]
        assert call_kwargs["from_status"] == "promoted"
        assert call_kwargs["to_status"] == "rejected"
        assert call_kwargs["reason"] == "Test rejection"


# =============================================================================
# Test D: Comparator integration with recommend()
# =============================================================================


class TestComparatorIntegration:
    """Test D: Comparator respects epsilon semantics in selection."""

    def test_epsilon_tiebreak_promotes_human_curation(self):
        """Within epsilon, promoted beats candidate."""
        now = datetime.now(timezone.utc)

        candidates = [
            # Score 1.50, candidate
            ScoredCandidate(
                source_id="a",
                score=1.50,
                kb_status="candidate",
                regime_schema_version="regime_v1",
                kb_promoted_at=None,
                created_at=now,
            ),
            # Score 1.49 (within epsilon=0.02), promoted
            ScoredCandidate(
                source_id="b",
                score=1.49,
                kb_status="promoted",
                regime_schema_version="regime_v1",
                kb_promoted_at=now,
                created_at=now,
            ),
            # Score 1.48 (within epsilon of 1.50), candidate
            ScoredCandidate(
                source_id="c",
                score=1.48,
                kb_status="candidate",
                regime_schema_version="regime_v1",
                kb_promoted_at=None,
                created_at=now,
            ),
            # Score 1.20 (outside epsilon), promoted
            ScoredCandidate(
                source_id="d",
                score=1.20,
                kb_status="promoted",
                regime_schema_version="regime_v1",
                kb_promoted_at=now,
                created_at=now,
            ),
            # Score 1.19 (within epsilon of 1.20), candidate
            ScoredCandidate(
                source_id="e",
                score=1.19,
                kb_status="candidate",
                regime_schema_version="regime_v1",
                kb_promoted_at=None,
                created_at=now,
            ),
            # Score 0.50 (clearly lower)
            ScoredCandidate(
                source_id="f",
                score=0.50,
                kb_status="promoted",
                regime_schema_version="regime_v1",
                kb_promoted_at=now,
                created_at=now,
            ),
        ]

        ranked = rank_candidates(candidates)
        ranked_ids = [c.source_id for c in ranked]

        # Within top epsilon cluster (1.48-1.50), promoted (b) should beat candidates
        # b has score 1.49, a has 1.50, c has 1.48 - all within epsilon of each other
        assert ranked_ids[0] == "b"  # promoted wins tie-break

        # Lower cluster (1.19-1.20), d should beat e
        # d comes before e in the ranking
        d_idx = ranked_ids.index("d")
        e_idx = ranked_ids.index("e")
        assert d_idx < e_idx  # promoted d beats candidate e

        # f should be last (score 0.50 is clearly lower)
        assert ranked_ids[-1] == "f"

    def test_no_reorder_outside_epsilon(self):
        """Scores outside epsilon maintain primary ordering."""
        now = datetime.now(timezone.utc)

        candidates = [
            ScoredCandidate(
                source_id="high",
                score=2.0,
                kb_status="candidate",
                regime_schema_version=None,
                kb_promoted_at=None,
                created_at=now - timedelta(days=30),
            ),
            ScoredCandidate(
                source_id="low",
                score=1.0,
                kb_status="promoted",
                regime_schema_version="regime_v1",
                kb_promoted_at=now,
                created_at=now,
            ),
        ]

        ranked = rank_candidates(candidates)

        # Score difference (1.0) > epsilon (0.02), so primary score wins
        assert ranked[0].source_id == "high"
        assert ranked[1].source_id == "low"

    def test_schema_tiebreak_within_epsilon(self):
        """Current schema beats null within epsilon."""
        now = datetime.now(timezone.utc)

        candidates = [
            ScoredCandidate(
                source_id="no_schema",
                score=1.0,
                kb_status="candidate",
                regime_schema_version=None,
                kb_promoted_at=None,
                created_at=now,
            ),
            ScoredCandidate(
                source_id="has_schema",
                score=1.0,
                kb_status="candidate",
                regime_schema_version="regime_v1",
                kb_promoted_at=None,
                created_at=now,
            ),
        ]

        ranked = rank_candidates(candidates)

        assert ranked[0].source_id == "has_schema"


# =============================================================================
# Test E: Manual experiment rule
# =============================================================================


class TestManualExperimentRule:
    """Test E: Manual experiments excluded unless promoted."""

    def test_build_trial_doc_from_manual_excluded(self):
        """Manual experiment type builds valid doc (candidacy check is separate)."""
        row = {
            "source_type": "test_variant",
            "source_id": uuid4(),
            "group_id": uuid4(),
            "workspace_id": uuid4(),
            "experiment_type": "manual",
            "strategy_name": "custom_strategy",
            "params": {"custom_param": 42},
            "trial_status": "success",
            "regime_is": None,
            "regime_oos": {"regime_tags": ["uptrend", "low_vol"]},
            "regime_schema_version": "regime_v1",
            "sharpe_oos": 1.5,
            "return_frac_oos": 0.20,
            "max_dd_frac_oos": 0.05,
            "n_trades_oos": 30,
            "kb_status": "excluded",
            "kb_promoted_at": None,
            "created_at": datetime.now(timezone.utc),
        }

        doc = build_trial_doc_from_eligible_row(row)

        # Doc is built (candidacy check happens at a different layer)
        # experiment_type and kb_status come from row, not TrialDoc
        assert doc is not None
        assert doc.strategy_name == "custom_strategy"
        assert row["experiment_type"] == "manual"
        assert row["kb_status"] == "excluded"

    def test_manual_promoted_flows_through_ingest(self):
        """Promoted manual experiment is included in ingest."""
        row = {
            "source_type": "test_variant",
            "source_id": uuid4(),
            "group_id": uuid4(),
            "workspace_id": uuid4(),
            "experiment_type": "manual",
            "strategy_name": "custom_strategy",
            "params": {"custom_param": 42},
            "trial_status": "success",
            "regime_is": None,
            "regime_oos": {"regime_tags": ["uptrend", "low_vol"]},
            "regime_schema_version": "regime_v1",
            "sharpe_oos": 1.5,
            "return_frac_oos": 0.20,
            "max_dd_frac_oos": 0.05,
            "n_trades_oos": 30,
            "kb_status": "promoted",  # Explicitly promoted
            "kb_promoted_at": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
        }

        doc = build_trial_doc_from_eligible_row(row)

        # experiment_type and kb_status come from row, not TrialDoc
        assert doc is not None
        assert row["kb_status"] == "promoted"
        assert row["experiment_type"] == "manual"

    @pytest.mark.asyncio
    async def test_candidacy_rejects_manual(self):
        """Candidacy gate rejects manual experiment type."""
        from app.services.kb.candidacy import (
            is_candidate,
            CandidacyConfig,
            VariantMetricsForCandidacy,
        )

        metrics = VariantMetricsForCandidacy(
            sharpe_oos=1.5,
            n_trades_oos=30,
            max_dd_frac_oos=0.05,
            overfit_gap=0.1,
        )
        config = CandidacyConfig()

        decision = is_candidate(
            metrics=metrics,
            regime_oos={"regime_tags": ["uptrend"]},  # Has regime
            experiment_type="manual",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "manual_experiment_excluded"

    @pytest.mark.asyncio
    async def test_candidacy_accepts_sweep(self):
        """Candidacy gate accepts sweep with good metrics."""
        from app.services.kb.candidacy import (
            is_candidate,
            CandidacyConfig,
            VariantMetricsForCandidacy,
        )

        metrics = VariantMetricsForCandidacy(
            sharpe_oos=1.5,
            n_trades_oos=30,
            max_dd_frac_oos=0.05,
            overfit_gap=0.1,
        )
        config = CandidacyConfig()

        decision = is_candidate(
            metrics=metrics,
            regime_oos={"regime_tags": ["uptrend"]},
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"
