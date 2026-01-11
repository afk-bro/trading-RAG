"""Unit tests for KB trial ingestion idempotency primitives."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import pytest

from app.services.kb.idempotency import (
    KB_NAMESPACE,
    BatchIngestResult,
    IndexEntry,
    IngestAction,
    IngestResult,
    compute_content_hash,
    compute_content_hash_from_trial,
    compute_point_id,
)


class TestComputePointId:
    """Tests for deterministic point ID generation."""

    def test_deterministic_same_inputs(self):
        """Same inputs produce same point ID."""
        workspace_id = UUID("11111111-1111-1111-1111-111111111111")
        source_type = "tune_run"
        source_id = UUID("22222222-2222-2222-2222-222222222222")

        result1 = compute_point_id(workspace_id, source_type, source_id)
        result2 = compute_point_id(workspace_id, source_type, source_id)

        assert result1 == result2

    def test_different_workspace_different_id(self):
        """Different workspace produces different point ID."""
        ws1 = UUID("11111111-1111-1111-1111-111111111111")
        ws2 = UUID("33333333-3333-3333-3333-333333333333")
        source_type = "tune_run"
        source_id = UUID("22222222-2222-2222-2222-222222222222")

        result1 = compute_point_id(ws1, source_type, source_id)
        result2 = compute_point_id(ws2, source_type, source_id)

        assert result1 != result2

    def test_different_source_type_different_id(self):
        """Different source type produces different point ID."""
        workspace_id = UUID("11111111-1111-1111-1111-111111111111")
        source_id = UUID("22222222-2222-2222-2222-222222222222")

        result1 = compute_point_id(workspace_id, "tune_run", source_id)
        result2 = compute_point_id(workspace_id, "test_variant", source_id)

        assert result1 != result2

    def test_different_source_id_different_id(self):
        """Different source ID produces different point ID."""
        workspace_id = UUID("11111111-1111-1111-1111-111111111111")
        source_type = "tune_run"
        src1 = UUID("22222222-2222-2222-2222-222222222222")
        src2 = UUID("44444444-4444-4444-4444-444444444444")

        result1 = compute_point_id(workspace_id, source_type, src1)
        result2 = compute_point_id(workspace_id, source_type, src2)

        assert result1 != result2

    def test_returns_uuid(self):
        """Returns a valid UUID."""
        result = compute_point_id(uuid4(), "tune_run", uuid4())
        assert isinstance(result, UUID)

    def test_uses_kb_namespace(self):
        """Uses the KB namespace for UUID5."""
        # KB_NAMESPACE is fixed, so we can verify the output is deterministic
        workspace_id = UUID("11111111-1111-1111-1111-111111111111")
        source_type = "tune_run"
        source_id = UUID("22222222-2222-2222-2222-222222222222")

        result = compute_point_id(workspace_id, source_type, source_id)

        # The result should be a UUID5 (version 5)
        assert result.version == 5


class TestComputeContentHash:
    """Tests for content hash computation."""

    def test_deterministic_same_inputs(self):
        """Same inputs produce same hash."""
        hash1 = compute_content_hash(
            embed_text="test text",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={"lookback": 52},
            sharpe_oos=1.5,
            return_frac_oos=0.15,
            max_dd_frac_oos=0.10,
            regime_schema_version="regime_v1",
        )
        hash2 = compute_content_hash(
            embed_text="test text",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={"lookback": 52},
            sharpe_oos=1.5,
            return_frac_oos=0.15,
            max_dd_frac_oos=0.10,
            regime_schema_version="regime_v1",
        )

        assert hash1 == hash2

    def test_different_text_different_hash(self):
        """Different embed text produces different hash."""
        hash1 = compute_content_hash(
            embed_text="text one",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={},
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )
        hash2 = compute_content_hash(
            embed_text="text two",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={},
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )

        assert hash1 != hash2

    def test_different_metrics_different_hash(self):
        """Different metrics produce different hash."""
        base_args = dict(
            embed_text="test",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={},
            regime_schema_version="regime_v1",
        )

        hash1 = compute_content_hash(
            **base_args,
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
        )
        hash2 = compute_content_hash(
            **base_args,
            sharpe_oos=2.0,  # Different sharpe
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
        )

        assert hash1 != hash2

    def test_different_params_different_hash(self):
        """Different strategy params produce different hash."""
        base_args = dict(
            embed_text="test",
            collection_name="kb_trials",
            strategy_name="breakout",
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )

        hash1 = compute_content_hash(**base_args, params={"lookback": 52})
        hash2 = compute_content_hash(**base_args, params={"lookback": 100})

        assert hash1 != hash2

    def test_returns_sha256_hex(self):
        """Returns 64-character hex string (SHA256)."""
        result = compute_content_hash(
            embed_text="test",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={},
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_params_order_independent(self):
        """Param dict order doesn't affect hash (JSON sort_keys)."""
        hash1 = compute_content_hash(
            embed_text="test",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={"a": 1, "b": 2},
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )
        hash2 = compute_content_hash(
            embed_text="test",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={"b": 2, "a": 1},  # Different order
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )

        assert hash1 == hash2

    def test_optional_fields_included(self):
        """Optional experiment_type and kb_status affect hash."""
        base_args = dict(
            embed_text="test",
            collection_name="kb_trials",
            strategy_name="breakout",
            params={},
            sharpe_oos=1.0,
            return_frac_oos=0.10,
            max_dd_frac_oos=0.05,
            regime_schema_version="regime_v1",
        )

        hash1 = compute_content_hash(**base_args)
        hash2 = compute_content_hash(**base_args, experiment_type="tune")
        hash3 = compute_content_hash(**base_args, kb_status="promoted")

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3


class TestComputeContentHashFromTrial:
    """Tests for compute_content_hash_from_trial wrapper."""

    @dataclass
    class MockRegime:
        schema_version: str = "regime_v1"

    @dataclass
    class MockTrial:
        strategy_name: str = "breakout"
        params: dict = None
        sharpe_oos: float = 1.5
        return_frac_oos: float = 0.15
        max_dd_frac_oos: float = 0.10
        regime_oos: Optional["TestComputeContentHashFromTrial.MockRegime"] = None

        def __post_init__(self):
            if self.params is None:
                self.params = {"lookback": 52}

    def test_extracts_regime_version(self):
        """Extracts regime schema version from trial."""
        trial = self.MockTrial(regime_oos=self.MockRegime(schema_version="regime_v2"))

        result = compute_content_hash_from_trial(
            trial=trial,
            collection_name="kb_trials",
            embed_text="test text",
        )

        # Verify it's a valid hash
        assert len(result) == 64

    def test_handles_missing_regime(self):
        """Handles trial without regime_oos."""
        trial = self.MockTrial(regime_oos=None)

        result = compute_content_hash_from_trial(
            trial=trial,
            collection_name="kb_trials",
            embed_text="test text",
        )

        assert len(result) == 64

    def test_passes_optional_args(self):
        """Passes experiment_type and kb_status."""
        trial = self.MockTrial()

        hash1 = compute_content_hash_from_trial(
            trial=trial,
            collection_name="kb_trials",
            embed_text="test text",
        )
        hash2 = compute_content_hash_from_trial(
            trial=trial,
            collection_name="kb_trials",
            embed_text="test text",
            experiment_type="tune",
            kb_status="promoted",
        )

        assert hash1 != hash2


class TestIngestAction:
    """Tests for IngestAction enum."""

    def test_all_actions_defined(self):
        """All expected actions are defined."""
        assert IngestAction.INSERTED.value == "inserted"
        assert IngestAction.UPDATED.value == "updated"
        assert IngestAction.SKIPPED.value == "skipped"
        assert IngestAction.UNARCHIVED.value == "unarchived"

    def test_is_string_enum(self):
        """Actions can be used as strings."""
        assert str(IngestAction.INSERTED) == "IngestAction.INSERTED"
        assert IngestAction.INSERTED.value == "inserted"


class TestIngestResult:
    """Tests for IngestResult dataclass."""

    def test_create_success_result(self):
        """Creates successful ingest result."""
        source_id = uuid4()
        point_id = uuid4()

        result = IngestResult(
            source_type="tune_run",
            source_id=source_id,
            action=IngestAction.INSERTED,
            point_id=point_id,
            content_hash="abc123",
        )

        assert result.source_type == "tune_run"
        assert result.source_id == source_id
        assert result.action == IngestAction.INSERTED
        assert result.point_id == point_id
        assert result.content_hash == "abc123"
        assert result.error is None

    def test_create_error_result(self):
        """Creates error ingest result."""
        source_id = uuid4()
        point_id = uuid4()

        result = IngestResult(
            source_type="tune_run",
            source_id=source_id,
            action=IngestAction.SKIPPED,
            point_id=point_id,
            error="Embedding failed",
        )

        assert result.error == "Embedding failed"


class TestIndexEntry:
    """Tests for IndexEntry dataclass."""

    def test_create_active_entry(self):
        """Creates active index entry."""
        entry = IndexEntry(
            id=uuid4(),
            workspace_id=uuid4(),
            source_type="tune_run",
            source_id=uuid4(),
            qdrant_point_id=uuid4(),
            content_hash="abc123",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name="kb_trials",
            ingested_at=datetime.utcnow(),
        )

        assert entry.archived_at is None
        assert entry.archived_reason is None
        assert entry.archived_by is None

    def test_create_archived_entry(self):
        """Creates archived index entry."""
        entry = IndexEntry(
            id=uuid4(),
            workspace_id=uuid4(),
            source_type="tune_run",
            source_id=uuid4(),
            qdrant_point_id=uuid4(),
            content_hash="abc123",
            content_hash_algo="sha256_v1",
            embed_model="nomic-embed-text",
            collection_name="kb_trials",
            ingested_at=datetime.utcnow(),
            archived_at=datetime.utcnow(),
            archived_reason="rejected",
            archived_by="admin:user123",
        )

        assert entry.archived_at is not None
        assert entry.archived_reason == "rejected"
        assert entry.archived_by == "admin:user123"


class TestBatchIngestResult:
    """Tests for BatchIngestResult dataclass."""

    def test_from_results_all_inserted(self):
        """Summarizes all-inserted batch."""
        results = [
            IngestResult(
                source_type="tune_run",
                source_id=uuid4(),
                action=IngestAction.INSERTED,
                point_id=uuid4(),
                content_hash="hash1",
            ),
            IngestResult(
                source_type="tune_run",
                source_id=uuid4(),
                action=IngestAction.INSERTED,
                point_id=uuid4(),
                content_hash="hash2",
            ),
        ]

        batch = BatchIngestResult.from_results(results)

        assert batch.total == 2
        assert batch.inserted == 2
        assert batch.updated == 0
        assert batch.skipped == 0
        assert batch.unarchived == 0
        assert batch.errors == 0
        assert batch.by_source == {"tune_run": 2}
        assert batch.error_details == []

    def test_from_results_mixed_actions(self):
        """Summarizes mixed action batch."""
        results = [
            IngestResult(
                source_type="tune_run",
                source_id=uuid4(),
                action=IngestAction.INSERTED,
                point_id=uuid4(),
            ),
            IngestResult(
                source_type="tune_run",
                source_id=uuid4(),
                action=IngestAction.UPDATED,
                point_id=uuid4(),
            ),
            IngestResult(
                source_type="test_variant",
                source_id=uuid4(),
                action=IngestAction.SKIPPED,
                point_id=uuid4(),
            ),
            IngestResult(
                source_type="test_variant",
                source_id=uuid4(),
                action=IngestAction.UNARCHIVED,
                point_id=uuid4(),
            ),
        ]

        batch = BatchIngestResult.from_results(results)

        assert batch.total == 4
        assert batch.inserted == 1
        assert batch.updated == 1
        assert batch.skipped == 1
        assert batch.unarchived == 1
        assert batch.errors == 0
        assert batch.by_source == {"tune_run": 2, "test_variant": 2}

    def test_from_results_with_errors(self):
        """Summarizes batch with errors."""
        source_id = uuid4()
        results = [
            IngestResult(
                source_type="tune_run",
                source_id=uuid4(),
                action=IngestAction.INSERTED,
                point_id=uuid4(),
            ),
            IngestResult(
                source_type="tune_run",
                source_id=source_id,
                action=IngestAction.SKIPPED,
                point_id=uuid4(),
                error="Embedding timeout",
            ),
        ]

        batch = BatchIngestResult.from_results(results)

        assert batch.total == 2
        assert batch.inserted == 1
        assert batch.errors == 1
        assert len(batch.error_details) == 1
        assert "Embedding timeout" in batch.error_details[0]
        assert str(source_id) in batch.error_details[0]

    def test_from_results_empty(self):
        """Handles empty results list."""
        batch = BatchIngestResult.from_results([])

        assert batch.total == 0
        assert batch.inserted == 0
        assert batch.updated == 0
        assert batch.skipped == 0
        assert batch.unarchived == 0
        assert batch.errors == 0
        assert batch.by_source == {}
        assert batch.error_details == []

    def test_to_dict(self):
        """Converts to API response dict."""
        batch = BatchIngestResult(
            total=10,
            inserted=5,
            updated=2,
            skipped=2,
            unarchived=1,
            errors=0,
            by_source={"tune_run": 6, "test_variant": 4},
            error_details=[],
        )

        result = batch.to_dict()

        assert result == {
            "total": 10,
            "inserted": 5,
            "updated": 2,
            "skipped": 2,
            "unarchived": 1,
            "errors": 0,
            "by_source": {"tune_run": 6, "test_variant": 4},
        }
        # error_details excluded from dict
        assert "error_details" not in result


class TestKBNamespace:
    """Tests for KB_NAMESPACE constant."""

    def test_is_fixed_uuid(self):
        """KB_NAMESPACE is a fixed UUID."""
        assert isinstance(KB_NAMESPACE, UUID)
        assert str(KB_NAMESPACE) == "c8f4e2a1-5b3d-4c7e-9f1a-2d8b6e0c3a5f"

    def test_namespace_stable_across_imports(self):
        """Namespace is stable across imports."""
        from app.services.kb.idempotency import KB_NAMESPACE as ns1
        from app.services.kb import KB_NAMESPACE as ns2

        assert ns1 == ns2
