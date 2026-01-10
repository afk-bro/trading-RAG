"""Unit tests for KB trials admin schemas."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.admin.kb_trials_schemas import (
    BulkStatusRequest,
    BulkStatusResponse,
    MarkCandidateRequest,
    PromotionPreviewResponse,
    PromotionPreviewSummary,
    StatusChangeResult,
    TrialPreviewItem,
)


class TestBulkStatusRequest:
    """Tests for BulkStatusRequest validation."""

    def test_valid_request(self):
        """Valid request with required fields."""
        req = BulkStatusRequest(
            source_type="tune_run",
            source_ids=[uuid4(), uuid4()],
        )
        assert req.source_type == "tune_run"
        assert len(req.source_ids) == 2
        assert req.reason is None
        assert req.trigger_ingest is True

    def test_with_reason(self):
        """Request with reason."""
        req = BulkStatusRequest(
            source_type="test_variant",
            source_ids=[uuid4()],
            reason="Manual promotion for testing",
        )
        assert req.reason == "Manual promotion for testing"

    def test_trigger_ingest_false(self):
        """Can disable trigger_ingest."""
        req = BulkStatusRequest(
            source_type="tune_run",
            source_ids=[uuid4()],
            trigger_ingest=False,
        )
        assert req.trigger_ingest is False

    def test_invalid_source_type(self):
        """Rejects invalid source type."""
        with pytest.raises(ValidationError):
            BulkStatusRequest(
                source_type="invalid",
                source_ids=[uuid4()],
            )

    def test_empty_source_ids(self):
        """Rejects empty source_ids list."""
        with pytest.raises(ValidationError):
            BulkStatusRequest(
                source_type="tune_run",
                source_ids=[],
            )

    def test_too_many_source_ids(self):
        """Rejects more than 100 source_ids."""
        with pytest.raises(ValidationError):
            BulkStatusRequest(
                source_type="tune_run",
                source_ids=[uuid4() for _ in range(101)],
            )

    def test_reason_max_length(self):
        """Rejects reason longer than 500 chars."""
        with pytest.raises(ValidationError):
            BulkStatusRequest(
                source_type="tune_run",
                source_ids=[uuid4()],
                reason="x" * 501,
            )


class TestStatusChangeResult:
    """Tests for StatusChangeResult."""

    def test_success_result(self):
        """Successful status change result."""
        source_id = uuid4()
        result = StatusChangeResult(
            source_id=source_id,
            status="promoted",
        )
        assert result.source_id == source_id
        assert result.status == "promoted"
        assert result.error is None

    def test_error_result(self):
        """Error status change result."""
        source_id = uuid4()
        result = StatusChangeResult(
            source_id=source_id,
            status="error",
            error="Trial not found",
        )
        assert result.error == "Trial not found"

    def test_with_group_id(self):
        """Result with group ID."""
        source_id = uuid4()
        group_id = uuid4()
        result = StatusChangeResult(
            source_id=source_id,
            group_id=group_id,
            status="promoted",
        )
        assert result.group_id == group_id


class TestBulkStatusResponse:
    """Tests for BulkStatusResponse."""

    def test_all_updated(self):
        """Response when all items updated."""
        response = BulkStatusResponse(
            updated=5,
            skipped=0,
            ingested=5,
            errors=0,
            results=[
                StatusChangeResult(source_id=uuid4(), status="promoted")
                for _ in range(5)
            ],
        )
        assert response.updated == 5
        assert response.skipped == 0
        assert response.ingested == 5
        assert response.errors == 0

    def test_mixed_results(self):
        """Response with mixed results."""
        response = BulkStatusResponse(
            updated=3,
            skipped=1,
            ingested=3,
            errors=1,
            results=[],
        )
        assert response.updated == 3
        assert response.skipped == 1
        assert response.errors == 1


class TestTrialPreviewItem:
    """Tests for TrialPreviewItem."""

    def test_minimal_item(self):
        """Minimal preview item."""
        source_id = uuid4()
        item = TrialPreviewItem(
            source_type="tune_run",
            source_id=source_id,
            experiment_type="tune",
            kb_status="candidate",
        )
        assert item.source_id == source_id
        assert item.passes_auto_gates is False
        assert item.can_promote is False
        assert item.is_eligible is False
        assert item.ineligibility_reasons == []

    def test_full_item(self):
        """Full preview item with all fields."""
        source_id = uuid4()
        group_id = uuid4()
        now = datetime.now(timezone.utc)

        item = TrialPreviewItem(
            source_type="tune_run",
            source_id=source_id,
            group_id=group_id,
            experiment_type="tune",
            strategy_name="breakout",
            kb_status="candidate",
            sharpe_oos=1.5,
            return_frac_oos=0.15,
            max_dd_frac_oos=0.08,
            n_trades_oos=25,
            passes_auto_gates=True,
            can_promote=True,
            is_eligible=True,
            has_regime_is=True,
            has_regime_oos=True,
            regime_schema_version="regime_v1",
            created_at=now,
        )
        assert item.strategy_name == "breakout"
        assert item.sharpe_oos == 1.5
        assert item.passes_auto_gates is True
        assert item.is_eligible is True


class TestPromotionPreviewSummary:
    """Tests for PromotionPreviewSummary."""

    def test_default_values(self):
        """Default summary values."""
        summary = PromotionPreviewSummary()
        assert summary.would_promote == 0
        assert summary.already_promoted == 0
        assert summary.would_skip == 0
        assert summary.missing_regime == 0

    def test_custom_values(self):
        """Custom summary values."""
        summary = PromotionPreviewSummary(
            would_promote=10,
            already_promoted=5,
            would_skip=3,
            missing_regime=2,
        )
        assert summary.would_promote == 10
        assert summary.already_promoted == 5


class TestPromotionPreviewResponse:
    """Tests for PromotionPreviewResponse."""

    def test_full_response(self):
        """Full preview response."""
        response = PromotionPreviewResponse(
            summary=PromotionPreviewSummary(would_promote=10),
            pagination={"limit": 50, "offset": 0, "total": 100},
            trials=[
                TrialPreviewItem(
                    source_type="tune_run",
                    source_id=uuid4(),
                    experiment_type="tune",
                    kb_status="candidate",
                )
            ],
        )
        assert response.summary.would_promote == 10
        assert response.pagination["total"] == 100
        assert len(response.trials) == 1


class TestMarkCandidateRequest:
    """Tests for MarkCandidateRequest."""

    def test_valid_request(self):
        """Valid mark candidate request."""
        req = MarkCandidateRequest(
            source_type="test_variant",
            source_ids=[uuid4(), uuid4()],
        )
        assert req.source_type == "test_variant"
        assert len(req.source_ids) == 2

    def test_invalid_source_type(self):
        """Rejects invalid source type."""
        with pytest.raises(ValidationError):
            MarkCandidateRequest(
                source_type="invalid",
                source_ids=[uuid4()],
            )
