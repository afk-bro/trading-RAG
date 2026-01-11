"""Unit tests for KB status service."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import pytest

from app.services.kb.status_service import (
    CurrentStatus,
    InvalidTransitionError,
    KBStatusResult,
    KBStatusService,
    TrialNotFoundError,
)
from app.services.kb.transitions import KBStatusTransition


class MockStatusRepository:
    """Mock implementation of KBStatusRepository."""

    def __init__(self):
        self.statuses: dict[tuple[str, UUID], CurrentStatus] = {}
        self.history: list[dict] = []
        self.update_calls: list[dict] = []
        self.promoted_at_calls: list[dict] = []

    def add_status(
        self,
        source_type: str,
        source_id: UUID,
        workspace_id: UUID,
        kb_status: str,
        kb_promoted_at: Optional[datetime] = None,
    ):
        """Add a status entry for testing."""
        self.statuses[(source_type, source_id)] = CurrentStatus(
            workspace_id=workspace_id,
            kb_status=kb_status,
            kb_promoted_at=kb_promoted_at,
        )

    async def get_current_status(
        self, source_type: str, source_id: UUID
    ) -> Optional[CurrentStatus]:
        return self.statuses.get((source_type, source_id))

    async def update_status(
        self,
        source_type: str,
        source_id: UUID,
        to_status: str,
        changed_by: Optional[str],
        reason: Optional[str],
    ) -> None:
        self.update_calls.append({
            "source_type": source_type,
            "source_id": source_id,
            "to_status": to_status,
            "changed_by": changed_by,
            "reason": reason,
        })
        # Update the in-memory status
        key = (source_type, source_id)
        if key in self.statuses:
            current = self.statuses[key]
            self.statuses[key] = CurrentStatus(
                workspace_id=current.workspace_id,
                kb_status=to_status,
                kb_promoted_at=current.kb_promoted_at,
            )

    async def set_promoted_at(
        self,
        source_type: str,
        source_id: UUID,
        promoted_by: Optional[str],
    ) -> None:
        self.promoted_at_calls.append({
            "source_type": source_type,
            "source_id": source_id,
            "promoted_by": promoted_by,
        })

    async def insert_history(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        from_status: str,
        to_status: str,
        actor_type: str,
        actor_id: Optional[str],
        reason: Optional[str],
    ) -> None:
        self.history.append({
            "workspace_id": workspace_id,
            "source_type": source_type,
            "source_id": source_id,
            "from_status": from_status,
            "to_status": to_status,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "reason": reason,
        })


class MockIndexRepository:
    """Mock implementation of KBIndexRepository."""

    def __init__(self):
        self.archive_calls: list[dict] = []
        self.unarchive_calls: list[dict] = []
        # Track which trials are "indexed" for archive/unarchive returns
        self.indexed_trials: set[tuple[str, UUID]] = set()

    def mark_indexed(self, source_type: str, source_id: UUID):
        """Mark a trial as indexed."""
        self.indexed_trials.add((source_type, source_id))

    async def archive_trial(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        reason: str,
        actor: Optional[str],
    ) -> bool:
        self.archive_calls.append({
            "workspace_id": workspace_id,
            "source_type": source_type,
            "source_id": source_id,
            "reason": reason,
            "actor": actor,
        })
        key = (source_type, source_id)
        if key in self.indexed_trials:
            self.indexed_trials.discard(key)
            return True
        return False

    async def unarchive_trial(
        self,
        source_type: str,
        source_id: UUID,
    ) -> bool:
        self.unarchive_calls.append({
            "source_type": source_type,
            "source_id": source_id,
        })
        # For testing, assume unarchive always succeeds if archived
        return True


class TestKBStatusService:
    """Tests for KBStatusService."""

    @pytest.fixture
    def status_repo(self):
        return MockStatusRepository()

    @pytest.fixture
    def index_repo(self):
        return MockIndexRepository()

    @pytest.fixture
    def service(self, status_repo, index_repo):
        return KBStatusService(
            status_repo=status_repo,
            index_repo=index_repo,
        )

    @pytest.fixture
    def workspace_id(self):
        return uuid4()

    @pytest.fixture
    def source_id(self):
        return uuid4()

    # --- Basic transition tests ---

    class TestBasicTransitions:
        """Tests for basic transition functionality."""

        @pytest.fixture
        def status_repo(self):
            return MockStatusRepository()

        @pytest.fixture
        def service(self, status_repo):
            return KBStatusService(status_repo=status_repo)

        @pytest.mark.asyncio
        async def test_excluded_to_candidate_auto(self, service, status_repo):
            """Auto can transition excluded → candidate."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "excluded"
            )

            result = await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="candidate",
                actor_type="auto",
            )

            assert result.from_status == "excluded"
            assert result.to_status == "candidate"
            assert result.skipped is False

        @pytest.mark.asyncio
        async def test_candidate_to_promoted_admin(self, service, status_repo):
            """Admin can transition candidate → promoted."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "candidate"
            )

            result = await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="promoted",
                actor_type="admin",
                actor_id="admin-123",
            )

            assert result.from_status == "candidate"
            assert result.to_status == "promoted"

        @pytest.mark.asyncio
        async def test_promotes_sets_promoted_at(self, service, status_repo):
            """Promotion sets promoted_at timestamp."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "excluded"
            )

            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="promoted",
                actor_type="admin",
                actor_id="admin-123",
            )

            assert len(status_repo.promoted_at_calls) == 1
            call = status_repo.promoted_at_calls[0]
            assert call["source_id"] == source_id
            assert call["promoted_by"] == "admin-123"

        @pytest.mark.asyncio
        async def test_candidate_does_not_set_promoted_at(
            self, service, status_repo
        ):
            """Candidate transition does not set promoted_at."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "excluded"
            )

            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="candidate",
                actor_type="auto",
            )

            assert len(status_repo.promoted_at_calls) == 0

    # --- History logging tests ---

    class TestHistoryLogging:
        """Tests for history/audit logging."""

        @pytest.fixture
        def status_repo(self):
            return MockStatusRepository()

        @pytest.fixture
        def service(self, status_repo):
            return KBStatusService(status_repo=status_repo)

        @pytest.mark.asyncio
        async def test_logs_history_on_transition(self, service, status_repo):
            """Transition logs to history."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "excluded"
            )

            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="candidate",
                actor_type="auto",
            )

            assert len(status_repo.history) == 1
            entry = status_repo.history[0]
            assert entry["workspace_id"] == workspace_id
            assert entry["source_type"] == "test_variant"
            assert entry["source_id"] == source_id
            assert entry["from_status"] == "excluded"
            assert entry["to_status"] == "candidate"
            assert entry["actor_type"] == "auto"

        @pytest.mark.asyncio
        async def test_logs_actor_id_and_reason(self, service, status_repo):
            """History includes actor_id and reason."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "candidate"
            )

            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="rejected",
                actor_type="admin",
                actor_id="admin-456",
                reason="low quality data",
            )

            entry = status_repo.history[0]
            assert entry["actor_id"] == "admin-456"
            assert entry["reason"] == "low quality data"

        @pytest.mark.asyncio
        async def test_no_history_on_skip(self, service, status_repo):
            """No history logged when transition is skipped (same status)."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "candidate"
            )

            result = await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="candidate",
                actor_type="admin",
            )

            assert result.skipped is True
            assert len(status_repo.history) == 0

    # --- Skipped transitions tests ---

    class TestSkippedTransitions:
        """Tests for transitions that are skipped (no-ops)."""

        @pytest.fixture
        def status_repo(self):
            return MockStatusRepository()

        @pytest.fixture
        def service(self, status_repo):
            return KBStatusService(status_repo=status_repo)

        @pytest.mark.asyncio
        async def test_same_status_skips(self, service, status_repo):
            """Same status transition is skipped."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "candidate"
            )

            result = await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="candidate",
                actor_type="admin",
            )

            assert result.skipped is True
            assert result.from_status == "candidate"
            assert result.to_status == "candidate"
            assert len(status_repo.update_calls) == 0

    # --- Error cases ---

    class TestErrorCases:
        """Tests for error handling."""

        @pytest.fixture
        def status_repo(self):
            return MockStatusRepository()

        @pytest.fixture
        def service(self, status_repo):
            return KBStatusService(status_repo=status_repo)

        @pytest.mark.asyncio
        async def test_not_found_raises(self, service, status_repo):
            """Not found trial raises TrialNotFoundError."""
            source_id = uuid4()

            with pytest.raises(TrialNotFoundError) as exc_info:
                await service.transition(
                    source_type="test_variant",
                    source_id=source_id,
                    to_status="candidate",
                    actor_type="auto",
                )

            assert "not found" in str(exc_info.value)

        @pytest.mark.asyncio
        async def test_invalid_transition_raises(self, service, status_repo):
            """Invalid transition raises InvalidTransitionError."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "rejected"
            )

            with pytest.raises(InvalidTransitionError) as exc_info:
                await service.transition(
                    source_type="test_variant",
                    source_id=source_id,
                    to_status="candidate",
                    actor_type="admin",
                )

            assert exc_info.value.error_code == "transition_rejected_to_candidate_not_allowed"

        @pytest.mark.asyncio
        async def test_auto_cannot_promote(self, service, status_repo):
            """Auto cannot promote."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "candidate"
            )

            with pytest.raises(InvalidTransitionError) as exc_info:
                await service.transition(
                    source_type="test_variant",
                    source_id=source_id,
                    to_status="promoted",
                    actor_type="auto",
                )

            assert "auto_cannot" in exc_info.value.error_code

        @pytest.mark.asyncio
        async def test_rejection_requires_reason(self, service, status_repo):
            """Rejection requires reason."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "candidate"
            )

            with pytest.raises(InvalidTransitionError) as exc_info:
                await service.transition(
                    source_type="test_variant",
                    source_id=source_id,
                    to_status="rejected",
                    actor_type="admin",
                )

            assert exc_info.value.error_code == "rejection_requires_reason"

    # --- Archive/unarchive tests ---

    class TestArchiveIntegration:
        """Tests for archive integration on rejection."""

        @pytest.fixture
        def status_repo(self):
            return MockStatusRepository()

        @pytest.fixture
        def index_repo(self):
            return MockIndexRepository()

        @pytest.fixture
        def service(self, status_repo, index_repo):
            return KBStatusService(
                status_repo=status_repo,
                index_repo=index_repo,
            )

        @pytest.mark.asyncio
        async def test_rejection_archives_indexed_trial(
            self, service, status_repo, index_repo
        ):
            """Rejection archives trial if it's in the index."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "promoted"
            )
            index_repo.mark_indexed("test_variant", source_id)

            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="rejected",
                actor_type="admin",
                actor_id="admin-789",
                reason="outlier data",
            )

            assert len(index_repo.archive_calls) == 1
            call = index_repo.archive_calls[0]
            assert call["workspace_id"] == workspace_id
            assert call["source_type"] == "test_variant"
            assert call["source_id"] == source_id
            assert call["reason"] == "outlier data"
            assert call["actor"] == "admin-789"

        @pytest.mark.asyncio
        async def test_unrejection_unarchives(
            self, service, status_repo, index_repo
        ):
            """Promotion from rejected unarchives."""
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "rejected"
            )

            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="promoted",
                actor_type="admin",
                actor_id="admin-999",
                reason="verified data is correct",
            )

            assert len(index_repo.unarchive_calls) == 1
            call = index_repo.unarchive_calls[0]
            assert call["source_type"] == "test_variant"
            assert call["source_id"] == source_id

        @pytest.mark.asyncio
        async def test_no_archive_without_index_repo(self, status_repo):
            """No archive calls if no index_repo is provided."""
            service = KBStatusService(status_repo=status_repo)
            workspace_id = uuid4()
            source_id = uuid4()
            status_repo.add_status(
                "test_variant", source_id, workspace_id, "promoted"
            )

            # Should not raise
            await service.transition(
                source_type="test_variant",
                source_id=source_id,
                to_status="rejected",
                actor_type="admin",
                reason="test",
            )

    # --- Bulk transition tests ---

    class TestBulkTransitions:
        """Tests for bulk transition functionality."""

        @pytest.fixture
        def status_repo(self):
            return MockStatusRepository()

        @pytest.fixture
        def service(self, status_repo):
            return KBStatusService(status_repo=status_repo)

        @pytest.mark.asyncio
        async def test_bulk_transitions(self, service, status_repo):
            """Bulk transitions process all items."""
            workspace_id = uuid4()
            ids = [uuid4() for _ in range(5)]
            for source_id in ids:
                status_repo.add_status(
                    "test_variant", source_id, workspace_id, "excluded"
                )

            results = await service.bulk_transition(
                source_type="test_variant",
                source_ids=ids,
                to_status="candidate",
                actor_type="auto",
            )

            assert len(results) == 5
            for result in results:
                assert result.from_status == "excluded"
                assert result.to_status == "candidate"

        @pytest.mark.asyncio
        async def test_bulk_continues_on_error(self, service, status_repo):
            """Bulk transition continues after individual failures."""
            workspace_id = uuid4()
            valid_id = uuid4()
            missing_id = uuid4()
            status_repo.add_status(
                "test_variant", valid_id, workspace_id, "excluded"
            )

            results = await service.bulk_transition(
                source_type="test_variant",
                source_ids=[missing_id, valid_id],
                to_status="candidate",
                actor_type="auto",
            )

            # Only the valid one should be in results
            assert len(results) == 1
            assert results[0].source_id == valid_id

        @pytest.mark.asyncio
        async def test_bulk_empty_list(self, service, status_repo):
            """Bulk transition with empty list returns empty."""
            results = await service.bulk_transition(
                source_type="test_variant",
                source_ids=[],
                to_status="candidate",
                actor_type="auto",
            )

            assert results == []


class TestKBStatusResult:
    """Tests for KBStatusResult dataclass."""

    def test_basic_result(self):
        """Basic result has expected fields."""
        source_id = uuid4()
        result = KBStatusResult(
            source_type="test_variant",
            source_id=source_id,
            from_status="excluded",
            to_status="candidate",
        )

        assert result.source_type == "test_variant"
        assert result.source_id == source_id
        assert result.from_status == "excluded"
        assert result.to_status == "candidate"
        assert result.skipped is False

    def test_skipped_result(self):
        """Skipped result has skipped=True."""
        source_id = uuid4()
        result = KBStatusResult(
            source_type="test_variant",
            source_id=source_id,
            from_status="candidate",
            to_status="candidate",
            skipped=True,
        )

        assert result.skipped is True


class TestInvalidTransitionError:
    """Tests for InvalidTransitionError."""

    def test_error_with_code(self):
        """Error includes error_code."""
        err = InvalidTransitionError("some_error_code")
        assert err.error_code == "some_error_code"
        assert "some_error_code" in str(err)

    def test_error_with_message(self):
        """Error can have custom message."""
        err = InvalidTransitionError("code", "Custom message")
        assert err.error_code == "code"
        assert err.message == "Custom message"
        assert "Custom message" in str(err)
