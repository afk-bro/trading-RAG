"""Unit tests for KB status transitions."""

import pytest

from app.services.kb.transitions import (
    KBStatus,
    KBStatusTransition,
    TransitionResult,
    transition_validator,
)


class TestKBStatus:
    """Tests for KBStatus enum."""

    def test_status_values(self):
        """All expected status values exist."""
        assert KBStatus.EXCLUDED.value == "excluded"
        assert KBStatus.CANDIDATE.value == "candidate"
        assert KBStatus.PROMOTED.value == "promoted"
        assert KBStatus.REJECTED.value == "rejected"

    def test_status_is_string_enum(self):
        """Status values can be compared to strings."""
        assert KBStatus.EXCLUDED == "excluded"
        assert KBStatus.CANDIDATE == "candidate"
        assert KBStatus.PROMOTED == "promoted"
        assert KBStatus.REJECTED == "rejected"


class TestTransitionResult:
    """Tests for TransitionResult dataclass."""

    def test_valid_result(self):
        """Valid result has no error."""
        result = TransitionResult(valid=True)
        assert result.valid is True
        assert result.error is None

    def test_invalid_result_with_error(self):
        """Invalid result includes error."""
        result = TransitionResult(valid=False, error="some_error")
        assert result.valid is False
        assert result.error == "some_error"


class TestKBStatusTransitionValidator:
    """Tests for KBStatusTransition validator."""

    @pytest.fixture
    def validator(self):
        """Create a fresh validator instance."""
        return KBStatusTransition()

    # --- Allowed transitions ---

    class TestExcludedToCandidate:
        """Tests for excluded → candidate transition."""

        def test_auto_can_transition(self):
            """Auto can transition excluded → candidate."""
            v = KBStatusTransition()
            result = v.validate("excluded", "candidate", "auto")
            assert result.valid is True

        def test_admin_can_transition(self):
            """Admin can transition excluded → candidate."""
            v = KBStatusTransition()
            result = v.validate("excluded", "candidate", "admin")
            assert result.valid is True

        def test_no_reason_required(self):
            """No reason required for this transition."""
            v = KBStatusTransition()
            result = v.validate("excluded", "candidate", "auto", reason=None)
            assert result.valid is True

    class TestExcludedToPromoted:
        """Tests for excluded → promoted transition."""

        def test_admin_can_transition(self):
            """Admin can transition excluded → promoted."""
            v = KBStatusTransition()
            result = v.validate("excluded", "promoted", "admin")
            assert result.valid is True

        def test_auto_cannot_transition(self):
            """Auto cannot transition excluded → promoted."""
            v = KBStatusTransition()
            result = v.validate("excluded", "promoted", "auto")
            assert result.valid is False
            assert "actor_auto_cannot" in result.error

    class TestCandidateToPromoted:
        """Tests for candidate → promoted transition."""

        def test_admin_can_transition(self):
            """Admin can transition candidate → promoted."""
            v = KBStatusTransition()
            result = v.validate("candidate", "promoted", "admin")
            assert result.valid is True

        def test_auto_cannot_transition(self):
            """Auto cannot transition candidate → promoted."""
            v = KBStatusTransition()
            result = v.validate("candidate", "promoted", "auto")
            assert result.valid is False

    class TestCandidateToRejected:
        """Tests for candidate → rejected transition."""

        def test_admin_can_transition_with_reason(self):
            """Admin can transition candidate → rejected with reason."""
            v = KBStatusTransition()
            result = v.validate("candidate", "rejected", "admin", reason="low quality")
            assert result.valid is True

        def test_admin_cannot_transition_without_reason(self):
            """Admin cannot transition candidate → rejected without reason."""
            v = KBStatusTransition()
            result = v.validate("candidate", "rejected", "admin")
            assert result.valid is False
            assert result.error == "rejection_requires_reason"

        def test_auto_cannot_transition(self):
            """Auto cannot transition candidate → rejected."""
            v = KBStatusTransition()
            result = v.validate("candidate", "rejected", "auto", reason="some reason")
            assert result.valid is False

    class TestPromotedToRejected:
        """Tests for promoted → rejected transition."""

        def test_admin_can_transition_with_reason(self):
            """Admin can transition promoted → rejected with reason."""
            v = KBStatusTransition()
            result = v.validate("promoted", "rejected", "admin", reason="bad data")
            assert result.valid is True

        def test_admin_cannot_transition_without_reason(self):
            """Admin cannot transition promoted → rejected without reason."""
            v = KBStatusTransition()
            result = v.validate("promoted", "rejected", "admin")
            assert result.valid is False
            assert result.error == "rejection_requires_reason"

    class TestRejectedToPromoted:
        """Tests for rejected → promoted transition (unrejection)."""

        def test_admin_can_transition_with_reason(self):
            """Admin can transition rejected → promoted with reason."""
            v = KBStatusTransition()
            result = v.validate("rejected", "promoted", "admin", reason="data verified")
            assert result.valid is True

        def test_admin_cannot_transition_without_reason(self):
            """Admin cannot transition rejected → promoted without reason."""
            v = KBStatusTransition()
            result = v.validate("rejected", "promoted", "admin")
            assert result.valid is False
            assert result.error == "unrejection_requires_reason"

        def test_auto_cannot_transition(self):
            """Auto cannot transition rejected → promoted."""
            v = KBStatusTransition()
            result = v.validate("rejected", "promoted", "auto", reason="verified")
            assert result.valid is False

    # --- Disallowed transitions ---

    class TestDisallowedTransitions:
        """Tests for transitions that are not allowed."""

        def test_rejected_to_candidate_not_allowed(self):
            """Cannot transition rejected → candidate."""
            v = KBStatusTransition()
            result = v.validate("rejected", "candidate", "admin")
            assert result.valid is False
            assert "not_allowed" in result.error

        def test_rejected_to_excluded_not_allowed(self):
            """Cannot transition rejected → excluded."""
            v = KBStatusTransition()
            result = v.validate("rejected", "excluded", "admin")
            assert result.valid is False

        def test_promoted_to_excluded_not_allowed(self):
            """Cannot transition promoted → excluded."""
            v = KBStatusTransition()
            result = v.validate("promoted", "excluded", "admin")
            assert result.valid is False

        def test_promoted_to_candidate_not_allowed(self):
            """Cannot transition promoted → candidate."""
            v = KBStatusTransition()
            result = v.validate("promoted", "candidate", "admin")
            assert result.valid is False

        def test_candidate_to_excluded_not_allowed(self):
            """Cannot transition candidate → excluded."""
            v = KBStatusTransition()
            result = v.validate("candidate", "excluded", "admin")
            assert result.valid is False

    # --- Same status (no-op) ---

    class TestSameStatusNoOp:
        """Tests for same status transitions (no-ops)."""

        @pytest.mark.parametrize(
            "status",
            ["excluded", "candidate", "promoted", "rejected"],
        )
        def test_same_status_is_valid(self, status):
            """Same status is a no-op, not an error."""
            v = KBStatusTransition()
            result = v.validate(status, status, "admin")
            assert result.valid is True

        @pytest.mark.parametrize(
            "status",
            ["excluded", "candidate", "promoted", "rejected"],
        )
        def test_same_status_auto_is_valid(self, status):
            """Same status is valid even for auto."""
            v = KBStatusTransition()
            result = v.validate(status, status, "auto")
            assert result.valid is True

    # --- Case insensitivity ---

    class TestCaseInsensitivity:
        """Tests for case handling."""

        def test_uppercase_from_status(self):
            """Uppercase from_status is normalized."""
            v = KBStatusTransition()
            result = v.validate("EXCLUDED", "candidate", "auto")
            assert result.valid is True

        def test_uppercase_to_status(self):
            """Uppercase to_status is normalized."""
            v = KBStatusTransition()
            result = v.validate("excluded", "CANDIDATE", "auto")
            assert result.valid is True

        def test_mixed_case(self):
            """Mixed case is normalized."""
            v = KBStatusTransition()
            result = v.validate("ExClUdEd", "CaNdIdAtE", "auto")
            assert result.valid is True

    # --- Helper methods ---

    class TestCanAutoCandidate:
        """Tests for can_auto_candidate helper."""

        def test_excluded_can_auto_candidate(self):
            """Excluded status can auto-candidate."""
            v = KBStatusTransition()
            assert v.can_auto_candidate("excluded") is True

        def test_candidate_cannot_auto_candidate(self):
            """Already candidate cannot auto-candidate."""
            v = KBStatusTransition()
            assert v.can_auto_candidate("candidate") is False

        def test_promoted_cannot_auto_candidate(self):
            """Promoted cannot auto-candidate."""
            v = KBStatusTransition()
            assert v.can_auto_candidate("promoted") is False

        def test_rejected_cannot_auto_candidate(self):
            """Rejected cannot auto-candidate."""
            v = KBStatusTransition()
            assert v.can_auto_candidate("rejected") is False

    class TestGetAllowedTargets:
        """Tests for get_allowed_targets helper."""

        def test_excluded_auto_targets(self):
            """Auto from excluded can go to candidate."""
            v = KBStatusTransition()
            targets = v.get_allowed_targets("excluded", "auto")
            assert targets == ["candidate"]

        def test_excluded_admin_targets(self):
            """Admin from excluded can go to candidate or promoted."""
            v = KBStatusTransition()
            targets = v.get_allowed_targets("excluded", "admin")
            assert targets == ["candidate", "promoted"]

        def test_candidate_admin_targets(self):
            """Admin from candidate can go to promoted or rejected."""
            v = KBStatusTransition()
            targets = v.get_allowed_targets("candidate", "admin")
            assert targets == ["promoted", "rejected"]

        def test_candidate_auto_targets(self):
            """Auto from candidate has no targets."""
            v = KBStatusTransition()
            targets = v.get_allowed_targets("candidate", "auto")
            assert targets == []

        def test_promoted_admin_targets(self):
            """Admin from promoted can go to rejected."""
            v = KBStatusTransition()
            targets = v.get_allowed_targets("promoted", "admin")
            assert targets == ["rejected"]

        def test_rejected_admin_targets(self):
            """Admin from rejected can go to promoted."""
            v = KBStatusTransition()
            targets = v.get_allowed_targets("rejected", "admin")
            assert targets == ["promoted"]


class TestTransitionValidatorSingleton:
    """Tests for the singleton validator instance."""

    def test_singleton_exists(self):
        """Singleton validator is available."""
        assert transition_validator is not None

    def test_singleton_is_validator(self):
        """Singleton is a KBStatusTransition instance."""
        assert isinstance(transition_validator, KBStatusTransition)

    def test_singleton_works(self):
        """Singleton can validate transitions."""
        result = transition_validator.validate("excluded", "candidate", "auto")
        assert result.valid is True
