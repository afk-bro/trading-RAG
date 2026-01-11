"""Unit tests for KB candidate comparator."""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.kb.comparator import (
    CURRENT_REGIME_SCHEMA,
    EPSILON,
    ScoredCandidate,
    candidates_within_epsilon,
    compare_candidates,
    rank_candidates,
)


def make_candidate(
    score: float = 1.0,
    kb_status: str = "candidate",
    regime_schema_version: str | None = CURRENT_REGIME_SCHEMA,
    kb_promoted_at: datetime | None = None,
    created_at: datetime | None = None,
    source_id: str = "test-id",
) -> ScoredCandidate:
    """Helper to create a ScoredCandidate with defaults."""
    return ScoredCandidate(
        source_id=source_id,
        score=score,
        kb_status=kb_status,
        regime_schema_version=regime_schema_version,
        kb_promoted_at=kb_promoted_at,
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestCompareByScore:
    """Tests for Rule 1: Primary score comparison."""

    def test_higher_score_ranks_first(self):
        """Higher score wins when difference > epsilon."""
        a = make_candidate(score=1.5)
        b = make_candidate(score=1.0)

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_score_difference_within_epsilon_triggers_tiebreak(self):
        """Within epsilon, should use tie-breaks (same status = 0)."""
        now = datetime.now(timezone.utc)
        # Use a difference clearly within epsilon to avoid floating-point issues
        a = make_candidate(score=1.0 + EPSILON / 2, created_at=now)
        b = make_candidate(score=1.0, created_at=now)

        # Same status, schema, no promoted_at, same created_at
        assert compare_candidates(a, b) == 0

    def test_score_difference_exceeds_epsilon(self):
        """Just over epsilon difference should use primary score."""
        a = make_candidate(score=1.0 + EPSILON + 0.001)
        b = make_candidate(score=1.0)

        assert compare_candidates(a, b) == -1


class TestCompareByStatus:
    """Tests for Rule 2: promoted > candidate."""

    def test_promoted_beats_candidate_within_epsilon(self):
        """Promoted status wins tie-break within epsilon."""
        a = make_candidate(score=1.0, kb_status="promoted")
        b = make_candidate(score=1.0 + EPSILON / 2, kb_status="candidate")

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_both_promoted_continues_to_next_rule(self):
        """When both promoted, proceed to schema comparison."""
        now = datetime.now(timezone.utc)
        a = make_candidate(
            score=1.0,
            kb_status="promoted",
            regime_schema_version=CURRENT_REGIME_SCHEMA,
            created_at=now,
        )
        b = make_candidate(
            score=1.0,
            kb_status="promoted",
            regime_schema_version="old_v0",
            created_at=now,
        )

        # Current schema ranks higher
        assert compare_candidates(a, b) == -1


class TestCompareBySchema:
    """Tests for Rule 3: Schema preference."""

    def test_current_schema_beats_other(self):
        """Current schema version wins over other versions."""
        now = datetime.now(timezone.utc)
        a = make_candidate(
            score=1.0, regime_schema_version=CURRENT_REGIME_SCHEMA, created_at=now
        )
        b = make_candidate(score=1.0, regime_schema_version="old_v0", created_at=now)

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_other_schema_beats_null(self):
        """Any schema beats null/missing."""
        now = datetime.now(timezone.utc)
        a = make_candidate(score=1.0, regime_schema_version="old_v0", created_at=now)
        b = make_candidate(score=1.0, regime_schema_version=None, created_at=now)

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_null_schemas_continue_to_next_rule(self):
        """When both null, proceed to promoted_at comparison."""
        now = datetime.now(timezone.utc)
        later = now + timedelta(hours=1)

        a = make_candidate(
            score=1.0,
            regime_schema_version=None,
            kb_promoted_at=later,
            created_at=now,
        )
        b = make_candidate(
            score=1.0,
            regime_schema_version=None,
            kb_promoted_at=now,
            created_at=now,
        )

        # More recent promotion wins
        assert compare_candidates(a, b) == -1


class TestCompareByPromotedAt:
    """Tests for Rule 4: Recent promotion."""

    def test_recent_promotion_beats_older(self):
        """More recent kb_promoted_at wins."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)

        a = make_candidate(score=1.0, kb_promoted_at=now, created_at=now)
        b = make_candidate(score=1.0, kb_promoted_at=old, created_at=now)

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_promoted_at_beats_null(self):
        """Having kb_promoted_at beats not having it."""
        now = datetime.now(timezone.utc)

        a = make_candidate(score=1.0, kb_promoted_at=now, created_at=now)
        b = make_candidate(score=1.0, kb_promoted_at=None, created_at=now)

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1


class TestCompareByCreatedAt:
    """Tests for Rule 5: Recency."""

    def test_newer_created_at_beats_older(self):
        """More recent created_at wins as final tie-breaker."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=7)

        a = make_candidate(score=1.0, created_at=now)
        b = make_candidate(score=1.0, created_at=old)

        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_identical_candidates_use_source_id_fallback(self):
        """Rule 6: When all else ties, use source_id for determinism."""
        now = datetime.now(timezone.utc)

        a = make_candidate(score=1.0, created_at=now, source_id="aaa")
        b = make_candidate(score=1.0, created_at=now, source_id="bbb")

        # Lower source_id ranks first (ascending order)
        assert compare_candidates(a, b) == -1
        assert compare_candidates(b, a) == 1

    def test_truly_identical_candidates_compare_equal(self):
        """Candidates with same source_id compare as 0."""
        now = datetime.now(timezone.utc)

        a = make_candidate(score=1.0, created_at=now, source_id="same")
        b = make_candidate(score=1.0, created_at=now, source_id="same")

        assert compare_candidates(a, b) == 0


class TestCompareBySourceId:
    """Tests for Rule 6: Deterministic fallback."""

    def test_source_id_provides_stable_sort(self):
        """Source ID ensures consistent ordering across runs."""
        now = datetime.now(timezone.utc)

        # All metadata identical except source_id
        candidates = [
            make_candidate(score=1.0, created_at=now, source_id="zebra"),
            make_candidate(score=1.0, created_at=now, source_id="apple"),
            make_candidate(score=1.0, created_at=now, source_id="mango"),
        ]

        ranked = rank_candidates(candidates)

        # Should be sorted by source_id ascending
        assert ranked[0].source_id == "apple"
        assert ranked[1].source_id == "mango"
        assert ranked[2].source_id == "zebra"


class TestRankCandidates:
    """Tests for rank_candidates function."""

    def test_ranks_by_score_primarily(self):
        """Ranking primarily by score."""
        c1 = make_candidate(score=1.5, source_id="high")
        c2 = make_candidate(score=0.5, source_id="low")
        c3 = make_candidate(score=1.0, source_id="mid")

        ranked = rank_candidates([c2, c3, c1])

        assert ranked[0].source_id == "high"
        assert ranked[1].source_id == "mid"
        assert ranked[2].source_id == "low"

    def test_applies_tiebreaks_within_epsilon(self):
        """Tie-breaks applied when scores within epsilon."""
        now = datetime.now(timezone.utc)

        c1 = make_candidate(
            score=1.0, kb_status="candidate", source_id="candidate", created_at=now
        )
        c2 = make_candidate(
            score=1.0 + EPSILON / 2,
            kb_status="promoted",
            source_id="promoted",
            created_at=now,
        )

        ranked = rank_candidates([c1, c2])

        # Promoted should rank first despite slightly lower score
        assert ranked[0].source_id == "promoted"

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        assert rank_candidates([]) == []

    def test_single_candidate_returns_self(self):
        """Single candidate returned as-is."""
        c = make_candidate(source_id="only")
        ranked = rank_candidates([c])

        assert len(ranked) == 1
        assert ranked[0].source_id == "only"


class TestCandidatesWithinEpsilon:
    """Tests for candidates_within_epsilon helper."""

    def test_nearly_at_epsilon_is_true(self):
        """Scores nearly epsilon apart are within epsilon."""
        a = make_candidate(score=1.0)
        # Use slightly less than epsilon to avoid floating-point boundary issues
        b = make_candidate(score=1.0 + EPSILON * 0.99)

        assert candidates_within_epsilon(a, b) is True

    def test_less_than_epsilon_is_true(self):
        """Scores less than epsilon apart are within epsilon."""
        a = make_candidate(score=1.0)
        b = make_candidate(score=1.0 + EPSILON / 2)

        assert candidates_within_epsilon(a, b) is True

    def test_exceeds_epsilon_is_false(self):
        """Scores more than epsilon apart are not within epsilon."""
        a = make_candidate(score=1.0)
        b = make_candidate(score=1.0 + EPSILON + 0.001)

        assert candidates_within_epsilon(a, b) is False

    def test_same_score_is_true(self):
        """Identical scores are within epsilon."""
        a = make_candidate(score=1.0)
        b = make_candidate(score=1.0)

        assert candidates_within_epsilon(a, b) is True


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_promotion_preview_scenario(self):
        """Simulate promotion preview ranking."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        last_week = now - timedelta(days=7)

        candidates = [
            # High score, candidate
            make_candidate(
                score=1.5,
                kb_status="candidate",
                source_id="high-candidate",
                created_at=last_week,
            ),
            # Similar score within epsilon (0.01 < 0.02), promoted - should win tie-break
            make_candidate(
                score=1.49,
                kb_status="promoted",
                kb_promoted_at=yesterday,
                source_id="close-promoted",
                created_at=last_week,
            ),
            # Low score
            make_candidate(
                score=0.8,
                kb_status="promoted",
                source_id="low-promoted",
                created_at=now,
            ),
        ]

        ranked = rank_candidates(candidates)

        # 1.5 and 1.49 are within epsilon (0.02)
        # Promoted should win the tie-break
        assert ranked[0].source_id == "close-promoted"
        assert ranked[1].source_id == "high-candidate"
        assert ranked[2].source_id == "low-promoted"

    def test_schema_migration_scenario(self):
        """Simulate ranking during schema migration."""
        now = datetime.now(timezone.utc)

        candidates = [
            make_candidate(
                score=1.0,
                regime_schema_version=None,
                source_id="no-schema",
                created_at=now,
            ),
            make_candidate(
                score=1.0,
                regime_schema_version="old_v0",
                source_id="old-schema",
                created_at=now,
            ),
            make_candidate(
                score=1.0,
                regime_schema_version=CURRENT_REGIME_SCHEMA,
                source_id="current-schema",
                created_at=now,
            ),
        ]

        ranked = rank_candidates(candidates)

        assert ranked[0].source_id == "current-schema"
        assert ranked[1].source_id == "old-schema"
        assert ranked[2].source_id == "no-schema"
