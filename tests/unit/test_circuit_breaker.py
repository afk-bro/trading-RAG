"""Unit tests for the candidacy circuit breaker."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

import pytest

from app.services.kb.circuit_breaker import (
    BreakerState,
    CandidacyCircuitBreaker,
)


# =============================================================================
# Mock Repository
# =============================================================================


class MockBreakerRepository:
    """Mock implementation of BreakerRepository for testing."""

    def __init__(
        self,
        state: Optional[BreakerState] = None,
        recent_decisions: Optional[list[str]] = None,
        candidate_count_24h: int = 0,
    ):
        self.state = state or BreakerState(
            kb_auto_candidacy_state="enabled",
            kb_auto_candidacy_disabled_until=None,
            kb_auto_candidacy_trip_reason=None,
            kb_auto_candidacy_tripped_at=None,
        )
        self.recent_decisions = recent_decisions or []
        self.candidate_count_24h = candidate_count_24h
        self.update_calls: list[dict] = []

    async def get_breaker_state(self, workspace_id: UUID) -> BreakerState:
        return self.state

    async def update_breaker_state(
        self,
        workspace_id: UUID,
        state: str,
        disabled_until: Optional[datetime],
        trip_reason: Optional[str],
    ) -> None:
        self.update_calls.append(
            {
                "workspace_id": workspace_id,
                "state": state,
                "disabled_until": disabled_until,
                "trip_reason": trip_reason,
            }
        )
        # Update the mock state
        self.state = BreakerState(
            kb_auto_candidacy_state=state,
            kb_auto_candidacy_disabled_until=disabled_until,
            kb_auto_candidacy_trip_reason=trip_reason,
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
        )

    async def get_recent_candidacy_decisions(
        self, workspace_id: UUID, limit: int
    ) -> list[str]:
        return self.recent_decisions[:limit]

    async def get_candidate_count_rolling_24h(self, workspace_id: UUID) -> int:
        return self.candidate_count_24h


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workspace_id() -> UUID:
    """Return a consistent workspace ID for tests."""
    return uuid4()


@pytest.fixture
def enabled_state() -> BreakerState:
    """Return an enabled breaker state."""
    return BreakerState(
        kb_auto_candidacy_state="enabled",
        kb_auto_candidacy_disabled_until=None,
        kb_auto_candidacy_trip_reason=None,
        kb_auto_candidacy_tripped_at=None,
    )


@pytest.fixture
def disabled_state() -> BreakerState:
    """Return a permanently disabled breaker state."""
    return BreakerState(
        kb_auto_candidacy_state="disabled",
        kb_auto_candidacy_disabled_until=None,
        kb_auto_candidacy_trip_reason="admin_disabled",
        kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
    )


# =============================================================================
# BreakerState Tests
# =============================================================================


class TestBreakerState:
    """Tests for the BreakerState dataclass."""

    def test_enabled_state_attributes(self, enabled_state):
        """Enabled state should have correct attributes."""
        assert enabled_state.kb_auto_candidacy_state == "enabled"
        assert enabled_state.kb_auto_candidacy_disabled_until is None
        assert enabled_state.kb_auto_candidacy_trip_reason is None
        assert enabled_state.kb_auto_candidacy_tripped_at is None

    def test_disabled_state_attributes(self, disabled_state):
        """Disabled state should have correct attributes."""
        assert disabled_state.kb_auto_candidacy_state == "disabled"
        assert disabled_state.kb_auto_candidacy_trip_reason == "admin_disabled"

    def test_degraded_state_with_cooldown(self):
        """Degraded state should have cooldown timestamp."""
        cooldown_until = datetime.now(timezone.utc) + timedelta(hours=6)
        state = BreakerState(
            kb_auto_candidacy_state="degraded",
            kb_auto_candidacy_disabled_until=cooldown_until,
            kb_auto_candidacy_trip_reason="rate_spike:0.35",
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
        )

        assert state.kb_auto_candidacy_state == "degraded"
        assert state.kb_auto_candidacy_disabled_until == cooldown_until
        assert "rate_spike" in state.kb_auto_candidacy_trip_reason


# =============================================================================
# Enabled State Tests
# =============================================================================


class TestEnabledState:
    """Tests for circuit breaker in enabled state."""

    @pytest.mark.asyncio
    async def test_allows_candidacy_when_enabled(self, workspace_id, enabled_state):
        """Should allow candidacy when breaker is enabled and limits not exceeded."""
        repo = MockBreakerRepository(state=enabled_state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_allows_with_low_rate(self, workspace_id, enabled_state):
        """Should allow when candidate rate is below threshold."""
        # 10 candidates out of 50 = 20% < 30% threshold
        decisions = ["candidate"] * 10 + ["excluded"] * 40
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_allows_with_low_volume(self, workspace_id, enabled_state):
        """Should allow when 24h volume is below cap."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=100,  # Below 200 cap
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True
        assert reason is None


# =============================================================================
# Disabled State Tests
# =============================================================================


class TestDisabledState:
    """Tests for circuit breaker in disabled state."""

    @pytest.mark.asyncio
    async def test_rejects_when_disabled(self, workspace_id, disabled_state):
        """Should reject candidacy when breaker is disabled."""
        repo = MockBreakerRepository(state=disabled_state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert reason == "disabled"

    @pytest.mark.asyncio
    async def test_disabled_ignores_rates(self, workspace_id, disabled_state):
        """Disabled state should skip rate checks entirely."""
        # Even with good rates, should be rejected
        decisions = ["excluded"] * 50
        repo = MockBreakerRepository(
            state=disabled_state,
            recent_decisions=decisions,
            candidate_count_24h=0,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert reason == "disabled"


# =============================================================================
# Degraded/Cooldown State Tests
# =============================================================================


class TestDegradedState:
    """Tests for circuit breaker in degraded (cooldown) state."""

    @pytest.mark.asyncio
    async def test_rejects_during_cooldown(self, workspace_id):
        """Should reject during active cooldown period."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=3)
        state = BreakerState(
            kb_auto_candidacy_state="degraded",
            kb_auto_candidacy_disabled_until=future_time,
            kb_auto_candidacy_trip_reason="rate_spike:0.35",
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
        )
        repo = MockBreakerRepository(state=state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert reason == "cooldown"

    @pytest.mark.asyncio
    async def test_allows_after_cooldown_expires(self, workspace_id):
        """Should allow after cooldown period expires."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        state = BreakerState(
            kb_auto_candidacy_state="degraded",
            kb_auto_candidacy_disabled_until=past_time,  # Expired
            kb_auto_candidacy_trip_reason="rate_spike:0.35",
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc)
            - timedelta(hours=7),
        )
        repo = MockBreakerRepository(state=state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_cooldown_exactly_at_boundary(self, workspace_id):
        """Should reject when exactly at cooldown boundary."""
        # This is a bit tricky due to timing, but we use a tiny future time
        now = datetime.now(timezone.utc)
        state = BreakerState(
            kb_auto_candidacy_state="degraded",
            kb_auto_candidacy_disabled_until=now + timedelta(seconds=1),
            kb_auto_candidacy_trip_reason="rate_spike:0.35",
            kb_auto_candidacy_tripped_at=now,
        )
        repo = MockBreakerRepository(state=state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert reason == "cooldown"


# =============================================================================
# Rate Spike Detection Tests
# =============================================================================


class TestRateSpikeDetection:
    """Tests for rate spike detection and tripping."""

    @pytest.mark.asyncio
    async def test_trips_on_rate_spike(self, workspace_id, enabled_state):
        """Should trip breaker when candidate rate exceeds threshold."""
        # 20 candidates out of 50 = 40% > 30% threshold
        decisions = ["candidate"] * 20 + ["excluded"] * 30
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert reason.startswith("rate_spike:")
        assert "0.40" in reason

    @pytest.mark.asyncio
    async def test_updates_state_on_rate_trip(self, workspace_id, enabled_state):
        """Should update repository state when tripped by rate."""
        decisions = ["candidate"] * 20 + ["excluded"] * 30
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
        )
        breaker = CandidacyCircuitBreaker(repo)

        await breaker.check(workspace_id)

        # Verify update was called
        assert len(repo.update_calls) == 1
        call = repo.update_calls[0]
        assert call["workspace_id"] == workspace_id
        assert call["state"] == "degraded"
        assert call["disabled_until"] is not None
        assert "rate_spike" in call["trip_reason"]

    @pytest.mark.asyncio
    async def test_rate_exactly_at_threshold_allows(self, workspace_id, enabled_state):
        """Rate exactly at 30% threshold should be allowed (not strictly greater)."""
        # 15 candidates out of 50 = 30% = threshold
        decisions = ["candidate"] * 15 + ["excluded"] * 35
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_rate_check_skipped_with_insufficient_history(
        self, workspace_id, enabled_state
    ):
        """Rate check should be skipped if less than window size history."""
        # Only 30 decisions, window is 50
        decisions = ["candidate"] * 30  # Would be 100% rate if counted
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        # Should pass because we don't have enough history
        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_rate_window_size_is_50(self, workspace_id, enabled_state):
        """Rate window should use exactly 50 most recent decisions."""
        # 49 decisions - just under threshold
        decisions = ["candidate"] * 49
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        # 49 < 50, so rate check is skipped
        assert allowed is True


# =============================================================================
# Daily Cap Tests
# =============================================================================


class TestDailyCapEnforcement:
    """Tests for 24h volume cap enforcement."""

    @pytest.mark.asyncio
    async def test_trips_on_daily_cap(self, workspace_id, enabled_state):
        """Should trip breaker when 24h count reaches cap."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=200,  # At cap
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert reason.startswith("daily_cap:")
        assert "200" in reason

    @pytest.mark.asyncio
    async def test_trips_on_exceeding_daily_cap(self, workspace_id, enabled_state):
        """Should trip breaker when 24h count exceeds cap."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=250,  # Over cap
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert "daily_cap" in reason

    @pytest.mark.asyncio
    async def test_allows_below_daily_cap(self, workspace_id, enabled_state):
        """Should allow when 24h count is below cap."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=199,  # Just below cap
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_updates_state_on_daily_cap_trip(self, workspace_id, enabled_state):
        """Should update repository state when tripped by daily cap."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=200,
        )
        breaker = CandidacyCircuitBreaker(repo)

        await breaker.check(workspace_id)

        assert len(repo.update_calls) == 1
        call = repo.update_calls[0]
        assert call["state"] == "degraded"
        assert "daily_cap" in call["trip_reason"]


# =============================================================================
# Cooldown Duration Tests
# =============================================================================


class TestCooldownDuration:
    """Tests for cooldown duration calculation."""

    @pytest.mark.asyncio
    async def test_cooldown_is_six_hours(self, workspace_id, enabled_state):
        """Cooldown should be 6 hours from trip time."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=200,  # Will trip
        )
        breaker = CandidacyCircuitBreaker(repo)

        before = datetime.now(timezone.utc)
        await breaker.check(workspace_id)
        after = datetime.now(timezone.utc)

        call = repo.update_calls[0]
        disabled_until = call["disabled_until"]

        # Should be approximately 6 hours from now
        expected_min = before + timedelta(hours=6)
        expected_max = after + timedelta(hours=6)

        assert expected_min <= disabled_until <= expected_max

    def test_cooldown_hours_constant(self):
        """COOLDOWN_HOURS should be 6."""
        assert CandidacyCircuitBreaker.COOLDOWN_HOURS == 6


# =============================================================================
# Threshold Constants Tests
# =============================================================================


class TestThresholdConstants:
    """Tests for circuit breaker threshold constants."""

    def test_max_candidate_rate_is_30_percent(self):
        """MAX_CANDIDATE_RATE should be 0.30."""
        assert CandidacyCircuitBreaker.MAX_CANDIDATE_RATE == 0.30

    def test_rate_window_size_is_50(self):
        """RATE_WINDOW_SIZE should be 50."""
        assert CandidacyCircuitBreaker.RATE_WINDOW_SIZE == 50

    def test_max_candidates_24h_is_200(self):
        """MAX_CANDIDATES_24H should be 200."""
        assert CandidacyCircuitBreaker.MAX_CANDIDATES_24H == 200


# =============================================================================
# Check Order Tests
# =============================================================================


class TestCheckOrder:
    """Tests for the order of checks in the circuit breaker."""

    @pytest.mark.asyncio
    async def test_disabled_checked_before_degraded(self, workspace_id):
        """Disabled state should be checked before degraded."""
        # Disabled state - should return immediately
        state = BreakerState(
            kb_auto_candidacy_state="disabled",
            kb_auto_candidacy_disabled_until=datetime.now(timezone.utc)
            + timedelta(hours=1),
            kb_auto_candidacy_trip_reason="admin_disabled",
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
        )
        repo = MockBreakerRepository(state=state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert reason == "disabled"  # Not "cooldown"

    @pytest.mark.asyncio
    async def test_cooldown_checked_before_rate(self, workspace_id):
        """Cooldown should be checked before rate spike."""
        state = BreakerState(
            kb_auto_candidacy_state="degraded",
            kb_auto_candidacy_disabled_until=datetime.now(timezone.utc)
            + timedelta(hours=1),
            kb_auto_candidacy_trip_reason="previous_trip",
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
        )
        # Even with low rates, should be in cooldown
        repo = MockBreakerRepository(
            state=state,
            recent_decisions=["excluded"] * 50,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert reason == "cooldown"

    @pytest.mark.asyncio
    async def test_rate_checked_before_daily_cap(self, workspace_id, enabled_state):
        """Rate spike should be checked before daily cap."""
        # High rate AND high daily count
        decisions = ["candidate"] * 25 + ["excluded"] * 25  # 50% rate
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=decisions,
            candidate_count_24h=250,  # Also over cap
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        # Should trip on rate first
        assert "rate_spike" in reason


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_decisions_allows(self, workspace_id, enabled_state):
        """Empty decision history should allow candidacy."""
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=[],
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True

    @pytest.mark.asyncio
    async def test_all_excluded_decisions_allows(self, workspace_id, enabled_state):
        """All excluded decisions should have 0% rate."""
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=["excluded"] * 50,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True

    @pytest.mark.asyncio
    async def test_all_candidate_decisions_trips(self, workspace_id, enabled_state):
        """All candidate decisions should have 100% rate and trip."""
        repo = MockBreakerRepository(
            state=enabled_state,
            recent_decisions=["candidate"] * 50,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is False
        assert "rate_spike:1.00" in reason

    @pytest.mark.asyncio
    async def test_none_disabled_until_in_degraded(self, workspace_id):
        """Degraded state with None disabled_until should not be in cooldown."""
        state = BreakerState(
            kb_auto_candidacy_state="degraded",
            kb_auto_candidacy_disabled_until=None,  # Edge case
            kb_auto_candidacy_trip_reason="previous_trip",
            kb_auto_candidacy_tripped_at=datetime.now(timezone.utc),
        )
        repo = MockBreakerRepository(state=state)
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        # None disabled_until means condition is falsy, so cooldown check passes
        assert allowed is True

    @pytest.mark.asyncio
    async def test_zero_candidate_count_allows(self, workspace_id, enabled_state):
        """Zero 24h candidate count should allow."""
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=0,
        )
        breaker = CandidacyCircuitBreaker(repo)

        allowed, reason = await breaker.check(workspace_id)

        assert allowed is True


# =============================================================================
# Multiple Workspace Tests
# =============================================================================


class TestMultipleWorkspaces:
    """Tests for multi-workspace isolation."""

    @pytest.mark.asyncio
    async def test_different_workspaces_independent(self):
        """Different workspaces should have independent state."""
        workspace1 = uuid4()
        workspace2 = uuid4()

        # Create separate repos with different states
        repo1 = MockBreakerRepository(
            state=BreakerState(
                kb_auto_candidacy_state="enabled",
                kb_auto_candidacy_disabled_until=None,
                kb_auto_candidacy_trip_reason=None,
                kb_auto_candidacy_tripped_at=None,
            )
        )
        repo2 = MockBreakerRepository(
            state=BreakerState(
                kb_auto_candidacy_state="disabled",
                kb_auto_candidacy_disabled_until=None,
                kb_auto_candidacy_trip_reason="admin",
                kb_auto_candidacy_tripped_at=None,
            )
        )

        breaker1 = CandidacyCircuitBreaker(repo1)
        breaker2 = CandidacyCircuitBreaker(repo2)

        allowed1, _ = await breaker1.check(workspace1)
        allowed2, _ = await breaker2.check(workspace2)

        assert allowed1 is True
        assert allowed2 is False

    @pytest.mark.asyncio
    async def test_workspace_id_passed_to_repo(self, enabled_state):
        """Workspace ID should be passed correctly to repository methods."""
        workspace_id = uuid4()
        repo = MockBreakerRepository(
            state=enabled_state,
            candidate_count_24h=200,  # Will trip
        )
        breaker = CandidacyCircuitBreaker(repo)

        await breaker.check(workspace_id)

        # Verify workspace_id was passed to update
        assert repo.update_calls[0]["workspace_id"] == workspace_id


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests that MockBreakerRepository satisfies BreakerRepository protocol."""

    def test_mock_implements_protocol(self):
        """Mock repository should implement the protocol interface."""
        repo = MockBreakerRepository()

        # Check all required methods exist
        assert hasattr(repo, "get_breaker_state")
        assert hasattr(repo, "update_breaker_state")
        assert hasattr(repo, "get_recent_candidacy_decisions")
        assert hasattr(repo, "get_candidate_count_rolling_24h")

        # Check they're callable
        assert callable(repo.get_breaker_state)
        assert callable(repo.update_breaker_state)
        assert callable(repo.get_recent_candidacy_decisions)
        assert callable(repo.get_candidate_count_rolling_24h)
