"""Circuit breaker for KB auto-candidacy rate limiting.

Prevents runaway auto-candidacy from flooding the KB with test variants
by applying rate limits and volume caps per workspace.

This is Phase 2 of the trial ingestion design.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BreakerState:
    """Workspace circuit breaker state.

    Stored in the workspaces table for persistence across restarts.

    Attributes:
        kb_auto_candidacy_state: Current state ("enabled", "degraded", "disabled")
        kb_auto_candidacy_disabled_until: When cooldown expires (for degraded state)
        kb_auto_candidacy_trip_reason: Why the breaker was tripped
        kb_auto_candidacy_tripped_at: When the breaker was tripped
    """

    kb_auto_candidacy_state: str  # "enabled" | "degraded" | "disabled"
    kb_auto_candidacy_disabled_until: Optional[datetime]
    kb_auto_candidacy_trip_reason: Optional[str]
    kb_auto_candidacy_tripped_at: Optional[datetime]


class BreakerRepository(Protocol):
    """Protocol for breaker state persistence.

    The actual implementation will be in the repository layer.
    This protocol allows the circuit breaker to be unit tested
    without database dependencies.
    """

    async def get_breaker_state(self, workspace_id: UUID) -> BreakerState:
        """Get current breaker state for a workspace.

        Args:
            workspace_id: The workspace to check

        Returns:
            BreakerState with current settings
        """
        ...

    async def update_breaker_state(
        self,
        workspace_id: UUID,
        state: str,
        disabled_until: Optional[datetime],
        trip_reason: Optional[str],
    ) -> None:
        """Update breaker state for a workspace.

        Args:
            workspace_id: The workspace to update
            state: New state ("enabled", "degraded", "disabled")
            disabled_until: Cooldown expiry (for degraded state)
            trip_reason: Why the breaker was tripped
        """
        ...

    async def get_recent_candidacy_decisions(
        self, workspace_id: UUID, limit: int
    ) -> list[str]:
        """Return list of recent kb_status values for successful runs.

        Used to calculate the candidacy rate over recent runs.

        Args:
            workspace_id: The workspace to check
            limit: Number of recent runs to fetch

        Returns:
            List of kb_status values (e.g., ["candidate", "excluded", "candidate"])
        """
        ...

    async def get_candidate_count_rolling_24h(self, workspace_id: UUID) -> int:
        """Count candidates created in last 24 hours.

        Args:
            workspace_id: The workspace to check

        Returns:
            Number of candidates created in the rolling 24h window
        """
        ...


class CandidacyCircuitBreaker:
    """Rate limiter for auto-candidacy to prevent KB flooding.

    The circuit breaker has three states:
    - enabled: Normal operation, candidacy allowed
    - degraded: In cooldown after rate spike, candidacy blocked
    - disabled: Permanently disabled by admin, candidacy blocked

    Triggers:
    - Rate spike: >30% of last 50 successful runs became candidates
    - Daily cap: >200 candidates in rolling 24h window

    Cooldown is 6 hours after a trip.
    """

    MAX_CANDIDATE_RATE = 0.30  # candidates / successes
    RATE_WINDOW_SIZE = 50  # last N successful runs
    MAX_CANDIDATES_24H = 200  # rolling 24h cap
    COOLDOWN_HOURS = 6

    def __init__(self, repo: BreakerRepository):
        """Initialize the circuit breaker.

        Args:
            repo: Repository for breaker state persistence
        """
        self._repo = repo

    async def check(self, workspace_id: UUID) -> tuple[bool, Optional[str]]:
        """Check if candidacy is allowed for this workspace.

        Evaluates:
        1. Permanent disable state
        2. Cooldown state
        3. Rate over recent successful runs
        4. Rolling 24h volume cap

        Args:
            workspace_id: The workspace to check

        Returns:
            Tuple of (allowed, trip_reason). trip_reason is None if allowed.
        """
        state = await self._repo.get_breaker_state(workspace_id)

        # Permanently disabled
        if state.kb_auto_candidacy_state == "disabled":
            return False, "disabled"

        # In cooldown
        if state.kb_auto_candidacy_state == "degraded":
            if (
                state.kb_auto_candidacy_disabled_until
                and datetime.now(timezone.utc) < state.kb_auto_candidacy_disabled_until
            ):
                return False, "cooldown"
            # Cooldown expired - state will be reset on next successful candidacy

        # Check rate over last N successful runs
        recent = await self._repo.get_recent_candidacy_decisions(
            workspace_id, self.RATE_WINDOW_SIZE
        )
        if len(recent) >= self.RATE_WINDOW_SIZE:
            candidate_count = sum(1 for d in recent if d == "candidate")
            rate = candidate_count / len(recent)
            if rate > self.MAX_CANDIDATE_RATE:
                trip_reason = f"rate_spike:{rate:.2f}"
                await self._trip_breaker(workspace_id, trip_reason)
                return False, trip_reason

        # Check rolling 24h volume
        count_24h = await self._repo.get_candidate_count_rolling_24h(workspace_id)
        if count_24h >= self.MAX_CANDIDATES_24H:
            trip_reason = f"daily_cap:{count_24h}"
            await self._trip_breaker(workspace_id, trip_reason)
            return False, trip_reason

        return True, None

    async def _trip_breaker(self, workspace_id: UUID, reason: str) -> None:
        """Trip the circuit breaker for this workspace.

        Sets the breaker to degraded state with a cooldown period.

        Args:
            workspace_id: The workspace to trip
            reason: Why the breaker was tripped
        """
        logger.warning(
            "circuit_breaker_tripped",
            workspace_id=str(workspace_id),
            reason=reason,
        )
        await self._repo.update_breaker_state(
            workspace_id,
            state="degraded",
            disabled_until=datetime.now(timezone.utc)
            + timedelta(hours=self.COOLDOWN_HOURS),
            trip_reason=reason,
        )


class BacktestBreakerAdapter:
    """Adapter that wraps BacktestRepository to implement BreakerRepository protocol.

    This allows the circuit breaker to use the backtest repository methods
    without tight coupling.
    """

    def __init__(self, backtest_repo):
        """Initialize the adapter.

        Args:
            backtest_repo: BacktestRepository instance
        """
        self._repo = backtest_repo

    async def get_breaker_state(self, workspace_id: UUID) -> BreakerState:
        """Get breaker state from workspace table."""
        state_dict = await self._repo.get_breaker_state(workspace_id)
        return BreakerState(
            kb_auto_candidacy_state=state_dict.get("kb_auto_candidacy_state", "enabled"),
            kb_auto_candidacy_disabled_until=state_dict.get(
                "kb_auto_candidacy_disabled_until"
            ),
            kb_auto_candidacy_trip_reason=state_dict.get("kb_auto_candidacy_trip_reason"),
            kb_auto_candidacy_tripped_at=state_dict.get("kb_auto_candidacy_tripped_at"),
        )

    async def update_breaker_state(
        self,
        workspace_id: UUID,
        state: str,
        disabled_until: Optional[datetime],
        trip_reason: Optional[str],
    ) -> None:
        """Update breaker state in workspace table."""
        await self._repo.update_breaker_state(
            workspace_id, state, disabled_until, trip_reason
        )

    async def get_recent_candidacy_decisions(
        self, workspace_id: UUID, limit: int
    ) -> list[str]:
        """Get recent candidacy decisions from backtest_runs."""
        return await self._repo.get_recent_candidacy_decisions(workspace_id, limit)

    async def get_candidate_count_rolling_24h(self, workspace_id: UUID) -> int:
        """Get 24h candidate count from backtest_runs."""
        return await self._repo.get_candidate_count_rolling_24h(workspace_id)
