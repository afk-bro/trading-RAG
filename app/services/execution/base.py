"""Abstract broker adapter interface."""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.schemas import (
    TradeIntent,
    ExecutionResult,
    PaperPosition,
    PaperState,
    ReconciliationResult,
)


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters.

    Defines the interface that both PaperBroker and future
    live broker implementations must follow.

    Key design decisions:
    - execute_intent() does NOT take a PolicyDecision - adapter re-evaluates internally
    - fill_price is required - execution emits facts, not guesses
    - All methods are async for consistency with codebase patterns
    """

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return execution mode (paper/live)."""
        pass

    @abstractmethod
    async def execute_intent(
        self,
        intent: TradeIntent,
        fill_price: float,
    ) -> ExecutionResult:
        """
        Execute an intent.

        The adapter will:
        1. Validate action is supported (OPEN_LONG, CLOSE_LONG only in PR1)
        2. Check idempotency (reject if intent_id already executed)
        3. Re-evaluate policy internally (don't trust caller)
        4. Execute and journal events

        Args:
            intent: The trade intent to execute
            fill_price: Required fill price (no lookup/slippage)

        Returns:
            ExecutionResult with order and position updates

        Raises:
            HTTPException 400: Unsupported action or validation failure
            HTTPException 409: Intent already executed (idempotency)
        """
        pass

    @abstractmethod
    async def get_positions(self, workspace_id: UUID) -> list[PaperPosition]:
        """Get all open positions for a workspace."""
        pass

    @abstractmethod
    async def get_position(
        self, workspace_id: UUID, symbol: str
    ) -> Optional[PaperPosition]:
        """Get position for a specific symbol."""
        pass

    @abstractmethod
    async def get_state(self, workspace_id: UUID) -> PaperState:
        """Get complete paper trading state."""
        pass

    @abstractmethod
    async def reconcile_from_journal(
        self, workspace_id: UUID
    ) -> ReconciliationResult:
        """
        Rebuild state from journal events.

        Replays ORDER_FILLED events only (source of truth).
        POSITION_* events are observability breadcrumbs.

        Deduplicates by order_id to handle duplicate journal entries.
        """
        pass

    @abstractmethod
    async def reset(self, workspace_id: UUID) -> None:
        """Reset state for a workspace (dev/test only)."""
        pass
