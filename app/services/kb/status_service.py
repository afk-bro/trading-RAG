"""KB Status service for managing status transitions.

Handles status transitions with validation, audit logging, and archive triggers.

This is Phase 3 of the trial ingestion design.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Protocol
from uuid import UUID

import structlog

from app.services.kb.transitions import KBStatusTransition

logger = structlog.get_logger(__name__)


def format_actor_id(actor_type: str, actor_id: Optional[str]) -> str:
    """Format actor identifier for audit logging.

    Standardized format: <type>:<id>
    - admin:<user_id> - Admin user action
    - token:<token_name> - API token action
    - service:<service_name> - Automated service action
    - auto:<gate_name> - Auto-candidacy gate

    Args:
        actor_type: Type of actor (auto, admin, token, service)
        actor_id: The actor's identifier

    Returns:
        Formatted actor string for audit logs
    """
    if actor_id is None:
        return f"{actor_type}:unknown"
    return f"{actor_type}:{actor_id}"


SourceType = Literal["tune_run", "test_variant"]


@dataclass
class CurrentStatus:
    """Current KB status of a trial.

    Attributes:
        workspace_id: Workspace owning the trial
        kb_status: Current status value
        kb_promoted_at: When the trial was promoted (if ever)
    """

    workspace_id: UUID
    kb_status: str
    kb_promoted_at: Optional[datetime]


@dataclass
class KBStatusResult:
    """Result of a status transition.

    Attributes:
        source_type: Type of source (tune_run or test_variant)
        source_id: ID of the source record
        from_status: Previous status
        to_status: New status
        skipped: True if no transition was needed (same status)
    """

    source_type: str
    source_id: UUID
    from_status: str
    to_status: str
    skipped: bool = False

    @property
    def transitioned(self) -> bool:
        """True if a transition actually occurred."""
        return not self.skipped


class InvalidTransitionError(Exception):
    """Raised when a status transition is not allowed."""

    def __init__(self, error_code: str, message: Optional[str] = None):
        self.error_code = error_code
        self.message = message or error_code
        super().__init__(self.message)


class TrialNotFoundError(Exception):
    """Raised when a trial is not found."""


class KBStatusRepository(Protocol):
    """Protocol for KB status persistence operations.

    Implementations handle the actual database operations.
    """

    async def get_current_status(
        self, source_type: SourceType, source_id: UUID
    ) -> Optional[CurrentStatus]:
        """Get current status of a trial.

        Args:
            source_type: Type of source (tune_run or test_variant)
            source_id: ID of the source record

        Returns:
            CurrentStatus or None if not found
        """
        ...

    async def update_status(
        self,
        source_type: SourceType,
        source_id: UUID,
        to_status: str,
        changed_by: Optional[str],
        reason: Optional[str],
    ) -> None:
        """Update trial status.

        Args:
            source_type: Type of source
            source_id: ID of the source record
            to_status: New status value
            changed_by: Who made the change
            reason: Reason for the change
        """
        ...

    async def set_promoted_at(
        self,
        source_type: SourceType,
        source_id: UUID,
        promoted_by: Optional[str],
    ) -> None:
        """Set the promoted_at timestamp.

        Args:
            source_type: Type of source
            source_id: ID of the source record
            promoted_by: Who promoted the trial
        """
        ...

    async def insert_history(
        self,
        workspace_id: UUID,
        source_type: SourceType,
        source_id: UUID,
        from_status: str,
        to_status: str,
        actor_type: str,
        actor_id: Optional[str],
        reason: Optional[str],
    ) -> None:
        """Insert a history record.

        Args:
            workspace_id: Workspace ID
            source_type: Type of source
            source_id: ID of the source record
            from_status: Previous status
            to_status: New status
            actor_type: Type of actor (auto or admin)
            actor_id: ID of the actor
            reason: Reason for the change
        """
        ...


class KBIndexRepository(Protocol):
    """Protocol for KB index operations (archive/unarchive)."""

    async def archive_trial(
        self,
        workspace_id: UUID,
        source_type: SourceType,
        source_id: UUID,
        reason: str,
        actor: Optional[str],
    ) -> bool:
        """Archive a trial from the KB index.

        Args:
            workspace_id: Workspace ID
            source_type: Type of source
            source_id: ID of the source record
            reason: Why the trial is being archived
            actor: Who is archiving

        Returns:
            True if archived, False if not found in index
        """
        ...

    async def unarchive_trial(
        self,
        source_type: SourceType,
        source_id: UUID,
    ) -> bool:
        """Unarchive a trial in the KB index.

        Args:
            source_type: Type of source
            source_id: ID of the source record

        Returns:
            True if unarchived, False if not found
        """
        ...


class KBStatusService:
    """Service for managing KB status transitions.

    Provides transactional status transitions with:
    - Validation via state machine rules
    - Audit logging to kb_status_history
    - Archive triggers on rejection
    - Unarchive + optional re-ingestion on promotion from rejected
    """

    def __init__(
        self,
        status_repo: KBStatusRepository,
        index_repo: Optional[KBIndexRepository] = None,
        validator: Optional[KBStatusTransition] = None,
    ):
        """Initialize the service.

        Args:
            status_repo: Repository for status operations
            index_repo: Repository for index operations (optional)
            validator: Transition validator (uses default if not provided)
        """
        self._status_repo = status_repo
        self._index_repo = index_repo
        self._validator = validator or KBStatusTransition()

    async def transition(
        self,
        source_type: SourceType,
        source_id: UUID,
        to_status: str,
        actor_type: Literal["auto", "admin"],
        actor_id: Optional[str] = None,
        reason: Optional[str] = None,
        trigger_ingest: bool = False,
    ) -> KBStatusResult:
        """Perform a status transition.

        This is the main entry point for all status changes.
        Validates the transition, updates the status, logs to history,
        and handles archive triggers.

        Args:
            source_type: Type of source (tune_run or test_variant)
            source_id: ID of the source record
            to_status: Target status value
            actor_type: Who is performing the transition
            actor_id: ID of the actor (e.g., admin user ID)
            reason: Reason for the transition (required for some)
            trigger_ingest: If promoting from rejected, re-ingest to KB

        Returns:
            KBStatusResult with transition details

        Raises:
            TrialNotFoundError: If the trial doesn't exist
            InvalidTransitionError: If the transition is not allowed
        """
        log = logger.bind(
            source_type=source_type,
            source_id=str(source_id),
            to_status=to_status,
            actor_type=actor_type,
        )

        # Get current status
        current = await self._status_repo.get_current_status(source_type, source_id)
        if current is None:
            raise TrialNotFoundError(f"{source_type}:{source_id} not found")

        from_status = current.kb_status

        # Skip if already at target status
        if from_status == to_status:
            log.debug("status_transition_skipped", reason="already_at_status")
            return KBStatusResult(
                source_type=source_type,
                source_id=source_id,
                from_status=from_status,
                to_status=to_status,
                skipped=True,
            )

        # Validate transition
        result = self._validator.validate(from_status, to_status, actor_type, reason)
        if not result.valid:
            log.warning("status_transition_rejected", error=result.error)
            raise InvalidTransitionError(result.error)

        # Update status
        await self._status_repo.update_status(
            source_type=source_type,
            source_id=source_id,
            to_status=to_status,
            changed_by=actor_id,
            reason=reason,
        )

        # Set promoted_at on promotion
        if to_status == "promoted":
            await self._status_repo.set_promoted_at(
                source_type=source_type,
                source_id=source_id,
                promoted_by=actor_id,
            )

        # Insert history record
        await self._status_repo.insert_history(
            workspace_id=current.workspace_id,
            source_type=source_type,
            source_id=source_id,
            from_status=from_status,
            to_status=to_status,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
        )

        # Archive on rejection
        if to_status == "rejected" and self._index_repo:
            archived = await self._index_repo.archive_trial(
                workspace_id=current.workspace_id,
                source_type=source_type,
                source_id=source_id,
                reason=reason or "rejected",
                actor=actor_id,
            )
            if archived:
                log.info("trial_archived", reason=reason)

        # Unarchive on promotion from rejected
        if from_status == "rejected" and to_status == "promoted" and self._index_repo:
            unarchived = await self._index_repo.unarchive_trial(
                source_type=source_type,
                source_id=source_id,
            )
            if unarchived:
                log.info("trial_unarchived")

            # TODO: Trigger re-ingestion if trigger_ingest is True
            # This would call the ingestion pipeline to re-add to Qdrant
            if trigger_ingest:
                log.info("trigger_ingest_requested", status="not_implemented")

        log.info(
            "status_transition_completed",
            from_status=from_status,
            to_status=to_status,
        )

        # Observability counter: emit for aggregation
        log.info(
            "kb_admin_status_change_total",
            transition=f"{from_status}_to_{to_status}",
            from_status=from_status,
            to_status=to_status,
            actor=format_actor_id(actor_type, actor_id),
            actor_type=actor_type,
            source_type=source_type,
            workspace_id=str(current.workspace_id),
        )

        return KBStatusResult(
            source_type=source_type,
            source_id=source_id,
            from_status=from_status,
            to_status=to_status,
        )

    async def bulk_transition(
        self,
        source_type: SourceType,
        source_ids: list[UUID],
        to_status: str,
        actor_type: Literal["auto", "admin"],
        actor_id: Optional[str] = None,
        reason: Optional[str] = None,
        trigger_ingest: bool = False,
    ) -> list[KBStatusResult]:
        """Perform bulk status transitions.

        Processes each transition independently. Failed transitions
        are logged but don't stop processing of remaining items.

        Args:
            source_type: Type of source (tune_run or test_variant)
            source_ids: List of source IDs to transition
            to_status: Target status value
            actor_type: Who is performing the transitions
            actor_id: ID of the actor
            reason: Reason for the transitions
            trigger_ingest: If promoting from rejected, re-ingest to KB

        Returns:
            List of KBStatusResult for each transition
        """
        results = []
        for source_id in source_ids:
            try:
                result = await self.transition(
                    source_type=source_type,
                    source_id=source_id,
                    to_status=to_status,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    reason=reason,
                    trigger_ingest=trigger_ingest,
                )
                results.append(result)
            except (TrialNotFoundError, InvalidTransitionError) as e:
                logger.warning(
                    "bulk_transition_item_failed",
                    source_type=source_type,
                    source_id=str(source_id),
                    error=str(e),
                )
                # Continue processing remaining items

        return results

    async def get_history(
        self,
        source_type: SourceType,
        source_id: UUID,
        limit: int = 50,
    ) -> list[dict]:
        """Get transition history for a trial.

        Note: This requires an additional repository method not defined
        in the protocol. Implementations should add this capability.

        Args:
            source_type: Type of source
            source_id: ID of the source record
            limit: Maximum number of history records

        Returns:
            List of history records (most recent first)
        """
        # This would be implemented by the repository
        # For now, return empty list
        return []
