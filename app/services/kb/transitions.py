"""KB Status transition validation.

Implements the state machine for KB status transitions with actor-based
access control and reason requirements.

This is Phase 3 of the trial ingestion design.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Set


class KBStatus(str, Enum):
    """KB status enum.

    Status progression:
    - excluded: Default for test variants - never ingest
    - candidate: Auto-eligible, pending batch ingestion
    - promoted: Explicitly approved - highest priority
    - rejected: Explicitly blocked - never recommend
    """

    EXCLUDED = "excluded"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"


ActorType = Literal["auto", "admin"]


@dataclass
class TransitionResult:
    """Result of a transition validation.

    Attributes:
        valid: Whether the transition is allowed
        error: Error reason if invalid
    """

    valid: bool
    error: Optional[str] = None


class KBStatusTransition:
    """Validates KB status transitions.

    State machine rules:
    - excluded → candidate: auto or admin
    - excluded → promoted: admin only
    - candidate → promoted: admin only
    - candidate → rejected: admin only (requires reason)
    - promoted → rejected: admin only (requires reason)
    - rejected → promoted: admin only (requires reason for override)

    Disallowed:
    - rejected → candidate (must go through promoted)
    - Any transition from/to same status
    - Auto cannot promote or reject
    """

    ALLOWED: dict[tuple[str, str], Set[ActorType]] = {
        ("excluded", "candidate"): {"auto", "admin"},
        ("excluded", "promoted"): {"admin"},
        ("candidate", "promoted"): {"admin"},
        ("candidate", "rejected"): {"admin"},
        ("promoted", "rejected"): {"admin"},
        ("rejected", "promoted"): {"admin"},
    }

    REQUIRES_REASON: set[tuple[str, str]] = {
        ("candidate", "rejected"),
        ("promoted", "rejected"),
        ("rejected", "promoted"),
    }

    def validate(
        self,
        from_status: str,
        to_status: str,
        actor: ActorType,
        reason: Optional[str] = None,
    ) -> TransitionResult:
        """Validate a status transition.

        Args:
            from_status: Current status value
            to_status: Target status value
            actor: Who is performing the transition ("auto" or "admin")
            reason: Reason for the transition (required for some transitions)

        Returns:
            TransitionResult with valid=True if allowed, else error reason
        """
        # Normalize to lowercase
        from_status = from_status.lower()
        to_status = to_status.lower()

        # Same status is a no-op, not invalid
        if from_status == to_status:
            return TransitionResult(valid=True)

        key = (from_status, to_status)

        # Check if transition exists
        if key not in self.ALLOWED:
            return TransitionResult(
                valid=False,
                error=f"transition_{from_status}_to_{to_status}_not_allowed",
            )

        # Check actor permission
        if actor not in self.ALLOWED[key]:
            return TransitionResult(
                valid=False,
                error=f"actor_{actor}_cannot_{from_status}_to_{to_status}",
            )

        # Check reason requirements
        if key in self.REQUIRES_REASON and not reason:
            if to_status == "rejected":
                return TransitionResult(
                    valid=False,
                    error="rejection_requires_reason",
                )
            elif from_status == "rejected" and to_status == "promoted":
                return TransitionResult(
                    valid=False,
                    error="unrejection_requires_reason",
                )

        return TransitionResult(valid=True)

    def can_auto_candidate(self, from_status: str) -> bool:
        """Check if auto can transition to candidate from current status.

        Args:
            from_status: Current status value

        Returns:
            True if auto can make the transition
        """
        from_status = from_status.lower()
        key = (from_status, "candidate")
        return key in self.ALLOWED and "auto" in self.ALLOWED[key]

    def get_allowed_targets(self, from_status: str, actor: ActorType) -> list[str]:
        """Get list of statuses this actor can transition to.

        Args:
            from_status: Current status value
            actor: Who is performing the transition

        Returns:
            List of allowed target status values
        """
        from_status = from_status.lower()
        targets = []
        for (src, tgt), actors in self.ALLOWED.items():
            if src == from_status and actor in actors:
                targets.append(tgt)
        return sorted(targets)


# Singleton instance for convenience
transition_validator = KBStatusTransition()
