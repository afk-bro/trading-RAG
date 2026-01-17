"""Event schemas for SSE notifications."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# Event topics
EventTopic = Literal[
    # Coverage events
    "coverage.weak_run.created",
    "coverage.weak_run.updated",
    # Backtest events
    "backtest.tune.started",
    "backtest.tune.progress",
    "backtest.tune.completed",
    "backtest.tune.failed",
]

# Topic categories for filtering
COVERAGE_TOPICS = {"coverage.weak_run.created", "coverage.weak_run.updated"}
BACKTEST_TOPICS = {
    "backtest.tune.started",
    "backtest.tune.progress",
    "backtest.tune.completed",
    "backtest.tune.failed",
}


class AdminEvent(BaseModel):
    """
    Event payload for SSE notifications.

    Designed for browser EventSource consumption with:
    - Monotonic ID for Last-Event-ID reconnection
    - Topic-based routing
    - Workspace scoping for multi-tenant isolation
    """

    id: str = Field(..., description="Monotonic event ID for reconnection")
    topic: EventTopic = Field(..., description="Event topic (e.g., 'coverage.weak_run.updated')")
    workspace_id: UUID = Field(..., description="Workspace scope for filtering")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC)",
    )
    payload: dict[str, Any] = Field(default_factory=dict, description="Event-specific data")

    def to_sse(self) -> str:
        """
        Format as SSE message with id: line for reconnection.

        Returns:
            SSE-formatted string:
                id: <event_id>
                event: <topic>
                data: <json_payload>

        """
        return f"id: {self.id}\nevent: {self.topic}\ndata: {self.model_dump_json()}\n\n"

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }


# Convenience constructors for common events


def coverage_run_created(
    event_id: str,
    workspace_id: UUID,
    run_id: UUID,
    priority_score: float,
    reason_code: str | None = None,
) -> AdminEvent:
    """Create a coverage.weak_run.created event."""
    return AdminEvent(
        id=event_id,
        topic="coverage.weak_run.created",
        workspace_id=workspace_id,
        payload={
            "run_id": str(run_id),
            "priority_score": priority_score,
            "reason_code": reason_code,
        },
    )


def coverage_run_updated(
    event_id: str,
    workspace_id: UUID,
    run_id: UUID,
    status: str,
    priority_score: float | None = None,
    acknowledged_by: str | None = None,
    resolved_by: str | None = None,
) -> AdminEvent:
    """Create a coverage.weak_run.updated event."""
    payload: dict[str, Any] = {
        "run_id": str(run_id),
        "status": status,
    }
    if priority_score is not None:
        payload["priority_score"] = priority_score
    if acknowledged_by:
        payload["acknowledged_by"] = acknowledged_by
    if resolved_by:
        payload["resolved_by"] = resolved_by

    return AdminEvent(
        id=event_id,
        topic="coverage.weak_run.updated",
        workspace_id=workspace_id,
        payload=payload,
    )


def tune_started(
    event_id: str,
    workspace_id: UUID,
    tune_id: UUID,
    total_trials: int,
) -> AdminEvent:
    """Create a backtest.tune.started event."""
    return AdminEvent(
        id=event_id,
        topic="backtest.tune.started",
        workspace_id=workspace_id,
        payload={
            "tune_id": str(tune_id),
            "total_trials": total_trials,
        },
    )


def tune_progress(
    event_id: str,
    workspace_id: UUID,
    tune_id: UUID,
    completed: int,
    total: int,
    best_score: float | None = None,
) -> AdminEvent:
    """Create a backtest.tune.progress event."""
    payload: dict[str, Any] = {
        "tune_id": str(tune_id),
        "completed": completed,
        "total": total,
    }
    if best_score is not None:
        payload["best_score"] = best_score

    return AdminEvent(
        id=event_id,
        topic="backtest.tune.progress",
        workspace_id=workspace_id,
        payload=payload,
    )


def tune_completed(
    event_id: str,
    workspace_id: UUID,
    tune_id: UUID,
    valid_trials: int,
    best_score: float | None = None,
) -> AdminEvent:
    """Create a backtest.tune.completed event."""
    payload: dict[str, Any] = {
        "tune_id": str(tune_id),
        "valid_trials": valid_trials,
    }
    if best_score is not None:
        payload["best_score"] = best_score

    return AdminEvent(
        id=event_id,
        topic="backtest.tune.completed",
        workspace_id=workspace_id,
        payload=payload,
    )


def tune_failed(
    event_id: str,
    workspace_id: UUID,
    tune_id: UUID,
    error: str,
) -> AdminEvent:
    """Create a backtest.tune.failed event."""
    return AdminEvent(
        id=event_id,
        topic="backtest.tune.failed",
        workspace_id=workspace_id,
        payload={
            "tune_id": str(tune_id),
            "error": error,
        },
    )
