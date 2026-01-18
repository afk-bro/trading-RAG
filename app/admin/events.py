"""SSE (Server-Sent Events) endpoint for real-time admin notifications.

Provides real-time updates for:
- Coverage cockpit (weak run status changes)
- Backtest tune progress
"""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Cookie, Depends, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.admin.sse_ticket import create_sse_ticket, get_sse_auth
from app.deps.security import require_admin_token
from app.services.events import get_event_bus

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/events", tags=["admin-sse"])


# ===========================================
# Response Models
# ===========================================


class SSETicketResponse(BaseModel):
    """Response containing SSE ticket."""

    ticket: str
    expires_in_seconds: int


# ===========================================
# Ticket Generation Endpoint
# ===========================================


@router.post(
    "/ticket",
    response_model=SSETicketResponse,
    summary="Generate SSE connection ticket",
    dependencies=[Depends(require_admin_token)],
)
async def generate_sse_ticket(
    workspace_id: Optional[UUID] = Query(
        default=None,
        description="Workspace to scope ticket to (None = all workspaces)",
    ),
) -> SSETicketResponse:
    """
    Generate a short-lived ticket for SSE connections.

    The ticket should be stored as an HttpOnly cookie and used
    to authenticate the SSE connection. This avoids exposing
    the raw admin token in long-lived connections.

    **Requires**: X-Admin-Token header

    **Returns**: Signed ticket valid for 5 minutes
    """
    from app.config import get_settings

    settings = get_settings()
    expiry_seconds = settings.sse_ticket_expiry_seconds

    ticket = create_sse_ticket(
        workspace_id=workspace_id,
        is_admin=True,
        expiry_seconds=expiry_seconds,
    )

    logger.info(
        "sse_ticket_generated",
        workspace_id=str(workspace_id) if workspace_id else "all",
        expires_in=expiry_seconds,
    )

    return SSETicketResponse(
        ticket=ticket,
        expires_in_seconds=expiry_seconds,
    )


# ===========================================
# SSE Stream Endpoint
# ===========================================


@router.get(
    "/stream",
    summary="SSE event stream",
    response_class=StreamingResponse,
)
async def event_stream(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to receive events for"),
    topics: str = Query(
        default="coverage,backtests",
        description="Comma-separated topics to subscribe to",
    ),
    sse_ticket: Optional[str] = Cookie(default=None, alias="sse_ticket"),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    """
    Server-Sent Events stream for real-time admin notifications.

    **Authentication** (one required):
    - `sse_ticket` cookie (preferred, from POST /events/ticket)
    - `X-Admin-Token` header (fallback for curl/testing)

    **Topics**:
    - `coverage` - Coverage gap updates (created, status changes)
    - `backtests` - Tune progress (started, progress, completed, failed)

    **Headers sent**:
    - `Content-Type: text/event-stream`
    - `Cache-Control: no-cache`
    - `Connection: keep-alive`
    - `X-Accel-Buffering: no` (disables nginx buffering)

    **Event format**:
    ```
    id: evt-123
    event: coverage.weak_run.updated
    data: {"id":"evt-123","topic":"...","workspace_id":"...","payload":{...}}

    ```

    **Reconnection**: Browser sends `Last-Event-ID` header on reconnect.
    Server replays missed events from buffer (5 min window).
    """
    # Authenticate
    claims = get_sse_auth(
        ticket_cookie=sse_ticket,
        admin_header=x_admin_token,
    )

    if not claims:
        logger.warning(
            "sse_auth_failed",
            has_cookie=sse_ticket is not None,
            has_header=x_admin_token is not None,
        )
        return Response(
            status_code=401,
            content="Unauthorized: Provide sse_ticket cookie or X-Admin-Token header",
        )

    # Check workspace access
    if claims.workspace_id and claims.workspace_id != workspace_id:
        if not claims.is_admin:
            return Response(
                status_code=403,
                content=f"Access denied to workspace {workspace_id}",
            )

    # Parse topics
    topic_set = set(t.strip() for t in topics.split(",") if t.strip())
    if not topic_set:
        topic_set = {"coverage", "backtests"}

    # Get Last-Event-ID from headers (browser sends on reconnect)
    last_event_id = request.headers.get("last-event-id")

    # Generate unique subscriber ID
    from uuid import uuid4

    subscriber_id = f"sse-{uuid4()}"

    logger.info(
        "sse_connection_started",
        subscriber_id=subscriber_id,
        workspace_id=str(workspace_id),
        topics=list(topic_set),
        last_event_id=last_event_id,
        is_reconnect=last_event_id is not None,
    )

    async def generate():
        """Generate SSE events."""
        bus = get_event_bus()

        # Send initial comment to establish connection
        yield ": connected\n\n"

        try:
            async for event in bus.subscribe(
                subscriber_id=subscriber_id,
                workspace_ids={workspace_id},
                topics=topic_set,
                last_event_id=last_event_id,
            ):
                yield event.to_sse()
        except Exception as e:
            logger.warning(
                "sse_stream_error",
                subscriber_id=subscriber_id,
                error=str(e),
            )
        finally:
            logger.info(
                "sse_connection_closed",
                subscriber_id=subscriber_id,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ===========================================
# Status Endpoint
# ===========================================


class RedisStatusInfo(BaseModel):
    """Redis-specific status information."""

    connected: bool
    stream_lengths: dict[str, int] = {}
    last_publish_ids: dict[str, str | None] = {}


class SSEStatusResponse(BaseModel):
    """SSE system status."""

    mode: str  # "memory" or "redis"
    subscriber_count: int
    buffer_size: int
    topics_available: list[str]
    redis: RedisStatusInfo | None = None


@router.get(
    "/status",
    response_model=SSEStatusResponse,
    summary="SSE system status",
    dependencies=[Depends(require_admin_token)],
)
async def get_sse_status() -> SSEStatusResponse:
    """
    Get SSE system status.

    **Requires**: X-Admin-Token header

    **Returns**:
    - mode: Event bus implementation ("memory" or "redis")
    - subscriber_count: Current number of active SSE subscribers
    - buffer_size: Maximum events buffered for reconnection
    - topics_available: List of subscribable topic names
    - redis: Redis-specific diagnostics (only when mode=redis)
      - connected: Redis connection status
      - stream_lengths: Current event count per workspace stream
      - last_publish_ids: Most recent event ID per workspace
    """
    from app.config import get_settings
    from app.services.events.schemas import COVERAGE_TOPICS, BACKTEST_TOPICS

    settings = get_settings()
    bus = get_event_bus()
    mode = settings.event_bus_mode

    redis_info: RedisStatusInfo | None = None

    if mode == "redis":
        from app.services.events.redis_bus import RedisEventBus

        if isinstance(bus, RedisEventBus):
            try:
                connected = await bus.ping()
                stream_lengths = await bus.get_stream_lengths() if connected else {}

                redis_info = RedisStatusInfo(
                    connected=connected,
                    stream_lengths=stream_lengths,
                    last_publish_ids={},  # Can be populated per-workspace if needed
                )
            except Exception as e:
                logger.warning("sse_status_redis_error", error=str(e))
                redis_info = RedisStatusInfo(connected=False)

    return SSEStatusResponse(
        mode=mode,
        subscriber_count=bus.subscriber_count(),
        buffer_size=bus.buffer_size() if hasattr(bus, "buffer_size") else 0,
        topics_available=list(COVERAGE_TOPICS | BACKTEST_TOPICS),
        redis=redis_info,
    )
