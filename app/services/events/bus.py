"""Event bus for real-time admin notifications.

Provides an abstract interface for event distribution with an in-memory
implementation for single-worker deployments. Future implementations can
use Redis or PostgreSQL NOTIFY for multi-worker support.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import AsyncIterator, Set
from uuid import UUID

import structlog

from app.services.events.schemas import (
    AdminEvent,
    BACKTEST_TOPICS,
    COVERAGE_TOPICS,
    PINE_TOPICS,
)

logger = structlog.get_logger(__name__)


class EventBus(ABC):
    """
    Abstract interface for event distribution.

    Current: InMemoryEventBus (single worker)
    Future: PgNotifyEventBus or RedisEventBus (multi-worker)
    """

    @abstractmethod
    def subscribe(
        self,
        subscriber_id: str,
        workspace_ids: Set[UUID],
        topics: Set[str],
        last_event_id: str | None = None,
    ) -> AsyncIterator[AdminEvent]:
        """
        Subscribe to events (async generator).

        Note: Abstract method declared without `async` for correct type annotation.
        Implementations use `async def` with `yield` to return AsyncIterator.

        Args:
            subscriber_id: Unique identifier for this subscriber (for cleanup)
            workspace_ids: Set of workspace IDs to filter events
            topics: Set of topic patterns to subscribe to
                    (e.g., {"coverage", "backtests"} or specific topics)
            last_event_id: Optional event ID for reconnection replay

        Yields:
            AdminEvent objects matching the filters.
        """
        ...

    @abstractmethod
    async def publish(self, event: AdminEvent) -> int:
        """
        Publish an event to all matching subscribers.

        Args:
            event: The event to publish

        Returns:
            Number of subscribers that received the event.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscriber_id: str) -> None:
        """
        Clean up subscriber on disconnect.

        Args:
            subscriber_id: The subscriber ID to remove
        """
        ...

    @abstractmethod
    def subscriber_count(self) -> int:
        """Return current number of active subscribers."""
        ...


def _expand_topics(topics: Set[str]) -> Set[str]:
    """
    Expand category names to specific topics.

    Args:
        topics: Set containing category names ("coverage", "backtests")
                or specific topics ("coverage.weak_run.updated")

    Returns:
        Set of specific topic names.
    """
    result: Set[str] = set()
    for topic in topics:
        if topic == "coverage":
            result.update(COVERAGE_TOPICS)
        elif topic == "backtests":
            result.update(BACKTEST_TOPICS)
        elif topic == "pine":
            result.update(PINE_TOPICS)
        else:
            result.add(topic)
    return result


class InMemoryEventBus(EventBus):
    """
    In-memory event bus implementation for single-worker deployments.

    Features:
    - Workspace-scoped event filtering
    - Topic-based subscription
    - Event buffer for reconnection (Last-Event-ID support)

    Limitations:
    - Events only reach subscribers in same process
    - Event buffer lost on restart (acceptable for admin tooling)
    """

    def __init__(self, buffer_size: int = 1000, buffer_ttl_seconds: int = 300):
        """
        Initialize the event bus.

        Args:
            buffer_size: Maximum events to buffer for reconnection
            buffer_ttl_seconds: Time to keep events in buffer (5 min default)
        """
        self._subscribers: dict[str, asyncio.Queue[AdminEvent]] = {}
        self._filters: dict[str, tuple[Set[UUID], Set[str]]] = {}
        self._event_buffer: deque[tuple[float, AdminEvent]] = deque(maxlen=buffer_size)
        self._buffer_ttl = buffer_ttl_seconds
        self._event_counter: int = 0
        self._lock = asyncio.Lock()

    def _generate_event_id(self) -> str:
        """Generate monotonic event ID."""
        self._event_counter += 1
        return f"evt-{self._event_counter}"

    def _match_event(
        self,
        event: AdminEvent,
        workspace_ids: Set[UUID],
        topics: Set[str],
    ) -> bool:
        """Check if event matches subscription filters."""
        return event.workspace_id in workspace_ids and event.topic in topics

    async def subscribe(
        self,
        subscriber_id: str,
        workspace_ids: Set[UUID],
        topics: Set[str],
        last_event_id: str | None = None,
    ) -> AsyncIterator[AdminEvent]:
        """
        Subscribe to events matching workspace and topic filters.

        Args:
            subscriber_id: Unique subscriber identifier
            workspace_ids: Workspaces to receive events for
            topics: Topic patterns to subscribe to
            last_event_id: Optional event ID for reconnection replay

        Yields:
            Matching AdminEvent objects.
        """
        expanded_topics = _expand_topics(topics)
        queue: asyncio.Queue[AdminEvent] = asyncio.Queue()

        async with self._lock:
            self._subscribers[subscriber_id] = queue
            self._filters[subscriber_id] = (workspace_ids, expanded_topics)

        logger.info(
            "sse_subscriber_added",
            subscriber_id=subscriber_id,
            workspaces=len(workspace_ids),
            topics=list(expanded_topics),
            total_subscribers=len(self._subscribers),
        )

        try:
            # Replay missed events if reconnecting
            if last_event_id:
                async for event in self._replay_from(
                    last_event_id, workspace_ids, expanded_topics
                ):
                    yield event

            # Stream new events
            while True:
                event = await queue.get()
                yield event
        finally:
            await self.unsubscribe(subscriber_id)

    async def _replay_from(
        self,
        last_event_id: str,
        workspace_ids: Set[UUID],
        topics: Set[str],
    ) -> AsyncIterator[AdminEvent]:
        """
        Replay events from buffer after a given event ID.

        Used for Last-Event-ID reconnection.
        """
        found_start = False
        cutoff = time.monotonic() - self._buffer_ttl

        for timestamp, event in self._event_buffer:
            # Skip expired events
            if timestamp < cutoff:
                continue

            # Skip until we find the last seen event
            if not found_start:
                if event.id == last_event_id:
                    found_start = True
                continue

            # Yield matching events after the last seen
            if self._match_event(event, workspace_ids, topics):
                yield event

    async def publish(self, event: AdminEvent) -> int:
        """
        Publish event to all matching subscribers.

        Args:
            event: Event to publish (ID will be assigned if not set)

        Returns:
            Number of subscribers that received the event.
        """
        # Assign event ID if not set
        if not event.id or event.id.startswith("evt-"):
            event.id = self._generate_event_id()

        # Add to buffer for replay
        async with self._lock:
            self._event_buffer.append((time.monotonic(), event))

        # Distribute to matching subscribers
        count = 0
        for sub_id, queue in list(self._subscribers.items()):
            try:
                ws_filter, topic_filter = self._filters.get(sub_id, (set(), set()))
                if self._match_event(event, ws_filter, topic_filter):
                    await queue.put(event)
                    count += 1
            except Exception as e:
                logger.warning(
                    "sse_publish_error",
                    subscriber_id=sub_id,
                    error=str(e),
                )

        if count > 0:
            logger.debug(
                "sse_event_published",
                event_id=event.id,
                topic=event.topic,
                subscriber_count=count,
            )

        return count

    async def unsubscribe(self, subscriber_id: str) -> None:
        """Remove subscriber and clean up resources."""
        async with self._lock:
            self._subscribers.pop(subscriber_id, None)
            self._filters.pop(subscriber_id, None)

        logger.info(
            "sse_subscriber_removed",
            subscriber_id=subscriber_id,
            remaining_subscribers=len(self._subscribers),
        )

    def subscriber_count(self) -> int:
        """Return current number of active subscribers."""
        return len(self._subscribers)

    def buffer_size(self) -> int:
        """Return current event buffer size."""
        return len(self._event_buffer)


# ===========================================
# Singleton instance
# ===========================================

_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the singleton event bus instance.

    Uses config to determine implementation:
    - memory: InMemoryEventBus (single worker, default)
    - redis: RedisEventBus (multi-worker, requires REDIS_URL)

    Returns:
        EventBus instance (singleton per process)

    Raises:
        ValueError: If event_bus_mode=redis but REDIS_URL not configured
    """
    global _event_bus
    if _event_bus is None:
        from app.config import get_settings

        settings = get_settings()

        if settings.event_bus_mode == "redis":
            if not settings.redis_url:
                raise ValueError(
                    "EVENT_BUS_MODE=redis requires REDIS_URL to be set. "
                    "Example: redis://localhost:6379/0"
                )
            from app.services.events.redis_bus import RedisEventBus

            _event_bus = RedisEventBus(
                redis_url=settings.redis_url,
                buffer_size=settings.event_bus_buffer_size,
            )
            logger.info(
                "event_bus_initialized",
                mode="redis",
                buffer_size=settings.event_bus_buffer_size,
            )
        else:
            _event_bus = InMemoryEventBus(
                buffer_size=settings.event_bus_buffer_size,
            )
            logger.info(
                "event_bus_initialized",
                mode="memory",
                buffer_size=settings.event_bus_buffer_size,
            )

    return _event_bus


def set_event_bus(bus: EventBus | None) -> None:
    """
    Set the event bus instance (for testing or runtime replacement).

    Args:
        bus: EventBus implementation to use, or None to reset
    """
    global _event_bus
    _event_bus = bus


def reset_event_bus() -> None:
    """
    Reset the event bus singleton (for testing).

    Forces re-initialization on next get_event_bus() call.
    """
    global _event_bus
    _event_bus = None
