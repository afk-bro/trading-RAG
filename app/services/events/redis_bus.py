"""Redis-backed event bus for multi-worker SSE deployments.

Uses Redis Streams for:
- Cross-worker event fanout
- Replay buffer with Last-Event-ID support
- Workspace-scoped stream isolation

Stream key pattern: sse:events:<workspace_id>
"""

import asyncio
import json
import time
from typing import AsyncIterator, Set
from uuid import UUID

import structlog

from app.services.events.bus import EventBus, _expand_topics
from app.services.events.schemas import AdminEvent

logger = structlog.get_logger(__name__)


class RedisEventBus(EventBus):
    """
    Redis Streams event bus for multi-worker deployments.

    Features:
    - Cross-process event delivery (multi-worker safe)
    - Workspace-scoped streams (isolation)
    - Last-Event-ID reconnection via Redis Stream IDs
    - Auto-trimming with XTRIM MAXLEN ~

    Stream key pattern: sse:events:{workspace_id}

    Event format in stream:
        XADD sse:events:<ws_id> * topic=<topic> payload=<json>
    """

    def __init__(
        self,
        redis_url: str,
        buffer_size: int = 2000,
        read_block_ms: int = 5000,
    ):
        """
        Initialize Redis event bus.

        Args:
            redis_url: Redis connection URL (redis://host:port/db or rediss://...)
            buffer_size: Max events per workspace stream (XTRIM MAXLEN ~)
            read_block_ms: XREAD BLOCK timeout in milliseconds
        """
        self._redis_url = redis_url
        self._buffer_size = buffer_size
        self._read_block_ms = read_block_ms
        self._redis: "redis.asyncio.Redis | None" = None
        self._subscribers: dict[str, asyncio.Task] = {}
        self._subscriber_queues: dict[str, asyncio.Queue[AdminEvent]] = {}
        self._filters: dict[str, tuple[Set[UUID], Set[str]]] = {}
        self._lock = asyncio.Lock()
        self._connected = False

    async def _get_redis(self) -> "redis.asyncio.Redis":
        """Get or create Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis_async

            self._redis = redis_async.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5.0,
                socket_keepalive=True,
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("redis_event_bus_connected", url=self._redis_url[:30] + "...")

        return self._redis

    async def close(self) -> None:
        """Close Redis connection and cancel all subscriber tasks."""
        async with self._lock:
            # Cancel all subscriber tasks
            for task in self._subscribers.values():
                task.cancel()
            self._subscribers.clear()
            self._subscriber_queues.clear()
            self._filters.clear()

            # Close Redis connection
            if self._redis:
                await self._redis.aclose()
                self._redis = None
                self._connected = False
                logger.info("redis_event_bus_closed")

    def _stream_key(self, workspace_id: UUID) -> str:
        """Get Redis stream key for workspace."""
        return f"sse:events:{workspace_id}"

    async def publish(self, event: AdminEvent) -> int:
        """
        Publish event to Redis stream.

        Args:
            event: Event to publish

        Returns:
            Number of local subscribers that received the event.
            (Remote subscribers receive via their own XREAD)
        """
        redis = await self._get_redis()
        stream_key = self._stream_key(event.workspace_id)

        # Serialize event payload
        payload = {
            "topic": event.topic,
            "workspace_id": str(event.workspace_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": json.dumps(event.payload),
        }

        # XADD with auto-ID, returns the stream ID (becomes event.id)
        stream_id = await redis.xadd(
            stream_key,
            payload,
            maxlen=self._buffer_size,
            approximate=True,  # MAXLEN ~ for performance
        )

        # Update event ID to Redis stream ID
        event.id = stream_id

        logger.debug(
            "redis_event_published",
            stream_key=stream_key,
            stream_id=stream_id,
            topic=event.topic,
        )

        # Count local subscribers (other workers receive via their own XREAD)
        local_count = 0
        for sub_id, queue in list(self._subscriber_queues.items()):
            ws_filter, topic_filter = self._filters.get(sub_id, (set(), set()))
            if self._match_event(event, ws_filter, topic_filter):
                try:
                    # Non-blocking put to avoid deadlock
                    queue.put_nowait(event)
                    local_count += 1
                except asyncio.QueueFull:
                    logger.warning(
                        "redis_subscriber_queue_full",
                        subscriber_id=sub_id,
                    )

        return local_count

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
        Subscribe to events from Redis streams.

        Args:
            subscriber_id: Unique subscriber identifier
            workspace_ids: Workspaces to receive events for
            topics: Topic patterns to subscribe to
            last_event_id: Optional Redis stream ID for reconnection replay

        Yields:
            Matching AdminEvent objects.
        """
        expanded_topics = _expand_topics(topics)
        queue: asyncio.Queue[AdminEvent] = asyncio.Queue(maxsize=1000)

        async with self._lock:
            self._subscriber_queues[subscriber_id] = queue
            self._filters[subscriber_id] = (workspace_ids, expanded_topics)

        logger.info(
            "redis_subscriber_added",
            subscriber_id=subscriber_id,
            workspaces=len(workspace_ids),
            topics=list(expanded_topics),
            last_event_id=last_event_id,
        )

        try:
            redis = await self._get_redis()

            # Build stream keys for all workspaces
            streams: dict[str, str] = {}
            for ws_id in workspace_ids:
                stream_key = self._stream_key(ws_id)
                # Use last_event_id if provided, otherwise start from now
                # Redis stream ID format: <milliseconds>-<seq>
                # "$" means "only new messages"
                streams[stream_key] = last_event_id if last_event_id else "$"

            # If reconnecting with last_event_id, first replay missed events
            if last_event_id:
                async for event in self._replay_from(
                    streams, expanded_topics, last_event_id
                ):
                    yield event
                # After replay, set all streams to listen from $ (new only)
                for key in streams:
                    streams[key] = "$"

            # Start background reader task
            reader_task = asyncio.create_task(
                self._stream_reader(subscriber_id, streams, expanded_topics, queue)
            )
            async with self._lock:
                self._subscribers[subscriber_id] = reader_task

            # Yield events from queue
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event
                except asyncio.TimeoutError:
                    # Keep-alive check - ensure reader is still running
                    if reader_task.done():
                        exc = reader_task.exception()
                        if exc:
                            logger.error(
                                "redis_reader_failed",
                                subscriber_id=subscriber_id,
                                error=str(exc),
                            )
                        break

        finally:
            await self.unsubscribe(subscriber_id)

    async def _stream_reader(
        self,
        subscriber_id: str,
        streams: dict[str, str],
        topics: Set[str],
        queue: asyncio.Queue[AdminEvent],
    ) -> None:
        """
        Background task that reads from Redis streams and pushes to queue.
        """
        redis = await self._get_redis()

        while True:
            try:
                # XREAD BLOCK - blocking read from multiple streams
                # Returns: [(stream_key, [(id, {fields}), ...]), ...]
                results = await redis.xread(
                    streams,
                    block=self._read_block_ms,
                    count=100,
                )

                if not results:
                    continue

                for stream_key, messages in results:
                    for msg_id, fields in messages:
                        # Update stream position for next read
                        streams[stream_key] = msg_id

                        # Parse event
                        event = self._parse_stream_event(msg_id, fields)
                        if event is None:
                            continue

                        # Filter by topic
                        if event.topic not in topics:
                            continue

                        # Push to subscriber queue
                        try:
                            queue.put_nowait(event)
                        except asyncio.QueueFull:
                            logger.warning(
                                "redis_queue_full_dropping_event",
                                subscriber_id=subscriber_id,
                                event_id=msg_id,
                            )

            except asyncio.CancelledError:
                logger.debug("redis_reader_cancelled", subscriber_id=subscriber_id)
                break
            except Exception as e:
                logger.error(
                    "redis_reader_error",
                    subscriber_id=subscriber_id,
                    error=str(e),
                )
                # Brief backoff before retry
                await asyncio.sleep(1.0)

    async def _replay_from(
        self,
        streams: dict[str, str],
        topics: Set[str],
        last_event_id: str,
    ) -> AsyncIterator[AdminEvent]:
        """
        Replay events from streams starting after last_event_id.
        """
        redis = await self._get_redis()

        for stream_key, start_id in streams.items():
            if start_id == "$":
                continue

            try:
                # XRANGE from last_event_id (exclusive) to latest
                # Use "(id" for exclusive start
                messages = await redis.xrange(
                    stream_key,
                    min=f"({last_event_id}",  # Exclusive (after this ID)
                    max="+",  # Up to latest
                    count=1000,  # Limit replay batch
                )

                for msg_id, fields in messages:
                    event = self._parse_stream_event(msg_id, fields)
                    if event and event.topic in topics:
                        yield event

            except Exception as e:
                logger.warning(
                    "redis_replay_error",
                    stream_key=stream_key,
                    start_id=start_id,
                    error=str(e),
                )

    def _parse_stream_event(
        self,
        stream_id: str,
        fields: dict,
    ) -> AdminEvent | None:
        """Parse Redis stream message into AdminEvent."""
        try:
            from datetime import datetime

            workspace_id = UUID(fields["workspace_id"])
            payload = json.loads(fields.get("payload", "{}"))

            # Parse timestamp, fallback to stream ID timestamp
            timestamp_str = fields.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                # Extract timestamp from stream ID (ms since epoch)
                ms = int(stream_id.split("-")[0])
                timestamp = datetime.fromtimestamp(ms / 1000)

            return AdminEvent(
                id=stream_id,
                topic=fields["topic"],
                workspace_id=workspace_id,
                timestamp=timestamp,
                payload=payload,
            )
        except Exception as e:
            logger.warning(
                "redis_event_parse_error",
                stream_id=stream_id,
                error=str(e),
            )
            return None

    async def unsubscribe(self, subscriber_id: str) -> None:
        """Remove subscriber and cancel reader task."""
        async with self._lock:
            # Cancel reader task
            task = self._subscribers.pop(subscriber_id, None)
            if task:
                task.cancel()

            # Remove queue and filters
            self._subscriber_queues.pop(subscriber_id, None)
            self._filters.pop(subscriber_id, None)

        logger.info(
            "redis_subscriber_removed",
            subscriber_id=subscriber_id,
            remaining_subscribers=len(self._subscribers),
        )

    def subscriber_count(self) -> int:
        """Return current number of local subscribers."""
        return len(self._subscriber_queues)

    def buffer_size(self) -> int:
        """Return configured buffer size (not actual stream length)."""
        return self._buffer_size

    async def get_stream_lengths(self) -> dict[str, int]:
        """Get current length of all workspace streams."""
        redis = await self._get_redis()
        result = {}

        # Get all stream keys matching pattern
        keys = await redis.keys("sse:events:*")
        for key in keys:
            try:
                length = await redis.xlen(key)
                result[key] = length
            except Exception:
                result[key] = -1

        return result

    async def get_last_publish_id(self, workspace_id: UUID) -> str | None:
        """Get the ID of the last published event for a workspace."""
        redis = await self._get_redis()
        stream_key = self._stream_key(workspace_id)

        try:
            # XREVRANGE to get latest entry
            entries = await redis.xrevrange(stream_key, count=1)
            if entries:
                return entries[0][0]  # Return the ID
        except Exception:
            pass
        return None

    @property
    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        return self._connected

    async def ping(self) -> bool:
        """Test Redis connection."""
        try:
            redis = await self._get_redis()
            await redis.ping()
            return True
        except Exception as e:
            logger.warning("redis_ping_failed", error=str(e))
            return False
