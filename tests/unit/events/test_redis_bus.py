"""Unit tests for RedisEventBus.

These tests mock the Redis client to verify behavior without requiring
a running Redis instance.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.events.schemas import coverage_run_updated


class TestRedisEventBusUnit:
    """Unit tests for RedisEventBus with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.ping = AsyncMock(return_value=True)
        redis.xadd = AsyncMock(return_value="1234567890-0")
        redis.xread = AsyncMock(return_value=[])
        redis.xrange = AsyncMock(return_value=[])
        redis.xlen = AsyncMock(return_value=0)
        redis.xrevrange = AsyncMock(return_value=[])
        redis.keys = AsyncMock(return_value=[])
        redis.aclose = AsyncMock()
        return redis

    @pytest.fixture
    def redis_bus(self, mock_redis):
        """Create RedisEventBus with mocked Redis."""
        from app.services.events.redis_bus import RedisEventBus

        bus = RedisEventBus(
            redis_url="redis://localhost:6379/0",
            buffer_size=100,
            read_block_ms=100,
        )
        bus._redis = mock_redis
        bus._connected = True
        return bus

    @pytest.mark.asyncio
    async def test_publish_calls_xadd_with_correct_stream_key(
        self, redis_bus, mock_redis
    ):
        """Publish should XADD to workspace-scoped stream."""
        workspace_id = uuid4()
        event = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="acknowledged",
        )

        await redis_bus.publish(event)

        # Verify XADD was called with correct stream key
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        stream_key = call_args[0][0]
        assert stream_key == f"sse:events:{workspace_id}"

    @pytest.mark.asyncio
    async def test_publish_uses_maxlen_approximate(self, redis_bus, mock_redis):
        """Publish should use XTRIM MAXLEN ~ for efficient trimming."""
        event = coverage_run_updated(
            event_id="",
            workspace_id=uuid4(),
            run_id=uuid4(),
            status="acknowledged",
        )

        await redis_bus.publish(event)

        call_kwargs = mock_redis.xadd.call_args[1]
        assert call_kwargs["maxlen"] == 100  # buffer_size
        assert call_kwargs["approximate"] is True

    @pytest.mark.asyncio
    async def test_publish_serializes_event_payload(self, redis_bus, mock_redis):
        """Publish should serialize event fields correctly."""
        workspace_id = uuid4()
        run_id = uuid4()
        event = coverage_run_updated(
            event_id="test-id",
            workspace_id=workspace_id,
            run_id=run_id,
            status="resolved",
            priority_score=0.85,
        )

        await redis_bus.publish(event)

        call_args = mock_redis.xadd.call_args[0]
        payload = call_args[1]
        assert payload["topic"] == "coverage.weak_run.updated"
        assert payload["workspace_id"] == str(workspace_id)
        assert "payload" in payload  # JSON-encoded

    @pytest.mark.asyncio
    async def test_publish_updates_event_id_from_stream(self, redis_bus, mock_redis):
        """Publish should update event.id to Redis stream ID."""
        mock_redis.xadd.return_value = "1705123456789-42"

        event = coverage_run_updated(
            event_id="",
            workspace_id=uuid4(),
            run_id=uuid4(),
            status="acknowledged",
        )

        await redis_bus.publish(event)

        assert event.id == "1705123456789-42"

    @pytest.mark.asyncio
    async def test_subscriber_count_tracks_local_subscribers(self, redis_bus):
        """subscriber_count should return count of local subscribers."""
        assert redis_bus.subscriber_count() == 0

        # Add some fake subscribers
        redis_bus._subscriber_queues["sub-1"] = asyncio.Queue()
        redis_bus._subscriber_queues["sub-2"] = asyncio.Queue()

        assert redis_bus.subscriber_count() == 2

    @pytest.mark.asyncio
    async def test_buffer_size_returns_configured_size(self, redis_bus):
        """buffer_size should return the configured max buffer size."""
        assert redis_bus.buffer_size() == 100

    @pytest.mark.asyncio
    async def test_get_stream_lengths_queries_all_streams(self, redis_bus, mock_redis):
        """get_stream_lengths should return lengths of all workspace streams."""
        mock_redis.keys.return_value = [
            "sse:events:uuid1",
            "sse:events:uuid2",
        ]
        mock_redis.xlen.side_effect = [50, 75]

        lengths = await redis_bus.get_stream_lengths()

        assert lengths == {
            "sse:events:uuid1": 50,
            "sse:events:uuid2": 75,
        }

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_connected(self, redis_bus, mock_redis):
        """ping should return True when Redis responds."""
        result = await redis_bus.ping()
        assert result is True
        mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_error(self, redis_bus, mock_redis):
        """ping should return False when Redis fails."""
        mock_redis.ping.side_effect = Exception("Connection refused")

        result = await redis_bus.ping()

        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscriber(self, redis_bus):
        """unsubscribe should clean up subscriber resources."""
        sub_id = "test-sub"
        redis_bus._subscriber_queues[sub_id] = asyncio.Queue()
        redis_bus._filters[sub_id] = ({uuid4()}, {"coverage"})

        await redis_bus.unsubscribe(sub_id)

        assert sub_id not in redis_bus._subscriber_queues
        assert sub_id not in redis_bus._filters

    @pytest.mark.asyncio
    async def test_close_clears_all_state(self, redis_bus, mock_redis):
        """close should clear all subscribers and close Redis connection."""
        # Setup some state
        redis_bus._subscriber_queues["sub-1"] = asyncio.Queue()
        redis_bus._filters["sub-1"] = ({uuid4()}, {"coverage"})

        await redis_bus.close()

        assert len(redis_bus._subscriber_queues) == 0
        assert len(redis_bus._filters) == 0
        assert redis_bus._connected is False
        mock_redis.aclose.assert_called_once()


class TestRedisEventBusEventParsing:
    """Tests for parsing Redis stream messages back into AdminEvent."""

    def test_parse_stream_event_creates_admin_event(self):
        """_parse_stream_event should create AdminEvent from stream message."""
        from app.services.events.redis_bus import RedisEventBus

        bus = RedisEventBus(redis_url="redis://localhost:6379/0")

        workspace_id = uuid4()
        stream_id = "1705123456789-0"
        fields = {
            "topic": "coverage.weak_run.updated",
            "workspace_id": str(workspace_id),
            "timestamp": "2025-01-13T12:00:00",
            "payload": '{"run_id": "abc", "status": "resolved"}',
        }

        event = bus._parse_stream_event(stream_id, fields)

        assert event is not None
        assert event.id == stream_id
        assert event.topic == "coverage.weak_run.updated"
        assert event.workspace_id == workspace_id
        assert event.payload["run_id"] == "abc"

    def test_parse_stream_event_handles_missing_timestamp(self):
        """_parse_stream_event should derive timestamp from stream ID if missing."""
        from app.services.events.redis_bus import RedisEventBus

        bus = RedisEventBus(redis_url="redis://localhost:6379/0")

        workspace_id = uuid4()
        # Stream ID: 1705123456789 ms = 2024-01-13 08:17:36.789
        stream_id = "1705123456789-0"
        fields = {
            "topic": "coverage.weak_run.created",
            "workspace_id": str(workspace_id),
            "payload": "{}",
        }

        event = bus._parse_stream_event(stream_id, fields)

        assert event is not None
        assert event.timestamp is not None

    def test_parse_stream_event_returns_none_on_invalid_data(self):
        """_parse_stream_event should return None for invalid messages."""
        from app.services.events.redis_bus import RedisEventBus

        bus = RedisEventBus(redis_url="redis://localhost:6379/0")

        # Invalid workspace_id
        fields = {
            "topic": "coverage.weak_run.updated",
            "workspace_id": "not-a-uuid",
            "payload": "{}",
        }

        event = bus._parse_stream_event("123-0", fields)
        assert event is None


class TestRedisEventBusFactoryIntegration:
    """Tests for get_event_bus() factory function with Redis config."""

    @pytest.fixture(autouse=True)
    def reset_bus(self):
        """Reset event bus singleton before each test."""
        from app.services.events.bus import reset_event_bus

        reset_event_bus()
        yield
        reset_event_bus()

    def test_get_event_bus_raises_if_redis_mode_without_url(self):
        """get_event_bus should raise ValueError if redis mode but no URL."""
        from app.services.events.bus import get_event_bus

        with patch("app.config.get_settings") as mock_settings:
            settings = MagicMock()
            settings.event_bus_mode = "redis"
            settings.redis_url = None
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="REDIS_URL"):
                get_event_bus()

    def test_get_event_bus_returns_memory_by_default(self):
        """get_event_bus should return InMemoryEventBus when mode=memory."""
        from app.services.events.bus import get_event_bus, InMemoryEventBus

        with patch("app.config.get_settings") as mock_settings:
            settings = MagicMock()
            settings.event_bus_mode = "memory"
            settings.event_bus_buffer_size = 1000
            mock_settings.return_value = settings

            bus = get_event_bus()
            assert isinstance(bus, InMemoryEventBus)

    def test_get_event_bus_returns_redis_when_configured(self):
        """get_event_bus should return RedisEventBus when mode=redis."""
        from app.services.events.bus import get_event_bus
        from app.services.events.redis_bus import RedisEventBus

        with patch("app.config.get_settings") as mock_settings:
            settings = MagicMock()
            settings.event_bus_mode = "redis"
            settings.redis_url = "redis://localhost:6379/0"
            settings.event_bus_buffer_size = 2000
            mock_settings.return_value = settings

            bus = get_event_bus()
            assert isinstance(bus, RedisEventBus)


class TestRedisEventBusTopicExpansion:
    """Tests for topic expansion in RedisEventBus subscriptions."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.ping = AsyncMock(return_value=True)
        redis.xread = AsyncMock(return_value=[])
        return redis

    @pytest.mark.asyncio
    async def test_subscribe_expands_coverage_topic(self, mock_redis):
        """subscribe should expand 'coverage' to specific topic names."""
        from app.services.events.redis_bus import RedisEventBus

        bus = RedisEventBus(redis_url="redis://localhost:6379/0")
        bus._redis = mock_redis
        bus._connected = True

        workspace_id = uuid4()
        subscriber_id = "test-sub"
        captured_filters: set = set()

        # Make xread capture filters then block
        async def capture_and_block(*args, **kwargs):
            # Capture filters while subscription is active
            _, topics = bus._filters.get(subscriber_id, (set(), set()))
            captured_filters.update(topics)
            await asyncio.sleep(0.05)
            raise asyncio.CancelledError()

        mock_redis.xread.side_effect = capture_and_block

        # Start subscription - will be cancelled by xread
        async def subscribe_briefly():
            async for _ in bus.subscribe(
                subscriber_id=subscriber_id,
                workspace_ids={workspace_id},
                topics={"coverage"},
            ):
                break

        # Run with timeout
        try:
            await asyncio.wait_for(subscribe_briefly(), timeout=0.2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # Check captured filters (taken while subscription was active)
        assert "coverage.weak_run.created" in captured_filters
        assert "coverage.weak_run.updated" in captured_filters
