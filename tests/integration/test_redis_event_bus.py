"""Integration tests for RedisEventBus with real Redis.

These tests require a running Redis instance. They are skipped if Redis
is not available.

Run with: pytest tests/integration/test_redis_event_bus.py -v
Requires: docker run -d -p 6379:6379 redis:7-alpine
"""

import asyncio
import os
from uuid import uuid4

import pytest

from app.services.events.schemas import AdminEvent, coverage_run_updated


# Check for Redis availability
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _redis_available_sync() -> bool:
    """Synchronous check if Redis is available."""
    try:
        import redis

        client = redis.from_url(REDIS_URL, socket_connect_timeout=2.0)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


# Skip all tests in this module if Redis is not available
pytestmark = [
    pytest.mark.requires_redis,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _redis_available_sync(),
        reason="Redis not available - skipping integration tests",
    ),
]


@pytest.fixture
async def redis_bus():
    """Create RedisEventBus connected to real Redis."""
    from app.services.events.redis_bus import RedisEventBus

    bus = RedisEventBus(
        redis_url=REDIS_URL,
        buffer_size=100,
        read_block_ms=500,
    )
    yield bus
    await bus.close()


@pytest.fixture
async def clean_streams():
    """Clean up test streams after each test."""
    import redis.asyncio as redis_async

    created_keys = []
    yield created_keys

    # Cleanup
    client = redis_async.from_url(REDIS_URL)
    for key in created_keys:
        await client.delete(key)
    await client.aclose()


class TestRedisEventBusIntegration:
    """Integration tests with real Redis."""

    @pytest.mark.asyncio
    async def test_publish_creates_stream_entry(self, redis_bus, clean_streams):
        """Publishing an event should create a Redis stream entry."""
        import redis.asyncio as redis_async

        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        event = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="acknowledged",
        )

        # Publish
        await redis_bus.publish(event)

        # Verify stream exists and has entry (use decode_responses=True)
        client = redis_async.from_url(REDIS_URL, decode_responses=True)
        length = await client.xlen(stream_key)
        assert length == 1

        # Read entry
        entries = await client.xrange(stream_key)
        assert len(entries) == 1

        msg_id, fields = entries[0]
        assert fields["topic"] == "coverage.weak_run.updated"
        assert fields["workspace_id"] == str(workspace_id)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_publish_assigns_stream_id_to_event(self, redis_bus, clean_streams):
        """Published event should get Redis stream ID assigned."""
        workspace_id = uuid4()
        clean_streams.append(f"sse:events:{workspace_id}")

        event = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="resolved",
        )

        await redis_bus.publish(event)

        # Event ID should be set to Redis stream ID format: <timestamp>-<seq>
        assert event.id is not None
        assert "-" in event.id
        parts = event.id.split("-")
        assert len(parts) == 2
        assert parts[0].isdigit()  # Timestamp in ms

    @pytest.mark.asyncio
    async def test_multiple_publishes_increment_stream(self, redis_bus, clean_streams):
        """Multiple publishes should add multiple entries to stream."""
        import redis.asyncio as redis_async

        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        # Publish 5 events
        for i in range(5):
            event = coverage_run_updated(
                event_id="",
                workspace_id=workspace_id,
                run_id=uuid4(),
                status="acknowledged",
            )
            await redis_bus.publish(event)

        # Verify count
        client = redis_async.from_url(REDIS_URL)
        length = await client.xlen(stream_key)
        assert length == 5
        await client.aclose()

    @pytest.mark.asyncio
    async def test_stream_trimming_respects_buffer_size(self, redis_bus, clean_streams):
        """Stream should be trimmed to buffer_size (approximately)."""
        import redis.asyncio as redis_async

        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        # Publish more than buffer_size (100)
        for i in range(150):
            event = coverage_run_updated(
                event_id="",
                workspace_id=workspace_id,
                run_id=uuid4(),
                status="acknowledged",
            )
            await redis_bus.publish(event)

        # Stream should be trimmed (MAXLEN ~ is approximate)
        client = redis_async.from_url(REDIS_URL)
        length = await client.xlen(stream_key)
        # MAXLEN ~ allows some slack, but should be close to 100
        assert length <= 120  # Allow 20% overhead for approximate trimming
        await client.aclose()

    @pytest.mark.asyncio
    async def test_ping_returns_true(self, redis_bus):
        """ping() should return True when connected."""
        result = await redis_bus.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_stream_lengths_returns_counts(self, redis_bus, clean_streams):
        """get_stream_lengths should return correct counts."""
        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        # Publish some events
        for _ in range(3):
            event = coverage_run_updated(
                event_id="",
                workspace_id=workspace_id,
                run_id=uuid4(),
                status="acknowledged",
            )
            await redis_bus.publish(event)

        lengths = await redis_bus.get_stream_lengths()
        assert stream_key in lengths
        assert lengths[stream_key] == 3

    @pytest.mark.asyncio
    async def test_get_last_publish_id(self, redis_bus, clean_streams):
        """get_last_publish_id should return the most recent event ID."""
        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        # Initially no events
        last_id = await redis_bus.get_last_publish_id(workspace_id)
        assert last_id is None

        # Publish an event
        event = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="acknowledged",
        )
        await redis_bus.publish(event)

        # Now should have last ID
        last_id = await redis_bus.get_last_publish_id(workspace_id)
        assert last_id is not None
        assert last_id == event.id


class TestRedisEventBusSubscription:
    """Integration tests for subscription functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_receives_published_events(self, redis_bus, clean_streams):
        """Subscriber should receive events published after subscription.

        Note: Due to dual-delivery (local queue + Redis stream), subscribers
        may receive duplicates. This test verifies both statuses are received.
        """
        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        received_events = []
        seen_statuses = set()

        async def subscriber():
            async for event in redis_bus.subscribe(
                subscriber_id="test-sub",
                workspace_ids={workspace_id},
                topics={"coverage"},
            ):
                received_events.append(event)
                seen_statuses.add(event.payload["status"])
                # Exit once we've seen both statuses
                if seen_statuses == {"acknowledged", "resolved"}:
                    break

        # Start subscriber in background
        sub_task = asyncio.create_task(subscriber())

        # Wait for subscriber to be ready (stream reader started)
        await asyncio.sleep(0.5)

        # Publish events with delay to ensure ordered delivery
        for status in ["acknowledged", "resolved"]:
            event = coverage_run_updated(
                event_id="",
                workspace_id=workspace_id,
                run_id=uuid4(),
                status=status,
            )
            await redis_bus.publish(event)
            await asyncio.sleep(0.1)  # Delay between publishes

        # Wait for subscriber to receive both (with timeout)
        try:
            await asyncio.wait_for(sub_task, timeout=5.0)
        except asyncio.TimeoutError:
            sub_task.cancel()
            pytest.fail(f"Only saw statuses: {seen_statuses}, events: {len(received_events)}")

        # Verify both statuses were received
        assert "acknowledged" in seen_statuses
        assert "resolved" in seen_statuses

    @pytest.mark.asyncio
    async def test_subscribe_filters_by_workspace(self, redis_bus, clean_streams):
        """Subscriber should only receive events for subscribed workspaces."""
        ws1 = uuid4()
        ws2 = uuid4()
        clean_streams.append(f"sse:events:{ws1}")
        clean_streams.append(f"sse:events:{ws2}")

        received_events = []

        async def subscriber():
            async for event in redis_bus.subscribe(
                subscriber_id="test-sub-ws",
                workspace_ids={ws1},  # Only subscribe to ws1
                topics={"coverage"},
            ):
                received_events.append(event)
                break

        sub_task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.2)

        # Publish to ws2 (should not be received)
        event_ws2 = coverage_run_updated(
            event_id="",
            workspace_id=ws2,
            run_id=uuid4(),
            status="acknowledged",
        )
        await redis_bus.publish(event_ws2)

        # Publish to ws1 (should be received)
        event_ws1 = coverage_run_updated(
            event_id="",
            workspace_id=ws1,
            run_id=uuid4(),
            status="resolved",
        )
        await redis_bus.publish(event_ws1)

        try:
            await asyncio.wait_for(sub_task, timeout=3.0)
        except asyncio.TimeoutError:
            sub_task.cancel()
            pytest.fail("Subscriber did not receive event")

        assert len(received_events) == 1
        assert received_events[0].workspace_id == ws1

    @pytest.mark.asyncio
    async def test_reconnection_replay(self, redis_bus, clean_streams):
        """Reconnecting with Last-Event-ID should replay missed events."""
        import redis.asyncio as redis_async

        workspace_id = uuid4()
        stream_key = f"sse:events:{workspace_id}"
        clean_streams.append(stream_key)

        # Publish some events directly to Redis (use decode_responses=True)
        client = redis_async.from_url(REDIS_URL, decode_responses=True)

        first_id = await client.xadd(
            stream_key,
            {
                "topic": "coverage.weak_run.updated",
                "workspace_id": str(workspace_id),
                "payload": '{"status": "first"}',
            },
        )

        second_id = await client.xadd(
            stream_key,
            {
                "topic": "coverage.weak_run.updated",
                "workspace_id": str(workspace_id),
                "payload": '{"status": "second"}',
            },
        )

        third_id = await client.xadd(
            stream_key,
            {
                "topic": "coverage.weak_run.updated",
                "workspace_id": str(workspace_id),
                "payload": '{"status": "third"}',
            },
        )
        await client.aclose()

        # Subscribe with last_event_id = first_id (should get second and third)
        received = []

        async def subscriber():
            async for event in redis_bus.subscribe(
                subscriber_id="test-replay",
                workspace_ids={workspace_id},
                topics={"coverage"},
                last_event_id=first_id,  # Now a string, not bytes
            ):
                received.append(event)
                if len(received) >= 2:
                    break

        try:
            await asyncio.wait_for(subscriber(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

        # Should have received second and third (not first)
        assert len(received) >= 2, f"Expected >=2, got {len(received)}: {received}"
        statuses = [e.payload.get("status") for e in received[:2]]
        assert "second" in statuses
        assert "third" in statuses
        assert "first" not in statuses
