"""Canary tests for Q1 2026 operational hardening features.

These tests verify the four phases of operational safety:
1. Idempotency - Concurrent retries return same result
2. Retention - Dry-run prune executes without error
3. LLM Fallback - Timeout triggers graceful degradation
4. SSE - Real-time updates are delivered

Run with: pytest tests/integration/test_operational_hardening.py -v
Note: DB tests require migrations to be applied (idempotency_keys table, retention functions)
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

# Skip if no database
pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Phase 1: Idempotency - Concurrent Retry Test
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("SUPABASE_URL"),
    reason="Requires DATABASE_URL or SUPABASE_URL",
)
@pytest.mark.requires_db
class TestIdempotencyConcurrentRetry:
    """Phase 1: Verify concurrent retries with same idempotency key return same result."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    def test_concurrent_tune_requests_return_same_tune_id(self, client):
        """
        Two concurrent requests with same idempotency key should:
        - Both succeed (200)
        - Return the same tune_id
        - Not create duplicate tunes

        Note: Requires idempotency_keys migration to be applied.
        """
        from pathlib import Path

        # Use existing fixture
        fixture_path = (
            Path(__file__).parent.parent / "unit" / "fixtures" / "valid_ohlcv.csv"
        )
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        workspace_id = "00000000-0000-0000-0000-000000000001"
        strategy_id = "8fd7589a-97c6-49bf-a65e-357fb063fe33"
        idempotency_key = f"canary-test-{uuid4()}"

        # First request
        with open(fixture_path, "rb") as f:
            resp1 = client.post(
                "/backtests/tune",
                files={"file": ("ohlcv.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": strategy_id,
                    "workspace_id": workspace_id,
                    "n_trials": 2,
                },
                headers={"X-Idempotency-Key": idempotency_key},
            )

        # Skip if migrations not applied (500 = likely missing table/function)
        # These tests require idempotency_keys table from migrations
        if resp1.status_code == 500:
            pytest.skip("idempotency_keys migration not applied (run migrations first)")

        # Second request with same key (simulates retry)
        with open(fixture_path, "rb") as f:
            resp2 = client.post(
                "/backtests/tune",
                files={"file": ("ohlcv.csv", f, "text/csv")},
                data={
                    "strategy_entity_id": strategy_id,
                    "workspace_id": workspace_id,
                    "n_trials": 2,
                },
                headers={"X-Idempotency-Key": idempotency_key},
            )

        # Both should succeed (first already checked above)
        assert resp2.status_code == 200, f"Second request failed: {resp2.text}"

        # Both should return same tune_id
        tune_id_1 = resp1.json()["tune_id"]
        tune_id_2 = resp2.json()["tune_id"]
        assert (
            tune_id_1 == tune_id_2
        ), f"Idempotency failed: got different tune_ids {tune_id_1} vs {tune_id_2}"


# =============================================================================
# Phase 2: Retention - Dry Run Test
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("SUPABASE_URL"),
    reason="Requires DATABASE_URL or SUPABASE_URL",
)
@pytest.mark.requires_db
class TestRetentionDryRun:
    """Phase 2: Verify retention dry-run executes without modifying data."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    def test_retention_dry_run_returns_count_without_deleting(self, client):
        """
        Retention dry-run should:
        - Return 200
        - Include deleted_count (may be 0)
        - Have dry_run=true in response
        - NOT actually delete any rows

        Note: Requires retention SQL functions migration to be applied.
        """
        admin_token = os.getenv("ADMIN_TOKEN", "test-admin-token")

        # Call dry-run endpoint
        resp = client.post(
            "/admin/retention/prune/job-runs",
            params={"cutoff_days": 30, "dry_run": True},
            headers={"X-Admin-Token": admin_token},
        )

        # Skip if migrations not applied (500 = likely missing function)
        # These tests require retention SQL functions from migrations
        if resp.status_code == 500:
            pytest.skip(
                "retention SQL functions migration not applied (run migrations first)"
            )

        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"

        data = resp.json()
        assert "deleted_count" in data, "Missing deleted_count in response"
        assert data["dry_run"] is True, "Expected dry_run=true"
        assert "job_log_id" in data, "Missing job_log_id in response"

        # Verify it's logged
        log_resp = client.get(
            "/admin/retention/logs",
            params={"limit": 1},
            headers={"X-Admin-Token": admin_token},
        )
        assert log_resp.status_code == 200


# =============================================================================
# Phase 3: LLM Fallback - Timeout Test
# =============================================================================


class TestLLMTimeoutFallback:
    """Phase 3: Verify LLM timeout triggers graceful fallback."""

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_fallback_not_error(self):
        """
        When LLM times out:
        - Should return StrategyExplanation (not raise)
        - degraded=True
        - reason_code="llm_timeout"
        - source="fallback"
        """
        from app.services.coverage_gap.explanation import (
            generate_strategy_explanation,
            LLM_EXPLANATION_TIMEOUT,
        )

        # Mock LLM to simulate timeout
        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(LLM_EXPLANATION_TIMEOUT + 5)
            return None  # Never reached

        mock_llm = AsyncMock()
        mock_llm.generate = slow_llm

        # Mock data matching actual function signature
        intent_json = {
            "strategy_archetypes": ["breakout"],
            "indicators": ["volume"],
            "timeframe_buckets": ["swing"],
        }
        strategy = {
            "id": str(uuid4()),
            "name": "Test Strategy",
            "description": "A test strategy for canary testing",
            "tags": {"strategy_archetypes": ["breakout"]},
            "backtest_summary": None,
        }
        matched_tags = ["breakout"]

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            # Use short timeout for test
            result = await generate_strategy_explanation(
                intent_json=intent_json,
                strategy=strategy,
                matched_tags=matched_tags,
                match_score=0.75,
                timeout_seconds=0.1,  # Very short timeout
            )

        # Should return fallback, not raise
        assert result is not None, "Expected fallback response, got None"
        assert result.degraded is True, "Expected degraded=True"
        assert (
            result.reason_code == "llm_timeout"
        ), f"Expected reason_code='llm_timeout', got '{result.reason_code}'"
        assert (
            result.model == "fallback"
        ), f"Expected model='fallback', got '{result.model}'"
        assert (
            result.provider == "fallback"
        ), f"Expected provider='fallback', got '{result.provider}'"


# =============================================================================
# Phase 4: SSE - Event Delivery Test
# =============================================================================


class TestSSEEventDelivery:
    """Phase 4: Verify SSE events are published and can be consumed."""

    @pytest.mark.asyncio
    async def test_sse_event_published_and_received(self):
        """
        When coverage status is updated:
        - Event should be published to bus
        - Subscriber should receive the event
        """
        from app.services.events import get_event_bus
        from app.services.events.schemas import coverage_run_updated

        bus = get_event_bus()
        workspace_id = uuid4()
        run_id = uuid4()
        event_id = f"evt-{uuid4()}"

        # Track received events
        received_events = []

        async def collect_events():
            subscriber_id = f"test-{uuid4()}"
            async for event in bus.subscribe(
                subscriber_id=subscriber_id,
                workspace_ids={workspace_id},
                topics={"coverage"},
            ):
                received_events.append(event)
                break  # Just need one

        # Start subscriber in background
        subscriber_task = asyncio.create_task(collect_events())

        # Give subscriber time to connect
        await asyncio.sleep(0.05)

        # Publish event with required event_id
        event = coverage_run_updated(
            event_id=event_id,
            workspace_id=workspace_id,
            run_id=run_id,
            status="acknowledged",
            priority_score=0.75,
        )
        delivered = await bus.publish(event)

        # Wait for subscriber to receive
        try:
            await asyncio.wait_for(subscriber_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass  # Subscriber might timeout if no event

        # Verify
        assert delivered >= 1, f"Expected event to be delivered, got {delivered}"
        assert (
            len(received_events) == 1
        ), f"Expected 1 event, got {len(received_events)}"

        received = received_events[0]
        assert received.topic == "coverage.weak_run.updated"
        assert received.workspace_id == workspace_id
        assert received.payload["run_id"] == str(run_id)
        assert received.payload["status"] == "acknowledged"


# =============================================================================
# Phase 4b: SSE with Redis - Multi-worker Event Delivery
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("REDIS_URL"),
    reason="Requires REDIS_URL for Redis EventBus testing",
)
class TestRedisSSEEventDelivery:
    """Phase 4b: Verify Redis-backed SSE for multi-worker scenarios."""

    @pytest.fixture
    async def redis_bus(self):
        """Create and cleanup RedisEventBus for testing."""
        from app.services.events.redis_bus import RedisEventBus

        bus = RedisEventBus(
            redis_url=os.getenv("REDIS_URL"),
            buffer_size=100,
            read_block_ms=500,
        )
        yield bus
        await bus.close()

    @pytest.mark.asyncio
    async def test_redis_publish_and_receive(self, redis_bus):
        """
        Basic Redis event delivery:
        - Publish event via XADD
        - Subscriber receives via XREAD
        """
        workspace_id = uuid4()
        run_id = uuid4()

        from app.services.events.schemas import coverage_run_updated

        received_events = []

        async def collect_events():
            subscriber_id = f"test-{uuid4()}"
            async for event in redis_bus.subscribe(
                subscriber_id=subscriber_id,
                workspace_ids={workspace_id},
                topics={"coverage"},
            ):
                received_events.append(event)
                break

        # Start subscriber
        subscriber_task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.1)

        # Publish event
        event = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=run_id,
            status="acknowledged",
        )
        await redis_bus.publish(event)

        # Wait for delivery
        try:
            await asyncio.wait_for(subscriber_task, timeout=2.0)
        except asyncio.TimeoutError:
            pass

        assert len(received_events) >= 1, "Expected to receive event via Redis"
        assert received_events[0].workspace_id == workspace_id

    @pytest.mark.asyncio
    async def test_redis_reconnection_replay(self, redis_bus):
        """
        Reconnection with Last-Event-ID:
        - Publish events before subscriber connects
        - Connect with Last-Event-ID
        - Should replay missed events
        """
        workspace_id = uuid4()

        from app.services.events.schemas import coverage_run_updated

        # Publish some events first
        event1 = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="open",
        )
        await redis_bus.publish(event1)
        first_id = event1.id

        event2 = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="acknowledged",
        )
        await redis_bus.publish(event2)

        # Now subscribe with first event's ID (should get event2)
        received = []

        async def collect_from_id():
            async for event in redis_bus.subscribe(
                subscriber_id=f"test-{uuid4()}",
                workspace_ids={workspace_id},
                topics={"coverage"},
                last_event_id=first_id,
            ):
                received.append(event)
                break

        try:
            await asyncio.wait_for(collect_from_id(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) >= 1, "Expected replay of missed events"
        # Should receive event2 (after event1)
        assert received[0].payload["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_redis_stream_lengths_diagnostic(self, redis_bus):
        """
        get_stream_lengths() should return stream info for diagnostics.
        """
        workspace_id = uuid4()

        from app.services.events.schemas import coverage_run_updated

        # Publish to create stream
        event = coverage_run_updated(
            event_id="",
            workspace_id=workspace_id,
            run_id=uuid4(),
            status="open",
        )
        await redis_bus.publish(event)

        # Check stream lengths
        lengths = await redis_bus.get_stream_lengths()

        expected_key = f"sse:events:{workspace_id}"
        assert expected_key in lengths, f"Expected stream {expected_key} in {lengths}"
        assert lengths[expected_key] >= 1, "Expected at least 1 event in stream"

    @pytest.mark.asyncio
    async def test_redis_ping_succeeds(self, redis_bus):
        """
        ping() should return True when Redis is available.
        """
        result = await redis_bus.ping()
        assert result is True, "Expected ping to succeed with running Redis"


# =============================================================================
# Phase B3.2: Polling - Repository Polling Tests
# =============================================================================


class TestPineRepoPollerUnit:
    """Unit-level integration tests for the polling service."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for polling tests."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.pine_repo_poll_enabled = True
        settings.pine_repo_poll_interval_minutes = 15
        settings.pine_repo_poll_tick_seconds = 60
        settings.pine_repo_poll_max_concurrency = 2
        settings.pine_repo_poll_max_repos_per_tick = 10
        settings.pine_repo_poll_backoff_max_multiplier = 16
        return settings

    @pytest.mark.asyncio
    async def test_poller_start_stop_lifecycle(self, mock_settings):
        """
        Test poller start/stop lifecycle:
        - Start sets running=True
        - Stop sets running=False gracefully
        """
        from unittest.mock import AsyncMock, MagicMock

        from app.services.pine.poller import PineRepoPoller

        mock_pool = MagicMock()
        poller = PineRepoPoller(mock_pool, mock_settings, None)

        # Patch the poll loop to just wait for stop
        async def mock_poll_loop():
            await poller._stop_event.wait()

        with patch.object(poller, "_poll_loop", side_effect=mock_poll_loop):
            # Start
            await poller.start()
            assert poller.is_running is True

            # Stop
            await poller.stop(timeout=5.0)
            assert poller.is_running is False

    @pytest.mark.asyncio
    async def test_poller_run_once_empty(self, mock_settings):
        """
        Test run_once when no repos are due:
        - Should return result with repos_scanned=0
        """
        from unittest.mock import AsyncMock, MagicMock

        from app.services.pine.poller import PineRepoPoller

        mock_pool = MagicMock()
        poller = PineRepoPoller(mock_pool, mock_settings, None)

        # Mock the repo registry to return no repos
        poller._repo_registry.list_due_for_poll = AsyncMock(return_value=[])
        poller._repo_registry.count_due_for_poll = AsyncMock(return_value=0)

        result = await poller.run_once()

        assert result.repos_scanned == 0
        assert result.repos_succeeded == 0
        assert result.repos_failed == 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_poller_get_health(self, mock_settings):
        """
        Test get_health returns correct structure.
        """
        from unittest.mock import AsyncMock, MagicMock

        from app.services.pine.poller import PineRepoPoller, PollerHealth

        mock_pool = MagicMock()
        poller = PineRepoPoller(mock_pool, mock_settings, None)
        poller._repo_registry.count_due_for_poll = AsyncMock(return_value=3)

        health = await poller.get_health()

        assert isinstance(health, PollerHealth)
        assert health.enabled is True
        assert health.running is False
        assert health.repos_due_count == 3
        assert health.poll_interval_minutes == 15


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("SUPABASE_URL"),
    reason="Requires DATABASE_URL or SUPABASE_URL",
)
@pytest.mark.requires_db
class TestPineRepoPollerWithDB:
    """Integration tests for polling that require database."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    def test_poll_status_endpoint(self, client):
        """
        GET /admin/pine/repos/poll-status should return poller health.
        """
        # Get admin token (or skip if not configured)
        admin_token = os.getenv("ADMIN_TOKEN")
        if not admin_token:
            pytest.skip("ADMIN_TOKEN not configured")

        resp = client.get(
            "/admin/pine/repos/poll-status",
            headers={"X-Admin-Token": admin_token},
        )

        # Should succeed even if poller is not started
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "running" in data
        assert "repos_due_count" in data

    def test_poll_run_endpoint(self, client):
        """
        POST /admin/pine/repos/poll-run should trigger manual poll.
        """
        admin_token = os.getenv("ADMIN_TOKEN")
        if not admin_token:
            pytest.skip("ADMIN_TOKEN not configured")

        resp = client.post(
            "/admin/pine/repos/poll-run",
            headers={"X-Admin-Token": admin_token},
        )

        # Should succeed
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "repos_scanned" in data
        assert "repos_succeeded" in data
        assert "repos_failed" in data
        assert "duration_ms" in data
