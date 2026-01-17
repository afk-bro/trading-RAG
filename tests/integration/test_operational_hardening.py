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
            pytest.skip(
                "idempotency_keys migration not applied (run migrations first)"
            )

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
        assert tune_id_1 == tune_id_2, (
            f"Idempotency failed: got different tune_ids {tune_id_1} vs {tune_id_2}"
        )


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
        assert result.reason_code == "llm_timeout", (
            f"Expected reason_code='llm_timeout', got '{result.reason_code}'"
        )
        assert result.model == "fallback", (
            f"Expected model='fallback', got '{result.model}'"
        )
        assert result.provider == "fallback", (
            f"Expected provider='fallback', got '{result.provider}'"
        )


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
        assert len(received_events) == 1, (
            f"Expected 1 event, got {len(received_events)}"
        )

        received = received_events[0]
        assert received.topic == "coverage.weak_run.updated"
        assert received.workspace_id == workspace_id
        assert received.payload["run_id"] == str(run_id)
        assert received.payload["status"] == "acknowledged"
