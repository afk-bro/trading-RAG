"""Tests for backtest run events (replay data)."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.run_events import RunEventsRepository


SAMPLE_WS_ID = uuid4()
SAMPLE_RUN_ID = uuid4()


SAMPLE_EVENTS = [
    {
        "type": "orb_range_update",
        "bar_index": 5,
        "ts": "2024-01-02T09:35:00Z",
        "high": 100.5,
        "low": 99.2,
    },
    {
        "type": "orb_range_locked",
        "bar_index": 30,
        "ts": "2024-01-02T10:00:00Z",
        "high": 101.0,
        "low": 99.0,
        "range": 2.0,
    },
    {
        "type": "setup_valid",
        "bar_index": 35,
        "ts": "2024-01-02T10:05:00Z",
        "direction": "long",
        "level": 101.0,
    },
    {
        "type": "entry_signal",
        "bar_index": 36,
        "ts": "2024-01-02T10:06:00Z",
        "side": "buy",
        "price": 101.05,
        "stop": 99.0,
        "target": 104.0,
    },
]


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


class TestRunEventsRepository:
    """Repository tests."""

    @pytest.mark.asyncio
    async def test_get_events_returns_list(self, mock_pool):
        pool, conn = mock_pool
        conn.fetchrow.return_value = {"events": SAMPLE_EVENTS}

        repo = RunEventsRepository(pool)
        result = await repo.get_events(SAMPLE_RUN_ID, SAMPLE_WS_ID)

        assert result is not None
        assert len(result) == 4
        assert result[0]["type"] == "orb_range_update"
        assert result[1]["type"] == "orb_range_locked"
        assert result[2]["type"] == "setup_valid"
        assert result[3]["type"] == "entry_signal"

    @pytest.mark.asyncio
    async def test_get_events_parses_json_string(self, mock_pool):
        pool, conn = mock_pool
        conn.fetchrow.return_value = {"events": json.dumps(SAMPLE_EVENTS)}

        repo = RunEventsRepository(pool)
        result = await repo.get_events(SAMPLE_RUN_ID, SAMPLE_WS_ID)

        assert result is not None
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_get_events_returns_none_when_missing(self, mock_pool):
        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        repo = RunEventsRepository(pool)
        result = await repo.get_events(SAMPLE_RUN_ID, SAMPLE_WS_ID)

        assert result is None

    @pytest.mark.asyncio
    async def test_save_events_upserts(self, mock_pool):
        pool, conn = mock_pool
        conn.execute.return_value = None

        repo = RunEventsRepository(pool)
        await repo.save_events(SAMPLE_RUN_ID, SAMPLE_WS_ID, SAMPLE_EVENTS)

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        assert "ON CONFLICT" in call_args[0]
        assert call_args[1] == SAMPLE_RUN_ID
        assert call_args[2] == SAMPLE_WS_ID

    @pytest.mark.asyncio
    async def test_save_events_sets_updated_at(self, mock_pool):
        pool, conn = mock_pool
        conn.execute.return_value = None

        repo = RunEventsRepository(pool)
        await repo.save_events(SAMPLE_RUN_ID, SAMPLE_WS_ID, SAMPLE_EVENTS)

        sql = conn.execute.call_args[0][0]
        assert "updated_at" in sql

    @pytest.mark.asyncio
    async def test_save_events_guards_workspace_id(self, mock_pool):
        """UPSERT includes workspace_id guard to prevent cross-ws overwrites."""
        pool, conn = mock_pool
        conn.execute.return_value = None

        repo = RunEventsRepository(pool)
        await repo.save_events(SAMPLE_RUN_ID, SAMPLE_WS_ID, SAMPLE_EVENTS)

        sql = conn.execute.call_args[0][0]
        assert "workspace_id = EXCLUDED.workspace_id" in sql

    @pytest.mark.asyncio
    async def test_get_event_count(self, mock_pool):
        pool, conn = mock_pool
        conn.fetchval.return_value = 4

        repo = RunEventsRepository(pool)
        count = await repo.get_event_count(SAMPLE_RUN_ID, SAMPLE_WS_ID)

        assert count == 4

    @pytest.mark.asyncio
    async def test_get_event_count_missing_returns_zero(self, mock_pool):
        pool, conn = mock_pool
        conn.fetchval.return_value = None

        repo = RunEventsRepository(pool)
        count = await repo.get_event_count(SAMPLE_RUN_ID, SAMPLE_WS_ID)

        assert count == 0


class TestEventStructure:
    """Event structure invariants."""

    def test_all_events_have_type(self):
        for event in SAMPLE_EVENTS:
            assert "type" in event

    def test_all_events_have_bar_index(self):
        for event in SAMPLE_EVENTS:
            assert "bar_index" in event

    def test_all_events_have_timestamp(self):
        for event in SAMPLE_EVENTS:
            assert "ts" in event

    def test_known_event_types(self):
        known = {"orb_range_update", "orb_range_locked", "setup_valid", "entry_signal"}
        for event in SAMPLE_EVENTS:
            assert event["type"] in known

    def test_bar_indices_are_monotonic(self):
        indices = [e["bar_index"] for e in SAMPLE_EVENTS]
        assert indices == sorted(indices)
