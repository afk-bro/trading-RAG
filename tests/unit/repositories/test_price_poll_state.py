"""Unit tests for price poll state repository (LP3)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.price_poll_state import (
    PollKey,
    PricePollStateRepository,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Setup acquire as async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm

    return pool, conn


@pytest.fixture
def repo(mock_pool):
    """Create repository with mock pool."""
    pool, _ = mock_pool
    return PricePollStateRepository(
        pool,
        interval_seconds=60,
        jitter_seconds=5,
        backoff_max_seconds=900,
    )


@pytest.fixture
def sample_key():
    """Sample poll key."""
    return PollKey(exchange_id="binance", symbol="BTC/USDT", timeframe="1m")


@pytest.fixture
def sample_state_row():
    """Sample database row for poll state."""
    return {
        "exchange_id": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "next_poll_at": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "failure_count": 0,
        "last_success_at": datetime(2024, 1, 15, 11, 59, 0, tzinfo=timezone.utc),
        "last_candle_ts": datetime(2024, 1, 15, 11, 58, 0, tzinfo=timezone.utc),
        "last_error": None,
        "created_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 1, 15, 11, 59, 0, tzinfo=timezone.utc),
    }


# =============================================================================
# Get State Tests
# =============================================================================


class TestGetState:
    """Tests for get_state method."""

    @pytest.mark.asyncio
    async def test_get_state_returns_state(
        self, mock_pool, sample_key, sample_state_row
    ):
        """Test get_state returns PollState when found."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=sample_state_row)

        repo = PricePollStateRepository(pool)
        state = await repo.get_state(sample_key)

        assert state is not None
        assert state.exchange_id == "binance"
        assert state.symbol == "BTC/USDT"
        assert state.timeframe == "1m"
        assert state.failure_count == 0

    @pytest.mark.asyncio
    async def test_get_state_returns_none_when_not_found(self, mock_pool, sample_key):
        """Test get_state returns None when not found."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        repo = PricePollStateRepository(pool)
        state = await repo.get_state(sample_key)

        assert state is None


# =============================================================================
# List Due Pairs Tests
# =============================================================================


class TestListDuePairs:
    """Tests for list_due_pairs method."""

    @pytest.mark.asyncio
    async def test_list_due_pairs_returns_states(self, mock_pool, sample_state_row):
        """Test list_due_pairs returns list of PollState."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[sample_state_row])

        repo = PricePollStateRepository(pool)
        states = await repo.list_due_pairs(limit=10)

        assert len(states) == 1
        assert states[0].exchange_id == "binance"
        assert states[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_list_due_pairs_empty(self, mock_pool):
        """Test list_due_pairs returns empty list when no due pairs."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = PricePollStateRepository(pool)
        states = await repo.list_due_pairs(limit=10)

        assert len(states) == 0


# =============================================================================
# Upsert State If Missing Tests
# =============================================================================


class TestUpsertStateIfMissing:
    """Tests for upsert_state_if_missing method."""

    @pytest.mark.asyncio
    async def test_upsert_returns_true_when_inserted(self, mock_pool, sample_key):
        """Test upsert returns True when row was inserted."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"exchange_id": "binance"})

        repo = PricePollStateRepository(pool)
        result = await repo.upsert_state_if_missing(sample_key)

        assert result is True

    @pytest.mark.asyncio
    async def test_upsert_returns_false_when_exists(self, mock_pool, sample_key):
        """Test upsert returns False when row already existed."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)  # ON CONFLICT DO NOTHING

        repo = PricePollStateRepository(pool)
        result = await repo.upsert_state_if_missing(sample_key)

        assert result is False


# =============================================================================
# Mark Success Tests
# =============================================================================


class TestMarkSuccess:
    """Tests for mark_success method."""

    @pytest.mark.asyncio
    async def test_mark_success_schedules_next_poll(self, mock_pool, sample_key):
        """Test mark_success schedules next poll at interval + jitter."""
        pool, conn = mock_pool
        conn.execute = AsyncMock()

        repo = PricePollStateRepository(
            pool, interval_seconds=60, jitter_seconds=5, backoff_max_seconds=900
        )
        last_candle_ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        await repo.mark_success(sample_key, last_candle_ts)

        # Verify execute was called
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        # Args: query, exchange_id, symbol, timeframe, next_poll, last_candle_ts

        # Check that next_poll_at is approximately interval + jitter from now
        next_poll_at = call_args[4]  # 5th positional arg (index 4)
        now = datetime.now(timezone.utc)
        expected_min = now + timedelta(seconds=60)
        expected_max = now + timedelta(seconds=65)

        assert expected_min <= next_poll_at <= expected_max


# =============================================================================
# Mark Failure Tests
# =============================================================================


class TestMarkFailure:
    """Tests for mark_failure method with exponential backoff."""

    @pytest.mark.asyncio
    async def test_mark_failure_first_failure(self, mock_pool, sample_key):
        """Test first failure uses 1x interval."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)  # No existing state
        conn.execute = AsyncMock()

        repo = PricePollStateRepository(
            pool, interval_seconds=60, jitter_seconds=0, backoff_max_seconds=900
        )

        await repo.mark_failure(sample_key, "Connection timeout")

        # Verify execute was called
        # Args: query, exchange_id, symbol, timeframe, next_poll, failure_count, error
        call_args = conn.execute.call_args[0]
        failure_count = call_args[5]  # 6th positional arg (index 5)
        assert failure_count == 1

        # next_poll_at should be ~60s from now (1x interval)
        next_poll_at = call_args[4]  # 5th positional arg (index 4)
        now = datetime.now(timezone.utc)
        expected = now + timedelta(seconds=60)
        assert abs((next_poll_at - expected).total_seconds()) < 2

    @pytest.mark.asyncio
    async def test_mark_failure_exponential_backoff(self, mock_pool, sample_key):
        """Test subsequent failures use exponential backoff."""
        pool, conn = mock_pool
        # Existing state with failure_count=2
        conn.fetchrow = AsyncMock(
            return_value={
                "exchange_id": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "next_poll_at": None,
                "failure_count": 2,
                "last_success_at": None,
                "last_candle_ts": None,
                "last_error": "Previous error",
            }
        )
        conn.execute = AsyncMock()

        repo = PricePollStateRepository(
            pool, interval_seconds=60, jitter_seconds=0, backoff_max_seconds=900
        )

        await repo.mark_failure(sample_key, "Another error")

        # Verify failure_count incremented to 3
        # Args: query, exchange_id, symbol, timeframe, next_poll, failure_count, error
        call_args = conn.execute.call_args[0]
        failure_count = call_args[5]  # 6th positional arg (index 5)
        assert failure_count == 3

        # Backoff: 60 * 2^(3-1) = 60 * 4 = 240 seconds
        next_poll_at = call_args[4]  # 5th positional arg (index 4)
        now = datetime.now(timezone.utc)
        expected = now + timedelta(seconds=240)
        assert abs((next_poll_at - expected).total_seconds()) < 2

    @pytest.mark.asyncio
    async def test_mark_failure_backoff_capped(self, mock_pool, sample_key):
        """Test backoff is capped at backoff_max_seconds."""
        pool, conn = mock_pool
        # Existing state with high failure_count
        conn.fetchrow = AsyncMock(
            return_value={
                "exchange_id": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "next_poll_at": None,
                "failure_count": 10,  # Would give 60 * 2^9 = 30720s without cap
                "last_success_at": None,
                "last_candle_ts": None,
                "last_error": "Previous error",
            }
        )
        conn.execute = AsyncMock()

        repo = PricePollStateRepository(
            pool, interval_seconds=60, jitter_seconds=0, backoff_max_seconds=900
        )

        await repo.mark_failure(sample_key, "Yet another error")

        # Backoff should be capped at 900 seconds
        # Args: query, exchange_id, symbol, timeframe, next_poll, failure_count, error
        call_args = conn.execute.call_args[0]
        next_poll_at = call_args[4]  # 5th positional arg (index 4)
        now = datetime.now(timezone.utc)
        expected = now + timedelta(seconds=900)  # Capped
        assert abs((next_poll_at - expected).total_seconds()) < 2


# =============================================================================
# Health Helper Tests
# =============================================================================


class TestHealthHelpers:
    """Tests for health helper methods."""

    @pytest.mark.asyncio
    async def test_count_due_pairs(self, mock_pool):
        """Test count_due_pairs returns correct count."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"cnt": 5})

        repo = PricePollStateRepository(pool)
        count = await repo.count_due_pairs()

        assert count == 5

    @pytest.mark.asyncio
    async def test_count_never_polled(self, mock_pool):
        """Test count_never_polled returns correct count."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"cnt": 3})

        repo = PricePollStateRepository(pool)
        count = await repo.count_never_polled()

        assert count == 3

    @pytest.mark.asyncio
    async def test_get_worst_staleness_returns_value(self, mock_pool):
        """Test get_worst_staleness returns staleness in seconds."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"staleness": 120.5})

        repo = PricePollStateRepository(pool)
        staleness = await repo.get_worst_staleness()

        assert staleness == 120.5

    @pytest.mark.asyncio
    async def test_get_worst_staleness_returns_none_when_no_data(self, mock_pool):
        """Test get_worst_staleness returns None when no polled pairs."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"staleness": None})

        repo = PricePollStateRepository(pool)
        staleness = await repo.get_worst_staleness()

        assert staleness is None

    @pytest.mark.asyncio
    async def test_count_total(self, mock_pool):
        """Test count_total returns total state rows."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"cnt": 10})

        repo = PricePollStateRepository(pool)
        count = await repo.count_total()

        assert count == 10
