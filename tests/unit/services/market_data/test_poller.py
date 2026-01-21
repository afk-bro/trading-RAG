"""Unit tests for live price poller service."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.market_data.poller import (
    LivePricePoller,
    PollKey,
    PollState,
    PollerHealth,
    get_poller,
    set_poller,
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
def mock_settings():
    """Mock settings with live price polling config."""
    settings = MagicMock()
    settings.live_price_poll_enabled = True
    settings.live_price_poll_interval_seconds = 60
    settings.live_price_poll_tick_seconds = 15
    settings.live_price_poll_jitter_seconds = 5
    settings.live_price_poll_backoff_max_seconds = 900
    settings.live_price_poll_max_concurrency_per_exchange = 3
    settings.live_price_poll_max_pairs_per_tick = 50
    settings.live_price_poll_lookback_candles = 5
    settings.live_price_poll_timeframes = ["1m"]
    return settings


@pytest.fixture
def sample_symbols():
    """Sample core symbols for testing."""
    from app.repositories.core_symbols import CoreSymbol

    return [
        CoreSymbol(
            exchange_id="binance",
            canonical_symbol="BTC/USDT",
            raw_symbol="BTCUSDT",
            timeframes=["1m", "5m"],
            is_enabled=True,
        ),
        CoreSymbol(
            exchange_id="binance",
            canonical_symbol="ETH/USDT",
            raw_symbol="ETHUSDT",
            timeframes=["1m"],
            is_enabled=True,
        ),
        CoreSymbol(
            exchange_id="kraken",
            canonical_symbol="BTC/USD",
            raw_symbol="XBTUSD",
            timeframes=None,  # Will use default
            is_enabled=True,
        ),
    ]


# =============================================================================
# PollKey Tests
# =============================================================================


class TestPollKey:
    """Tests for PollKey namedtuple."""

    def test_poll_key_creation(self):
        """Test PollKey can be created with all fields."""
        key = PollKey(
            exchange_id="binance",
            symbol="BTC/USDT",
            timeframe="1m",
        )
        assert key.exchange_id == "binance"
        assert key.symbol == "BTC/USDT"
        assert key.timeframe == "1m"

    def test_poll_key_hashable(self):
        """Test PollKey is hashable for use in sets/dicts."""
        key1 = PollKey("binance", "BTC/USDT", "1m")
        key2 = PollKey("binance", "BTC/USDT", "1m")
        key3 = PollKey("binance", "ETH/USDT", "1m")

        assert hash(key1) == hash(key2)
        assert key1 == key2
        assert key1 != key3

        # Can use in set
        keys = {key1, key2, key3}
        assert len(keys) == 2

    def test_poll_key_tuple_unpacking(self):
        """Test PollKey can be unpacked like a tuple."""
        key = PollKey("kraken", "BTC/USD", "1h")
        exchange, symbol, tf = key
        assert exchange == "kraken"
        assert symbol == "BTC/USD"
        assert tf == "1h"


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestPollerLifecycle:
    """Tests for poller start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self, mock_pool, mock_settings):
        """Test poller doesn't start when disabled."""
        pool, _ = mock_pool
        mock_settings.live_price_poll_enabled = False

        poller = LivePricePoller(pool, mock_settings)
        await poller.start()

        assert not poller.is_running

    @pytest.mark.asyncio
    async def test_start_when_enabled(self, mock_pool, mock_settings, sample_symbols):
        """Test poller starts when enabled."""
        pool, conn = mock_pool

        # Mock list_symbols to return empty (no pairs to poll)
        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            await poller.start()

            assert poller.is_running

            # Stop it
            await poller.stop()
            assert not poller.is_running

    @pytest.mark.asyncio
    async def test_double_start_ignored(self, mock_pool, mock_settings):
        """Test calling start twice doesn't create multiple tasks."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            await poller.start()
            task1 = poller._task

            await poller.start()  # Second start
            task2 = poller._task

            assert task1 is task2  # Same task

            await poller.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, mock_pool, mock_settings):
        """Test stop is safe when not running."""
        pool, _ = mock_pool
        poller = LivePricePoller(pool, mock_settings)

        # Should not raise
        await poller.stop()


# =============================================================================
# Semaphore Tests
# =============================================================================


class TestSemaphores:
    """Tests for per-exchange semaphore management."""

    def test_semaphore_creation(self, mock_pool, mock_settings):
        """Test semaphores are created per exchange."""
        pool, _ = mock_pool
        poller = LivePricePoller(pool, mock_settings)

        sem1 = poller._get_semaphore("binance")
        sem2 = poller._get_semaphore("binance")
        sem3 = poller._get_semaphore("kraken")

        # Same exchange returns same semaphore
        assert sem1 is sem2

        # Different exchange returns different semaphore
        assert sem1 is not sem3

    def test_semaphore_concurrency_limit(self, mock_pool, mock_settings):
        """Test semaphores have correct concurrency limit."""
        pool, _ = mock_pool
        mock_settings.live_price_poll_max_concurrency_per_exchange = 5

        poller = LivePricePoller(pool, mock_settings)
        sem = poller._get_semaphore("binance")

        # Semaphore should have initial value of 5
        # In asyncio, Semaphore._value is the internal counter
        assert sem._value == 5


# =============================================================================
# Selection Logic Tests
# =============================================================================


class TestDuePairSelection:
    """Tests for due pair selection logic."""

    @pytest.mark.asyncio
    async def test_expands_symbol_timeframes(self, mock_pool, mock_settings, sample_symbols):
        """Test selection expands symbols to (exchange, symbol, tf) pairs."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            due_pairs = await poller._get_due_pairs(limit=100)

            # BTC/USDT has ["1m", "5m"] but only 1m is in active_timeframes
            # ETH/USDT has ["1m"]
            # BTC/USD has None -> uses default ["1m"]
            # Total: 3 pairs (all 1m since that's the only active timeframe)
            assert len(due_pairs) == 3

            # Verify all pairs are 1m
            for pair, state in due_pairs:
                assert pair.timeframe == "1m"

    @pytest.mark.asyncio
    async def test_filters_to_active_timeframes(self, mock_pool, mock_settings, sample_symbols):
        """Test selection filters to configured active timeframes."""
        pool, _ = mock_pool
        mock_settings.live_price_poll_timeframes = ["1m", "5m"]

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            due_pairs = await poller._get_due_pairs(limit=100)

            # BTC/USDT: 1m, 5m (both active)
            # ETH/USDT: 1m only
            # BTC/USD: defaults to ["1m", "5m"] (both active)
            # Total: 2 + 1 + 2 = 5 pairs
            assert len(due_pairs) == 5

    @pytest.mark.asyncio
    async def test_respects_limit(self, mock_pool, mock_settings, sample_symbols):
        """Test selection respects the limit parameter."""
        pool, _ = mock_pool
        mock_settings.live_price_poll_timeframes = ["1m", "5m", "1h"]

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            # Create more symbols to exceed limit
            many_symbols = sample_symbols * 10  # 30 symbols
            mock_repo.list_symbols = AsyncMock(return_value=many_symbols)
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            due_pairs = await poller._get_due_pairs(limit=5)

            assert len(due_pairs) == 5

    @pytest.mark.asyncio
    async def test_empty_symbols(self, mock_pool, mock_settings):
        """Test selection handles no enabled symbols."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            due_pairs = await poller._get_due_pairs(limit=50)

            assert len(due_pairs) == 0


# =============================================================================
# Health Tests
# =============================================================================


class TestPollerHealth:
    """Tests for poller health endpoint."""

    @pytest.mark.asyncio
    async def test_health_when_not_running(self, mock_pool, mock_settings):
        """Test health returns disabled state when not running."""
        pool, _ = mock_pool
        mock_settings.live_price_poll_enabled = False

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            health = await poller.get_health()

            assert health.enabled is False
            assert health.running is False

    @pytest.mark.asyncio
    async def test_health_computes_pairs_enabled(self, mock_pool, mock_settings, sample_symbols):
        """Test health computes pairs_enabled from symbols × timeframes."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            health = await poller.get_health()

            # BTC/USDT: 2 timeframes
            # ETH/USDT: 1 timeframe
            # BTC/USD: None -> 1 default timeframe
            # Total: 4 pairs
            assert health.pairs_enabled == 4

    @pytest.mark.asyncio
    async def test_health_includes_config(self, mock_pool, mock_settings):
        """Test health includes configuration values."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            health = await poller.get_health()

            assert health.poll_interval_seconds == 60
            assert health.active_timeframes == ["1m"]


# =============================================================================
# Module Singleton Tests
# =============================================================================


class TestModuleSingleton:
    """Tests for module-level singleton management."""

    def test_get_set_poller(self, mock_pool, mock_settings):
        """Test get/set poller singleton."""
        pool, _ = mock_pool

        # Initially None
        assert get_poller() is None

        # Set a poller
        poller = LivePricePoller(pool, mock_settings)
        set_poller(poller)
        assert get_poller() is poller

        # Clear it
        set_poller(None)
        assert get_poller() is None


# =============================================================================
# Poll Tick Tests
# =============================================================================


class TestPollTick:
    """Tests for poll tick execution."""

    @pytest.mark.asyncio
    async def test_poll_tick_logs_selection(self, mock_pool, mock_settings, sample_symbols):
        """Test poll tick logs number of selected pairs."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            result = await poller.run_once()

            # Should have fetched 3 pairs (all 1m since that's the only active tf)
            assert result.pairs_fetched == 3
            assert result.pairs_succeeded == 3
            assert result.pairs_failed == 0
            assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_poll_tick_empty(self, mock_pool, mock_settings):
        """Test poll tick with no symbols."""
        pool, _ = mock_pool

        with patch(
            "app.services.market_data.poller.CoreSymbolsRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.list_symbols = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            poller = LivePricePoller(pool, mock_settings)
            result = await poller.run_once()

            assert result.pairs_fetched == 0
            assert result.duration_ms >= 0
