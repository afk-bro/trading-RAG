"""Tests for connection resilience and recovery logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.resilience import (
    CircuitState,
    RetryConfig,
    _calculate_backoff,
    _is_transient_db_error,
    _is_transient_qdrant_error,
    get_circuit_status,
    reset_circuits,
    with_db_retry,
    with_qdrant_retry,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay_seconds == 0.5
        assert config.max_delay_seconds == 10.0
        assert config.exponential_base == 2.0
        assert config.jitter_factor == 0.25

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
        )
        assert config.max_attempts == 5
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 30.0


class TestBackoffCalculation:
    """Tests for exponential backoff calculation."""

    def test_backoff_increases_with_attempts(self):
        """Test that backoff increases exponentially."""
        config = RetryConfig(jitter_factor=0)  # No jitter for predictable test

        delay_0 = _calculate_backoff(0, config)
        delay_1 = _calculate_backoff(1, config)
        delay_2 = _calculate_backoff(2, config)

        assert delay_0 == 0.5  # base_delay
        assert delay_1 == 1.0  # base_delay * 2
        assert delay_2 == 2.0  # base_delay * 4

    def test_backoff_capped_at_max(self):
        """Test that backoff is capped at max_delay."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            jitter_factor=0,
        )

        # Attempt 10 would be 1 * 2^10 = 1024, but capped at 5
        delay = _calculate_backoff(10, config)
        assert delay == 5.0

    def test_backoff_includes_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(jitter_factor=0.25)

        # Run multiple times to verify jitter varies
        delays = [_calculate_backoff(1, config) for _ in range(10)]

        # Base delay at attempt 1 is 1.0, jitter adds 0-25%
        assert all(1.0 <= d <= 1.25 for d in delays)
        # Should have some variation (not all identical)
        assert len(set(round(d, 4) for d in delays)) > 1


class TestTransientErrorDetection:
    """Tests for transient error classification."""

    def test_connection_refused_is_transient(self):
        """Test ConnectionRefusedError is transient."""
        assert _is_transient_db_error(ConnectionRefusedError())
        assert _is_transient_qdrant_error(ConnectionRefusedError())

    def test_connection_reset_is_transient(self):
        """Test ConnectionResetError is transient."""
        assert _is_transient_db_error(ConnectionResetError())
        assert _is_transient_qdrant_error(ConnectionResetError())

    def test_timeout_is_transient(self):
        """Test timeout errors are transient."""
        assert _is_transient_db_error(TimeoutError())
        assert _is_transient_db_error(asyncio.TimeoutError())
        assert _is_transient_qdrant_error(TimeoutError())
        assert _is_transient_qdrant_error(asyncio.TimeoutError())

    def test_os_error_is_transient(self):
        """Test OSError (network) is transient."""
        assert _is_transient_db_error(OSError("Network unreachable"))
        assert _is_transient_qdrant_error(OSError("Network unreachable"))

    def test_value_error_not_transient(self):
        """Test ValueError is NOT transient."""
        assert not _is_transient_db_error(ValueError("bad value"))
        assert not _is_transient_qdrant_error(ValueError("bad value"))


class TestDatabaseRetry:
    """Tests for database retry logic."""

    @pytest.fixture(autouse=True)
    def reset_circuit(self):
        """Reset circuits before each test."""
        reset_circuits()
        yield
        reset_circuits()

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Test successful operation doesn't retry."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 1})

        # Mock the context manager
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"))

        assert result == {"id": 1}
        assert conn.fetchrow.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Test retry occurs on transient error."""
        pool = MagicMock()
        conn = AsyncMock()

        # Fail first, succeed second
        conn.fetchrow = AsyncMock(
            side_effect=[ConnectionRefusedError(), {"id": 1}]
        )

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        config = RetryConfig(base_delay_seconds=0.01)  # Fast retry for test
        result = await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"), config)

        assert result == {"id": 1}
        assert conn.fetchrow.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient_error(self):
        """Test no retry on non-transient error."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=ValueError("bad query"))

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(ValueError, match="bad query"):
            await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"))

        assert conn.fetchrow.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        """Test exception raised after max retries."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=ConnectionRefusedError())

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)

        with pytest.raises(ConnectionRefusedError):
            await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"), config)

        assert conn.fetchrow.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=ConnectionRefusedError())

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        config = RetryConfig(max_attempts=2, base_delay_seconds=0.01)

        # Exhaust retries 5 times to trigger circuit (threshold is 5)
        # Each exhausted retry cycle records 1 failure to the circuit
        for _ in range(5):
            try:
                await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"), config)
            except ConnectionRefusedError:
                pass

        # Circuit should be open now (5 consecutive failures)
        status = get_circuit_status()
        assert status["supabase"]["is_open"] is True


class TestQdrantRetry:
    """Tests for Qdrant retry logic."""

    @pytest.fixture(autouse=True)
    def reset_circuit(self):
        """Reset circuits before each test."""
        reset_circuits()
        yield
        reset_circuits()

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Test successful Qdrant operation doesn't retry."""
        client = AsyncMock()
        client.search = AsyncMock(return_value=[{"id": 1}])

        result = await with_qdrant_retry(client, lambda c: c.search())

        assert result == [{"id": 1}]
        assert client.search.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry on Qdrant connection error."""
        client = AsyncMock()
        client.search = AsyncMock(
            side_effect=[ConnectionRefusedError(), [{"id": 1}]]
        )

        config = RetryConfig(base_delay_seconds=0.01)
        result = await with_qdrant_retry(client, lambda c: c.search(), config)

        assert result == [{"id": 1}]
        assert client.search.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        """Test exception raised after max Qdrant retries."""
        client = AsyncMock()
        client.search = AsyncMock(side_effect=ConnectionRefusedError())

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)

        with pytest.raises(ConnectionRefusedError):
            await with_qdrant_retry(client, lambda c: c.search(), config)

        assert client.search.call_count == 3


class TestCircuitStatus:
    """Tests for circuit breaker status reporting."""

    @pytest.fixture(autouse=True)
    def reset_circuit(self):
        """Reset circuits before each test."""
        reset_circuits()
        yield
        reset_circuits()

    def test_initial_status_healthy(self):
        """Test initial circuit status is healthy."""
        status = get_circuit_status()

        assert status["supabase"]["failures"] == 0
        assert status["supabase"]["is_open"] is False
        assert status["supabase"]["last_failure"] is None

        assert status["qdrant"]["failures"] == 0
        assert status["qdrant"]["is_open"] is False
        assert status["qdrant"]["last_failure"] is None

    def test_reset_clears_status(self):
        """Test reset_circuits clears all state."""
        # Manually set some state
        from app.core.resilience import _db_circuit

        _db_circuit.failures = 5
        _db_circuit.is_open = True

        reset_circuits()

        status = get_circuit_status()
        assert status["supabase"]["failures"] == 0
        assert status["supabase"]["is_open"] is False


class TestRecoveryScenarios:
    """Integration-style tests for recovery scenarios."""

    @pytest.fixture(autouse=True)
    def reset_circuit(self):
        """Reset circuits before each test."""
        reset_circuits()
        yield
        reset_circuits()

    @pytest.mark.asyncio
    async def test_recovery_after_transient_failure(self):
        """Test service recovers after transient failure."""
        pool = MagicMock()
        conn = AsyncMock()

        # Simulate: fail -> fail -> succeed
        conn.fetchrow = AsyncMock(
            side_effect=[
                ConnectionRefusedError(),
                ConnectionRefusedError(),
                {"recovered": True},
            ]
        )

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.01)
        result = await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"), config)

        assert result == {"recovered": True}

        # Circuit should be healthy after recovery
        status = get_circuit_status()
        assert status["supabase"]["failures"] == 0
        assert status["supabase"]["is_open"] is False

    @pytest.mark.asyncio
    async def test_prolonged_outage_opens_circuit(self):
        """Test prolonged outage opens circuit breaker."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=ConnectionRefusedError())

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        config = RetryConfig(max_attempts=2, base_delay_seconds=0.01)

        # Multiple failed operations
        failures = 0
        for _ in range(5):
            try:
                await with_db_retry(pool, lambda c: c.fetchrow("SELECT 1"), config)
            except (ConnectionRefusedError, RuntimeError):
                failures += 1

        # Eventually circuit opens and blocks requests
        status = get_circuit_status()
        assert status["supabase"]["is_open"] is True
        assert failures == 5
