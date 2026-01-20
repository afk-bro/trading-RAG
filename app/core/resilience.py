"""Connection resilience utilities for database and vector store recovery.

Provides retry logic with exponential backoff for transient connection failures.
Designed to allow services to recover from temporary outages without restart.

Usage:
    from app.core.resilience import with_db_retry, with_qdrant_retry

    # Wrap database operations
    result = await with_db_retry(pool, lambda conn: conn.fetchrow(query))

    # Wrap Qdrant operations
    result = await with_qdrant_retry(client, lambda c: c.search(...))
"""

import asyncio
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypeVar

import asyncpg
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.25  # Add up to 25% random jitter


@dataclass
class CircuitState:
    """Track circuit breaker state for a service."""

    failures: int = 0
    last_failure: Optional[datetime] = None
    is_open: bool = False
    open_until: Optional[datetime] = None

    # Circuit opens after this many consecutive failures
    failure_threshold: int = 5
    # Circuit stays open for this many seconds before half-open test
    reset_timeout_seconds: float = 30.0


# Global circuit states for each service
_db_circuit = CircuitState()
_qdrant_circuit = CircuitState()


def _calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds before next retry
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay_seconds * (config.exponential_base**attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay_seconds)

    # Add jitter to prevent thundering herd
    jitter = delay * config.jitter_factor * random.random()
    delay += jitter

    return delay


def _is_transient_db_error(error: Exception) -> bool:
    """Check if database error is transient and worth retrying.

    Returns True for connection errors, timeouts, and pool exhaustion.
    Returns False for query errors, constraint violations, etc.
    """
    # Connection-related errors are transient
    if isinstance(
        error,
        (
            asyncpg.InterfaceError,  # Connection interface issues
            asyncpg.InternalClientError,  # Client-side connection issues
            ConnectionRefusedError,
            ConnectionResetError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,  # Network-level errors
        ),
    ):
        return True

    # Pool exhaustion is transient
    if isinstance(error, asyncpg.TooManyConnectionsError):
        return True

    # Connection timeout during acquire
    if isinstance(error, asyncpg.InterfaceError):
        error_msg = str(error).lower()
        if "timeout" in error_msg or "connection" in error_msg:
            return True

    # Query errors, constraint violations, etc. are NOT transient
    if isinstance(
        error,
        (
            asyncpg.PostgresError,  # Base class for Postgres errors
        ),
    ):
        # Check specific error codes for transient issues
        error_code = getattr(error, "sqlstate", None)
        transient_codes = {
            "08000",  # connection_exception
            "08003",  # connection_does_not_exist
            "08006",  # connection_failure
            "08001",  # sqlclient_unable_to_establish_sqlconnection
            "08004",  # sqlserver_rejected_establishment_of_sqlconnection
            "57P01",  # admin_shutdown
            "57P02",  # crash_shutdown
            "57P03",  # cannot_connect_now
            "40001",  # serialization_failure (can retry)
            "40P01",  # deadlock_detected (can retry)
        }
        return error_code in transient_codes

    return False


def _is_transient_qdrant_error(error: Exception) -> bool:
    """Check if Qdrant error is transient and worth retrying."""
    # Network-level errors are transient
    if isinstance(
        error,
        (
            ConnectionRefusedError,
            ConnectionResetError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        ),
    ):
        return True

    # Qdrant-specific errors
    if isinstance(error, (ResponseHandlingException, UnexpectedResponse)):
        error_msg = str(error).lower()
        # Connection/timeout issues are transient
        if any(
            keyword in error_msg
            for keyword in ["timeout", "connection", "unavailable", "reset"]
        ):
            return True

    # HTTP client errors (aiohttp/httpx)
    error_type = type(error).__name__
    if error_type in (
        "ClientConnectorError",
        "ServerDisconnectedError",
        "ClientOSError",
    ):
        return True

    return False


def _check_circuit(circuit: CircuitState, service_name: str) -> bool:
    """Check if circuit breaker allows the request.

    Returns True if request should proceed, False if circuit is open.
    """
    now = datetime.now(timezone.utc)

    if circuit.is_open:
        # Check if reset timeout has passed (half-open state)
        if circuit.open_until and now >= circuit.open_until:
            logger.info(
                "circuit_half_open",
                service=service_name,
                failures=circuit.failures,
            )
            return True  # Allow one request through to test
        else:
            return False  # Circuit still open

    return True


def _record_success(circuit: CircuitState, service_name: str) -> None:
    """Record successful operation, reset circuit breaker."""
    if circuit.failures > 0 or circuit.is_open:
        logger.info(
            "circuit_closed",
            service=service_name,
            previous_failures=circuit.failures,
        )
    circuit.failures = 0
    circuit.last_failure = None
    circuit.is_open = False
    circuit.open_until = None


def _record_failure(circuit: CircuitState, service_name: str) -> None:
    """Record failed operation, possibly open circuit breaker."""
    now = datetime.now(timezone.utc)
    circuit.failures += 1
    circuit.last_failure = now

    if circuit.failures >= circuit.failure_threshold:
        circuit.is_open = True
        circuit.open_until = datetime.fromtimestamp(
            now.timestamp() + circuit.reset_timeout_seconds,
            tz=timezone.utc,
        )
        logger.warning(
            "circuit_opened",
            service=service_name,
            failures=circuit.failures,
            reset_at=circuit.open_until.isoformat(),
        )


async def with_db_retry(
    pool: asyncpg.Pool,
    operation: Callable[[asyncpg.Connection], Any],
    config: Optional[RetryConfig] = None,
) -> Any:
    """Execute database operation with retry on transient failures.

    Args:
        pool: asyncpg connection pool
        operation: Async callable that takes a connection and returns result
        config: Optional retry configuration

    Returns:
        Result of the operation

    Raises:
        Exception: If all retries exhausted or non-transient error
    """
    if config is None:
        config = RetryConfig()

    # Check circuit breaker
    if not _check_circuit(_db_circuit, "supabase"):
        raise RuntimeError(
            "Database circuit breaker is open - service recovering from outage"
        )

    last_error: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            async with pool.acquire() as conn:
                result = await operation(conn)
                _record_success(_db_circuit, "supabase")
                return result

        except Exception as e:
            last_error = e

            if not _is_transient_db_error(e):
                # Non-transient error - don't retry
                logger.warning(
                    "db_non_transient_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            # Log retry attempt
            delay = _calculate_backoff(attempt, config)
            logger.warning(
                "db_retry_attempt",
                attempt=attempt + 1,
                max_attempts=config.max_attempts,
                delay_seconds=round(delay, 2),
                error=str(e),
                error_type=type(e).__name__,
            )

            if attempt < config.max_attempts - 1:
                await asyncio.sleep(delay)

    # All retries exhausted
    _record_failure(_db_circuit, "supabase")
    logger.error(
        "db_retries_exhausted",
        attempts=config.max_attempts,
        error=str(last_error),
    )
    raise last_error  # type: ignore


async def with_qdrant_retry(
    client: AsyncQdrantClient,
    operation: Callable[[AsyncQdrantClient], Any],
    config: Optional[RetryConfig] = None,
) -> Any:
    """Execute Qdrant operation with retry on transient failures.

    Args:
        client: Qdrant async client
        operation: Async callable that takes the client and returns result
        config: Optional retry configuration

    Returns:
        Result of the operation

    Raises:
        Exception: If all retries exhausted or non-transient error
    """
    if config is None:
        config = RetryConfig()

    # Check circuit breaker
    if not _check_circuit(_qdrant_circuit, "qdrant"):
        raise RuntimeError(
            "Qdrant circuit breaker is open - service recovering from outage"
        )

    last_error: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            result = await operation(client)
            _record_success(_qdrant_circuit, "qdrant")
            return result

        except Exception as e:
            last_error = e

            if not _is_transient_qdrant_error(e):
                # Non-transient error - don't retry
                logger.warning(
                    "qdrant_non_transient_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            # Log retry attempt
            delay = _calculate_backoff(attempt, config)
            logger.warning(
                "qdrant_retry_attempt",
                attempt=attempt + 1,
                max_attempts=config.max_attempts,
                delay_seconds=round(delay, 2),
                error=str(e),
                error_type=type(e).__name__,
            )

            if attempt < config.max_attempts - 1:
                await asyncio.sleep(delay)

    # All retries exhausted
    _record_failure(_qdrant_circuit, "qdrant")
    logger.error(
        "qdrant_retries_exhausted",
        attempts=config.max_attempts,
        error=str(last_error),
    )
    raise last_error  # type: ignore


@asynccontextmanager
async def resilient_db_connection(
    pool: asyncpg.Pool,
    config: Optional[RetryConfig] = None,
):
    """Context manager for resilient database connection acquisition.

    Usage:
        async with resilient_db_connection(pool) as conn:
            await conn.fetchrow(query)
    """
    if config is None:
        config = RetryConfig()

    # Check circuit breaker
    if not _check_circuit(_db_circuit, "supabase"):
        raise RuntimeError(
            "Database circuit breaker is open - service recovering from outage"
        )

    last_error: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            async with pool.acquire() as conn:
                _record_success(_db_circuit, "supabase")
                yield conn
                return

        except Exception as e:
            last_error = e

            if not _is_transient_db_error(e):
                raise

            delay = _calculate_backoff(attempt, config)
            logger.warning(
                "db_acquire_retry",
                attempt=attempt + 1,
                max_attempts=config.max_attempts,
                delay_seconds=round(delay, 2),
                error=str(e),
            )

            if attempt < config.max_attempts - 1:
                await asyncio.sleep(delay)

    _record_failure(_db_circuit, "supabase")
    raise last_error  # type: ignore


def get_circuit_status() -> dict:
    """Get current circuit breaker status for health checks."""
    return {
        "supabase": {
            "failures": _db_circuit.failures,
            "is_open": _db_circuit.is_open,
            "last_failure": (
                _db_circuit.last_failure.isoformat()
                if _db_circuit.last_failure
                else None
            ),
        },
        "qdrant": {
            "failures": _qdrant_circuit.failures,
            "is_open": _qdrant_circuit.is_open,
            "last_failure": (
                _qdrant_circuit.last_failure.isoformat()
                if _qdrant_circuit.last_failure
                else None
            ),
        },
    }


def reset_circuits() -> None:
    """Reset all circuit breakers. Used for testing."""
    global _db_circuit, _qdrant_circuit
    _db_circuit = CircuitState()
    _qdrant_circuit = CircuitState()
