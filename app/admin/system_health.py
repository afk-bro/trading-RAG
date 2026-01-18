"""System health dashboard for operations.

Single page that answers "what's broken?" without opening logs.

Endpoints:
- GET /admin/system/health - HTML dashboard with status cards
- GET /admin/system/health.json - Machine-readable JSON
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.deps.security import require_admin_token

router = APIRouter(prefix="/system", tags=["admin-system"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


# =============================================================================
# Health Snapshot Schemas
# =============================================================================


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    status: str = Field(..., description="ok, degraded, error, unknown")
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: dict = Field(default_factory=dict)


class DBHealth(ComponentHealth):
    """Database health details."""

    pool_size: Optional[int] = None
    pool_available: Optional[int] = None
    pool_acquire_ms: Optional[float] = None
    query_latency_ms: Optional[float] = None
    connection_errors_5m: int = 0


class QdrantHealth(ComponentHealth):
    """Qdrant health details."""

    vectors_count: Optional[int] = None
    segments_count: Optional[int] = None
    collection: Optional[str] = None
    last_error_at: Optional[datetime] = None


class LLMHealth(ComponentHealth):
    """LLM provider health details."""

    provider_configured: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    degraded_count_1h: int = 0
    error_count_1h: int = 0
    last_success_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None


class IngestionHealth(BaseModel):
    """Ingestion pipeline health."""

    status: str = "unknown"
    youtube_last_success: Optional[datetime] = None
    youtube_last_failure: Optional[datetime] = None
    pdf_last_success: Optional[datetime] = None
    pdf_last_failure: Optional[datetime] = None
    pine_last_success: Optional[datetime] = None
    pine_last_failure: Optional[datetime] = None
    pending_jobs: int = 0


class SSEHealth(ComponentHealth):
    """SSE event bus health."""

    subscribers: int = 0
    events_published_1h: int = 0
    queue_drops_1h: int = 0
    buffer_size: int = 0
    bus_type: str = "memory"  # memory or redis


class RedisHealth(ComponentHealth):
    """Redis health (for multi-worker event bus)."""

    connected: bool = False
    configured: bool = False
    stream_count: int = 0
    total_stream_length: int = 0
    ping_latency_ms: Optional[float] = None


class IdempotencyHealth(ComponentHealth):
    """Idempotency key hygiene status."""

    total_keys: int = 0
    expired_pending: int = 0
    pending_requests: int = 0
    oldest_pending_age_minutes: Optional[float] = None
    oldest_expired_age_hours: Optional[float] = None


class RetentionHealth(ComponentHealth):
    """Retention job health."""

    last_run_at: Optional[datetime] = None
    last_run_ok: Optional[bool] = None
    last_run_job: Optional[str] = None
    rows_deleted_last: int = 0
    consecutive_failures: int = 0
    pg_cron_available: bool = False
    jobs_24h: dict = Field(default_factory=dict)  # Per-job stats


class TuneHealth(ComponentHealth):
    """Backtest tune health."""

    active_tunes: int = 0
    completed_24h: int = 0
    failed_24h: int = 0
    avg_duration_ms: Optional[float] = None


class PineDiscoveryHealth(ComponentHealth):
    """Pine script discovery health."""

    scripts_by_status: dict = Field(default_factory=dict)  # {status: count}
    scripts_by_ingest_status: dict = Field(
        default_factory=dict
    )  # {ingest_status: count} - values: pending, ok, error
    total_scripts: int = 0
    pending_ingest: int = 0
    stale_scripts: int = 0
    stale_ratio: float = 0.0  # stale_scripts / active_total (0.0 when no scripts)
    stale_cutoff_days: int = 7  # Time anchor
    recent_ingest_errors: int = 0
    window_ingest_errors_hours: int = 24  # Time anchor
    last_discovery_at: Optional[datetime] = None
    last_run_ts: Optional[int] = None  # Unix epoch seconds from gauge
    last_success_ts: Optional[int] = None  # Unix epoch seconds from gauge
    notes: list[str] = Field(default_factory=list)  # Reasons for degraded/error


class SystemHealthSnapshot(BaseModel):
    """Complete system health snapshot."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall_status: str = Field(..., description="ok, degraded, error")
    version: str = "unknown"
    git_sha: Optional[str] = None

    # Component health
    database: DBHealth
    qdrant: QdrantHealth
    llm: LLMHealth
    ingestion: IngestionHealth
    sse: SSEHealth
    redis: Optional[RedisHealth] = None  # Only present when event_bus_mode=redis
    retention: RetentionHealth
    idempotency: IdempotencyHealth
    tunes: TuneHealth
    pine_discovery: PineDiscoveryHealth

    # Summary
    components_ok: int = 0
    components_degraded: int = 0
    components_error: int = 0
    issues: list[str] = Field(default_factory=list)


# =============================================================================
# Health Collection Functions
# =============================================================================


async def _check_database() -> DBHealth:
    """Check database connectivity and pool health."""
    if _db_pool is None:
        return DBHealth(status="error", error="Pool not initialized")

    start = time.perf_counter()
    try:
        # Check pool stats
        pool_size = _db_pool.get_size()
        pool_available = _db_pool.get_idle_size()

        # Test acquire + query
        acquire_start = time.perf_counter()
        async with _db_pool.acquire() as conn:
            acquire_ms = (time.perf_counter() - acquire_start) * 1000

            query_start = time.perf_counter()
            await conn.fetchval("SELECT 1")
            query_ms = (time.perf_counter() - query_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000

        # Determine status based on latency thresholds
        status = "ok"
        if acquire_ms > 100 or query_ms > 50:
            status = "degraded"
        if acquire_ms > 500 or query_ms > 200:
            status = "error"

        return DBHealth(
            status=status,
            latency_ms=total_ms,
            pool_size=pool_size,
            pool_available=pool_available,
            pool_acquire_ms=acquire_ms,
            query_latency_ms=query_ms,
        )
    except Exception as e:
        return DBHealth(
            status="error",
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e)[:200],
        )


async def _check_qdrant(settings: Settings) -> QdrantHealth:
    """Check Qdrant connectivity and collection stats."""
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            health_resp = await client.get(f"{settings.qdrant_url}/healthz")
            if health_resp.status_code != 200:
                return QdrantHealth(
                    status="error",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    error=f"Health check returned {health_resp.status_code}",
                )

            # Collection stats
            collection = settings.qdrant_collection_active
            coll_resp = await client.get(
                f"{settings.qdrant_url}/collections/{collection}"
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if coll_resp.status_code == 200:
                data = coll_resp.json().get("result", {})
                vectors_count = data.get("vectors_count", 0)
                segments_count = data.get("segments_count", 0)

                return QdrantHealth(
                    status="ok",
                    latency_ms=latency_ms,
                    vectors_count=vectors_count,
                    segments_count=segments_count,
                    collection=collection,
                )
            elif coll_resp.status_code == 404:
                return QdrantHealth(
                    status="degraded",
                    latency_ms=latency_ms,
                    error=f"Collection '{collection}' not found",
                    collection=collection,
                )
            else:
                return QdrantHealth(
                    status="error",
                    latency_ms=latency_ms,
                    error=f"Collection check returned {coll_resp.status_code}",
                )
    except Exception as e:
        return QdrantHealth(
            status="error",
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e)[:200],
        )


async def _check_llm() -> LLMHealth:
    """Check LLM provider configuration and recent error rates."""
    try:
        from app.services.llm_factory import get_llm_status

        status_obj = get_llm_status()

        if not status_obj.enabled:
            return LLMHealth(
                status="ok",  # Not an error - just not configured
                provider_configured=False,
                details={"reason": "LLM disabled or no API key"},
            )

        # LLM is configured
        # TODO: Track actual error/degraded counts with prometheus counters
        return LLMHealth(
            status="ok",
            provider_configured=True,
            provider=status_obj.provider_resolved,
            model=status_obj.answer_model,
        )
    except Exception as e:
        return LLMHealth(
            status="error",
            error=str(e)[:200],
        )


async def _check_ingestion() -> IngestionHealth:
    """Check ingestion pipeline status."""
    if _db_pool is None:
        return IngestionHealth(status="unknown")

    try:
        async with _db_pool.acquire() as conn:
            # Get last success/failure per source type
            rows = await conn.fetch(
                """
                SELECT
                    source_type,
                    MAX(CASE WHEN status = 'active' THEN updated_at END) as last_success,
                    MAX(CASE WHEN status = 'failed' THEN updated_at END) as last_failure
                FROM documents
                WHERE source_type IN ('youtube', 'pdf', 'pine')
                  AND updated_at > NOW() - INTERVAL '7 days'
                GROUP BY source_type
            """
            )

            result = IngestionHealth(status="ok")
            for row in rows:
                st = row["source_type"]
                if st == "youtube":
                    result.youtube_last_success = row["last_success"]
                    result.youtube_last_failure = row["last_failure"]
                elif st == "pdf":
                    result.pdf_last_success = row["last_success"]
                    result.pdf_last_failure = row["last_failure"]
                elif st == "pine":
                    result.pine_last_success = row["last_success"]
                    result.pine_last_failure = row["last_failure"]

            # Check for recent failures without success
            for _st, last_fail, last_success in [
                ("youtube", result.youtube_last_failure, result.youtube_last_success),
                ("pdf", result.pdf_last_failure, result.pdf_last_success),
                ("pine", result.pine_last_failure, result.pine_last_success),
            ]:
                if last_fail and (not last_success or last_fail > last_success):
                    result.status = "degraded"

            return result
    except Exception:
        return IngestionHealth(status="error", pending_jobs=-1)


async def _check_sse(settings: Settings) -> SSEHealth:
    """Check SSE event bus health."""
    try:
        from app.services.events import get_event_bus

        bus = get_event_bus()
        subscribers = bus.subscriber_count()
        buffer_size = bus.buffer_size() if hasattr(bus, "buffer_size") else 0
        bus_type = settings.event_bus_mode

        return SSEHealth(
            status="ok",
            subscribers=subscribers,
            buffer_size=buffer_size,
            bus_type=bus_type,
        )
    except Exception as e:
        return SSEHealth(status="error", error=str(e)[:200])


async def _check_redis(settings: Settings) -> Optional[RedisHealth]:
    """Check Redis health (only when event_bus_mode=redis)."""
    if settings.event_bus_mode != "redis":
        return None  # Redis not configured

    if not settings.redis_url:
        return RedisHealth(
            status="error",
            configured=False,
            error="EVENT_BUS_MODE=redis but REDIS_URL not set",
        )

    start = time.perf_counter()
    try:
        from app.services.events import get_event_bus
        from app.services.events.redis_bus import RedisEventBus

        bus = get_event_bus()

        if not isinstance(bus, RedisEventBus):
            return RedisHealth(
                status="error",
                configured=True,
                connected=False,
                error="Event bus is not RedisEventBus",
            )

        # Ping Redis
        ping_ok = await bus.ping()
        ping_ms = (time.perf_counter() - start) * 1000

        if not ping_ok:
            return RedisHealth(
                status="error",
                configured=True,
                connected=False,
                ping_latency_ms=ping_ms,
                error="Redis ping failed",
            )

        # Get stream stats
        stream_lengths = await bus.get_stream_lengths()
        stream_count = len(stream_lengths)
        total_length = sum(stream_lengths.values())

        # Determine status
        status = "ok"
        if ping_ms > 100:
            status = "degraded"
        if ping_ms > 500:
            status = "error"

        return RedisHealth(
            status=status,
            configured=True,
            connected=True,
            stream_count=stream_count,
            total_stream_length=total_length,
            ping_latency_ms=ping_ms,
        )
    except Exception as e:
        return RedisHealth(
            status="error",
            configured=True,
            connected=False,
            ping_latency_ms=(time.perf_counter() - start) * 1000,
            error=str(e)[:200],
        )


async def _check_retention() -> RetentionHealth:
    """Check retention job status from logs."""
    if _db_pool is None:
        return RetentionHealth(status="unknown", error="Pool not initialized")

    try:
        async with _db_pool.acquire() as conn:
            # Check if pg_cron is available
            pg_cron_available = False
            try:
                result = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'pg_cron'"
                )
                pg_cron_available = result is not None
            except Exception:
                pass

            # Get last retention run
            row = await conn.fetchrow(
                """
                SELECT job_name, started_at, finished_at, rows_deleted, ok, error
                FROM retention_job_log
                ORDER BY started_at DESC
                LIMIT 1
            """
            )

            if not row:
                return RetentionHealth(
                    status="ok",
                    pg_cron_available=pg_cron_available,
                    details={"message": "No retention runs yet"},
                )

            # Count consecutive failures
            failures = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT ok
                    FROM retention_job_log
                    ORDER BY started_at DESC
                    LIMIT 5
                ) recent
                WHERE ok = false
            """
            )

            status = "ok"
            if not row["ok"]:
                status = "error" if failures >= 3 else "degraded"

            return RetentionHealth(
                status=status,
                last_run_at=row["started_at"],
                last_run_ok=row["ok"],
                last_run_job=row["job_name"],
                rows_deleted_last=row["rows_deleted"] or 0,
                consecutive_failures=failures or 0,
                pg_cron_available=pg_cron_available,
                error=row["error"] if not row["ok"] else None,
            )
    except Exception as e:
        # Table might not exist yet
        if "does not exist" in str(e):
            return RetentionHealth(
                status="ok",
                details={"message": "Retention not configured (table missing)"},
            )
        return RetentionHealth(status="error", error=str(e)[:200])


async def _check_tunes() -> TuneHealth:
    """Check backtest tune health."""
    if _db_pool is None:
        return TuneHealth(status="unknown", error="Pool not initialized")

    try:
        async with _db_pool.acquire() as conn:
            # Get tune stats for last 24h
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'running') as active,
                    COUNT(*) FILTER (WHERE status = 'completed'
                        AND updated_at > NOW() - INTERVAL '24 hours') as completed_24h,
                    COUNT(*) FILTER (WHERE status = 'failed'
                        AND updated_at > NOW() - INTERVAL '24 hours') as failed_24h,
                    AVG(EXTRACT(EPOCH FROM (finished_at - started_at)) * 1000)
                        FILTER (WHERE finished_at IS NOT NULL
                            AND updated_at > NOW() - INTERVAL '24 hours') as avg_duration_ms
                FROM backtest_tunes
            """
            )

            if not stats:
                return TuneHealth(status="ok")

            active = stats["active"] or 0
            failed = stats["failed_24h"] or 0
            completed = stats["completed_24h"] or 0

            # Determine status
            status = "ok"
            if failed > 0 and failed > completed:
                status = "degraded"
            if active > 10:  # Too many active tunes
                status = "degraded"

            return TuneHealth(
                status=status,
                active_tunes=active,
                completed_24h=completed,
                failed_24h=failed,
                avg_duration_ms=stats["avg_duration_ms"],
            )
    except Exception as e:
        # Table might not exist
        if "does not exist" in str(e):
            return TuneHealth(status="ok", details={"message": "Tunes not configured"})
        return TuneHealth(status="error", error=str(e)[:200])


async def _check_pine_discovery() -> PineDiscoveryHealth:
    """Check Pine script discovery health and update Prometheus metrics.

    Gauge updates are guarded: only updated on successful DB query.
    On failure, gauges retain last-known-good values.
    """
    from app.routers.metrics import (
        PINE_DISCOVERY_LAST_RUN_TIMESTAMP,
        PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP,
        set_pine_pending_ingest,
        set_pine_scripts_metrics,
    )

    # Time anchors (constants for documentation/alerting)
    STALE_CUTOFF_DAYS = 7
    INGEST_ERROR_WINDOW_HOURS = 24

    if _db_pool is None:
        return PineDiscoveryHealth(status="unknown", error="Pool not initialized")

    try:
        async with _db_pool.acquire() as conn:
            # Get script counts by discovery status
            rows = await conn.fetch(
                """
                SELECT status, COUNT(*) as count
                FROM strategy_scripts
                GROUP BY status
            """
            )
            status_counts = {row["status"]: row["count"] for row in rows}
            total = sum(status_counts.values())

            # Get script counts by ingest status (NULL = pending)
            # Canonical values: pending (NULL), ok, error
            ingest_rows = await conn.fetch(
                """
                SELECT
                    COALESCE(ingest_status, 'pending') as ingest_status,
                    COUNT(*) as count
                FROM strategy_scripts
                WHERE status != 'archived'
                GROUP BY ingest_status
            """
            )
            ingest_status_counts = {
                row["ingest_status"]: row["count"] for row in ingest_rows
            }

            # Get pending ingest count:
            # - Never ingested (ingest_status IS NULL)
            # - Content changed (sha256 != last_ingested_sha)
            # - Error and needs retry (ingest_status = 'error')
            pending_ingest = (
                await conn.fetchval(
                    """
                SELECT COUNT(*)
                FROM strategy_scripts
                WHERE status != 'archived'
                  AND (
                    ingest_status IS NULL
                    OR last_ingested_sha IS DISTINCT FROM sha256
                    OR ingest_status = 'error'
                  )
            """
                )
                or 0
            )

            # Get last discovery timestamp
            last_seen = await conn.fetchval(
                """
                SELECT MAX(last_seen_at)
                FROM strategy_scripts
            """
            )

            # Get stale script count (not seen in N days)
            # "seen" = last_seen_at updated on every discovery scan hit
            stale_count = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*)
                FROM strategy_scripts
                WHERE status != 'archived'
                  AND last_seen_at < NOW() - INTERVAL '{STALE_CUTOFF_DAYS} days'
            """
                )
                or 0
            )

            # Calculate stale ratio (0.0 when no active scripts)
            active_total = sum(c for s, c in status_counts.items() if s != "archived")
            stale_ratio = stale_count / active_total if active_total > 0 else 0.0

            # Get recent ingest errors (last N hours)
            # Use last_ingested_at for accurate timing, status='error' (canonical)
            recent_errors = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*)
                FROM strategy_scripts
                WHERE ingest_status = 'error'
                  AND last_ingested_at >= NOW() - INTERVAL '{INGEST_ERROR_WINDOW_HOURS} hours'
            """
                )
                or 0
            )

            # --- DB queries succeeded, safe to update gauges ---
            set_pine_scripts_metrics(status_counts)
            set_pine_pending_ingest(pending_ingest)

            # Read timestamp gauges (0 means never set)
            last_run_ts = PINE_DISCOVERY_LAST_RUN_TIMESTAMP._value.get()
            last_success_ts = PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP._value.get()

            # Determine health status with notes
            status = "ok"
            notes: list[str] = []

            # Degraded if stale ratio > 50%
            if stale_ratio is not None and stale_ratio > 0.5:
                status = "degraded"
                notes.append(f"stale_ratio={stale_ratio:.1%} > 50%")

            # Degraded if pending ingest > 50
            if pending_ingest > 50:
                status = "degraded"
                notes.append(f"pending_ingest={pending_ingest} > 50")

            # Degraded/error if recent ingest failures
            if recent_errors > 10:
                status = "error"
                notes.append(f"recent_ingest_errors={recent_errors} > 10")
            elif recent_errors > 0:
                status = "degraded"
                notes.append(f"recent_ingest_errors={recent_errors} > 0")

            return PineDiscoveryHealth(
                status=status,
                scripts_by_status=status_counts,
                scripts_by_ingest_status=ingest_status_counts,
                total_scripts=total,
                pending_ingest=pending_ingest,
                stale_scripts=stale_count,
                stale_ratio=round(stale_ratio, 3),
                stale_cutoff_days=STALE_CUTOFF_DAYS,
                recent_ingest_errors=recent_errors,
                window_ingest_errors_hours=INGEST_ERROR_WINDOW_HOURS,
                last_discovery_at=last_seen,
                # Cast to int for cleaner dashboard/alerting (epoch seconds)
                last_run_ts=int(last_run_ts) if last_run_ts > 0 else None,
                last_success_ts=int(last_success_ts) if last_success_ts > 0 else None,
                notes=notes,
            )
    except Exception as e:
        # Table might not exist yet
        if "does not exist" in str(e):
            return PineDiscoveryHealth(
                status="ok",
                details={"message": "Pine discovery not configured (table missing)"},
            )
        # On DB error, don't update gauges (leave last-known-good)
        return PineDiscoveryHealth(status="error", error=str(e)[:200])


async def _check_idempotency() -> IdempotencyHealth:
    """Check idempotency key table hygiene."""
    from app.routers.metrics import set_idempotency_metrics

    if _db_pool is None:
        return IdempotencyHealth(status="unknown", error="Pool not initialized")

    try:
        async with _db_pool.acquire() as conn:
            # Get idempotency key stats
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_keys,
                    COUNT(*) FILTER (WHERE expires_at < NOW()) as expired_pending,
                    COUNT(*) FILTER (
                        WHERE status = 'pending' AND expires_at >= NOW()
                    ) as pending_requests,
                    EXTRACT(EPOCH FROM (NOW() - MIN(created_at)))::FLOAT / 60.0
                        FILTER (WHERE status = 'pending' AND expires_at >= NOW())
                        as oldest_pending_age_minutes,
                    EXTRACT(EPOCH FROM (NOW() - MIN(expires_at)))::FLOAT / 3600.0
                        FILTER (WHERE expires_at < NOW())
                        as oldest_expired_age_hours
                FROM idempotency_keys
            """
            )

            if not stats:
                set_idempotency_metrics(0, 0, 0, None, None)
                return IdempotencyHealth(status="ok", total_keys=0)

            total = stats["total_keys"] or 0
            expired = stats["expired_pending"] or 0
            pending = stats["pending_requests"] or 0
            oldest_pending = stats["oldest_pending_age_minutes"]
            oldest_expired = stats["oldest_expired_age_hours"]

            # Update Prometheus metrics
            set_idempotency_metrics(
                total, expired, pending, oldest_pending, oldest_expired
            )

            # Determine status based on hygiene thresholds
            status = "ok"

            # Expired keys not being pruned is a warning
            if expired > 100:
                status = "degraded"
            if expired > 1000:
                status = "error"

            # Very old expired keys indicate pg_cron failure
            if oldest_expired and oldest_expired > 48:  # > 48 hours old
                status = "error"

            # Stuck pending requests (> 30 min) indicate issues
            if oldest_pending and oldest_pending > 30:
                status = "degraded"

            return IdempotencyHealth(
                status=status,
                total_keys=total,
                expired_pending=expired,
                pending_requests=pending,
                oldest_pending_age_minutes=oldest_pending,
                oldest_expired_age_hours=oldest_expired,
            )
    except Exception as e:
        # Table might not exist yet
        if "does not exist" in str(e):
            return IdempotencyHealth(
                status="ok",
                details={"message": "Idempotency not configured (table missing)"},
            )
        return IdempotencyHealth(status="error", error=str(e)[:200])


async def collect_system_health(settings: Settings) -> SystemHealthSnapshot:
    """
    Collect complete system health snapshot.

    This is the core function - endpoints just wrap this.
    """
    from app import __version__

    # Run all checks concurrently
    (
        db,
        qdrant,
        llm,
        ingestion,
        sse,
        redis,
        retention,
        idempotency,
        tunes,
        pine_discovery,
    ) = await asyncio.gather(
        _check_database(),
        _check_qdrant(settings),
        _check_llm(),
        _check_ingestion(),
        _check_sse(settings),
        _check_redis(settings),
        _check_retention(),
        _check_idempotency(),
        _check_tunes(),
        _check_pine_discovery(),
        return_exceptions=True,
    )

    # Handle any exceptions that slipped through
    def safe_result(result, default_cls):
        if isinstance(result, Exception):
            return default_cls(status="error", error=str(result)[:200])
        return result

    db = safe_result(db, DBHealth)
    qdrant = safe_result(qdrant, QdrantHealth)
    llm = safe_result(llm, LLMHealth)
    ingestion = safe_result(ingestion, IngestionHealth)
    sse = safe_result(sse, SSEHealth)
    # Redis can be None if not configured
    if isinstance(redis, Exception):
        redis = RedisHealth(status="error", error=str(redis)[:200])
    retention = safe_result(retention, RetentionHealth)
    idempotency = safe_result(idempotency, IdempotencyHealth)
    tunes = safe_result(tunes, TuneHealth)
    pine_discovery = safe_result(pine_discovery, PineDiscoveryHealth)

    # Calculate overall status (only include Redis if configured)
    components = [db, qdrant, llm, sse, retention, idempotency, tunes, pine_discovery]
    if redis is not None:
        components.append(redis)
    statuses = [c.status for c in components]

    ok_count = statuses.count("ok")
    degraded_count = statuses.count("degraded")
    error_count = statuses.count("error")

    if error_count > 0:
        overall = "error"
    elif degraded_count > 0:
        overall = "degraded"
    else:
        overall = "ok"

    # Collect issues
    issues = []
    if db.status != "ok":
        issues.append(f"Database: {db.error or db.status}")
    if qdrant.status != "ok":
        issues.append(f"Qdrant: {qdrant.error or qdrant.status}")
    if llm.status == "error":
        issues.append(f"LLM: {llm.error or 'error'}")
    if ingestion.status != "ok":
        issues.append("Ingestion: recent failures detected")
    if sse.status != "ok":
        issues.append(f"SSE: {sse.error or sse.status}")
    if redis is not None and redis.status != "ok":
        issues.append(f"Redis: {redis.error or redis.status}")
    if retention.status != "ok":
        issues.append(f"Retention: {retention.error or 'failing'}")
    if idempotency.status != "ok":
        issues.append(
            f"Idempotency: {idempotency.expired_pending} expired keys pending"
        )
    if tunes.status != "ok":
        issues.append(f"Tunes: {tunes.error or tunes.status}")
    if pine_discovery.status != "ok":
        issues.append(f"Pine Discovery: {pine_discovery.error or 'stale scripts'}")

    return SystemHealthSnapshot(
        overall_status=overall,
        version=__version__,
        git_sha=settings.git_sha,
        database=db,
        qdrant=qdrant,
        llm=llm,
        ingestion=ingestion,
        sse=sse,
        redis=redis,
        retention=retention,
        idempotency=idempotency,
        tunes=tunes,
        pine_discovery=pine_discovery,
        components_ok=ok_count,
        components_degraded=degraded_count,
        components_error=error_count,
        issues=issues,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health.json", response_model=SystemHealthSnapshot)
async def system_health_json(
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
) -> SystemHealthSnapshot:
    """
    Machine-readable system health snapshot.

    Returns complete health status for all subsystems.
    Suitable for automation, CI checks, and alerting integrations.
    """
    return await collect_system_health(settings)


@router.get("/health", response_class=HTMLResponse)
async def system_health_html(
    request: Request,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
):
    """
    Human-friendly system health dashboard.

    One page that answers "what's broken?" without opening logs.
    Auto-refreshes every 30 seconds.
    """
    snapshot = await collect_system_health(settings)

    # Status badge colors
    def status_color(s: str) -> str:
        return {"ok": "green", "degraded": "orange", "error": "red"}.get(s, "gray")

    def status_icon(s: str) -> str:
        return {"ok": "&#10003;", "degraded": "&#9888;", "error": "&#10007;"}.get(
            s, "?"
        )

    # Format datetime
    def fmt_dt(dt: Optional[datetime]) -> str:
        if not dt:
            return "-"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    def fmt_ms(ms: Optional[float]) -> str:
        if ms is None:
            return "-"
        return f"{ms:.1f}ms"

    def metric(label: str, value: str, error: bool = False) -> str:
        style = ' style="color:#f87171;"' if error else ""
        return (
            f'<div class="metric">'
            f'<span class="metric-label">{label}</span>'
            f'<span class="metric-value"{style}>{value}</span>'
            f"</div>"
        )

    def card_header(title: str, status: str) -> str:
        return (
            f'<div class="card-header">'
            f'<span class="card-title">{title}</span>'
            f'<span class="badge {status}">{status_icon(status)} {status}</span>'
            f"</div>"
        )

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>System Health - Trading RAG</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0; padding: 20px;
        }}
        .header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #334155;
        }}
        .header h1 {{ font-size: 1.5rem; }}
        .overall {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 16px; border-radius: 8px;
            font-weight: 600; font-size: 1.1rem;
        }}
        .overall.ok {{ background: #166534; }}
        .overall.degraded {{ background: #92400e; }}
        .overall.error {{ background: #991b1b; }}
        .grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: #1e293b; border-radius: 12px; padding: 16px;
            border: 1px solid #334155;
        }}
        .card-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 12px;
        }}
        .card-title {{ font-weight: 600; font-size: 1.1rem; }}
        .badge {{
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 10px; border-radius: 12px; font-size: 0.85rem;
        }}
        .badge.ok {{ background: #166534; }}
        .badge.degraded {{ background: #92400e; }}
        .badge.error {{ background: #991b1b; }}
        .badge.unknown {{ background: #475569; }}
        .metric {{
            display: flex; justify-content: space-between;
            padding: 6px 0; border-bottom: 1px solid #334155;
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #94a3b8; }}
        .metric-value {{ font-family: monospace; }}
        .issues {{
            background: #7f1d1d; border-radius: 8px; padding: 12px;
            margin-bottom: 20px;
        }}
        .issues h3 {{ margin-bottom: 8px; }}
        .issues ul {{ margin-left: 20px; }}
        .meta {{
            margin-top: 20px; padding-top: 15px; border-top: 1px solid #334155;
            color: #64748b; font-size: 0.85rem;
            display: flex; justify-content: space-between;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>System Health</h1>
        <div class="overall {snapshot.overall_status}">
            {status_icon(snapshot.overall_status)} {snapshot.overall_status.upper()}
        </div>
    </div>
"""

    # Issues section
    if snapshot.issues:
        html += """<div class="issues"><h3>Issues</h3><ul>"""
        for issue in snapshot.issues:
            html += f"<li>{issue}</li>"
        html += "</ul></div>"

    html += '<div class="grid">'

    # Database card
    db = snapshot.database
    html += '<div class="card">'
    html += card_header("Database", db.status)
    html += metric("Pool Size", str(db.pool_size or "-"))
    html += metric("Available", str(db.pool_available or "-"))
    html += metric("Acquire Time", fmt_ms(db.pool_acquire_ms))
    html += metric("Query Latency", fmt_ms(db.query_latency_ms))
    if db.error:
        html += metric("Error", db.error, error=True)
    html += "</div>"

    # Qdrant card
    qd = snapshot.qdrant
    html += '<div class="card">'
    html += card_header("Qdrant", qd.status)
    html += metric("Collection", qd.collection or "-")
    html += metric("Vectors", f"{qd.vectors_count:,}" if qd.vectors_count else "-")
    html += metric("Segments", str(qd.segments_count or "-"))
    html += metric("Latency", fmt_ms(qd.latency_ms))
    if qd.error:
        html += metric("Error", qd.error, error=True)
    html += "</div>"

    # LLM card
    llm_h = snapshot.llm
    html += '<div class="card">'
    html += card_header("LLM", llm_h.status)
    html += metric("Configured", "Yes" if llm_h.provider_configured else "No")
    html += metric("Provider", llm_h.provider or "-")
    html += metric("Model", llm_h.model or "-")
    html += metric("Degraded (1h)", str(llm_h.degraded_count_1h))
    html += metric("Errors (1h)", str(llm_h.error_count_1h))
    html += "</div>"

    # SSE card
    sse_h = snapshot.sse
    html += '<div class="card">'
    html += card_header("SSE Events", sse_h.status)
    html += metric("Bus Type", sse_h.bus_type)
    html += metric("Subscribers", str(sse_h.subscribers))
    html += metric("Events (1h)", str(sse_h.events_published_1h))
    html += metric("Buffer Size", str(sse_h.buffer_size))
    html += metric("Queue Drops", str(sse_h.queue_drops_1h))
    html += "</div>"

    # Redis card (only shown when configured)
    if snapshot.redis is not None:
        rd = snapshot.redis
        html += '<div class="card">'
        html += card_header("Redis", rd.status)
        html += metric("Configured", "Yes" if rd.configured else "No")
        html += metric("Connected", "Yes" if rd.connected else "No")
        html += metric("Ping Latency", fmt_ms(rd.ping_latency_ms))
        html += metric("Stream Count", str(rd.stream_count))
        html += metric("Total Events", f"{rd.total_stream_length:,}")
        if rd.error:
            html += metric("Error", rd.error, error=True)
        html += "</div>"

    # Retention card
    ret = snapshot.retention
    html += '<div class="card">'
    html += card_header("Retention", ret.status)
    html += metric("pg_cron", "Yes" if ret.pg_cron_available else "No")
    html += metric("Last Run", fmt_dt(ret.last_run_at))
    html += metric("Last Job", ret.last_run_job or "-")
    html += metric("Rows Deleted", f"{ret.rows_deleted_last:,}")
    html += metric("Consecutive Fails", str(ret.consecutive_failures))
    html += "</div>"

    # Idempotency card
    idem = snapshot.idempotency
    html += '<div class="card">'
    html += card_header("Idempotency", idem.status)
    html += metric("Total Keys", f"{idem.total_keys:,}")
    html += metric(
        "Expired Pending", str(idem.expired_pending), error=idem.expired_pending > 100
    )
    html += metric("Pending Requests", str(idem.pending_requests))
    if idem.oldest_pending_age_minutes:
        html += metric("Oldest Pending", f"{idem.oldest_pending_age_minutes:.1f} min")
    if idem.oldest_expired_age_hours:
        html += metric(
            "Oldest Expired",
            f"{idem.oldest_expired_age_hours:.1f} hrs",
            error=idem.oldest_expired_age_hours > 48,
        )
    if idem.error:
        html += metric("Error", idem.error, error=True)
    html += "</div>"

    # Tunes card
    tn = snapshot.tunes
    html += '<div class="card">'
    html += card_header("Backtest Tunes", tn.status)
    html += metric("Active", str(tn.active_tunes))
    html += metric("Completed (24h)", str(tn.completed_24h))
    html += metric("Failed (24h)", str(tn.failed_24h))
    html += metric("Avg Duration", fmt_ms(tn.avg_duration_ms))
    html += "</div>"

    # Ingestion card
    ing = snapshot.ingestion
    html += '<div class="card">'
    html += card_header("Ingestion", ing.status)
    html += metric("YouTube Last OK", fmt_dt(ing.youtube_last_success))
    html += metric("YouTube Last Fail", fmt_dt(ing.youtube_last_failure))
    html += metric("PDF Last OK", fmt_dt(ing.pdf_last_success))
    html += metric("Pine Last OK", fmt_dt(ing.pine_last_success))
    html += "</div>"

    # Pine Discovery card
    pd = snapshot.pine_discovery
    html += '<div class="card">'
    html += card_header("Pine Discovery", pd.status)
    html += metric("Total Scripts", f"{pd.total_scripts:,}")
    html += metric(
        "Pending Ingest", f"{pd.pending_ingest:,}", error=pd.pending_ingest > 50
    )
    stale_pct = f"{pd.stale_ratio:.1%}"
    stale_error = pd.stale_ratio > 0.5
    html += metric(
        f"Stale ({pd.stale_cutoff_days}d+)",
        f"{pd.stale_scripts:,} ({stale_pct})",
        error=stale_error,
    )
    html += metric(
        f"Ingest Errors ({pd.window_ingest_errors_hours}h)",
        str(pd.recent_ingest_errors),
        error=pd.recent_ingest_errors > 0,
    )

    # Format timestamps from gauges
    def fmt_ts(ts: Optional[float]) -> str:
        if ts is None:
            return "-"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    html += metric("Last Run", fmt_ts(pd.last_run_ts))
    html += metric("Last Success", fmt_ts(pd.last_success_ts))
    # Show ingest status breakdown
    if pd.scripts_by_ingest_status:
        for ingest_status, count in pd.scripts_by_ingest_status.items():
            html += metric(f"  {ingest_status}", str(count))
    # Show notes if any
    if pd.notes:
        html += metric("Notes", "; ".join(pd.notes), error=True)
    if pd.error:
        html += metric("Error", pd.error, error=True)
    html += "</div>"

    html += "</div>"  # grid

    # Footer
    html += f"""
    <div class="meta">
        <span>Version: {snapshot.version} | SHA: {snapshot.git_sha or 'dev'}</span>
        <span>Generated: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
    </div>
</body>
</html>
"""

    return HTMLResponse(content=html)
