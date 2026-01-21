"""Health check functions for system health dashboard.

All check functions are pure (no globals) - pool is passed as parameter.
"""

import asyncio
import time
from typing import Any, Optional

import httpx

from app.admin.services.health_models import (
    DBHealth,
    IdempotencyHealth,
    IngestionHealth,
    LLMHealth,
    PineDiscoveryHealth,
    PinePollerHealth,
    PineReposHealth,
    QdrantHealth,
    RedisHealth,
    RetentionHealth,
    SSEHealth,
    SystemHealthSnapshot,
    TuneHealth,
)
from app.config import Settings


async def check_database(pool: Any) -> DBHealth:
    """Check database connectivity and pool health."""
    if pool is None:
        return DBHealth(status="error", error="Pool not initialized")

    start = time.perf_counter()
    try:
        # Check pool stats
        pool_size = pool.get_size()
        pool_available = pool.get_idle_size()

        # Test acquire + query
        acquire_start = time.perf_counter()
        async with pool.acquire() as conn:
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


async def check_qdrant(settings: Settings) -> QdrantHealth:
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


async def check_llm() -> LLMHealth:
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


async def check_ingestion(pool: Any) -> IngestionHealth:
    """Check ingestion pipeline status."""
    if pool is None:
        return IngestionHealth(status="unknown")

    try:
        async with pool.acquire() as conn:
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


async def check_sse(settings: Settings) -> SSEHealth:
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


async def check_redis(settings: Settings) -> Optional[RedisHealth]:
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


async def check_retention(pool: Any) -> RetentionHealth:
    """Check retention job status from logs."""
    if pool is None:
        return RetentionHealth(status="unknown", error="Pool not initialized")

    try:
        async with pool.acquire() as conn:
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


async def check_tunes(pool: Any) -> TuneHealth:
    """Check backtest tune health."""
    if pool is None:
        return TuneHealth(status="unknown", error="Pool not initialized")

    try:
        async with pool.acquire() as conn:
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


async def check_pine_repos(pool: Any) -> PineReposHealth:
    """Check Pine repository registry health."""
    from app.routers.metrics import set_pine_repos_metrics

    if pool is None:
        return PineReposHealth(status="unknown", error="Pool not initialized")

    try:
        async with pool.acquire() as conn:
            # Get repo health stats
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE enabled = TRUE) as enabled,
                    COUNT(*) FILTER (WHERE last_pull_ok = FALSE) as pull_failed,
                    COUNT(*) FILTER (
                        WHERE last_scan_at IS NOT NULL
                        AND last_scan_at < NOW() - INTERVAL '7 days'
                    ) as stale,
                    EXTRACT(EPOCH FROM (NOW() - MIN(last_scan_at))) / 3600.0 as oldest_scan_hours
                FROM pine_repos
            """
            )

            if not stats or stats["total"] == 0:
                return PineReposHealth(
                    status="ok",
                    repos_total=0,
                    details={"message": "No repos registered yet"},
                )

            total = stats["total"] or 0
            enabled = stats["enabled"] or 0
            pull_failed = stats["pull_failed"] or 0
            stale = stats["stale"] or 0
            oldest_hours = stats["oldest_scan_hours"]

            # Determine status
            status = "ok"

            # Degraded if any repos have pull failures
            if pull_failed > 0:
                status = "degraded"

            # Degraded if stale repos > 50% of enabled
            if enabled > 0 and stale > enabled / 2:
                status = "degraded"

            # Error if all enabled repos have pull failures
            if enabled > 0 and pull_failed == enabled:
                status = "error"

            # Update Prometheus metrics on successful DB query
            set_pine_repos_metrics(total, enabled, pull_failed, stale, oldest_hours)

            return PineReposHealth(
                status=status,
                repos_total=total,
                repos_enabled=enabled,
                repos_pull_failed=pull_failed,
                repos_stale=stale,
                oldest_scan_age_hours=oldest_hours,
            )
    except Exception as e:
        # Table might not exist yet
        if "does not exist" in str(e):
            return PineReposHealth(
                status="ok",
                details={"message": "Pine repos not configured (table missing)"},
            )
        return PineReposHealth(status="error", error=str(e)[:200])


async def check_pine_poller(settings: Settings) -> PinePollerHealth:
    """Check Pine repository poller health."""
    from app.services.pine.poller import get_poller

    poller = get_poller()

    if not settings.pine_repo_poll_enabled:
        return PinePollerHealth(
            status="ok",
            enabled=False,
            running=False,
            details={"message": "Polling disabled (PINE_REPO_POLL_ENABLED=false)"},
        )

    if poller is None:
        return PinePollerHealth(
            status="error",
            enabled=True,
            running=False,
            error="Poller not initialized",
        )

    try:
        health = await poller.get_health()

        # Determine status
        status = "ok"

        # Degraded if poller should be running but isn't
        if health.enabled and not health.running:
            status = "error"

        # Degraded if repos are due but no recent runs
        if health.repos_due_count > 5:
            status = "degraded"

        # Degraded if last run had errors
        if health.last_run_errors > 0:
            status = "degraded"

        return PinePollerHealth(
            status=status,
            enabled=health.enabled,
            running=health.running,
            last_run_at=health.last_run_at,
            last_run_repos_scanned=health.last_run_repos_scanned,
            last_run_errors=health.last_run_errors,
            repos_due_count=health.repos_due_count,
            poll_interval_minutes=health.poll_interval_minutes,
            poll_max_concurrency=health.poll_max_concurrency,
            poll_tick_seconds=health.poll_tick_seconds,
        )

    except Exception as e:
        return PinePollerHealth(
            status="error",
            enabled=settings.pine_repo_poll_enabled,
            error=str(e)[:200],
        )


async def check_pine_discovery(pool: Any) -> PineDiscoveryHealth:
    """Check Pine script discovery health and update Prometheus metrics."""
    from app.routers.metrics import (
        PINE_DISCOVERY_LAST_RUN_TIMESTAMP,
        PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP,
        set_pine_pending_ingest,
        set_pine_scripts_metrics,
    )

    # Time anchors (constants for documentation/alerting)
    STALE_CUTOFF_DAYS = 7
    INGEST_ERROR_WINDOW_HOURS = 24

    if pool is None:
        return PineDiscoveryHealth(status="unknown", error="Pool not initialized")

    try:
        async with pool.acquire() as conn:
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

            # Get pending ingest count
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

            # Get stale scripts count (not updated in STALE_CUTOFF_DAYS)
            stale_scripts = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*)
                FROM strategy_scripts
                WHERE status != 'archived'
                  AND updated_at < NOW() - INTERVAL '{STALE_CUTOFF_DAYS} days'
            """
                )
                or 0
            )

            # Get recent ingest errors (last INGEST_ERROR_WINDOW_HOURS)
            recent_ingest_errors = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*)
                FROM strategy_scripts
                WHERE ingest_status = 'error'
                  AND updated_at > NOW() - INTERVAL '{INGEST_ERROR_WINDOW_HOURS} hours'
            """
                )
                or 0
            )

            # Get last discovery timestamp
            last_discovery = await conn.fetchval(
                """
                SELECT MAX(created_at)
                FROM strategy_scripts
            """
            )

            # Calculate stale ratio
            active_total = sum(v for k, v in status_counts.items() if k != "archived")
            stale_ratio = stale_scripts / active_total if active_total > 0 else 0.0

            # Update Prometheus gauges
            set_pine_scripts_metrics(status_counts)
            set_pine_pending_ingest(pending_ingest)

            # Determine status and collect notes
            status = "ok"
            notes: list[str] = []

            # Degraded if stale ratio > 20%
            if stale_ratio > 0.2:
                status = "degraded"
                notes.append(f"Stale ratio {stale_ratio:.1%} > 20%")

            # Degraded if recent ingest errors
            if recent_ingest_errors > 0:
                status = "degraded"
                notes.append(
                    f"{recent_ingest_errors} ingest errors in {INGEST_ERROR_WINDOW_HOURS}h"
                )

            # Error if stale ratio > 50%
            if stale_ratio > 0.5:
                status = "error"
                notes.append(f"Stale ratio {stale_ratio:.1%} > 50%")

            # Get timestamps from Prometheus gauges
            last_run_ts = None
            last_success_ts = None
            try:
                # Prometheus gauges store values as floats
                last_run_ts = int(
                    PINE_DISCOVERY_LAST_RUN_TIMESTAMP._value.get()  # type: ignore
                )
                last_success_ts = int(
                    PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP._value.get()  # type: ignore
                )
            except Exception:
                pass

            return PineDiscoveryHealth(
                status=status,
                scripts_by_status=status_counts,
                scripts_by_ingest_status=ingest_status_counts,
                total_scripts=total,
                pending_ingest=pending_ingest,
                stale_scripts=stale_scripts,
                stale_ratio=stale_ratio,
                stale_cutoff_days=STALE_CUTOFF_DAYS,
                recent_ingest_errors=recent_ingest_errors,
                window_ingest_errors_hours=INGEST_ERROR_WINDOW_HOURS,
                last_discovery_at=last_discovery,
                last_run_ts=last_run_ts,
                last_success_ts=last_success_ts,
                notes=notes,
            )
    except Exception as e:
        # Table might not exist yet
        if "does not exist" in str(e):
            return PineDiscoveryHealth(
                status="ok",
                details={"message": "Pine discovery not configured (table missing)"},
            )
        return PineDiscoveryHealth(status="error", error=str(e)[:200])


async def check_idempotency(pool: Any) -> IdempotencyHealth:
    """Check idempotency key hygiene status."""
    if pool is None:
        return IdempotencyHealth(status="unknown", error="Pool not initialized")

    try:
        async with pool.acquire() as conn:
            # Get idempotency stats
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (
                        WHERE status = 'pending'
                        AND created_at < NOW() - INTERVAL '5 minutes'
                    ) as expired_pending,
                    MIN(CASE WHEN status = 'pending' THEN created_at END) as oldest_pending,
                    MIN(CASE WHEN status = 'pending'
                        AND created_at < NOW() - INTERVAL '5 minutes'
                        THEN created_at END) as oldest_expired
                FROM idempotency_keys
            """
            )

            if not stats or stats["total"] == 0:
                return IdempotencyHealth(
                    status="ok",
                    total_keys=0,
                    details={"message": "No idempotency keys yet"},
                )

            total = stats["total"] or 0
            pending = stats["pending"] or 0
            expired_pending = stats["expired_pending"] or 0

            # Calculate ages
            oldest_pending_age = None
            oldest_expired_age = None

            if stats["oldest_pending"]:
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc)
                oldest_pending_age = (
                    now - stats["oldest_pending"].replace(tzinfo=timezone.utc)
                ).total_seconds() / 60  # minutes

            if stats["oldest_expired"]:
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc)
                oldest_expired_age = (
                    now - stats["oldest_expired"].replace(tzinfo=timezone.utc)
                ).total_seconds() / 3600  # hours

            # Determine status
            status = "ok"
            if expired_pending > 0:
                status = "degraded"
            if expired_pending > 10:
                status = "error"

            return IdempotencyHealth(
                status=status,
                total_keys=total,
                expired_pending=expired_pending,
                pending_requests=pending,
                oldest_pending_age_minutes=oldest_pending_age,
                oldest_expired_age_hours=oldest_expired_age,
            )
    except Exception as e:
        # Table might not exist yet
        if "does not exist" in str(e):
            return IdempotencyHealth(
                status="ok",
                details={"message": "Idempotency not configured (table missing)"},
            )
        return IdempotencyHealth(status="error", error=str(e)[:200])


async def collect_system_health(settings: Settings, pool: Any) -> SystemHealthSnapshot:
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
        pine_repos,
        pine_discovery,
        pine_poller,
    ) = await asyncio.gather(
        check_database(pool),
        check_qdrant(settings),
        check_llm(),
        check_ingestion(pool),
        check_sse(settings),
        check_redis(settings),
        check_retention(pool),
        check_idempotency(pool),
        check_tunes(pool),
        check_pine_repos(pool),
        check_pine_discovery(pool),
        check_pine_poller(settings),
        return_exceptions=True,
    )

    # Handle any exceptions that slipped through
    # mypy: asyncio.gather with return_exceptions=True returns T | BaseException union
    def safe_result(result: Any, default_cls: type) -> Any:
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
    pine_repos = safe_result(pine_repos, PineReposHealth)
    pine_discovery = safe_result(pine_discovery, PineDiscoveryHealth)
    pine_poller = safe_result(pine_poller, PinePollerHealth)

    # Calculate overall status (only include Redis if configured)
    components = [
        db,
        qdrant,
        llm,
        sse,
        retention,
        idempotency,
        tunes,
        pine_repos,
        pine_discovery,
    ]
    if pine_poller.enabled:
        components.append(pine_poller)
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
    if redis is not None and redis.status != "ok":  # type: ignore[union-attr]
        issues.append(f"Redis: {redis.error or redis.status}")  # type: ignore[union-attr]
    if retention.status != "ok":
        issues.append(f"Retention: {retention.error or 'failing'}")
    if idempotency.status != "ok":
        issues.append(
            f"Idempotency: {idempotency.expired_pending} expired keys pending"
        )
    if tunes.status != "ok":
        issues.append(f"Tunes: {tunes.error or tunes.status}")
    if pine_repos.status != "ok":
        msg = (
            f"{pine_repos.repos_pull_failed} pull failures"
            if pine_repos.repos_pull_failed
            else pine_repos.status
        )
        issues.append(f"Pine Repos: {pine_repos.error or msg}")
    if pine_discovery.status != "ok":
        issues.append(f"Pine Discovery: {pine_discovery.error or 'stale scripts'}")
    if pine_poller.enabled and pine_poller.status != "ok":
        issues.append(
            f"Pine Poller: {pine_poller.error or f'{pine_poller.repos_due_count} repos due'}"
        )

    return SystemHealthSnapshot(
        overall_status=overall,
        version=__version__,
        git_sha=settings.git_sha,
        database=db,
        qdrant=qdrant,
        llm=llm,
        ingestion=ingestion,
        sse=sse,
        redis=redis,  # type: ignore[arg-type]
        retention=retention,
        idempotency=idempotency,
        tunes=tunes,
        pine_repos=pine_repos,
        pine_discovery=pine_discovery,
        pine_poller=pine_poller,
        components_ok=ok_count,
        components_degraded=degraded_count,
        components_error=error_count,
        issues=issues,
    )
