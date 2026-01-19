"""Pydantic models for system health status."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


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


class PineReposHealth(ComponentHealth):
    """Pine repository registry health."""

    repos_total: int = 0
    repos_enabled: int = 0
    repos_pull_failed: int = 0
    repos_stale: int = 0  # Not scanned in 7+ days
    oldest_scan_age_hours: Optional[float] = None


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


class PinePollerHealth(ComponentHealth):
    """Pine repository polling background service health."""

    enabled: bool = False
    running: bool = False
    last_run_at: Optional[datetime] = None
    last_run_repos_scanned: int = 0
    last_run_errors: int = 0
    repos_due_count: int = 0
    poll_interval_minutes: int = 15
    poll_max_concurrency: int = 2
    poll_tick_seconds: int = 60


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
    pine_repos: PineReposHealth
    pine_discovery: PineDiscoveryHealth
    pine_poller: PinePollerHealth

    # Summary
    components_ok: int = 0
    components_degraded: int = 0
    components_error: int = 0
    issues: list[str] = Field(default_factory=list)
