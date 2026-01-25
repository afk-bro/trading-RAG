"""System health dashboard for operations.

Single page that answers "what's broken?" without opening logs.

Endpoints:
- GET /admin/system/health - HTML dashboard with status cards
- GET /admin/system/health.json - Machine-readable JSON
"""

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from app.admin.services.health_checks import collect_system_health
from app.admin.services.health_models import SystemHealthSnapshot
from app.config import Settings, get_settings
from app.deps.security import require_admin_token

router = APIRouter(prefix="/system", tags=["admin-system"])
logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

AUTO_REFRESH_SECONDS = 30

STATUS_COLORS = {
    "ok": "green",
    "degraded": "orange",
    "error": "red",
}

STATUS_ICONS = {
    "ok": "&#10003;",
    "degraded": "&#9888;",
    "error": "&#10007;",
}

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


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
    return await collect_system_health(settings, _db_pool)


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
    snapshot = await collect_system_health(settings, _db_pool)

    def status_color(s: str) -> str:
        return STATUS_COLORS.get(s, "gray")

    def status_icon(s: str) -> str:
        return STATUS_ICONS.get(s, "?")

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
    <meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">
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

    # Pine Repos card
    pr = snapshot.pine_repos
    html += '<div class="card">'
    html += card_header("Pine Repos", pr.status)
    html += metric("Total Repos", str(pr.repos_total))
    html += metric("Enabled", str(pr.repos_enabled))
    html += metric(
        "Pull Failures", str(pr.repos_pull_failed), error=pr.repos_pull_failed > 0
    )
    html += metric("Stale (7d+)", str(pr.repos_stale), error=pr.repos_stale > 0)
    if pr.oldest_scan_age_hours:
        html += metric("Oldest Scan", f"{pr.oldest_scan_age_hours:.1f} hrs")
    if pr.error:
        html += metric("Error", pr.error, error=True)
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

    # Pine Poller card
    pp = snapshot.pine_poller
    html += '<div class="card">'
    html += card_header("Pine Poller", pp.status)
    html += metric("Enabled", "Yes" if pp.enabled else "No")
    html += metric("Running", "Yes" if pp.running else "No")
    if pp.last_run_at:
        html += metric("Last Run", fmt_dt(pp.last_run_at))
    if pp.last_run_repos_scanned is not None:
        html += metric("Repos Scanned", str(pp.last_run_repos_scanned))
    if pp.last_run_errors is not None:
        html += metric(
            "Last Errors", str(pp.last_run_errors), error=pp.last_run_errors > 0
        )
    if pp.repos_due_count is not None:
        html += metric(
            "Repos Due", str(pp.repos_due_count), error=pp.repos_due_count > 5
        )
    if pp.error:
        html += metric("Error", pp.error, error=True)
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
