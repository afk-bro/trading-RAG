"""Admin UI router composition - assembles all admin sub-routers."""

from pathlib import Path

import structlog
from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

from app.admin import analytics as analytics_router
from app.admin import alerts as alerts_router
from app.admin import ops_alerts as ops_alerts_router
from app.admin import coverage as coverage_router
from app.admin import retention as retention_router
from app.admin import events as events_router
from app.admin import system_health as system_health_router
from app.admin import ops as ops_router
from app.admin import ops_alerts as ops_alerts_router
from app.admin import trade_events as trade_events_router
from app.admin import evals as evals_router
from app.admin import kb_admin as kb_admin_router
from app.admin import kb_trials as kb_trials_router
from app.admin import backtests as backtests_router
from app.admin import run_plans as run_plans_router
from app.admin import jobs as jobs_router
from app.admin import pine_discovery as pine_discovery_router
from app.admin import pine_repos as pine_repos_router
from app.admin import ingest as ingest_router

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)

# =============================================================================
# Include All Sub-Routers
# =============================================================================

# Analytics
router.include_router(analytics_router.router)

# Alerts
router.include_router(alerts_router.router)

# Ops Alerts
router.include_router(ops_alerts_router.router)

# Coverage
router.include_router(coverage_router.router)

# Retention
router.include_router(retention_router.router)

# Events (SSE)
router.include_router(events_router.router)

# System Health
router.include_router(system_health_router.router)

# Ops
router.include_router(ops_router.router)

# Ops Alerts (operational alerting - health, coverage, drift)
router.include_router(ops_alerts_router.router)

# Trade Events
router.include_router(trade_events_router.router)

# Evals
router.include_router(evals_router.router)

# KB Admin
router.include_router(kb_admin_router.router)

# KB Trials
router.include_router(kb_trials_router.router)

# Backtests
router.include_router(backtests_router.router)

# Run Plans
router.include_router(run_plans_router.router)

# Jobs
router.include_router(jobs_router.router)

# Pine Discovery
router.include_router(pine_discovery_router.router)

# Pine Repos (GitHub repository management)
router.include_router(pine_repos_router.router)

# Ingest (unified content ingestion UI)
router.include_router(ingest_router.router)

# =============================================================================
# Shared Resources
# =============================================================================

# Setup Jinja2 templates (shared by sub-routers that import from here)
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


# =============================================================================
# Pool Propagation
# =============================================================================


def set_db_pool(pool):
    """Set the database pool for all admin routes."""
    analytics_router.set_db_pool(pool)
    alerts_router.set_db_pool(pool)
    ops_alerts_router.set_db_pool(pool)
    coverage_router.set_db_pool(pool)
    retention_router.set_db_pool(pool)
    system_health_router.set_db_pool(pool)
    ops_router.set_db_pool(pool)
    ops_alerts_router.set_db_pool(pool)
    trade_events_router.set_db_pool(pool)
    evals_router.set_db_pool(pool)
    kb_admin_router.set_db_pool(pool)
    kb_trials_router.set_db_pool(pool)
    backtests_router.set_db_pool(pool)
    run_plans_router.set_db_pool(pool)
    jobs_router.set_db_pool(pool)
    pine_discovery_router.set_db_pool(pool)
    pine_repos_router.set_db_pool(pool)
    ingest_router.set_db_pool(pool)


def set_qdrant_client(client):
    """Set the Qdrant client for admin routes that need it."""
    kb_trials_router.set_qdrant_client(client)
    pine_repos_router.set_qdrant_client(client)
