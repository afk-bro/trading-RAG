"""Backtest router package.

This package contains the backtest API endpoints split into modules:
- runs.py: Backtest run endpoints (POST /run, GET /{run_id}, etc.)
- tunes.py: Parameter tuning endpoints (POST /tune, GET /tunes, etc.)
- wfo.py: Walk-forward optimization endpoints (POST /wfo, GET /wfo, etc.)
- chart.py: Chart data endpoints for visualization
- schemas.py: Pydantic request/response models
"""

from fastapi import APIRouter

from . import chart, runs, tunes, wfo, wfo_chart

# Create combined router with prefix
router = APIRouter(prefix="/backtests", tags=["backtests"])

# Include sub-routers
router.include_router(runs.router)
router.include_router(tunes.router)
router.include_router(wfo.router)
router.include_router(chart.router)
router.include_router(wfo_chart.router)


def set_db_pool(pool):
    """Set the database pool for all backtest routers."""
    runs.set_db_pool(pool)
    tunes.set_db_pool(pool)
    wfo.set_db_pool(pool)
    chart.set_db_pool(pool)
    wfo_chart.set_db_pool(pool)


# Export for backward compatibility
__all__ = ["router", "set_db_pool"]
