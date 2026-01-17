"""KB Trials Admin Service - orchestration layer for admin endpoints.

This package extracts business logic from kb_trials router for better testability.
All functions return dicts suitable for JSON responses.

Usage:
    from app.services.kb_trials_admin import compute_kb_trials_stats
    from app.services.kb_trials_admin import get_qdrant_collections
"""

from app.services.kb_trials_admin.stats import (
    compute_kb_trials_stats,
    compute_ingestion_status,
)
from app.services.kb_trials_admin.qdrant import get_qdrant_collections
from app.services.kb_trials_admin.warnings import get_top_warnings
from app.services.kb_trials_admin.samples import get_trial_samples
from app.services.kb_trials_admin.promotions import compute_promotion_preview

__all__ = [
    "compute_kb_trials_stats",
    "compute_ingestion_status",
    "get_qdrant_collections",
    "get_top_warnings",
    "get_trial_samples",
    "compute_promotion_preview",
]
