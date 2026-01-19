"""Admin service layer - pure functions for query/response building.

Public API:
- Health checks: collect_system_health
- Coverage queries: fetch_weak_coverage_runs, hydrate_strategy_cards, etc.
- Models: SystemHealthSnapshot, WeakCoverageItem, etc.

Internal helpers are NOT exported to prevent coupling.
"""

# Health check services
from app.admin.services.health_checks import collect_system_health

# Coverage query services
from app.admin.services.coverage_queries import (
    build_template_items,
    collect_candidate_ids,
    fetch_weak_coverage_runs,
    hydrate_strategy_cards,
    hydrate_strategy_cards_for_template,
    parse_json_field,
)

# Models (re-export for convenience)
from app.admin.services.health_models import SystemHealthSnapshot
from app.admin.services.coverage_models import (
    CoverageStatusEnum,
    WeakCoverageItem,
    WeakCoverageResponse,
)

__all__ = [
    # Health
    "collect_system_health",
    "SystemHealthSnapshot",
    # Coverage
    "fetch_weak_coverage_runs",
    "hydrate_strategy_cards",
    "hydrate_strategy_cards_for_template",
    "collect_candidate_ids",
    "build_template_items",
    "parse_json_field",
    "CoverageStatusEnum",
    "WeakCoverageItem",
    "WeakCoverageResponse",
]
