"""Admin endpoints for coverage gap inspection."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.admin.services.coverage_models import (
    CoverageStatusEnum,
    CoverageStatusUpdateRequest,
    CoverageStatusUpdateResponse,
    ExplainStrategyRequest,
    ExplainStrategyResponse,
    SeedCoverageResponse,
    WeakCoverageItem,
    WeakCoverageResponse,
)
from app.admin.services.coverage_queries import (
    build_template_items,
    collect_candidate_ids,
    fetch_weak_coverage_runs,
    get_default_workspace_id,
    get_workspace_from_run,
    hydrate_strategy_cards,
    hydrate_strategy_cards_for_template,
)
from app.admin.utils import parse_json_field, require_db_pool
from app.deps.security import require_admin_token
from app.schemas import StrategyCard

# Templates
_template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_template_dir))

router = APIRouter(prefix="/coverage", tags=["admin-coverage"])
logger = structlog.get_logger(__name__)

# =============================================================================
# Seed Fixture Constants
# =============================================================================

SEED_TOP_K = 10
SEED_TOTAL_SEARCHED = 100
SEED_WORKSPACE_NAME = "Dev Workspace"
SEED_WORKSPACE_SLUG = "dev"


# =============================================================================
# Database Pool Management
# =============================================================================

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for coverage routes."""
    global _db_pool
    _db_pool = pool


def _get_pool():
    """Get database pool, raising 503 if unavailable."""
    return require_db_pool(_db_pool, "Database")


# =============================================================================
# Template Context Builder
# =============================================================================


@dataclass
class CockpitTemplateContext:
    """Context for cockpit template rendering."""

    request: Request
    items: list = field(default_factory=list)
    strategy_cards: dict = field(default_factory=dict)
    missing_strategy_ids: list = field(default_factory=list)
    workspace_id: Optional[str] = None
    status_filter: str = "open"
    sort_by: str = "priority"
    admin_token: str = ""
    selected_run_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict for template rendering."""
        return {
            "request": self.request,
            "items": self.items,
            "strategy_cards": self.strategy_cards,
            "missing_strategy_ids": self.missing_strategy_ids,
            "workspace_id": self.workspace_id,
            "status_filter": self.status_filter,
            "sort_by": self.sort_by,
            "admin_token": self.admin_token,
            "selected_run_id": self.selected_run_id,
            "error": self.error,
        }


# =============================================================================
# Coverage Endpoints
# =============================================================================


@router.get(
    "/weak",
    response_model=WeakCoverageResponse,
    summary="List recent weak coverage runs",
    description="Returns match runs with weak coverage, shaped for cockpit UI.",
)
async def list_weak_coverage(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    since: Optional[str] = Query(
        None, description="ISO timestamp filter (e.g., 2026-01-01T00:00:00Z)"
    ),
    status: Optional[Literal["open", "acknowledged", "resolved", "all"]] = Query(
        None, description="Filter by status (default: open)"
    ),
    include_candidate_cards: bool = Query(
        True, description="Include hydrated strategy cards (default: true for cockpit)"
    ),
    _: bool = Depends(require_admin_token),
) -> WeakCoverageResponse:
    """
    List recent weak coverage runs for cockpit display.

    Returns rows shaped for UI with:
    - run_id, created_at, intent_signature
    - script_type (from filters)
    - weak_reason_codes[]
    - best_score, num_above_threshold
    - candidate_strategy_ids[] (point-in-time snapshot, ordered by recommendation)
    - query_preview (first ~120 chars)
    - source_ref (youtube:ID or doc:UUID)
    - coverage_status (open, acknowledged, resolved)
    - priority_score (higher = more urgent, sorted desc)

    By default (include_candidate_cards=true), also returns:
    - strategy_cards_by_id: {uuid: StrategyCard} for all candidates across items
    - missing_strategy_ids: IDs referenced but not found (deleted/archived)

    Filters:
    - status: open (default), acknowledged, resolved, or all
    - Hydration capped at 300 unique IDs to prevent payload bloat
    - Results sorted by priority_score descending (most actionable first)
    """
    pool = _get_pool()

    try:
        items = await fetch_weak_coverage_runs(
            pool=pool,
            workspace_id=workspace_id,
            limit=limit,
            since=since,
            status=status,
        )
    except Exception as e:
        logger.error("weak_coverage_list_failed", error=str(e))
        raise HTTPException(500, f"Failed to list weak coverage: {e}")

    response_items = [WeakCoverageItem(**item) for item in items]

    # Hydrate strategy cards (default on for cockpit)
    strategy_cards_by_id: Optional[dict[str, StrategyCard]] = None
    missing_strategy_ids: list[UUID] = []

    if include_candidate_cards and items:
        candidate_ids = collect_candidate_ids(items)
        if candidate_ids:
            strategy_cards_by_id, missing_strategy_ids = await hydrate_strategy_cards(
                pool, workspace_id, candidate_ids
            )

    return WeakCoverageResponse(
        items=response_items,
        count=len(response_items),
        strategy_cards_by_id=strategy_cards_by_id,
        missing_strategy_ids=missing_strategy_ids,
    )


@router.patch(
    "/weak/{run_id}",
    response_model=CoverageStatusUpdateResponse,
    summary="Update coverage status",
    description="Mark a weak coverage run as acknowledged or resolved.",
)
async def update_coverage_status(
    run_id: UUID,
    request: CoverageStatusUpdateRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> CoverageStatusUpdateResponse:
    """
    Update coverage status for a match run.

    Use this to triage weak coverage items:
    - acknowledged: Someone is looking at it
    - resolved: The coverage gap has been addressed
    - open: Reopen if needed
    """
    pool = _get_pool()

    from app.services.coverage_gap import MatchRunRepository

    repo = MatchRunRepository(pool)

    try:
        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status=request.status.value,
            note=request.note,
            updated_by="admin",  # Could be extracted from token in future
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("coverage_status_update_failed", run_id=str(run_id), error=str(e))
        raise HTTPException(500, f"Failed to update coverage status: {e}")

    if not result:
        raise HTTPException(404, "Match run not found")

    return CoverageStatusUpdateResponse(
        run_id=result["id"],
        coverage_status=CoverageStatusEnum(result["coverage_status"]),
        acknowledged_at=(
            result["acknowledged_at"].isoformat()
            if result.get("acknowledged_at")
            else None
        ),
        acknowledged_by=result.get("acknowledged_by"),
        resolved_at=(
            result["resolved_at"].isoformat() if result.get("resolved_at") else None
        ),
        resolved_by=result.get("resolved_by"),
        resolution_note=result.get("resolution_note"),
    )


@router.post(
    "/explain",
    response_model=ExplainStrategyResponse,
    summary="Generate strategy explanation",
    description="Generate LLM-powered explanation of why a strategy matches an intent.",
)
async def explain_strategy_match(
    request: ExplainStrategyRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> ExplainStrategyResponse:
    """
    Generate an LLM-powered explanation of why a strategy matches an intent.

    This endpoint uses the intent from a match run and strategy metadata
    to generate a natural language explanation of the match.

    Features:
    - Caches explanations per (run_id, strategy_id, verbosity) in match_runs JSONB
    - Invalidates cache if strategy has been updated since generation
    - Verbosity: "short" (2-4 sentences) or "detailed" (2-3 paragraphs)
    - Includes deterministic confidence qualifier based on tag overlap and backtest

    Requires LLM to be configured (ANTHROPIC_API_KEY, OPENAI_API_KEY, or
    OPENROUTER_API_KEY).
    """
    pool = _get_pool()

    from app.services.coverage_gap import MatchRunRepository

    repo = MatchRunRepository(pool)

    # Fetch match run to get intent_json, candidate_scores, and cache
    run_row = await repo.get_match_run_for_explanation(request.run_id, workspace_id)

    if not run_row:
        raise HTTPException(404, "Match run not found")

    # Parse JSON fields (may be string or dict from DB)
    intent_json = parse_json_field(run_row["intent_json"])
    candidate_scores = parse_json_field(run_row["candidate_scores"])
    explanations_cache = parse_json_field(run_row["explanations_cache"])

    # Ensure we have dicts (parse_json_field from utils returns Any)
    if not isinstance(intent_json, dict):
        intent_json = {}
    if not isinstance(candidate_scores, dict):
        candidate_scores = {}
    if not isinstance(explanations_cache, dict):
        explanations_cache = {}

    strategy_id_str = str(request.strategy_id)
    strategy_score_data = candidate_scores.get(strategy_id_str, {})
    matched_tags = strategy_score_data.get("matched_tags", [])
    match_score = strategy_score_data.get("score")

    # Fetch strategy
    from app.services.strategy import StrategyRepository

    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(request.strategy_id, workspace_id)

    if not strategy:
        raise HTTPException(404, "Strategy not found")

    strategy_updated_at = strategy.get("updated_at")
    if strategy_updated_at is not None and hasattr(strategy_updated_at, "isoformat"):
        strategy_updated_at_str: Optional[str] = strategy_updated_at.isoformat()
    elif strategy_updated_at is not None:
        strategy_updated_at_str = str(strategy_updated_at)
    else:
        strategy_updated_at_str = None

    # Build cache key: strategy_id:verbosity
    cache_key = f"{strategy_id_str}:{request.verbosity}"

    # Check cache
    from app.services.coverage_gap import CachedExplanation

    cached = explanations_cache.get(cache_key)
    if cached:
        cached_entry = CachedExplanation.from_dict(cached)
        # Validate: invalidate if strategy changed since generation
        cache_valid = True
        if strategy_updated_at_str and cached_entry.strategy_updated_at:
            if strategy_updated_at_str > cached_entry.strategy_updated_at:
                cache_valid = False
                logger.info(
                    "explanation_cache_invalidated",
                    run_id=str(request.run_id),
                    strategy_id=strategy_id_str,
                    reason="strategy_updated",
                )

        if cache_valid:
            logger.info(
                "explanation_cache_hit",
                run_id=str(request.run_id),
                strategy_id=strategy_id_str,
                verbosity=request.verbosity,
            )
            return ExplainStrategyResponse(
                run_id=request.run_id,
                strategy_id=request.strategy_id,
                strategy_name=strategy.get("name", "Unknown"),
                explanation=cached_entry.explanation,
                confidence_qualifier=cached_entry.confidence_qualifier,
                model=cached_entry.model,
                provider=cached_entry.provider,
                verbosity=cached_entry.verbosity,
                latency_ms=cached_entry.latency_ms,
                cache_hit=True,
            )

    # Generate explanation
    from app.services.coverage_gap import (
        ExplanationError,
        generate_strategy_explanation,
    )

    try:
        result = await generate_strategy_explanation(
            intent_json=intent_json,
            strategy=strategy,
            matched_tags=matched_tags,
            match_score=match_score,
            verbosity=request.verbosity,
        )
    except ExplanationError as e:
        logger.warning("explanation_generation_failed", error=str(e))
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.error("explanation_generation_error", error=str(e))
        raise HTTPException(500, f"Failed to generate explanation: {e}")

    # Store in cache
    cache_entry = CachedExplanation(
        explanation=result.explanation,
        confidence_qualifier=result.confidence_qualifier,
        model=result.model,
        provider=result.provider,
        verbosity=result.verbosity,
        latency_ms=result.latency_ms,
        generated_at=result.generated_at or "",
        strategy_updated_at=strategy_updated_at_str,
    )
    explanations_cache[cache_key] = cache_entry.to_dict()

    try:
        success = await repo.update_explanations_cache(
            run_id=request.run_id,
            workspace_id=workspace_id,
            cache_data=explanations_cache,
        )
        if success:
            logger.info(
                "explanation_cached",
                run_id=str(request.run_id),
                strategy_id=strategy_id_str,
                verbosity=request.verbosity,
            )
    except Exception as e:
        # Don't fail the request if caching fails
        logger.warning("explanation_cache_write_failed", error=str(e))

    return ExplainStrategyResponse(
        run_id=request.run_id,
        strategy_id=request.strategy_id,
        strategy_name=result.strategy_name,
        explanation=result.explanation,
        confidence_qualifier=result.confidence_qualifier,
        model=result.model,
        provider=result.provider,
        verbosity=result.verbosity,
        latency_ms=result.latency_ms,
        cache_hit=False,
    )


# =============================================================================
# Cockpit UI Endpoints
# =============================================================================


async def _build_cockpit_context(
    request: Request,
    pool: Any,
    workspace_id: Optional[UUID],
    status_filter: str,
    sort_by: str,
    selected_run_id: Optional[str] = None,
    fetch_limit: int = 50,
) -> CockpitTemplateContext:
    """
    Build template context for cockpit rendering.

    Fetches coverage items, hydrates strategy cards, and prepares
    all data needed for template rendering.

    Args:
        request: FastAPI request
        pool: Database pool
        workspace_id: Workspace UUID
        status_filter: Status filter (open, acknowledged, resolved, all)
        sort_by: Sort order (priority or newest)
        selected_run_id: Pre-selected run ID for deep links
        fetch_limit: Max items to fetch

    Returns:
        CockpitTemplateContext ready for template rendering
    """
    admin_token = os.environ.get("ADMIN_TOKEN", "")

    # Handle missing workspace
    if not workspace_id:
        workspace_id = await get_default_workspace_id(pool)

    if not workspace_id:
        return CockpitTemplateContext(
            request=request,
            status_filter=status_filter,
            sort_by=sort_by,
            selected_run_id=selected_run_id,
            error="No workspace found.",
        )

    # Fetch coverage items
    try:
        items = await fetch_weak_coverage_runs(
            pool=pool,
            workspace_id=workspace_id,
            limit=fetch_limit,
            since=None,
            status=status_filter if status_filter != "all" else "all",
        )
    except Exception as e:
        logger.error("coverage_cockpit_fetch_failed", error=str(e))
        return CockpitTemplateContext(
            request=request,
            workspace_id=str(workspace_id),
            status_filter=status_filter,
            sort_by=sort_by,
            admin_token=admin_token,
            selected_run_id=selected_run_id,
            error=f"Failed to load coverage data: {e}",
        )

    # Sort by newest if requested (default is priority from repo)
    if sort_by == "newest":
        items = sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    # Hydrate strategy cards
    candidate_ids = collect_candidate_ids(items)
    strategy_cards, missing_strategy_ids = await hydrate_strategy_cards_for_template(
        pool, workspace_id, candidate_ids
    )

    # Convert items for template
    template_items = build_template_items(items)

    return CockpitTemplateContext(
        request=request,
        items=template_items,
        strategy_cards=strategy_cards,
        missing_strategy_ids=missing_strategy_ids,
        workspace_id=str(workspace_id),
        status_filter=status_filter,
        sort_by=sort_by,
        admin_token=admin_token,
        selected_run_id=selected_run_id,
    )


@router.get("/cockpit", response_class=HTMLResponse)
async def coverage_cockpit(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Workspace ID"),
    status: Optional[Literal["open", "acknowledged", "resolved", "all"]] = Query(
        "open", description="Filter by status"
    ),
    sort: Optional[str] = Query("priority", description="Sort by: priority or newest"),
    _: bool = Depends(require_admin_token),
):
    """
    Coverage cockpit UI - two-panel triage interface.

    Left panel: Queue of weak coverage items sorted by priority.
    Right panel: Selected item detail with triage controls.
    """
    pool = _get_pool()

    context = await _build_cockpit_context(
        request=request,
        pool=pool,
        workspace_id=workspace_id,
        status_filter=status or "open",
        sort_by=sort or "priority",
    )

    return templates.TemplateResponse("coverage_cockpit.html", context.to_dict())


@router.get("/cockpit/{run_id}", response_class=HTMLResponse)
async def coverage_cockpit_detail(
    request: Request,
    run_id: UUID,
    workspace_id: Optional[UUID] = Query(None, description="Workspace ID"),
    _: bool = Depends(require_admin_token),
):
    """
    Deep link to coverage cockpit with a specific run pre-selected.

    If the run exists and belongs to the workspace, it will be shown
    in the detail panel. The queue shows all items (status=all) to
    ensure the linked item is visible.
    """
    pool = _get_pool()

    # Get workspace from run if not specified
    if not workspace_id:
        workspace_id = await get_workspace_from_run(pool, run_id)

    context = await _build_cockpit_context(
        request=request,
        pool=pool,
        workspace_id=workspace_id,
        status_filter="all",
        sort_by="priority",
        selected_run_id=str(run_id),
        fetch_limit=100,
    )

    # Check if requested run is in the list (for deep link warning)
    if context.items:
        run_found = any(item.get("run_id") == str(run_id) for item in context.items)
        if not run_found:
            logger.warning("deep_link_run_not_in_list", run_id=str(run_id))

    return templates.TemplateResponse("coverage_cockpit.html", context.to_dict())


# =============================================================================
# Dev-Only Seed Endpoint
# =============================================================================


def _is_dev_mode() -> bool:
    """Check if running in dev mode (allow seeding)."""
    profile = os.environ.get("CONFIG_PROFILE", "development")
    return profile.lower() in ("development", "dev", "local", "test")


# =============================================================================
# Seed Helper Functions
# =============================================================================


async def _get_or_create_seed_workspace(
    conn: Any,
    workspace_id: Optional[UUID],
) -> UUID:
    """
    Get existing or create new workspace for seeding.

    Args:
        conn: Database connection
        workspace_id: Optional workspace ID to use

    Returns:
        Workspace UUID

    Raises:
        HTTPException: If specified workspace not found
    """
    if workspace_id:
        ws_row = await conn.fetchrow(
            "SELECT id FROM workspaces WHERE id = $1", workspace_id
        )
        if not ws_row:
            raise HTTPException(404, f"Workspace {workspace_id} not found")
        return workspace_id

    # Try to get default, or create one
    ws_row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
    if ws_row:
        return ws_row["id"]

    # Create a dev workspace
    ws_row = await conn.fetchrow(
        """
        INSERT INTO workspaces (name, slug, owner_id, is_active, ingestion_enabled)
        VALUES ($1, $2, 'seed-script', true, true)
        ON CONFLICT (slug) DO UPDATE SET name = $1
        RETURNING id
        """,
        SEED_WORKSPACE_NAME,
        SEED_WORKSPACE_SLUG,
    )
    ws_id = ws_row["id"]
    logger.info("seed_created_workspace", workspace_id=str(ws_id))
    return ws_id


async def _clear_seed_data(conn: Any, workspace_id: UUID) -> None:
    """
    Clear existing seed data for workspace.

    Deletes match_runs and strategies with seed markers.

    Args:
        conn: Database connection
        workspace_id: Workspace UUID
    """
    # Delete match_runs with seed marker in intent_signature
    await conn.execute(
        """
        DELETE FROM match_runs
        WHERE workspace_id = $1 AND intent_signature LIKE 'seed-%%'
        """,
        workspace_id,
    )
    # Delete seeded strategies
    await conn.execute(
        """
        DELETE FROM strategies
        WHERE workspace_id = $1 AND slug LIKE 'seed-%%'
        """,
        workspace_id,
    )
    logger.info("seed_cleared_existing", workspace_id=str(workspace_id))


def _get_strategy_fixtures() -> list[dict[str, Any]]:
    """Return list of strategy fixture data for seeding."""
    return [
        {
            "name": "52-Week Breakout Strategy",
            "slug": "seed-breakout-52w",
            "description": "Enters when price breaks 52-week high with volume confirmation",
            "engine": "pine",
            "status": "active",
            "tags": {
                "strategy_archetypes": ["breakout", "momentum"],
                "indicators": ["volume", "atr", "ma"],
                "timeframe_buckets": ["swing"],
                "topics": ["stocks", "crypto"],
            },
            "backtest_summary": {
                "status": "validated",
                "best_oos_score": 1.45,
                "max_drawdown": 0.12,
                "num_trades": 156,
            },
        },
        {
            "name": "RSI Mean Reversion",
            "slug": "seed-rsi-reversion",
            "description": "Buy oversold RSI bounces with tight stops",
            "engine": "pine",
            "status": "active",
            "tags": {
                "strategy_archetypes": ["mean_reversion", "oscillator"],
                "indicators": ["rsi", "bollinger"],
                "timeframe_buckets": ["intraday", "swing"],
                "topics": ["forex"],
            },
            "backtest_summary": {
                "status": "validated",
                "best_oos_score": 0.95,
                "max_drawdown": 0.18,
                "num_trades": 312,
            },
        },
        {
            "name": "MACD Trend Follow",
            "slug": "seed-macd-trend",
            "description": "Follow MACD crossovers with ATR trailing stops",
            "engine": "python",
            "status": "active",
            "tags": {
                "strategy_archetypes": ["trend_following"],
                "indicators": ["macd", "atr", "ema"],
                "timeframe_buckets": ["swing", "position"],
                "topics": ["futures", "commodities"],
            },
            "backtest_summary": {
                "status": "complete",
                "best_oos_score": 1.12,
                "max_drawdown": 0.22,
                "num_trades": 89,
            },
        },
        {
            "name": "Bollinger Squeeze",
            "slug": "seed-bb-squeeze",
            "description": "Enter on Bollinger Band squeeze breakout",
            "engine": "pine",
            "status": "draft",
            "tags": {
                "strategy_archetypes": ["breakout", "volatility"],
                "indicators": ["bollinger", "keltner", "volume"],
                "timeframe_buckets": ["intraday"],
                "topics": ["stocks"],
            },
            "backtest_summary": {
                "status": "never",
            },
        },
        {
            "name": "Volume Profile Scalp",
            "slug": "seed-vp-scalp",
            "description": "Scalp off volume profile POC levels",
            "engine": "vectorbt",
            "status": "active",
            "tags": {
                "strategy_archetypes": ["scalping", "support_resistance"],
                "indicators": ["volume_profile", "vwap"],
                "timeframe_buckets": ["scalp", "intraday"],
                "topics": ["futures", "crypto"],
            },
            "backtest_summary": {
                "status": "validated",
                "best_oos_score": 0.78,
                "max_drawdown": 0.08,
                "num_trades": 1245,
            },
        },
    ]


async def _seed_strategies(
    conn: Any,
    workspace_id: UUID,
    strategy_fixtures: list[dict[str, Any]],
) -> list[UUID]:
    """
    Insert strategy fixtures and return IDs.

    Args:
        conn: Database connection
        workspace_id: Workspace UUID
        strategy_fixtures: List of strategy fixture dicts

    Returns:
        List of created strategy UUIDs
    """
    strategy_ids: list[UUID] = []

    for strat in strategy_fixtures:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO strategies (
                    workspace_id, name, slug, description, engine, status,
                    tags, backtest_summary
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (workspace_id, slug) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    tags = EXCLUDED.tags,
                    backtest_summary = EXCLUDED.backtest_summary,
                    updated_at = NOW()
                RETURNING id
                """,
                workspace_id,
                strat["name"],
                strat["slug"],
                strat["description"],
                strat["engine"],
                strat["status"],
                json.dumps(strat["tags"]),
                json.dumps(strat["backtest_summary"]),
            )
            strategy_ids.append(row["id"])
        except Exception as e:
            logger.warning(
                "seed_strategy_insert_failed", slug=strat["slug"], error=str(e)
            )

    return strategy_ids


def _get_match_run_fixtures(
    strategy_ids: list[UUID],
    missing_strategy_id: UUID,
    now: datetime,
) -> list[dict[str, Any]]:
    """
    Build match run fixture data.

    Args:
        strategy_ids: List of seeded strategy UUIDs
        missing_strategy_id: Fake strategy ID for testing missing warning
        now: Current timestamp

    Returns:
        List of match run fixture dicts
    """
    return [
        # High priority: NO_MATCHES, recent, no candidates
        {
            "intent_signature": "seed-001-no-matches",
            "query_used": (
                "I need a strategy for trading Fibonacci retracements "
                "on 4-hour charts with strict risk management"
            ),
            "intent_json": {
                "strategy_archetypes": ["fibonacci", "retracement"],
                "indicators": ["fibonacci"],
                "timeframe_buckets": ["swing"],
                "topics": ["forex"],
            },
            "reason_codes": ["NO_MATCHES"],
            "best_score": None,
            "num_above_threshold": 0,
            "candidate_strategy_ids": [],
            "coverage_status": "open",
            "created_at": now - timedelta(hours=2),
            "video_id": "abc123xyz",
        },
        # Medium priority: NO_STRONG_MATCHES, has candidates
        {
            "intent_signature": "seed-002-weak-matches",
            "query_used": (
                "Looking for momentum breakout strategy for crypto "
                "with volume confirmation"
            ),
            "intent_json": {
                "strategy_archetypes": ["breakout", "momentum"],
                "indicators": ["volume"],
                "timeframe_buckets": ["swing"],
                "topics": ["crypto"],
            },
            "reason_codes": ["NO_STRONG_MATCHES"],
            "best_score": 0.38,
            "num_above_threshold": 1,
            "candidate_strategy_ids": (
                [strategy_ids[0], strategy_ids[2]] if len(strategy_ids) >= 3 else []
            ),
            "candidate_scores": (
                {
                    str(strategy_ids[0]): {
                        "score": 0.38,
                        "matched_tags": ["breakout", "momentum", "volume"],
                    },
                    str(strategy_ids[2]): {
                        "score": 0.25,
                        "matched_tags": ["momentum"],
                    },
                }
                if len(strategy_ids) >= 3
                else {}
            ),
            "coverage_status": "open",
            "created_at": now - timedelta(hours=6),
            "video_id": "def456uvw",
        },
        # Acknowledged item
        {
            "intent_signature": "seed-003-acknowledged",
            "query_used": "Need RSI divergence strategy for swing trading stocks",
            "intent_json": {
                "strategy_archetypes": ["divergence", "oscillator"],
                "indicators": ["rsi"],
                "timeframe_buckets": ["swing"],
                "topics": ["stocks"],
            },
            "reason_codes": ["NO_STRONG_MATCHES", "LOW_SIGNAL_INPUT"],
            "best_score": 0.42,
            "num_above_threshold": 2,
            "candidate_strategy_ids": (
                [strategy_ids[1]] if len(strategy_ids) >= 2 else []
            ),
            "candidate_scores": (
                {
                    str(strategy_ids[1]): {
                        "score": 0.42,
                        "matched_tags": ["rsi", "swing"],
                    },
                }
                if len(strategy_ids) >= 2
                else {}
            ),
            "coverage_status": "acknowledged",
            "created_at": now - timedelta(days=1),
        },
        # Resolved item
        {
            "intent_signature": "seed-004-resolved",
            "query_used": "MACD crossover strategy for commodities with position sizing",
            "intent_json": {
                "strategy_archetypes": ["trend_following"],
                "indicators": ["macd"],
                "timeframe_buckets": ["position"],
                "topics": ["commodities"],
            },
            "reason_codes": ["NO_STRONG_MATCHES"],
            "best_score": 0.55,
            "num_above_threshold": 3,
            "candidate_strategy_ids": (
                [strategy_ids[2]] if len(strategy_ids) >= 3 else []
            ),
            "coverage_status": "resolved",
            "resolution_note": "Added MACD Trend Follow strategy",
            "created_at": now - timedelta(days=3),
        },
        # Item with MISSING strategy ID (tests warning UI)
        {
            "intent_signature": "seed-005-missing-strategy",
            "query_used": "Scalping strategy for ES futures using volume profile",
            "intent_json": {
                "strategy_archetypes": ["scalping"],
                "indicators": ["volume_profile", "vwap"],
                "timeframe_buckets": ["scalp"],
                "topics": ["futures"],
            },
            "reason_codes": ["NO_STRONG_MATCHES"],
            "best_score": 0.35,
            "num_above_threshold": 1,
            "candidate_strategy_ids": (
                [strategy_ids[4], missing_strategy_id]
                if len(strategy_ids) >= 5
                else [missing_strategy_id]
            ),
            "candidate_scores": (
                {
                    str(strategy_ids[4]): {
                        "score": 0.35,
                        "matched_tags": ["scalping", "volume_profile"],
                    },
                    str(missing_strategy_id): {
                        "score": 0.30,
                        "matched_tags": ["scalping"],
                    },
                }
                if len(strategy_ids) >= 5
                else {
                    str(missing_strategy_id): {
                        "score": 0.30,
                        "matched_tags": ["scalping"],
                    },
                }
            ),
            "coverage_status": "open",
            "created_at": now - timedelta(hours=12),
        },
        # Low score, multiple candidates
        {
            "intent_signature": "seed-006-low-score",
            "query_used": "Bollinger band breakout for intraday stocks with squeeze detection",
            "intent_json": {
                "strategy_archetypes": ["breakout", "volatility"],
                "indicators": ["bollinger"],
                "timeframe_buckets": ["intraday"],
                "topics": ["stocks"],
            },
            "reason_codes": ["LOW_SIGNAL_INPUT"],
            "best_score": 0.28,
            "num_above_threshold": 0,
            "candidate_strategy_ids": (
                [strategy_ids[0], strategy_ids[3]] if len(strategy_ids) >= 4 else []
            ),
            "candidate_scores": (
                {
                    str(strategy_ids[0]): {
                        "score": 0.28,
                        "matched_tags": ["breakout"],
                    },
                    str(strategy_ids[3]): {
                        "score": 0.22,
                        "matched_tags": ["bollinger", "breakout"],
                    },
                }
                if len(strategy_ids) >= 4
                else {}
            ),
            "coverage_status": "open",
            "created_at": now - timedelta(days=2),
        },
        # Older acknowledged
        {
            "intent_signature": "seed-007-old-acknowledged",
            "query_used": "Mean reversion strategy using RSI and Bollinger for forex",
            "intent_json": {
                "strategy_archetypes": ["mean_reversion"],
                "indicators": ["rsi", "bollinger"],
                "timeframe_buckets": ["swing"],
                "topics": ["forex"],
            },
            "reason_codes": ["NO_STRONG_MATCHES"],
            "best_score": 0.48,
            "num_above_threshold": 2,
            "candidate_strategy_ids": (
                [strategy_ids[1]] if len(strategy_ids) >= 2 else []
            ),
            "coverage_status": "acknowledged",
            "created_at": now - timedelta(days=5),
        },
        # Recent high-priority
        {
            "intent_signature": "seed-008-recent-high",
            "query_used": "Elliott wave strategy for crypto with Fibonacci targets",
            "intent_json": {
                "strategy_archetypes": ["elliott_wave", "fibonacci"],
                "indicators": ["fibonacci"],
                "timeframe_buckets": ["position"],
                "topics": ["crypto"],
            },
            "reason_codes": ["NO_MATCHES", "LOW_SIGNAL_INPUT"],
            "best_score": None,
            "num_above_threshold": 0,
            "candidate_strategy_ids": [],
            "coverage_status": "open",
            "created_at": now - timedelta(minutes=30),
        },
    ]


async def _seed_match_runs(
    conn: Any,
    workspace_id: UUID,
    match_run_fixtures: list[dict[str, Any]],
) -> int:
    """
    Insert match run fixtures and return count.

    Args:
        conn: Database connection
        workspace_id: Workspace UUID
        match_run_fixtures: List of match run fixture dicts

    Returns:
        Number of match runs created
    """
    match_runs_created = 0

    for run in match_run_fixtures:
        try:
            candidate_scores_json = (
                json.dumps(run.get("candidate_scores"))
                if run.get("candidate_scores")
                else None
            )

            await conn.execute(
                """
                INSERT INTO match_runs (
                    workspace_id, source_type, video_id,
                    intent_signature, intent_json, query_used, filters_applied,
                    top_k, total_searched, best_score, avg_top_k_score,
                    num_above_threshold, weak_coverage, reason_codes,
                    candidate_strategy_ids, candidate_scores,
                    coverage_status, created_at, resolution_note
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
                    $17::coverage_status, $18, $19
                )
                ON CONFLICT DO NOTHING
                """,
                workspace_id,
                "youtube",
                run.get("video_id"),
                run["intent_signature"],
                json.dumps(run["intent_json"]),
                run["query_used"],
                json.dumps({"script_type": "strategy"}),
                SEED_TOP_K,
                SEED_TOTAL_SEARCHED,
                run["best_score"],
                run["best_score"],  # avg = best for simplicity
                run["num_above_threshold"],
                True,  # weak_coverage
                run["reason_codes"],
                run["candidate_strategy_ids"],
                candidate_scores_json,
                run["coverage_status"],
                run["created_at"],
                run.get("resolution_note"),
            )
            match_runs_created += 1
        except Exception as e:
            logger.warning(
                "seed_match_run_insert_failed",
                signature=run["intent_signature"],
                error=str(e),
            )

    return match_runs_created


@router.post(
    "/seed",
    response_model=SeedCoverageResponse,
    summary="Seed fixture data (dev only)",
    description="Create deterministic strategies and match_runs for UI testing.",
)
async def seed_coverage_fixtures(
    request: Request,
    workspace_id: Optional[UUID] = Query(
        None, description="Workspace ID (creates if missing)"
    ),
    clear_existing: bool = Query(
        False, description="Delete existing seeded data first"
    ),
    _: bool = Depends(require_admin_token),
) -> SeedCoverageResponse:
    """
    Seed deterministic fixture data for coverage cockpit testing.

    **Dev-only**: Only available when CONFIG_PROFILE is development/dev/local/test.

    Creates:
    - 1 workspace (if not provided)
    - 5 strategies with varied tags and backtest status
    - 8 match_runs with varied reason codes, candidates, and triage status
    - 1 match_run referencing a non-existent strategy (tests missing warning)

    Use `clear_existing=true` to delete previous seed data before inserting.
    """
    if not _is_dev_mode():
        raise HTTPException(
            403,
            "Seed endpoint only available in development mode. "
            f"Current CONFIG_PROFILE: {os.environ.get('CONFIG_PROFILE', 'not set')}",
        )

    # Log seed invocation with caller IP
    caller_ip = request.client.host if request.client else "unknown"
    logger.info(
        "SEED CALLED",
        caller_ip=caller_ip,
        workspace_id=str(workspace_id) if workspace_id else None,
        clear_existing=clear_existing,
    )

    pool = _get_pool()

    async with pool.acquire() as conn:
        # Get or create workspace
        workspace_id = await _get_or_create_seed_workspace(conn, workspace_id)

        # Clear existing seed data if requested
        if clear_existing:
            await _clear_seed_data(conn, workspace_id)

        # Seed strategies
        strategy_fixtures = _get_strategy_fixtures()
        strategy_ids = await _seed_strategies(conn, workspace_id, strategy_fixtures)

        # Create a fake "missing" strategy ID that won't exist
        missing_strategy_id = uuid4()

        # Seed match_runs
        now = datetime.now(timezone.utc)
        match_run_fixtures = _get_match_run_fixtures(
            strategy_ids, missing_strategy_id, now
        )
        match_runs_created = await _seed_match_runs(
            conn, workspace_id, match_run_fixtures
        )

    strategies_created = len(strategy_ids)

    logger.info(
        "seed_coverage_complete",
        workspace_id=str(workspace_id),
        strategies_created=strategies_created,
        match_runs_created=match_runs_created,
    )

    return SeedCoverageResponse(
        status="success",
        workspace_id=workspace_id,
        strategies_created=strategies_created,
        match_runs_created=match_runs_created,
        message=(
            f"Seeded {strategies_created} strategies and "
            f"{match_runs_created} match_runs for testing"
        ),
    )
