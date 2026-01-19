"""Coverage query and hydration helpers.

Extracted from coverage.py to keep router thin.
All functions accept dependencies explicitly (no globals).
"""

import json
from typing import Any, Optional
from uuid import UUID

import structlog

from app.schemas import (
    BacktestSummaryStatus,
    StrategyCard,
    StrategyEngine,
    StrategyStatus,
    StrategyTags,
)

logger = structlog.get_logger(__name__)

# Max unique strategy IDs to hydrate (prevents payload bloat)
MAX_HYDRATION_IDS = 300


def collect_candidate_ids(
    items: list[dict[str, Any]],
    max_ids: int = MAX_HYDRATION_IDS,
) -> list[UUID]:
    """
    Collect unique candidate strategy IDs from coverage items.

    Preserves order (first occurrence) and caps at max_ids.
    Used by both API endpoint and cockpit template rendering.

    Args:
        items: List of coverage items with candidate_strategy_ids field
        max_ids: Maximum number of IDs to collect (default 300)

    Returns:
        Deduplicated list of UUIDs in order of first appearance
    """
    all_candidate_ids: list[UUID] = []
    seen: set[UUID] = set()

    for item in items:
        for cid in item.get("candidate_strategy_ids", []):
            if cid and cid not in seen:
                seen.add(cid)
                all_candidate_ids.append(cid)
                if len(all_candidate_ids) >= max_ids:
                    return all_candidate_ids

    return all_candidate_ids


async def hydrate_strategy_cards(
    pool: Any,
    workspace_id: UUID,
    candidate_ids: list[UUID],
) -> tuple[dict[str, StrategyCard], list[UUID]]:
    """
    Hydrate strategy cards from database for API responses.

    Returns StrategyCard Pydantic models suitable for JSON serialization.
    Also tracks missing IDs (deleted/archived strategies).

    Args:
        pool: Database connection pool
        workspace_id: Workspace UUID
        candidate_ids: List of strategy UUIDs to fetch

    Returns:
        Tuple of (cards_by_id dict, missing_ids list)
    """
    if not candidate_ids:
        return {}, []

    from app.services.strategy import StrategyRepository

    strategy_cards_by_id: dict[str, StrategyCard] = {}
    missing_strategy_ids: list[UUID] = []

    try:
        strategy_repo = StrategyRepository(pool)
        cards_dict = await strategy_repo.get_cards_by_ids(workspace_id, candidate_ids)

        for uuid_str, card in cards_dict.items():
            tags_data = card.get("tags") or {}
            strategy_cards_by_id[uuid_str] = StrategyCard(
                id=card["id"],
                name=card["name"],
                slug=card["slug"],
                engine=StrategyEngine(card["engine"]),
                status=StrategyStatus(card["status"]),
                tags=StrategyTags(**tags_data) if tags_data else StrategyTags(),
                backtest_status=(
                    BacktestSummaryStatus(card["backtest_status"])
                    if card.get("backtest_status")
                    else None
                ),
                last_backtest_at=card.get("last_backtest_at"),
                best_oos_score=card.get("best_oos_score"),
                max_drawdown=card.get("max_drawdown"),
            )

        # Track missing IDs (deleted/archived strategies)
        found_ids = set(cards_dict.keys())
        for cid in candidate_ids:
            if str(cid) not in found_ids:
                missing_strategy_ids.append(cid)

        logger.info(
            "hydrated_candidate_cards",
            requested=len(candidate_ids),
            found=len(strategy_cards_by_id),
            missing=len(missing_strategy_ids),
        )

    except Exception as e:
        logger.warning("candidate_card_hydration_failed", error=str(e))
        # Return empty results but don't fail the request

    return strategy_cards_by_id, missing_strategy_ids


async def hydrate_strategy_cards_for_template(
    pool: Any,
    workspace_id: UUID,
    candidate_ids: list[UUID],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Hydrate strategy cards from database for template rendering.

    Returns plain dicts suitable for Jinja2 templates.
    Also tracks missing IDs (deleted/archived strategies).

    Args:
        pool: Database connection pool
        workspace_id: Workspace UUID
        candidate_ids: List of strategy UUIDs to fetch

    Returns:
        Tuple of (cards_by_id dict, missing_ids list as strings)
    """
    if not candidate_ids:
        return {}, []

    from app.services.strategy import StrategyRepository

    strategy_cards: dict[str, dict[str, Any]] = {}
    missing_strategy_ids: list[str] = []

    try:
        strategy_repo = StrategyRepository(pool)
        cards_dict = await strategy_repo.get_cards_by_ids(workspace_id, candidate_ids)

        for uuid_str, card in cards_dict.items():
            tags_data = card.get("tags") or {}
            strategy_cards[uuid_str] = {
                "id": str(card["id"]),
                "name": card["name"],
                "slug": card["slug"],
                "engine": card["engine"],
                "status": card["status"],
                "tags": tags_data,
                "backtest_status": card.get("backtest_status"),
                "last_backtest_at": (
                    card["last_backtest_at"].isoformat()
                    if card.get("last_backtest_at")
                    else None
                ),
                "best_oos_score": card.get("best_oos_score"),
                "max_drawdown": card.get("max_drawdown"),
            }

        found_ids = set(cards_dict.keys())
        for cid in candidate_ids:
            if str(cid) not in found_ids:
                missing_strategy_ids.append(str(cid))

    except Exception as e:
        logger.warning("template_card_hydration_failed", error=str(e))

    return strategy_cards, missing_strategy_ids


def build_template_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw coverage item to template-friendly format.

    Ensures UUIDs are strings for JSON serialization in templates.

    Args:
        item: Raw coverage item from repository

    Returns:
        Dict suitable for Jinja2 template rendering
    """
    return {
        "run_id": str(item["run_id"]),
        "created_at": item["created_at"],
        "intent_signature": item.get("intent_signature", ""),
        "script_type": item.get("script_type"),
        "weak_reason_codes": item.get("weak_reason_codes", []),
        "best_score": item.get("best_score"),
        "num_above_threshold": item.get("num_above_threshold", 0),
        "candidate_strategy_ids": [
            str(c) for c in item.get("candidate_strategy_ids", [])
        ],
        "candidate_scores": item.get("candidate_scores", {}),
        "query_preview": item.get("query_preview", ""),
        "source_ref": item.get("source_ref"),
        "coverage_status": item.get("coverage_status", "open"),
        "priority_score": item.get("priority_score", 0.0),
        "resolution_note": item.get("resolution_note"),
    }


def build_template_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of raw coverage items to template-friendly format.

    Args:
        items: List of raw coverage items from repository

    Returns:
        List of dicts suitable for Jinja2 template rendering
    """
    return [build_template_item(item) for item in items]


def parse_json_field(value: Any) -> dict[str, Any]:
    """
    Safely parse a JSON field that may be string, dict, or None.

    Used for intent_json, candidate_scores, explanations_cache fields
    that may come from the database as either JSON string or parsed dict.

    Args:
        value: Field value (str, dict, or None)

    Returns:
        Parsed dict, or empty dict if parsing fails
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


async def fetch_weak_coverage_runs(
    pool: Any,
    workspace_id: UUID,
    limit: int = 50,
    since: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Fetch weak coverage runs from database.

    Thin wrapper around MatchRunRepository for service-layer access.

    Args:
        pool: Database connection pool
        workspace_id: Workspace UUID
        limit: Maximum results
        since: ISO timestamp filter
        status: Status filter (open, acknowledged, resolved, all)

    Returns:
        List of coverage run dicts
    """
    from app.services.coverage_gap import MatchRunRepository

    repo = MatchRunRepository(pool)
    return await repo.list_weak_coverage_for_cockpit(
        workspace_id=workspace_id,
        limit=limit,
        since=since,
        status=status,
    )


async def get_default_workspace_id(pool: Any) -> Optional[UUID]:
    """
    Get the first workspace ID from database.

    Used when workspace_id is not provided in request.

    Args:
        pool: Database connection pool

    Returns:
        Workspace UUID or None if no workspaces exist
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
            if row:
                return row["id"]
    except Exception as e:
        logger.warning("could_not_fetch_default_workspace", error=str(e))
    return None


async def get_workspace_from_run(pool: Any, run_id: UUID) -> Optional[UUID]:
    """
    Get workspace ID from a match run.

    Used for deep-link URLs where workspace_id may not be provided.

    Args:
        pool: Database connection pool
        run_id: Match run UUID

    Returns:
        Workspace UUID or None if run not found
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT workspace_id FROM match_runs WHERE id = $1",
                run_id,
            )
            if row:
                return row["workspace_id"]
    except Exception as e:
        logger.warning("could_not_fetch_workspace_from_run", error=str(e))
    return None
