"""KB trial warning aggregation."""

from typing import Optional
from uuid import UUID


async def get_top_warnings(
    pool,
    workspace_id: Optional[UUID],
    limit: int,
) -> dict:
    """
    Get top warning types across KB trials.

    Returns aggregated warning counts sorted by frequency.
    """
    async with pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id, limit] if workspace_id else [limit]
        limit_param = "$2" if workspace_id else "$1"

        query = f"""
            SELECT
                warning,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (
                    SELECT COUNT(*) FROM kb_trial_vectors
                    WHERE 1=1 {workspace_filter}
                ) as pct
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            warnings = [
                {
                    "warning": row["warning"],
                    "count": row["count"],
                    "pct": round(row["pct"], 2) if row["pct"] else 0,
                }
                for row in rows
            ]
        except Exception:
            warnings = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "warnings": warnings,
    }
