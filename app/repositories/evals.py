"""Evaluation repository for query compare evals."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EvalSummary:
    """Summary stats for query compare evaluations."""

    total: int
    impacted_count: int
    impact_rate: float
    p50_rerank_ms: Optional[float]
    p95_rerank_ms: Optional[float]
    fallback_count: int
    fallback_rate: float
    timeout_count: int
    timeout_rate: float


@dataclass
class ConfigStats:
    """Stats grouped by config fingerprint."""

    config_fingerprint: str
    rerank_method: Optional[str]
    rerank_model: Optional[str]
    candidates_k: int
    top_k: int
    total: int
    impact_rate: float
    p50_rerank_ms: Optional[float]
    p95_rerank_ms: Optional[float]
    fallback_rate: float


@dataclass
class ImpactedQuery:
    """A query where reranking had significant impact."""

    created_at: datetime
    question_hash: str
    question_preview: Optional[str]
    jaccard: float
    spearman: Optional[float]
    rank_delta_mean: Optional[float]
    rank_delta_max: Optional[int]
    vector_top5_ids: list[str]
    reranked_top5_ids: list[str]


def compute_question_hash(question: str) -> str:
    """Compute SHA256 hash of normalized question."""
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def compute_config_fingerprint(
    candidates_k: int,
    top_k: int,
    rerank_method: Optional[str],
    rerank_model: Optional[str],
    skip_neighbors: bool,
) -> str:
    """Compute fingerprint for config grouping."""
    config = {
        "candidates_k": candidates_k,
        "top_k": top_k,
        "rerank_method": rerank_method,
        "rerank_model": rerank_model,
        "skip_neighbors": skip_neighbors,
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class EvalRepository:
    """Repository for query compare evaluation storage."""

    def __init__(self, pool):
        """Initialize repository with asyncpg pool."""
        self.pool = pool

    async def insert(
        self,
        workspace_id: UUID,
        question: str,
        candidates_k: int,
        top_k: int,
        share_candidates: bool,
        skip_neighbors: bool,
        rerank_method: Optional[str],
        rerank_model: Optional[str],
        jaccard: float,
        spearman: Optional[float],
        rank_delta_mean: Optional[float],
        rank_delta_max: Optional[int],
        overlap_count: int,
        union_count: int,
        embed_ms: int,
        search_ms: int,
        vector_total_ms: int,
        rerank_ms: Optional[int],
        rerank_total_ms: Optional[int],
        rerank_state: str,
        rerank_timeout: bool,
        rerank_fallback: bool,
        vector_top5_ids: list[str],
        reranked_top5_ids: list[str],
        store_question_preview: bool = False,
    ) -> UUID:
        """Insert a query compare evaluation."""
        question_hash = compute_question_hash(question)
        question_preview = question[:80] if store_question_preview else None
        config_fingerprint = compute_config_fingerprint(
            candidates_k, top_k, rerank_method, rerank_model, skip_neighbors
        )

        payload = {
            "workspace_id": str(workspace_id),
            "candidates_k": candidates_k,
            "top_k": top_k,
            "share_candidates": share_candidates,
            "skip_neighbors": skip_neighbors,
            "rerank_method": rerank_method,
            "rerank_model": rerank_model,
            "jaccard": jaccard,
            "spearman": spearman,
            "rank_delta_mean": rank_delta_mean,
            "rank_delta_max": rank_delta_max,
            "overlap_count": overlap_count,
            "union_count": union_count,
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "vector_total_ms": vector_total_ms,
            "rerank_ms": rerank_ms,
            "rerank_total_ms": rerank_total_ms,
            "rerank_state": rerank_state,
            "rerank_timeout": rerank_timeout,
            "rerank_fallback": rerank_fallback,
            "vector_top5_ids": vector_top5_ids,
            "reranked_top5_ids": reranked_top5_ids,
        }

        query = """
            INSERT INTO query_compare_evals (
                workspace_id, question_hash, question_preview, config_fingerprint,
                rerank_method, rerank_model, candidates_k, top_k,
                share_candidates, skip_neighbors,
                jaccard, spearman, rank_delta_mean, rank_delta_max,
                overlap_count, union_count,
                embed_ms, search_ms, vector_total_ms, rerank_ms, rerank_total_ms,
                rerank_state, rerank_timeout, rerank_fallback,
                vector_top5_ids, reranked_top5_ids, payload
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                $22, $23, $24, $25, $26, $27
            )
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                question_hash,
                question_preview,
                config_fingerprint,
                rerank_method,
                rerank_model,
                candidates_k,
                top_k,
                share_candidates,
                skip_neighbors,
                jaccard,
                spearman,
                rank_delta_mean,
                rank_delta_max,
                overlap_count,
                union_count,
                embed_ms,
                search_ms,
                vector_total_ms,
                rerank_ms,
                rerank_total_ms,
                rerank_state,
                rerank_timeout,
                rerank_fallback,
                vector_top5_ids,
                reranked_top5_ids,
                json.dumps(payload),
            )
            return row["id"]

    async def get_summary(
        self,
        workspace_id: UUID,
        since_hours: int = 24,
    ) -> EvalSummary:
        """Get summary stats for a workspace."""
        since = datetime.utcnow() - timedelta(hours=since_hours)

        query = """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE jaccard < 0.8) as impacted_count,
                COUNT(*) FILTER (WHERE rerank_fallback = true) as fallback_count,
                COUNT(*) FILTER (WHERE rerank_timeout = true) as timeout_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rerank_total_ms)
                    FILTER (WHERE rerank_total_ms IS NOT NULL) as p50_rerank_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rerank_total_ms)
                    FILTER (WHERE rerank_total_ms IS NOT NULL) as p95_rerank_ms
            FROM query_compare_evals
            WHERE workspace_id = $1 AND created_at >= $2
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, since)

        total = row["total"] or 0
        impacted = row["impacted_count"] or 0
        fallbacks = row["fallback_count"] or 0
        timeouts = row["timeout_count"] or 0

        return EvalSummary(
            total=total,
            impacted_count=impacted,
            impact_rate=impacted / total if total > 0 else 0.0,
            p50_rerank_ms=row["p50_rerank_ms"],
            p95_rerank_ms=row["p95_rerank_ms"],
            fallback_count=fallbacks,
            fallback_rate=fallbacks / total if total > 0 else 0.0,
            timeout_count=timeouts,
            timeout_rate=timeouts / total if total > 0 else 0.0,
        )

    async def get_by_config(
        self,
        workspace_id: UUID,
        since_hours: int = 168,  # 7 days
    ) -> list[ConfigStats]:
        """Get stats grouped by config fingerprint."""
        since = datetime.utcnow() - timedelta(hours=since_hours)

        query = """
            SELECT
                config_fingerprint,
                rerank_method,
                rerank_model,
                candidates_k,
                top_k,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE jaccard < 0.8)::float / COUNT(*) as impact_rate,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rerank_total_ms)
                    FILTER (WHERE rerank_total_ms IS NOT NULL) as p50_rerank_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rerank_total_ms)
                    FILTER (WHERE rerank_total_ms IS NOT NULL) as p95_rerank_ms,
                COUNT(*) FILTER (WHERE rerank_fallback = true)::float / COUNT(*) as fallback_rate
            FROM query_compare_evals
            WHERE workspace_id = $1 AND created_at >= $2
            GROUP BY config_fingerprint, rerank_method, rerank_model, candidates_k, top_k
            ORDER BY total DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, since)

        return [
            ConfigStats(
                config_fingerprint=row["config_fingerprint"],
                rerank_method=row["rerank_method"],
                rerank_model=row["rerank_model"],
                candidates_k=row["candidates_k"],
                top_k=row["top_k"],
                total=row["total"],
                impact_rate=row["impact_rate"] or 0.0,
                p50_rerank_ms=row["p50_rerank_ms"],
                p95_rerank_ms=row["p95_rerank_ms"],
                fallback_rate=row["fallback_rate"] or 0.0,
            )
            for row in rows
        ]

    async def get_most_impacted(
        self,
        workspace_id: UUID,
        since_hours: int = 168,  # 7 days
        limit: int = 20,
    ) -> list[ImpactedQuery]:
        """Get queries with lowest jaccard (highest impact)."""
        since = datetime.utcnow() - timedelta(hours=since_hours)

        query = """
            SELECT
                created_at, question_hash, question_preview,
                jaccard, spearman, rank_delta_mean, rank_delta_max,
                vector_top5_ids, reranked_top5_ids
            FROM query_compare_evals
            WHERE workspace_id = $1 AND created_at >= $2 AND jaccard < 0.8
            ORDER BY jaccard ASC, created_at DESC
            LIMIT $3
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, since, limit)

        return [
            ImpactedQuery(
                created_at=row["created_at"],
                question_hash=row["question_hash"],
                question_preview=row["question_preview"],
                jaccard=row["jaccard"],
                spearman=row["spearman"],
                rank_delta_mean=row["rank_delta_mean"],
                rank_delta_max=row["rank_delta_max"],
                vector_top5_ids=row["vector_top5_ids"] or [],
                reranked_top5_ids=row["reranked_top5_ids"] or [],
            )
            for row in rows
        ]

    async def delete_older_than(
        self,
        days: int = 90,
    ) -> int:
        """Delete evaluations older than N days. Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = """
            DELETE FROM query_compare_evals
            WHERE created_at < $1
        """

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, cutoff)
            # Result is like "DELETE 42"
            count = int(result.split()[-1]) if result else 0
            return count
