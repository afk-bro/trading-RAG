"""
Neighbor expansion for RAG retrieval.

Expands seed chunks with neighboring chunks from the same document
to provide better context continuity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.repositories.chunks import ChunkRepository

from app.services.reranker import RerankResult

logger = logging.getLogger(__name__)


@dataclass
class ExpandedChunk:
    """A chunk after neighbor expansion."""

    chunk_id: str
    document_id: str
    chunk_index: int
    rerank_score: float
    rerank_rank: int
    vector_score: float
    source_type: str | None
    is_neighbor: bool
    neighbor_of: str | None  # chunk_id of the seed this is a neighbor of


async def expand_neighbors(
    seeds: list[RerankResult],
    chunk_repo: "ChunkRepository",
    config: dict,
    already_have_ids: set[str] | None = None,
) -> tuple[list[ExpandedChunk], list[str]]:
    """
    Expand seeds with neighboring chunks.

    Args:
        seeds: Reranked seed chunks
        chunk_repo: Repository for chunk data
        config: Neighbor configuration dict with:
            - enabled: bool (default True)
            - window: int (default 1) - neighbor window for non-PDF
            - pdf_window: int (default 2) - neighbor window for PDF
            - min_chars: int (default 200) - minimum chars for valid neighbor
            - max_total: int (default 20) - soft cap on total chunks
        already_have_ids: Set of chunk IDs already fetched (for deduplication)

    Returns:
        Tuple of:
            - expanded list ordered by (best_doc_rank, doc_id, chunk_index)
            - list of new chunk_ids that need fetching from DB
    """
    already_have_ids = already_have_ids or set()

    if not seeds:
        return [], []

    enabled = config.get("enabled", True)

    # Convert seeds to ExpandedChunk
    seed_expanded = [
        ExpandedChunk(
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            chunk_index=s.chunk_index,
            rerank_score=s.rerank_score,
            rerank_rank=s.rerank_rank,
            vector_score=s.vector_score,
            source_type=s.source_type,
            is_neighbor=False,
            neighbor_of=None,
        )
        for s in seeds
    ]

    if not enabled:
        return seed_expanded, []

    window = config.get("window", 1)
    pdf_window = config.get("pdf_window", 2)
    min_chars = config.get("min_chars", 200)
    max_total = config.get("max_total", 20)

    # Build deduped neighbor requests
    # For overlapping neighbors, best seed (lowest rank) wins attribution
    seed_set = {s.chunk_id for s in seeds}
    req_map: dict[tuple[str, int], tuple[int, str]] = {}  # (doc_id, idx) -> (rank, seed_id)

    for seed in seeds:
        st = (seed.source_type or "").lower()
        w = pdf_window if st == "pdf" else window

        for offset in range(-w, w + 1):
            if offset == 0:
                continue
            neighbor_idx = seed.chunk_index + offset
            if neighbor_idx < 0:
                continue

            key = (seed.document_id, neighbor_idx)
            cur = req_map.get(key)
            if cur is None or seed.rerank_rank < cur[0]:
                req_map[key] = (seed.rerank_rank, seed.chunk_id)

    neighbor_requests = [
        (doc_id, idx, seed_id)
        for (doc_id, idx), (_, seed_id) in req_map.items()
    ]

    if not neighbor_requests:
        return seed_expanded, []

    # Fetch neighbors from repository
    neighbors = await chunk_repo.get_neighbors_by_doc_indices(neighbor_requests)

    # Filter and build neighbor ExpandedChunks
    valid_neighbors: list[ExpandedChunk] = []
    seen_ids = set(seed_set) | already_have_ids

    for n in neighbors:
        if n.chunk_id in seen_ids:
            continue
        if len(n.text) < min_chars:
            continue
        seen_ids.add(n.chunk_id)

        valid_neighbors.append(
            ExpandedChunk(
                chunk_id=n.chunk_id,
                document_id=n.document_id,
                chunk_index=n.chunk_index,
                rerank_score=0.0,
                rerank_rank=-1,
                vector_score=0.0,
                source_type=n.source_type,
                is_neighbor=True,
                neighbor_of=n.seed_chunk_id,
            )
        )

    # Compute best seed rank per document
    best_doc_rank: dict[str, int] = {}
    for s in seeds:
        doc_id = s.document_id
        best_doc_rank[doc_id] = min(best_doc_rank.get(doc_id, 10**9), s.rerank_rank)

    all_expanded = seed_expanded + valid_neighbors

    # Sort: best doc first, then by chunk_index within doc
    all_expanded.sort(
        key=lambda x: (
            best_doc_rank.get(x.document_id, 10**9),
            x.document_id,
            x.chunk_index,
        )
    )

    # Soft cap: preserves all seeds, trims neighbors only
    if len(all_expanded) > max_total:
        kept_seeds = [e for e in all_expanded if not e.is_neighbor]

        if len(kept_seeds) >= max_total:
            # Edge case: more seeds than max_total, keep all seeds
            all_expanded = kept_seeds
        else:
            kept_neighbors = [e for e in all_expanded if e.is_neighbor]
            remaining = max_total - len(kept_seeds)
            all_expanded = kept_seeds + kept_neighbors[:remaining]

        # Re-sort after trimming
        all_expanded.sort(
            key=lambda x: (
                best_doc_rank.get(x.document_id, 10**9),
                x.document_id,
                x.chunk_index,
            )
        )

    # Identify new chunk IDs for caller to fetch from DB
    new_ids = [
        e.chunk_id
        for e in all_expanded
        if e.is_neighbor and e.chunk_id not in already_have_ids
    ]

    return all_expanded, new_ids
