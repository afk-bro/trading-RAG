"""Admin document detail view - inspect ingested content and validate."""

import re
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.admin.concepts_constants import (
    NON_TICKERS,
    SQL_GET_CHUNK_BY_ID,
    SQL_GET_CHUNK_CONTENT,
    SQL_GET_CHUNK_VALIDATIONS,
    SQL_GET_CHUNKS,
    SQL_GET_DOCUMENT,
    SQL_GET_DOCUMENT_TITLE,
    SQL_UPSERT_VALIDATION,
    TICKER_PATTERN,
    TRADING_CONCEPTS,
    VALID_STATUSES,
)
from app.admin.utils import require_db_pool
from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for document routes."""
    global _db_pool
    _db_pool = pool


# =============================================================================
# Key Concepts Extraction
# =============================================================================


def _sort_key(x: tuple[str, dict[str, object]]) -> int:
    """Sort key for concept frequency sorting."""
    return int(x[1].get("count", 0) or 0)  # type: ignore[call-overload]


def extract_key_concepts(chunks: list[dict]) -> dict:
    """Extract key trading concepts and tickers from chunk content."""
    combined_text = " ".join(c.get("content", "") for c in chunks).lower()

    # Find matching concepts
    found_concepts: dict[str, dict[str, object]] = {}
    for term, description in TRADING_CONCEPTS.items():
        # Use word boundary matching
        pattern = r"\b" + re.escape(term) + r"\b"
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        if matches:
            found_concepts[term] = {
                "description": description,
                "count": len(matches),
            }

    # Sort by frequency
    found_concepts = dict(sorted(found_concepts.items(), key=_sort_key, reverse=True))

    # Find potential tickers
    combined_upper = " ".join(c.get("content", "") for c in chunks)
    ticker_matches = TICKER_PATTERN.findall(combined_upper)
    tickers: dict[str, int] = {}
    for t in ticker_matches:
        if t not in NON_TICKERS and len(t) >= 2:
            tickers[t] = tickers.get(t, 0) + 1

    # Sort tickers by frequency
    tickers = dict(sorted(tickers.items(), key=lambda x: x[1], reverse=True))

    return {
        "concepts": found_concepts,
        "tickers": tickers,
    }


# =============================================================================
# Routes
# =============================================================================


@router.get("/documents/{doc_id}", response_class=HTMLResponse)
async def document_detail_page(
    request: Request,
    doc_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Document detail page showing all extracted content and key concepts.

    Features:
    - Document metadata (title, author, source, timestamps)
    - All chunks with content and timestamps
    - Auto-detected key concepts and tickers
    - Chunk validation UI
    """
    pool = require_db_pool(_db_pool)

    async with pool.acquire() as conn:
        # Fetch document
        doc = await conn.fetchrow(SQL_GET_DOCUMENT, doc_id)

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )

        # Fetch chunks
        chunks = await conn.fetch(SQL_GET_CHUNKS, doc_id)

        # Fetch chunk validation status if exists
        chunk_validations = await conn.fetch(
            SQL_GET_CHUNK_VALIDATIONS,
            [c["id"] for c in chunks],
        )
        validation_map = {v["chunk_id"]: dict(v) for v in chunk_validations}

    # Convert to dicts
    doc_dict = dict(doc)
    chunks_list = [dict(c) for c in chunks]

    # Add validation status to chunks
    for chunk in chunks_list:
        chunk["validation"] = validation_map.get(chunk["id"])

    # Extract key concepts
    key_concepts = extract_key_concepts(chunks_list)

    # Calculate stats
    total_tokens = sum(c.get("token_count", 0) or 0 for c in chunks_list)
    total_duration = None
    if chunks_list and chunks_list[-1].get("time_end_secs"):
        total_duration = chunks_list[-1]["time_end_secs"]

    return templates.TemplateResponse(
        "document_detail.html",
        {
            "request": request,
            "document": doc_dict,
            "chunks": chunks_list,
            "key_concepts": key_concepts,
            "total_tokens": total_tokens,
            "total_duration": total_duration,
            "chunk_count": len(chunks_list),
        },
    )


# =============================================================================
# API Endpoints for Chunk Validation
# =============================================================================


class ChunkValidationRequest(BaseModel):
    """Request to validate a chunk."""

    status: str  # "verified", "needs_review", "garbage"
    notes: Optional[str] = None


@router.post("/documents/chunks/{chunk_id}/validate")
async def validate_chunk(
    chunk_id: UUID,
    validation: ChunkValidationRequest,
    _: bool = Depends(require_admin_token),
):
    """
    Mark a chunk with validation status.

    Status options:
    - verified: Content is accurate and useful
    - needs_review: Content needs manual review/correction
    - garbage: Content is noise (sponsor, engagement, etc.)
    """
    pool = require_db_pool(_db_pool)

    if validation.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid status. Must be: verified, needs_review, garbage",
        )

    async with pool.acquire() as conn:
        # Verify chunk exists
        chunk = await conn.fetchrow(SQL_GET_CHUNK_BY_ID, chunk_id)
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found",
            )

        # Upsert validation
        await conn.execute(
            SQL_UPSERT_VALIDATION,
            chunk_id,
            validation.status,
            validation.notes,
        )

    return {
        "chunk_id": str(chunk_id),
        "status": validation.status,
        "message": "Validation saved",
    }


@router.get("/documents/{doc_id}/concepts")
async def get_document_concepts(
    doc_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Get extracted key concepts for a document (JSON API).
    """
    pool = require_db_pool(_db_pool)

    async with pool.acquire() as conn:
        # Verify document exists
        doc = await conn.fetchrow(SQL_GET_DOCUMENT_TITLE, doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )

        # Fetch chunks
        chunks = await conn.fetch(SQL_GET_CHUNK_CONTENT, doc_id)

    chunks_list = [dict(c) for c in chunks]
    key_concepts = extract_key_concepts(chunks_list)

    return {
        "doc_id": str(doc_id),
        "title": doc["title"],
        "concepts": key_concepts["concepts"],
        "tickers": key_concepts["tickers"],
    }
