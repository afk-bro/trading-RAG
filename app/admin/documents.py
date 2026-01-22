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

# Trading-related terms to detect
TRADING_CONCEPTS = {
    # Price action
    "breakout": "Price breaking through support/resistance",
    "false breakout": "Failed breakout that reverses",
    "consolidation": "Price trading in a range",
    "support": "Price level where buying pressure emerges",
    "resistance": "Price level where selling pressure emerges",
    "trend": "Directional price movement",
    "reversal": "Change in trend direction",
    "pullback": "Temporary price retracement",
    "retracement": "Temporary move against the trend",
    # Volume
    "volume": "Trading activity/shares traded",
    "volume spike": "Sudden increase in trading volume",
    "relative volume": "Volume compared to average",
    # Patterns
    "range": "Price bounded between levels",
    "channel": "Parallel support/resistance lines",
    "flag": "Continuation pattern",
    "wedge": "Converging trendlines pattern",
    "double top": "Reversal pattern at highs",
    "double bottom": "Reversal pattern at lows",
    "head and shoulders": "Reversal pattern",
    # Timeframes
    "1-minute": "1-minute chart timeframe",
    "5-minute": "5-minute chart timeframe",
    "15-minute": "15-minute chart timeframe",
    "daily": "Daily chart timeframe",
    "weekly": "Weekly chart timeframe",
    # Indicators
    "moving average": "Average price over period",
    "ema": "Exponential moving average",
    "sma": "Simple moving average",
    "vwap": "Volume weighted average price",
    "rsi": "Relative strength index",
    "macd": "Moving average convergence divergence",
    # Risk/Position
    "stop loss": "Exit point to limit losses",
    "take profit": "Exit point to secure gains",
    "risk reward": "Ratio of potential profit to loss",
    "position size": "Amount of capital in trade",
    # Market structure
    "higher high": "Price making higher peaks",
    "higher low": "Price making higher troughs",
    "lower high": "Price making lower peaks",
    "lower low": "Price making lower troughs",
    # Order flow
    "bid": "Buy order price",
    "ask": "Sell order price",
    "spread": "Difference between bid and ask",
    "liquidity": "Ease of executing trades",
}

# Ticker pattern (1-5 uppercase letters, optionally with $ prefix)
TICKER_PATTERN = re.compile(r"\$?([A-Z]{1,5})\b")

# Common non-ticker words to exclude
NON_TICKERS = {
    "I",
    "A",
    "THE",
    "AND",
    "OR",
    "BUT",
    "FOR",
    "WITH",
    "THIS",
    "THAT",
    "ARE",
    "WAS",
    "WERE",
    "BEEN",
    "HAVE",
    "HAS",
    "HAD",
    "DO",
    "DOES",
    "DID",
    "WILL",
    "WOULD",
    "COULD",
    "SHOULD",
    "MAY",
    "MIGHT",
    "MUST",
    "CAN",
    "NOW",
    "THEN",
    "HERE",
    "THERE",
    "WHERE",
    "WHEN",
    "WHY",
    "HOW",
    "ALL",
    "ANY",
    "BOTH",
    "EACH",
    "FEW",
    "MORE",
    "MOST",
    "OTHER",
    "SOME",
    "SUCH",
    "NO",
    "NOT",
    "ONLY",
    "OWN",
    "SAME",
    "SO",
    "THAN",
    "TOO",
    "VERY",
    "JUST",
    "IF",
    "BECAUSE",
    "AS",
    "UNTIL",
    "WHILE",
    "OF",
    "AT",
    "BY",
    "FROM",
    "UP",
    "ABOUT",
    "INTO",
    "OVER",
    "AFTER",
    "BELOW",
    "TO",
    "ABOVE",
    "BETWEEN",
    "BE",
    "IT",
    "HE",
    "SHE",
    "WE",
    "THEY",
    "YOU",
    "YOUR",
    "MY",
    "HIS",
    "HER",
    "ITS",
    "OUR",
    "THEIR",
    "WHAT",
    "WHICH",
    "WHO",
    "WHOM",
    "THESE",
    "THOSE",
    "AM",
    "IS",
    "AN",
    "ON",
    "IN",
    # Common words that look like tickers
    "US",
    "UK",
    "EU",
    "CEO",
    "CFO",
    "IPO",
    "ETF",
    "SEC",
    "FDA",
    "AI",
    "API",
    "URL",
    "PDF",
    "OK",
    "VS",
    "ETC",
    "LLC",
    "INC",
    "LTD",
    "USD",
    "PM",
    "AM",
    "TV",
    "PC",
    "ID",
    "PR",
    "HR",
    "IT",
    "VP",
    # Trading indicators (not tickers)
    "EMA",
    "SMA",
    "VWAP",
    "RSI",
    "MACD",
    "ATR",
    "ADX",
    "CCI",
    "MFI",
    "OBV",
    "ROC",
    "TEMA",
    "WMA",
    "DEMA",
    "HMA",
    "KAMA",
    "BB",  # Bollinger Bands
    "KC",  # Keltner Channel
    "SAR",  # Parabolic SAR
    "DMI",  # Directional Movement Index
    "PPO",  # Percentage Price Oscillator
    "TSI",  # True Strength Index
    "UO",  # Ultimate Oscillator
    "WR",  # Williams %R
    "CMF",  # Chaikin Money Flow
    "EMV",  # Ease of Movement
    "FI",  # Force Index
    "NVI",  # Negative Volume Index
    "PVI",  # Positive Volume Index
    "TRIX",
    "VROC",  # Volume Rate of Change
    "VI",  # Vortex Indicator
    # Trading terms/abbreviations (not tickers)
    "SMB",  # SMB Capital (company name)
    "HOD",  # High of day
    "LOD",  # Low of day
    "ATH",  # All time high
    "ATL",  # All time low
    "EOD",  # End of day
    "RTH",  # Regular trading hours
    "AH",  # After hours
    "PM",  # Pre-market
    "NYSE",
    "NASD",
    "OTC",
    "DMA",  # Direct market access
    "HFT",  # High frequency trading
    "MM",  # Market maker
    "TA",  # Technical analysis
    "FA",  # Fundamental analysis
    "DD",  # Due diligence
    "PE",  # Price to earnings
    "EPS",  # Earnings per share
    "ROI",  # Return on investment
    "PNL",  # Profit and loss
    "RR",  # Risk reward
    "SL",  # Stop loss
    "TP",  # Take profit
    "BE",  # Break even
    "ES",  # E-mini S&P (futures, but commonly used as abbreviation)
    "NQ",  # E-mini Nasdaq
    "YM",  # E-mini Dow
    "CL",  # Crude oil futures
    "GC",  # Gold futures
    "ZB",  # Bond futures
    "IV",  # Implied volatility
    "HV",  # Historical volatility
    "OI",  # Open interest
    "DTE",  # Days to expiration
    "ITM",  # In the money
    "OTM",  # Out of the money
    "ATM",  # At the money
}


def extract_key_concepts(chunks: list[dict]) -> dict:
    """Extract key trading concepts and tickers from chunk content."""
    combined_text = " ".join(c.get("content", "") for c in chunks).lower()

    # Find matching concepts
    found_concepts = {}
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
    found_concepts = dict(
        sorted(found_concepts.items(), key=lambda x: x[1]["count"], reverse=True)
    )

    # Find potential tickers
    combined_upper = " ".join(c.get("content", "") for c in chunks)
    ticker_matches = TICKER_PATTERN.findall(combined_upper)
    tickers = {}
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
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        # Fetch document
        doc = await conn.fetchrow(
            """
            SELECT
                id, workspace_id, title, author, source_type, source_url,
                canonical_url, language, duration_secs, published_at,
                created_at, updated_at, status, content_hash
            FROM documents
            WHERE id = $1
            """,
            doc_id,
        )

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )

        # Fetch chunks
        chunks = await conn.fetch(
            """
            SELECT
                id, chunk_index, content, token_count,
                time_start_secs, time_end_secs,
                page_start, page_end
            FROM chunks
            WHERE doc_id = $1
            ORDER BY chunk_index
            """,
            doc_id,
        )

        # Fetch chunk validation status if exists
        chunk_validations = await conn.fetch(
            """
            SELECT chunk_id, status, notes, validated_at
            FROM chunk_validations
            WHERE chunk_id = ANY($1::uuid[])
            """,
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
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    if validation.status not in ("verified", "needs_review", "garbage"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid status. Must be: verified, needs_review, garbage",
        )

    async with _db_pool.acquire() as conn:
        # Verify chunk exists
        chunk = await conn.fetchrow(
            "SELECT id, doc_id FROM chunks WHERE id = $1", chunk_id
        )
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found",
            )

        # Upsert validation
        await conn.execute(
            """
            INSERT INTO chunk_validations (chunk_id, status, notes, validated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (chunk_id) DO UPDATE SET
                status = EXCLUDED.status,
                notes = EXCLUDED.notes,
                validated_at = NOW()
            """,
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
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        # Verify document exists
        doc = await conn.fetchrow(
            "SELECT id, title FROM documents WHERE id = $1", doc_id
        )
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )

        # Fetch chunks
        chunks = await conn.fetch(
            "SELECT content FROM chunks WHERE doc_id = $1",
            doc_id,
        )

    chunks_list = [dict(c) for c in chunks]
    key_concepts = extract_key_concepts(chunks_list)

    return {
        "doc_id": str(doc_id),
        "title": doc["title"],
        "concepts": key_concepts["concepts"],
        "tickers": key_concepts["tickers"],
    }
