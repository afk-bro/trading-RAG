"""Constants for key concepts extraction in admin documents.

Contains trading terminology dictionaries and ticker exclusion lists
used for automatic concept detection from ingested content.
"""

import re
from enum import Enum

# =============================================================================
# Validation Status Enum
# =============================================================================


class ValidationStatus(str, Enum):
    """Validation status for chunks."""

    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    GARBAGE = "garbage"


VALID_STATUSES = frozenset(s.value for s in ValidationStatus)


# =============================================================================
# Trading Concepts Dictionary
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


# =============================================================================
# Ticker Detection
# =============================================================================

# Ticker pattern (1-5 uppercase letters, optionally with $ prefix)
# Uses negative lookbehind to prevent matching mid-word (e.g., "RRENT" from "CURRENT")
TICKER_PATTERN = re.compile(r"(?<![A-Za-z])\$?([A-Z]{1,5})\b")

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
    # Finance/regulatory terms
    "PDT",  # Pattern Day Trader
    "FX",  # Forex/Foreign Exchange
    "FINRA",  # Financial Industry Regulatory Authority
    "CMT",  # Chartered Market Technician
    "CPA",  # Certified Public Accountant
    "CFP",  # Certified Financial Planner
    "CFA",  # Chartered Financial Analyst
    "CFTC",  # Commodity Futures Trading Commission
    "FDIC",  # Federal Deposit Insurance Corporation
    "SIPC",  # Securities Investor Protection Corporation
    "DTCC",  # Depository Trust & Clearing Corporation
    "CBOE",  # Chicago Board Options Exchange (exchange, not ticker)
    "CME",  # Chicago Mercantile Exchange
    "ICE",  # Intercontinental Exchange
    "NYMEX",  # New York Mercantile Exchange
}


# =============================================================================
# SQL Queries
# =============================================================================

SQL_GET_DOCUMENT = """
SELECT
    id, workspace_id, title, author, source_type, source_url,
    canonical_url, language, duration_secs, published_at,
    created_at, updated_at, status, content_hash
FROM documents
WHERE id = $1
"""

SQL_GET_CHUNKS = """
SELECT
    id, chunk_index, content, token_count,
    time_start_secs, time_end_secs,
    page_start, page_end
FROM chunks
WHERE doc_id = $1
ORDER BY chunk_index
"""

SQL_GET_CHUNK_VALIDATIONS = """
SELECT chunk_id, status, notes, validated_at
FROM chunk_validations
WHERE chunk_id = ANY($1::uuid[])
"""

SQL_GET_CHUNK_BY_ID = "SELECT id, doc_id FROM chunks WHERE id = $1"

SQL_UPSERT_VALIDATION = """
INSERT INTO chunk_validations (chunk_id, status, notes, validated_at)
VALUES ($1, $2, $3, NOW())
ON CONFLICT (chunk_id) DO UPDATE SET
    status = EXCLUDED.status,
    notes = EXCLUDED.notes,
    validated_at = NOW()
"""

SQL_GET_DOCUMENT_TITLE = "SELECT id, title FROM documents WHERE id = $1"

SQL_GET_CHUNK_CONTENT = "SELECT content FROM chunks WHERE doc_id = $1"


__all__ = [
    # Enums
    "ValidationStatus",
    "VALID_STATUSES",
    # Concepts
    "TRADING_CONCEPTS",
    # Ticker detection
    "TICKER_PATTERN",
    "NON_TICKERS",
    # SQL queries
    "SQL_GET_DOCUMENT",
    "SQL_GET_CHUNKS",
    "SQL_GET_CHUNK_VALIDATIONS",
    "SQL_GET_CHUNK_BY_ID",
    "SQL_UPSERT_VALIDATION",
    "SQL_GET_DOCUMENT_TITLE",
    "SQL_GET_CHUNK_CONTENT",
]
