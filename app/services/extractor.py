"""Metadata extraction service for symbols, entities, and topics."""

import re
from dataclasses import dataclass, field

# Common stock ticker patterns
TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
TICKER_WITH_DOLLAR = re.compile(r"\$([A-Za-z]{1,5})\b")

# Known valid tickers (expandable allowlist)
VALID_TICKERS = {
    # Major indices/ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VOO",
    "VXX",
    "UVXY",
    "SQQQ",
    "TQQQ",
    # Mega caps
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "BRK",
    # Tech
    "AMD",
    "INTC",
    "AVGO",
    "ORCL",
    "CRM",
    "ADBE",
    "NFLX",
    "PYPL",
    "SQ",
    "SHOP",
    # Finance
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "BLK",
    "SCHW",
    "AXP",
    "V",
    "MA",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "MRK",
    "ABBV",
    "LLY",
    "TMO",
    "ABT",
    "BMY",
    "GILD",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "MPC",
    "VLO",
    "PSX",
    "OXY",
    "HAL",
    # Consumer
    "WMT",
    "COST",
    "HD",
    "TGT",
    "LOW",
    "SBUX",
    "MCD",
    "NKE",
    "DIS",
    "CMCSA",
    # Industrial
    "CAT",
    "DE",
    "BA",
    "HON",
    "UPS",
    "FDX",
    "GE",
    "MMM",
    "LMT",
    "RTX",
    # Crypto related
    "COIN",
    "MSTR",
    "RIOT",
    "MARA",
    # Other popular
    "GME",
    "AMC",
    "PLTR",
    "SOFI",
    "RIVN",
    "LCID",
}

# Common false positives to exclude
EXCLUDED_WORDS = {
    "I",
    "A",
    "AN",
    "AM",
    "AS",
    "AT",
    "BE",
    "BY",
    "DO",
    "GO",
    "IF",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "NO",
    "OF",
    "ON",
    "OR",
    "SO",
    "TO",
    "UP",
    "US",
    "WE",
    "AI",
    "CEO",
    "CFO",
    "CTO",
    "IPO",
    "GDP",
    "CPI",
    "PPI",
    "PCE",
    "FED",
    "SEC",
    "FBI",
    "CIA",
    "USA",
    "UK",
    "EU",
    "UN",
    "IMF",
    "WHO",
    "THE",
    "AND",
    "FOR",
    "NOT",
    "BUT",
    "ARE",
    "WAS",
    "HAS",
    "HAD",
    "HIS",
    "HER",
    "YOU",
    "ALL",
    "CAN",
    "HER",
    "WAS",
    "ONE",
    "OUR",
    "OUT",
    "DAY",
    "GET",
    "HAS",
    "HIM",
    "HOW",
    "ITS",
    "LET",
    "MAY",
    "NEW",
    "NOW",
    "OLD",
    "SEE",
    "WAY",
    "WHO",
    "BOY",
    "DID",
    "OWN",
    "SAY",
    "SHE",
    "TOO",
    "USE",
    "PM",
    "EST",
    "PST",
    "UTC",
    "YTD",
    "QOQ",
    "MOM",
    "YOY",
    "ATH",
    "ATL",
}

# Entity keywords
ENTITY_KEYWORDS = {
    # Central banks
    "fed": "Federal Reserve",
    "federal reserve": "Federal Reserve",
    "fomc": "FOMC",
    "ecb": "ECB",
    "boj": "Bank of Japan",
    "pboc": "PBOC",
    # Key figures
    "powell": "Jerome Powell",
    "yellen": "Janet Yellen",
    "lagarde": "Christine Lagarde",
    "buffett": "Warren Buffett",
    "dimon": "Jamie Dimon",
    "musk": "Elon Musk",
    "bezos": "Jeff Bezos",
    # Institutions
    "treasury": "US Treasury",
    "congress": "US Congress",
    "sec": "SEC",
    "cftc": "CFTC",
    # Major companies
    "apple inc": "Apple Inc.",
    "apple inc.": "Apple Inc.",
    "microsoft corporation": "Microsoft Corporation",
    "microsoft corp": "Microsoft Corporation",
    "alphabet inc": "Alphabet Inc.",
    "amazon.com": "Amazon.com Inc.",
    "amazon inc": "Amazon.com Inc.",
    "meta platforms": "Meta Platforms Inc.",
    "nvidia corporation": "NVIDIA Corporation",
    "nvidia corp": "NVIDIA Corporation",
    "tesla inc": "Tesla Inc.",
    "tesla, inc": "Tesla Inc.",
    "berkshire hathaway": "Berkshire Hathaway",
    "jpmorgan chase": "JPMorgan Chase",
    "jp morgan": "JPMorgan Chase",
    "goldman sachs": "Goldman Sachs",
    "morgan stanley": "Morgan Stanley",
    "bank of america": "Bank of America",
    "wells fargo": "Wells Fargo",
    "citigroup": "Citigroup",
    "blackrock": "BlackRock",
}

# Topic keywords
TOPIC_KEYWORDS = {
    "macro": [
        "inflation",
        "gdp",
        "unemployment",
        "employment",
        "jobs",
        "recession",
        "economy",
        "economic",
        "monetary",
        "fiscal",
        "deficit",
        "debt",
        "stimulus",
        "tightening",
        "easing",
    ],
    "rates": [
        "interest rate",
        "rate hike",
        "rate cut",
        "fed funds",
        "yield",
        "yields",
        "treasury",
        "bonds",
        "duration",
        "basis points",
        "bps",
        "fomc",
        "dot plot",
    ],
    "earnings": [
        "earnings",
        "revenue",
        "eps",
        "guidance",
        "outlook",
        "beat",
        "miss",
        "quarterly",
        "q1",
        "q2",
        "q3",
        "q4",
        "profit",
        "margin",
        "growth",
        "yoy",
        "qoq",
    ],
    "tech": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "cloud",
        "software",
        "saas",
        "semiconductor",
        "chip",
        "gpu",
        "data center",
        "technology",
        "tech",
        "innovation",
        "startup",
    ],
    "crypto": [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "crypto",
        "cryptocurrency",
        "blockchain",
        "defi",
        "nft",
        "web3",
        "altcoin",
    ],
    "options": [
        "options",
        "calls",
        "puts",
        "strike",
        "expiry",
        "expiration",
        "iv",
        "implied volatility",
        "gamma",
        "delta",
        "theta",
        "vega",
        "spread",
        "straddle",
        "strangle",
        "iron condor",
    ],
    "markets": [
        "market",
        "stock",
        "equity",
        "index",
        "s&p",
        "nasdaq",
        "dow",
        "russell",
        "bull",
        "bear",
        "rally",
        "selloff",
        "correction",
        "volatility",
        "vix",
    ],
}


@dataclass
class ExtractedMetadata:
    """Extracted metadata from content."""

    symbols: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    speaker: str | None = None
    quality_score: float = 1.0


class MetadataExtractor:
    """Extract symbols, entities, and topics from text content."""

    def __init__(
        self,
        valid_tickers: set[str] | None = None,
        excluded_words: set[str] | None = None,
    ):
        """
        Initialize extractor.

        Args:
            valid_tickers: Allowlist of valid ticker symbols
            excluded_words: Words to exclude from symbol detection
        """
        self.valid_tickers = valid_tickers or VALID_TICKERS
        self.excluded_words = excluded_words or EXCLUDED_WORDS

    def extract(self, text: str) -> ExtractedMetadata:
        """
        Extract all metadata from text.

        Args:
            text: Content to analyze

        Returns:
            ExtractedMetadata with symbols, entities, topics
        """
        return ExtractedMetadata(
            symbols=self.extract_symbols(text),
            entities=self.extract_entities(text),
            topics=self.extract_topics(text),
            quality_score=self.estimate_quality(text),
        )

    def extract_symbols(self, text: str) -> list[str]:
        """
        Extract stock symbols from text.

        Uses allowlist to avoid false positives.
        """
        symbols = set()

        # Find $TICKER patterns (high confidence)
        for match in TICKER_WITH_DOLLAR.finditer(text):
            ticker = match.group(1).upper()
            if ticker in self.valid_tickers:
                symbols.add(ticker)

        # Find standalone tickers (validate against allowlist)
        for match in TICKER_PATTERN.finditer(text):
            ticker = match.group(1).upper()
            if ticker in self.valid_tickers and ticker not in self.excluded_words:
                symbols.add(ticker)

        return sorted(symbols)

    def extract_entities(self, text: str) -> list[str]:
        """
        Extract named entities from text.

        Uses keyword matching for key financial figures and institutions.
        """
        entities = set()
        text_lower = text.lower()

        for keyword, entity_name in ENTITY_KEYWORDS.items():
            if keyword in text_lower:
                entities.add(entity_name)

        return sorted(entities)

    def extract_topics(self, text: str) -> list[str]:
        """
        Classify content into topics.

        Uses keyword matching for topic classification.
        """
        topics = set()
        text_lower = text.lower()

        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.add(topic)
                    break  # One match per topic is enough

        return sorted(topics)

    def estimate_quality(self, text: str) -> float:
        """
        Estimate transcript/content quality score.

        Factors:
        - Text length
        - Sentence structure
        - Word diversity
        """
        if not text:
            return 0.0

        words = text.split()
        if len(words) < 10:
            return 0.3

        # Check for sentence structure (periods, capital letters)
        sentences = text.count(".") + text.count("!") + text.count("?")
        sentence_ratio = sentences / max(
            len(words) / 15, 1
        )  # Expect ~15 words/sentence

        # Word diversity
        unique_words = len(set(w.lower() for w in words))
        diversity = unique_words / len(words)

        # Calculate score (0.0 - 1.0)
        score = min(1.0, (sentence_ratio * 0.3) + (diversity * 0.7))

        return round(score, 2)

    def detect_speaker(self, text: str) -> str | None:
        """
        Detect speaker from text patterns.

        Looks for patterns like "Speaker:", "[Name]:", etc.
        """
        # Common speaker patterns
        patterns = [
            r"^([A-Z][a-z]+ [A-Z][a-z]+):",  # "John Smith:"
            r"^\[([A-Z][a-z]+)\]",  # "[John]"
            r"^Speaker\s*(\d+):",  # "Speaker 1:"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1)

        return None


# Singleton instance
_extractor: MetadataExtractor | None = None


def get_extractor() -> MetadataExtractor:
    """Get or create extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = MetadataExtractor()
    return _extractor
