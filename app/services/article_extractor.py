"""Web article extraction service using trafilatura."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx
import structlog
import trafilatura

logger = structlog.get_logger(__name__)

# Maximum response size (10 MB)
MAX_RESPONSE_SIZE = 10_000_000

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 15.0

# User agent to avoid blocks
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class ExtractedArticle:
    """Extracted article content and metadata."""

    url: str
    title: Optional[str]
    text: str
    author: Optional[str]
    published_at: Optional[datetime]
    html: Optional[str] = None  # For debugging


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime, returning None on failure."""
    if not date_str:
        return None

    # Common date formats to try
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue

    return None


def _extract_metadata(html: str) -> dict:
    """
    Extract metadata from HTML defensively.

    Wraps trafilatura metadata extraction to handle API variations.
    """
    metadata = {"title": None, "author": None, "date": None}

    try:
        # Try trafilatura's metadata extraction
        from trafilatura.metadata import extract_metadata

        result = extract_metadata(html)
        if result:
            metadata["title"] = getattr(result, "title", None)
            metadata["author"] = getattr(result, "author", None)
            metadata["date"] = getattr(result, "date", None)
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("Metadata extraction fallback", error=str(e))

    return metadata


async def extract_article(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    include_html: bool = False,
) -> ExtractedArticle:
    """
    Extract article content from a URL.

    Args:
        url: Article URL to extract
        timeout: HTTP request timeout in seconds
        include_html: Include raw HTML in response (for debugging)

    Returns:
        ExtractedArticle with text content and metadata

    Raises:
        ValueError: If extraction fails or response too large
        httpx.HTTPError: On HTTP request failures
    """
    logger.info("Extracting article", url=url)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        response.raise_for_status()

        # Guard against huge responses
        content_length = len(response.content)
        if content_length > MAX_RESPONSE_SIZE:
            raise ValueError(
                f"Response too large: {content_length} bytes "
                f"(max {MAX_RESPONSE_SIZE})"
            )

        html = response.text

        # Extract main content
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )

        if not text:
            raise ValueError(f"Unable to extract article content from {url}")

        # Extract metadata defensively
        metadata = _extract_metadata(html)

        logger.info(
            "Article extracted",
            url=str(response.url),
            title=metadata.get("title"),
            text_length=len(text),
        )

        return ExtractedArticle(
            url=str(response.url),  # Use final URL after redirects
            title=metadata.get("title"),
            text=text,
            author=metadata.get("author"),
            published_at=_parse_date(metadata.get("date")),
            html=html if include_html else None,
        )
