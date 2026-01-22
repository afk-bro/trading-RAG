"""Text and Markdown ingestion service."""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import structlog
from fastapi import UploadFile

from app.schemas import IngestResponse, SourceType

logger = structlog.get_logger(__name__)

# Maximum title length
MAX_TITLE_LENGTH = 200


@dataclass
class TextContent:
    """Extracted text content and metadata."""

    text: str
    title: Optional[str]
    is_markdown: bool
    content_hash: str


def extract_markdown_title(content: str) -> Optional[str]:
    """
    Extract title from markdown content.

    Looks for first heading (# Title) in the content.
    """
    # Match first # heading (not ## or more)
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        # Remove any trailing markdown formatting
        title = re.sub(r"\*+|_+|`+", "", title)
        return title[:MAX_TITLE_LENGTH] if len(title) > MAX_TITLE_LENGTH else title

    return None


def extract_text_title(content: str) -> Optional[str]:
    """
    Extract title from plain text content.

    Uses first non-empty line, truncated.
    """
    for line in content.split("\n"):
        line = line.strip()
        if line:
            return line[:MAX_TITLE_LENGTH] if len(line) > MAX_TITLE_LENGTH else line

    return None


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_markdown_file(filename: str) -> bool:
    """Check if filename indicates markdown content."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in ("md", "markdown")


async def extract_text_content(
    file: Optional[UploadFile] = None,
    content: Optional[str] = None,
    filename: Optional[str] = None,
    title_override: Optional[str] = None,
) -> TextContent:
    """
    Extract text content from file upload or raw content.

    Args:
        file: Uploaded text/markdown file
        content: Raw text/markdown content
        filename: Filename (for detecting markdown)
        title_override: Override auto-detected title

    Returns:
        TextContent with text, title, and metadata

    Raises:
        ValueError: If no content provided
    """
    # Get content from file or raw input
    if file:
        raw_bytes = await file.read()
        text = raw_bytes.decode("utf-8")
        filename = filename or file.filename or "unknown.txt"
    elif content:
        text = content
        filename = filename or "content.txt"
    else:
        raise ValueError("Must provide file or content")

    # Detect markdown
    is_markdown = is_markdown_file(filename)

    # Extract title
    if title_override:
        title = title_override
    elif is_markdown:
        title = extract_markdown_title(text)
    else:
        title = extract_text_title(text)

    # Compute hash
    content_hash = compute_content_hash(text)

    logger.info(
        "Text content extracted",
        filename=filename,
        is_markdown=is_markdown,
        title=title,
        text_length=len(text),
    )

    return TextContent(
        text=text,
        title=title,
        is_markdown=is_markdown,
        content_hash=content_hash,
    )


async def ingest_text_file(
    workspace_id: UUID,
    file: UploadFile,
    title_override: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    ingest_pipeline_func=None,
    settings=None,
) -> IngestResponse:
    """
    Ingest a text or markdown file.

    Args:
        workspace_id: Target workspace
        file: Uploaded file
        title_override: Override auto-detected title
        idempotency_key: Idempotency key for deduplication
        ingest_pipeline_func: Ingest pipeline function to call
        settings: App settings

    Returns:
        IngestResponse from pipeline
    """
    # Extract content
    extracted = await extract_text_content(
        file=file,
        title_override=title_override,
    )

    # Build canonical URL
    canonical_url = f"text://{extracted.content_hash[:32]}"

    # Call ingest pipeline
    return await ingest_pipeline_func(
        workspace_id=workspace_id,
        content=extracted.text,
        source_type=(
            SourceType.NOTE if not extracted.is_markdown else SourceType.ARTICLE
        ),
        source_url=None,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or canonical_url,
        content_hash=extracted.content_hash,
        title=extracted.title,
        author=None,
        published_at=None,
        language="en",
        settings=settings,
    )


async def ingest_text_content(
    workspace_id: UUID,
    content: str,
    title_override: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    ingest_pipeline_func=None,
    settings=None,
) -> IngestResponse:
    """
    Ingest raw text/markdown content.

    Args:
        workspace_id: Target workspace
        content: Raw text content
        title_override: Override auto-detected title
        idempotency_key: Idempotency key for deduplication
        ingest_pipeline_func: Ingest pipeline function to call
        settings: App settings

    Returns:
        IngestResponse from pipeline
    """
    # Extract content
    extracted = await extract_text_content(
        content=content,
        title_override=title_override,
    )

    # Build canonical URL
    canonical_url = f"text://{extracted.content_hash[:32]}"

    # Call ingest pipeline
    return await ingest_pipeline_func(
        workspace_id=workspace_id,
        content=extracted.text,
        source_type=SourceType.NOTE,
        source_url=None,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or canonical_url,
        content_hash=extracted.content_hash,
        title=extracted.title,
        author=None,
        published_at=None,
        language="en",
        settings=settings,
    )
