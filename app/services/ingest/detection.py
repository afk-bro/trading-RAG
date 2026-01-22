"""Source type detection for unified ingestion endpoint."""

from enum import Enum
from urllib.parse import urlparse

from fastapi import HTTPException


class DetectedSource(str, Enum):
    """Detected source type for unified ingestion."""

    YOUTUBE = "youtube"
    PDF_URL = "pdf_url"
    ARTICLE_URL = "article_url"
    PDF_FILE = "pdf_file"
    TEXT_FILE = "text_file"
    PINE_FILE = "pine_file"
    TEXT_CONTENT = "text_content"


# Sources that require URL input
URL_SOURCES = {
    DetectedSource.YOUTUBE,
    DetectedSource.PDF_URL,
    DetectedSource.ARTICLE_URL,
}

# Sources that require file input
FILE_SOURCES = {
    DetectedSource.PDF_FILE,
    DetectedSource.TEXT_FILE,
    DetectedSource.PINE_FILE,
}


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video or playlist URL."""
    try:
        parsed = urlparse(url.lower())
        host = parsed.netloc.replace("www.", "")

        youtube_hosts = {
            "youtube.com",
            "m.youtube.com",
            "youtu.be",
            "music.youtube.com",
        }

        return host in youtube_hosts
    except Exception:
        return False


def url_is_pdf(url: str) -> bool:
    """Check if URL points to a PDF file (ignoring querystring)."""
    try:
        parsed = urlparse(url.lower())
        # Get path without querystring and check extension
        path = parsed.path.rstrip("/")
        return path.endswith(".pdf")
    except Exception:
        return False


def validate_override(source_type_override: str) -> DetectedSource:
    """Validate and convert source type override to DetectedSource enum."""
    try:
        return DetectedSource(source_type_override.lower())
    except ValueError:
        valid = [s.value for s in DetectedSource]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type override: {source_type_override}. "
            f"Valid values: {valid}",
        )


def detect_source_type(
    url: str | None,
    filename: str | None,
    content: str | None,
    source_type_override: str | None,
) -> DetectedSource:
    """
    Detect the source type based on input parameters.

    Args:
        url: URL to ingest (YouTube, article, or PDF link)
        filename: Filename of uploaded file
        content: Raw text/markdown content
        source_type_override: Override auto-detection

    Returns:
        DetectedSource enum value

    Raises:
        HTTPException: 400 for invalid override or unsupported file type
        HTTPException: 422 if no valid input provided
    """
    # Override validation: reject impossible combos
    if source_type_override:
        override = validate_override(source_type_override)

        # Check input matches override type
        if override in FILE_SOURCES:
            if not filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"Override '{override.value}' requires file upload",
                )
        elif override in URL_SOURCES:
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail=f"Override '{override.value}' requires url",
                )
        elif override == DetectedSource.TEXT_CONTENT:
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"Override '{override.value}' requires content",
                )

        return override

    # Auto-detect from URL
    if url:
        if is_youtube_url(url):
            return DetectedSource.YOUTUBE
        if url_is_pdf(url):
            return DetectedSource.PDF_URL
        return DetectedSource.ARTICLE_URL

    # Auto-detect from filename
    if filename:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            return DetectedSource.PDF_FILE
        elif ext in ("txt", "md", "markdown"):
            return DetectedSource.TEXT_FILE
        elif ext in ("pine", "pinescript"):
            return DetectedSource.PINE_FILE
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{ext}. "
                "Supported: .pdf, .txt, .md, .pine",
            )

    # Raw content
    if content:
        return DetectedSource.TEXT_CONTENT

    # No valid input
    raise HTTPException(
        status_code=422,
        detail="Must provide exactly one of: url, file, or content",
    )
