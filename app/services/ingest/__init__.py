"""Ingest services for document processing and health validation."""

from app.services.ingest.detection import (
    DetectedSource,
    detect_source_type,
    is_youtube_url,
    url_is_pdf,
)
from app.services.ingest.health import (
    HealthCheckResult,
    HealthResult,
    HealthStatus,
    get_source_health_summary,
    validate_source_health,
)
from app.services.ingest.text import (
    TextContent,
    extract_markdown_title,
    extract_text_content,
    extract_text_title,
    ingest_text_content,
    ingest_text_file,
)

__all__ = [
    # Detection
    "DetectedSource",
    "detect_source_type",
    "is_youtube_url",
    "url_is_pdf",
    # Health
    "HealthCheckResult",
    "HealthResult",
    "HealthStatus",
    "get_source_health_summary",
    "validate_source_health",
    # Text
    "TextContent",
    "extract_markdown_title",
    "extract_text_content",
    "extract_text_title",
    "ingest_text_content",
    "ingest_text_file",
]
