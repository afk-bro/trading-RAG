"""Tests for unified ingest detection logic."""

import pytest
from fastapi import HTTPException

from app.services.ingest.detection import (
    DetectedSource,
    detect_source_type,
    is_youtube_url,
    url_is_pdf,
)


class TestIsYoutubeUrl:
    """Tests for YouTube URL detection."""

    def test_standard_youtube_url(self):
        """Standard youtube.com watch URL."""
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("http://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_without_www(self):
        """YouTube URL without www prefix."""
        assert is_youtube_url("https://youtube.com/watch?v=abc123")

    def test_mobile_youtube_url(self):
        """Mobile YouTube URL (m.youtube.com)."""
        assert is_youtube_url("https://m.youtube.com/watch?v=abc123")

    def test_youtu_be_short_url(self):
        """Short youtu.be URL."""
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_youtube_url("http://youtu.be/abc123")

    def test_youtube_music(self):
        """YouTube Music URL."""
        assert is_youtube_url("https://music.youtube.com/watch?v=abc123")

    def test_youtube_playlist(self):
        """YouTube playlist URL."""
        assert is_youtube_url("https://www.youtube.com/playlist?list=PLxyz123")

    def test_youtube_embed(self):
        """YouTube embed URL."""
        assert is_youtube_url("https://www.youtube.com/embed/abc123")

    def test_non_youtube_url(self):
        """Non-YouTube URLs should return False."""
        assert not is_youtube_url("https://vimeo.com/123456")
        assert not is_youtube_url("https://example.com")
        assert not is_youtube_url("https://youtube.example.com/video")

    def test_case_insensitive(self):
        """URL detection should be case insensitive."""
        assert is_youtube_url("HTTPS://WWW.YOUTUBE.COM/watch?v=abc")
        assert is_youtube_url("https://YOUTU.BE/abc123")


class TestUrlIsPdf:
    """Tests for PDF URL detection."""

    def test_simple_pdf_url(self):
        """Simple .pdf URL."""
        assert url_is_pdf("https://example.com/document.pdf")

    def test_pdf_url_with_path(self):
        """PDF URL with path segments."""
        assert url_is_pdf("https://example.com/docs/files/report.pdf")

    def test_pdf_url_with_querystring(self):
        """PDF URL with querystring should still be detected."""
        assert url_is_pdf("https://example.com/file.pdf?download=1")
        assert url_is_pdf("https://example.com/doc.pdf?token=abc&dl=true")

    def test_pdf_url_case_insensitive(self):
        """PDF detection should be case insensitive."""
        assert url_is_pdf("https://example.com/DOC.PDF")
        assert url_is_pdf("https://example.com/file.Pdf")

    def test_non_pdf_url(self):
        """Non-PDF URLs should return False."""
        assert not url_is_pdf("https://example.com/document.docx")
        assert not url_is_pdf("https://example.com/file.txt")
        assert not url_is_pdf("https://example.com/")

    def test_pdf_in_path_but_not_extension(self):
        """URL with 'pdf' in path but not as extension."""
        assert not url_is_pdf("https://example.com/pdf/viewer")
        assert not url_is_pdf("https://example.com/pdf-files/doc.html")


class TestDetectSourceType:
    """Tests for source type detection."""

    # URL detection tests

    def test_detect_youtube_url(self):
        """YouTube URLs should be detected."""
        result = detect_source_type(
            url="https://www.youtube.com/watch?v=abc123",
            filename=None,
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.YOUTUBE

    def test_detect_pdf_url(self):
        """PDF URLs should be detected."""
        result = detect_source_type(
            url="https://example.com/doc.pdf",
            filename=None,
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.PDF_URL

    def test_detect_pdf_url_ignores_querystring(self):
        """PDF URL detection should ignore querystring."""
        result = detect_source_type(
            url="https://example.com/x.pdf?download=1",
            filename=None,
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.PDF_URL

    def test_detect_article_url(self):
        """Non-YouTube, non-PDF URLs should be article."""
        result = detect_source_type(
            url="https://example.com/blog/article",
            filename=None,
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.ARTICLE_URL

    # File detection tests

    def test_detect_pdf_file(self):
        """PDF files should be detected."""
        result = detect_source_type(
            url=None,
            filename="document.pdf",
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.PDF_FILE

    def test_detect_text_file(self):
        """Text files should be detected."""
        result = detect_source_type(
            url=None,
            filename="notes.txt",
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.TEXT_FILE

    def test_detect_markdown_file(self):
        """Markdown files should be detected as text."""
        result = detect_source_type(
            url=None,
            filename="readme.md",
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.TEXT_FILE

        result = detect_source_type(
            url=None,
            filename="doc.markdown",
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.TEXT_FILE

    def test_detect_pine_file(self):
        """Pine script files should be detected."""
        result = detect_source_type(
            url=None,
            filename="strategy.pine",
            content=None,
            source_type_override=None,
        )
        assert result == DetectedSource.PINE_FILE

    def test_detect_unsupported_file(self):
        """Unsupported file types should raise 400."""
        with pytest.raises(HTTPException) as exc:
            detect_source_type(
                url=None,
                filename="doc.docx",
                content=None,
                source_type_override=None,
            )
        assert exc.value.status_code == 400
        assert "Unsupported file type" in exc.value.detail

    # Content detection tests

    def test_detect_text_content(self):
        """Raw content should be detected as text."""
        result = detect_source_type(
            url=None,
            filename=None,
            content="This is some text content",
            source_type_override=None,
        )
        assert result == DetectedSource.TEXT_CONTENT

    # Validation tests

    def test_no_input_raises_422(self):
        """No input should raise 422."""
        with pytest.raises(HTTPException) as exc:
            detect_source_type(
                url=None,
                filename=None,
                content=None,
                source_type_override=None,
            )
        assert exc.value.status_code == 422

    # Override tests

    def test_override_youtube(self):
        """Override with valid YouTube type."""
        result = detect_source_type(
            url="https://youtube.com/watch?v=abc",
            filename=None,
            content=None,
            source_type_override="youtube",
        )
        assert result == DetectedSource.YOUTUBE

    def test_override_requires_matching_input_file(self):
        """Override pdf_file without file should raise 400."""
        with pytest.raises(HTTPException) as exc:
            detect_source_type(
                url="https://example.com",
                filename=None,
                content=None,
                source_type_override="pdf_file",
            )
        assert exc.value.status_code == 400
        assert "requires file upload" in exc.value.detail

    def test_override_requires_matching_input_url(self):
        """Override youtube without url should raise 400."""
        with pytest.raises(HTTPException) as exc:
            detect_source_type(
                url=None,
                filename="file.txt",
                content=None,
                source_type_override="youtube",
            )
        assert exc.value.status_code == 400
        assert "requires url" in exc.value.detail

    def test_invalid_override_value(self):
        """Invalid override value should raise 400."""
        with pytest.raises(HTTPException) as exc:
            detect_source_type(
                url="https://example.com",
                filename=None,
                content=None,
                source_type_override="invalid_type",
            )
        assert exc.value.status_code == 400
        assert "Invalid source_type override" in exc.value.detail
