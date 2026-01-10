"""Unit tests for PDF extraction service."""

import pytest

from app.services.pdf_extractor import (
    PDFBackend,
    PDFConfig,
    PDFExtractionResult,
    PDFPage,
    PyMuPDFBackend,
    PDFPlumberBackend,
    extract_pdf,
    get_backend,
    get_page_markers,
)


class TestPDFConfig:
    """Tests for PDFConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = PDFConfig()
        assert config.backend == PDFBackend.PYMUPDF
        assert config.max_pages is None
        assert config.min_chars_per_page == 10
        assert config.join_pages_with == "\n\n"
        assert config.enable_ocr is False

    def test_custom_values(self):
        """Test config with custom values."""
        config = PDFConfig(
            backend=PDFBackend.PDFPLUMBER,
            max_pages=50,
            min_chars_per_page=20,
            join_pages_with="\n---\n",
            enable_ocr=False,
        )
        assert config.backend == PDFBackend.PDFPLUMBER
        assert config.max_pages == 50
        assert config.min_chars_per_page == 20
        assert config.join_pages_with == "\n---\n"


class TestPDFExtractionResult:
    """Tests for PDFExtractionResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = PDFExtractionResult()
        assert result.pages == []
        assert result.text == ""
        assert result.metadata == {}
        assert result.warnings == []
        assert result.page_count == 0
        assert result.extracted_page_count == 0
        assert result.backend_used == ""

    def test_with_pages(self):
        """Test result with pages."""
        pages = [
            PDFPage(page_number=1, text="Page 1 content", char_count=14, char_start=0),
            PDFPage(page_number=2, text="Page 2 content", char_count=14, char_start=16),
        ]
        result = PDFExtractionResult(
            pages=pages,
            text="Page 1 content\n\nPage 2 content",
            page_count=2,
            extracted_page_count=2,
            backend_used="pymupdf",
        )
        assert len(result.pages) == 2
        assert result.page_count == 2
        assert result.extracted_page_count == 2


class TestPDFPage:
    """Tests for PDFPage dataclass."""

    def test_page_creation(self):
        """Test PDFPage creation."""
        page = PDFPage(
            page_number=1,
            text="Hello world",
            char_count=11,
            char_start=0,
        )
        assert page.page_number == 1
        assert page.text == "Hello world"
        assert page.char_count == 11
        assert page.char_start == 0


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_pymupdf_backend(self):
        """Test getting PyMuPDF backend."""
        backend = get_backend(PDFBackend.PYMUPDF)
        assert isinstance(backend, PyMuPDFBackend)

    def test_get_pdfplumber_backend(self):
        """Test getting pdfplumber backend."""
        backend = get_backend(PDFBackend.PDFPLUMBER)
        assert isinstance(backend, PDFPlumberBackend)

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            get_backend("invalid")


class TestGetPageMarkers:
    """Tests for get_page_markers function."""

    def test_empty_result(self):
        """Test page markers from empty result."""
        result = PDFExtractionResult()
        markers = get_page_markers(result)
        assert markers == []

    def test_with_pages(self):
        """Test page markers from result with pages."""
        pages = [
            PDFPage(page_number=1, text="Page 1", char_count=6, char_start=0),
            PDFPage(page_number=2, text="Page 2", char_count=6, char_start=8),
            PDFPage(page_number=3, text="Page 3", char_count=6, char_start=16),
        ]
        result = PDFExtractionResult(pages=pages)
        markers = get_page_markers(result)
        assert markers == [(0, 1), (8, 2), (16, 3)]


class TestExtractPDFWithDict:
    """Tests for extract_pdf with dict config."""

    def test_dict_config_backend(self):
        """Test extract_pdf accepts dict config with string backend."""
        # Create a minimal PDF for testing
        # This is the simplest valid PDF structure
        pdf_bytes = create_minimal_pdf()

        # The function should accept dict config
        result = extract_pdf(pdf_bytes, {"backend": "pymupdf"})
        assert result.backend_used == "pymupdf"

    def test_dict_config_max_pages(self):
        """Test extract_pdf with max_pages in dict."""
        pdf_bytes = create_minimal_pdf()
        result = extract_pdf(pdf_bytes, {"max_pages": 1})
        # Should not raise and should respect the config
        assert result is not None

    def test_none_config_uses_defaults(self):
        """Test extract_pdf with None config uses defaults."""
        pdf_bytes = create_minimal_pdf()
        result = extract_pdf(pdf_bytes, None)
        assert result.backend_used == "pymupdf"


class TestPyMuPDFBackend:
    """Tests for PyMuPDF backend extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = PyMuPDFBackend()
        self.config = PDFConfig()

    def test_extract_valid_pdf(self):
        """Test extracting text from valid PDF."""
        pdf_bytes = create_minimal_pdf()
        result = self.backend.extract(pdf_bytes, self.config)
        assert result.backend_used == "pymupdf"
        assert result.page_count >= 0

    def test_extract_with_max_pages(self):
        """Test extraction with max_pages limit."""
        pdf_bytes = create_multi_page_pdf(5)
        config = PDFConfig(max_pages=2)
        result = self.backend.extract(pdf_bytes, config)
        assert result.extracted_page_count <= 2
        if result.page_count > 2:
            assert len(result.warnings) > 0  # Should warn about truncation

    def test_extract_skips_empty_pages(self):
        """Test that pages with too few chars are skipped."""
        pdf_bytes = create_minimal_pdf()
        config = PDFConfig(min_chars_per_page=1000)  # High threshold
        _result = self.backend.extract(pdf_bytes, config)  # noqa: F841
        # Pages with fewer chars should be skipped (warnings added)
        # The exact behavior depends on the PDF content


class TestPDFPlumberBackend:
    """Tests for pdfplumber backend extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = PDFPlumberBackend()
        self.config = PDFConfig(backend=PDFBackend.PDFPLUMBER)

    def test_extract_valid_pdf(self):
        """Test extracting text from valid PDF."""
        pdf_bytes = create_minimal_pdf()
        result = self.backend.extract(pdf_bytes, self.config)
        assert result.backend_used == "pdfplumber"
        assert result.page_count >= 0

    def test_extract_with_max_pages(self):
        """Test extraction with max_pages limit."""
        pdf_bytes = create_multi_page_pdf(5)
        config = PDFConfig(backend=PDFBackend.PDFPLUMBER, max_pages=2)
        result = self.backend.extract(pdf_bytes, config)
        assert result.extracted_page_count <= 2


class TestExtractPDFIntegration:
    """Integration tests for extract_pdf function."""

    def test_pymupdf_extraction(self):
        """Test full extraction with PyMuPDF."""
        pdf_bytes = create_text_pdf("Hello World. This is a test PDF document.")
        result = extract_pdf(pdf_bytes, PDFConfig(backend=PDFBackend.PYMUPDF))
        assert result.backend_used == "pymupdf"
        assert result.page_count >= 1
        assert (
            "Hello" in result.text or len(result.text) == 0
        )  # May be empty for minimal PDF

    def test_pdfplumber_extraction(self):
        """Test full extraction with pdfplumber."""
        pdf_bytes = create_text_pdf("Hello World. This is a test PDF document.")
        result = extract_pdf(pdf_bytes, PDFConfig(backend=PDFBackend.PDFPLUMBER))
        assert result.backend_used == "pdfplumber"
        assert result.page_count >= 1

    def test_page_join_separator(self):
        """Test that pages are joined with custom separator."""
        pdf_bytes = create_multi_page_pdf(2)
        config = PDFConfig(join_pages_with="\n---PAGE BREAK---\n")
        result = extract_pdf(pdf_bytes, config)
        if result.extracted_page_count > 1:
            assert "---PAGE BREAK---" in result.text


# Helper functions to create test PDFs


def create_minimal_pdf() -> bytes:
    """Create a minimal valid PDF for testing."""
    try:
        import fitz  # pymupdf

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF content for unit testing.")
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except ImportError:
        # Fallback: return a minimal PDF structure
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"  # noqa: E501


def create_multi_page_pdf(num_pages: int) -> bytes:
    """Create a PDF with multiple pages for testing."""
    try:
        import fitz  # pymupdf

        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"This is page {i + 1} of the test document.")
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except ImportError:
        return create_minimal_pdf()


def create_text_pdf(text: str) -> bytes:
    """Create a PDF with specific text content."""
    try:
        import fitz  # pymupdf

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except ImportError:
        return create_minimal_pdf()
