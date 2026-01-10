"""PDF extraction service with swappable backends.

Provides a unified interface for extracting text from PDFs.
Currently supports PyMuPDF and pdfplumber backends.
Designed to be swapped to MinerU or external services later.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class PDFBackend(str, Enum):
    """Available PDF extraction backends."""

    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"


@dataclass
class PDFConfig:
    """Configuration for PDF extraction.

    Matches workspace.config.pdf schema.
    """

    backend: PDFBackend = PDFBackend.PYMUPDF
    max_pages: int | None = None  # None = no limit
    min_chars_per_page: int = 10  # Skip pages with fewer chars
    join_pages_with: str = "\n\n"
    enable_ocr: bool = False  # Reserved for future use


@dataclass
class PDFPage:
    """Extracted content from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    char_count: int
    char_start: int  # Character offset in full document


@dataclass
class PDFExtractionResult:
    """Result of PDF extraction."""

    pages: list[PDFPage] = field(default_factory=list)
    text: str = ""  # Full joined text
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    page_count: int = 0
    extracted_page_count: int = 0
    backend_used: str = ""


class PDFExtractorBackend(ABC):
    """Abstract base class for PDF extraction backends."""

    @abstractmethod
    def extract(self, file_bytes: bytes, config: PDFConfig) -> PDFExtractionResult:
        """Extract text from PDF bytes."""


class PyMuPDFBackend(PDFExtractorBackend):
    """PyMuPDF (fitz) based extraction backend."""

    def extract(self, file_bytes: bytes, config: PDFConfig) -> PDFExtractionResult:
        """Extract text using PyMuPDF."""
        try:
            import fitz  # pymupdf
        except ImportError as e:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf") from e

        result = PDFExtractionResult(backend_used="pymupdf")
        warnings: list[str] = []

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            result.page_count = len(doc)
            result.metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "mod_date": doc.metadata.get("modDate", ""),
            }

            pages_to_extract = result.page_count
            if config.max_pages:
                pages_to_extract = min(pages_to_extract, config.max_pages)
                if result.page_count > config.max_pages:
                    warnings.append(
                        f"Truncated: {result.page_count} pages, extracted {config.max_pages}"
                    )

            char_offset = 0
            for page_num in range(pages_to_extract):
                page = doc[page_num]
                page_text = page.get_text("text")
                char_count = len(page_text.strip())

                if char_count < config.min_chars_per_page:
                    warnings.append(
                        f"Page {page_num + 1}: skipped (only {char_count} chars)"
                    )
                    continue

                result.pages.append(
                    PDFPage(
                        page_number=page_num + 1,
                        text=page_text,
                        char_count=char_count,
                        char_start=char_offset,
                    )
                )
                char_offset += char_count + len(config.join_pages_with)

            doc.close()

        except Exception as e:
            logger.error("PyMuPDF extraction failed", error=str(e))
            warnings.append(f"Extraction error: {str(e)}")

        result.extracted_page_count = len(result.pages)
        result.text = config.join_pages_with.join(p.text for p in result.pages)
        result.warnings = warnings

        return result


class PDFPlumberBackend(PDFExtractorBackend):
    """pdfplumber based extraction backend."""

    def extract(self, file_bytes: bytes, config: PDFConfig) -> PDFExtractionResult:
        """Extract text using pdfplumber."""
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError(
                "pdfplumber not installed. Run: pip install pdfplumber"
            ) from e

        import io

        result = PDFExtractionResult(backend_used="pdfplumber")
        warnings: list[str] = []

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                result.page_count = len(pdf.pages)
                result.metadata = pdf.metadata or {}

                pages_to_extract = result.page_count
                if config.max_pages:
                    pages_to_extract = min(pages_to_extract, config.max_pages)
                    if result.page_count > config.max_pages:
                        warnings.append(
                            f"Truncated: {result.page_count} pages, extracted {config.max_pages}"
                        )

                char_offset = 0
                for page_num in range(pages_to_extract):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text() or ""
                    char_count = len(page_text.strip())

                    if char_count < config.min_chars_per_page:
                        warnings.append(
                            f"Page {page_num + 1}: skipped (only {char_count} chars)"
                        )
                        continue

                    result.pages.append(
                        PDFPage(
                            page_number=page_num + 1,
                            text=page_text,
                            char_count=char_count,
                            char_start=char_offset,
                        )
                    )
                    char_offset += char_count + len(config.join_pages_with)

        except Exception as e:
            logger.error("pdfplumber extraction failed", error=str(e))
            warnings.append(f"Extraction error: {str(e)}")

        result.extracted_page_count = len(result.pages)
        result.text = config.join_pages_with.join(p.text for p in result.pages)
        result.warnings = warnings

        return result


# Backend registry for future extensibility
_BACKENDS: dict[PDFBackend, type[PDFExtractorBackend]] = {
    PDFBackend.PYMUPDF: PyMuPDFBackend,
    PDFBackend.PDFPLUMBER: PDFPlumberBackend,
}


def get_backend(backend: PDFBackend) -> PDFExtractorBackend:
    """Get an instance of the specified backend."""
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    return _BACKENDS[backend]()


def extract_pdf(
    file_bytes: bytes,
    config: PDFConfig | dict[str, Any] | None = None,
) -> PDFExtractionResult:
    """
    Extract text from a PDF file.

    This is the main entry point - use this function.

    Args:
        file_bytes: Raw PDF file bytes
        config: PDFConfig or dict with config values.
                If dict, keys should match PDFConfig fields.
                If None, uses defaults.

    Returns:
        PDFExtractionResult with pages, text, metadata, warnings

    Example:
        result = extract_pdf(pdf_bytes, {"max_pages": 50, "backend": "pymupdf"})
        print(result.text)
        for warning in result.warnings:
            print(f"Warning: {warning}")
    """
    # Parse config
    if config is None:
        pdf_config = PDFConfig()
    elif isinstance(config, dict):
        # Convert dict to PDFConfig
        backend = config.get("backend", PDFBackend.PYMUPDF)
        if isinstance(backend, str):
            backend = PDFBackend(backend)
        pdf_config = PDFConfig(
            backend=backend,
            max_pages=config.get("max_pages"),
            min_chars_per_page=config.get("min_chars_per_page", 10),
            join_pages_with=config.get("join_pages_with", "\n\n"),
            enable_ocr=config.get("enable_ocr", False),
        )
    else:
        pdf_config = config

    # Get backend and extract
    backend_instance = get_backend(pdf_config.backend)

    logger.info(
        "Extracting PDF",
        backend=pdf_config.backend.value,
        max_pages=pdf_config.max_pages,
        min_chars=pdf_config.min_chars_per_page,
    )

    result = backend_instance.extract(file_bytes, pdf_config)

    logger.info(
        "PDF extraction complete",
        total_pages=result.page_count,
        extracted_pages=result.extracted_page_count,
        text_length=len(result.text),
        warnings_count=len(result.warnings),
    )

    return result


def get_page_markers(result: PDFExtractionResult) -> list[tuple[int, int]]:
    """
    Convert extraction result to page markers for chunker.

    Returns list of (char_start, page_number) tuples for use with
    Chunker.chunk_with_pages().
    """
    return [(page.char_start, page.page_number) for page in result.pages]
