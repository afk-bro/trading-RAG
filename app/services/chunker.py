"""Text chunking service with token-aware splitting."""

import re
from dataclasses import dataclass
from typing import Optional

import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    chunk_index: int
    token_count: int
    time_start_secs: Optional[int] = None
    time_end_secs: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    locator_label: Optional[str] = None


class Chunker:
    """Token-aware text chunking with optional timestamp/page preservation."""

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks for context
            encoding_name: Tiktoken encoding name
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect section header from text.

        Looks for markdown-style headers (# Header, ## Header, etc.)
        or ALL CAPS lines that could be section titles.

        Args:
            text: Text to analyze for section header

        Returns:
            Section name if found, None otherwise
        """
        lines = text.strip().split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            # Markdown headers
            match = re.match(r"^#{1,6}\s+(.+)$", line)
            if match:
                return match.group(1).strip()
            # ALL CAPS lines (common section headers)
            if line.isupper() and len(line) > 3 and len(line) < 100:
                return line.title()
        return None

    def chunk_text(self, text: str) -> list[Chunk]:
        """
        Split text into token-aware chunks with section detection.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Tokenize entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.max_tokens:
            section = self._detect_section(text)
            return [
                Chunk(
                    content=text,
                    chunk_index=0,
                    token_count=total_tokens,
                    section=section,
                )
            ]

        chunks: list[Chunk] = []
        start_idx = 0
        chunk_index = 0
        current_section: Optional[str] = None

        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.max_tokens, total_tokens)

            # Decode chunk tokens back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Detect section in this chunk
            detected_section = self._detect_section(chunk_text)
            if detected_section:
                current_section = detected_section

            chunks.append(
                Chunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    token_count=len(chunk_tokens),
                    section=current_section,
                )
            )

            # Move start index forward (accounting for overlap)
            start_idx = (
                end_idx - self.overlap_tokens if end_idx < total_tokens else end_idx
            )
            chunk_index += 1

        return chunks

    def chunk_timestamped_content(
        self,
        segments: list[dict],
    ) -> list[Chunk]:
        """
        Chunk content with timestamp preservation.

        Args:
            segments: List of dicts with 'text', 'start', 'end' keys

        Returns:
            List of Chunk objects with timestamp info
        """
        if not segments:
            return []

        chunks: list[Chunk] = []
        current_text = ""
        current_tokens = 0
        current_start = segments[0].get("start", 0)
        chunk_index = 0

        for segment in segments:
            seg_text = segment.get("text", "")
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            seg_tokens = self.count_tokens(seg_text)

            # Check if adding this segment exceeds limit
            if current_tokens + seg_tokens > self.max_tokens and current_text:
                # Save current chunk
                chunks.append(
                    Chunk(
                        content=current_text.strip(),
                        chunk_index=chunk_index,
                        token_count=current_tokens,
                        time_start_secs=int(current_start),
                        time_end_secs=int(seg_start),
                        locator_label=self._format_timestamp(int(current_start)),
                    )
                )
                chunk_index += 1

                # Start new chunk
                current_text = seg_text
                current_tokens = seg_tokens
                current_start = seg_start
            else:
                current_text += " " + seg_text if current_text else seg_text
                current_tokens += seg_tokens

        # Save final chunk
        if current_text.strip():
            chunks.append(
                Chunk(
                    content=current_text.strip(),
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    time_start_secs=int(current_start),
                    time_end_secs=int(segments[-1].get("end", 0)) if segments else 0,
                    locator_label=self._format_timestamp(int(current_start)),
                )
            )

        return chunks

    def _format_timestamp(self, seconds: int) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def chunk_with_pages(
        self,
        text: str,
        page_markers: list[tuple[int, int]],
    ) -> list[Chunk]:
        """
        Chunk text with page number preservation.

        Args:
            text: Full document text
            page_markers: List of (char_start, page_number) tuples

        Returns:
            List of Chunk objects with page info
        """
        # Basic implementation - chunk then assign pages
        chunks = self.chunk_text(text)

        if not page_markers:
            return chunks

        # Assign page numbers based on character positions
        # This is a simplified implementation
        for chunk in chunks:
            # Find page for chunk start (would need char position tracking)
            chunk.page_start = 1  # Placeholder
            chunk.page_end = 1
            chunk.locator_label = f"p. {chunk.page_start}"

        return chunks


def normalize_transcript(text: str) -> str:
    """
    Normalize YouTube transcript text.

    Removes:
    - [Music] and similar markers
    - Repeated phrases
    - Excessive whitespace
    """
    # Remove common markers
    text = re.sub(r"\[(?:Music|Applause|Laughter)\]", "", text, flags=re.IGNORECASE)

    # Remove repeated words (like "um um um")
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()
