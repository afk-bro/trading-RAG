"""Unit tests for text chunking service."""

from app.services.chunker import Chunk, Chunker, normalize_transcript


class TestChunker:
    """Tests for Chunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = Chunker(max_tokens=100, overlap_tokens=10)

    # Token counting tests
    def test_count_tokens_empty(self):
        """Test token count for empty string."""
        assert self.chunker.count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Test token count for simple text."""
        count = self.chunker.count_tokens("Hello world")
        assert count > 0
        assert count < 10  # Simple phrase should be few tokens

    def test_count_tokens_longer_text(self):
        """Test token count for longer text."""
        text = "This is a longer piece of text with multiple words and sentences."
        count = self.chunker.count_tokens(text)
        assert count > 10

    # Basic chunking tests
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = self.chunker.chunk_text("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunks = self.chunker.chunk_text("   \n\t  ")
        assert chunks == []

    def test_chunk_short_text_single_chunk(self):
        """Test that short text produces single chunk."""
        text = "This is a short text."
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_chunk_long_text_multiple_chunks(self):
        """Test that long text produces multiple chunks."""
        # Create text that will exceed max_tokens
        text = " ".join(["word"] * 500)  # ~500 tokens
        chunker = Chunker(max_tokens=100, overlap_tokens=10)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.token_count <= 100

    def test_chunk_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "This is the original text that should be preserved in chunks."
        chunks = self.chunker.chunk_text(text)
        # For short text, content should be exactly preserved
        assert chunks[0].content == text

    def test_chunk_index_sequential(self):
        """Test that chunk indices are sequential."""
        text = " ".join(["word"] * 500)
        chunks = self.chunker.chunk_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    # Timestamp chunking tests
    def test_chunk_timestamped_empty(self):
        """Test timestamped chunking with empty segments."""
        chunks = self.chunker.chunk_timestamped_content([])
        assert chunks == []

    def test_chunk_timestamped_single_segment(self):
        """Test timestamped chunking with single segment."""
        segments = [{"text": "Hello world", "start": 0.0, "end": 5.0}]
        chunks = self.chunker.chunk_timestamped_content(segments)
        assert len(chunks) == 1
        assert chunks[0].time_start_secs == 0
        assert chunks[0].time_end_secs == 5
        assert chunks[0].locator_label == "0:00"

    def test_chunk_timestamped_multiple_segments(self):
        """Test timestamped chunking with multiple segments."""
        segments = [
            {"text": "First segment", "start": 0.0, "end": 10.0},
            {"text": "Second segment", "start": 10.0, "end": 20.0},
            {"text": "Third segment", "start": 20.0, "end": 30.0},
        ]
        chunks = self.chunker.chunk_timestamped_content(segments)
        assert len(chunks) >= 1
        assert chunks[0].time_start_secs is not None
        assert chunks[0].time_end_secs is not None

    def test_chunk_timestamped_preserves_timestamps(self):
        """Test that timestamps are correctly preserved."""
        segments = [
            {"text": "First part", "start": 60.0, "end": 120.0},
            {"text": "Second part", "start": 120.0, "end": 180.0},
        ]
        chunks = self.chunker.chunk_timestamped_content(segments)
        assert chunks[0].time_start_secs == 60

    # Timestamp formatting tests
    def test_format_timestamp_seconds_only(self):
        """Test timestamp formatting for < 1 minute."""
        assert self.chunker._format_timestamp(45) == "0:45"

    def test_format_timestamp_minutes(self):
        """Test timestamp formatting for minutes."""
        assert self.chunker._format_timestamp(125) == "2:05"

    def test_format_timestamp_hours(self):
        """Test timestamp formatting with hours."""
        assert self.chunker._format_timestamp(3665) == "1:01:05"

    def test_format_timestamp_zero(self):
        """Test timestamp formatting for zero."""
        assert self.chunker._format_timestamp(0) == "0:00"

    # Chunk dataclass tests
    def test_chunk_dataclass_defaults(self):
        """Test Chunk dataclass default values."""
        chunk = Chunk(content="test", chunk_index=0, token_count=1)
        assert chunk.time_start_secs is None
        assert chunk.time_end_secs is None
        assert chunk.page_start is None
        assert chunk.page_end is None
        assert chunk.section is None
        assert chunk.locator_label is None


class TestNormalizeTranscript:
    """Tests for normalize_transcript function."""

    def test_removes_music_marker(self):
        """Test removal of [Music] markers."""
        text = "Hello [Music] world"
        result = normalize_transcript(text)
        assert "[Music]" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_applause_marker(self):
        """Test removal of [Applause] markers."""
        text = "Thank you [Applause] very much"
        result = normalize_transcript(text)
        assert "[Applause]" not in result

    def test_removes_laughter_marker(self):
        """Test removal of [Laughter] markers."""
        text = "That was funny [Laughter]"
        result = normalize_transcript(text)
        assert "[Laughter]" not in result

    def test_marker_removal_case_insensitive(self):
        """Test that marker removal is case-insensitive."""
        text = "Hello [MUSIC] [music] [Music] world"
        result = normalize_transcript(text)
        assert "[MUSIC]" not in result
        assert "[music]" not in result
        assert "[Music]" not in result

    def test_removes_repeated_words(self):
        """Test removal of repeated words."""
        text = "um um um I think think that"
        result = normalize_transcript(text)
        # Should collapse repeated words
        assert "um um um" not in result
        assert "think think" not in result

    def test_normalizes_whitespace(self):
        """Test normalization of excessive whitespace."""
        text = "Hello    world\n\n\ttest"
        result = normalize_transcript(text)
        assert "    " not in result
        assert result == "Hello world test"

    def test_strips_text(self):
        """Test that result is stripped."""
        text = "   Hello world   "
        result = normalize_transcript(text)
        assert result == "Hello world"

    def test_empty_text(self):
        """Test normalizing empty text."""
        result = normalize_transcript("")
        assert result == ""

    def test_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "This is a normal sentence without any markers."
        result = normalize_transcript(text)
        assert result == text


class TestChunkerConfiguration:
    """Tests for Chunker configuration options."""

    def test_custom_max_tokens(self):
        """Test chunker with custom max_tokens."""
        chunker = Chunker(max_tokens=50)
        assert chunker.max_tokens == 50

    def test_custom_overlap_tokens(self):
        """Test chunker with custom overlap_tokens."""
        chunker = Chunker(overlap_tokens=20)
        assert chunker.overlap_tokens == 20

    def test_default_values(self):
        """Test chunker default values."""
        chunker = Chunker()
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50

    def test_max_tokens_constraint_enforced(self):
        """Test that max_tokens constraint is enforced."""
        chunker = Chunker(max_tokens=50, overlap_tokens=5)
        # Create text that will produce multiple chunks
        text = " ".join(["word"] * 200)
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            assert chunk.token_count <= 50


class TestChunkingContentIntegrity:
    """Tests for verifying chunking preserves content integrity."""

    def test_short_content_fully_preserved(self):
        """Test that short content is fully preserved in single chunk."""
        text = "The quick brown fox jumps over the lazy dog."
        chunker = Chunker(max_tokens=512)
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_long_content_all_words_present(self):
        """Test that all words from long content appear in at least one chunk."""
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        text = " ".join(words * 50)  # Create long text
        chunker = Chunker(max_tokens=50, overlap_tokens=10)
        chunks = chunker.chunk_text(text)

        # Concatenate all chunk content
        all_chunk_content = " ".join(chunk.content for chunk in chunks)

        # All original words should be present
        for word in words:
            assert word in all_chunk_content

    def test_timestamped_content_all_segments_preserved(self):
        """Test that all timestamped segment texts appear in chunks."""
        segments = [
            {"text": "First unique segment content", "start": 0.0, "end": 10.0},
            {"text": "Second segment with different words", "start": 10.0, "end": 20.0},
            {"text": "Third segment contains more text", "start": 20.0, "end": 30.0},
        ]
        chunker = Chunker(max_tokens=512)
        chunks = chunker.chunk_timestamped_content(segments)

        # Concatenate all chunk content
        all_chunk_content = " ".join(chunk.content for chunk in chunks)

        # All segment texts should appear
        for segment in segments:
            # Check that key words from each segment are present
            for word in segment["text"].split():
                assert word in all_chunk_content

    def test_unicode_content_preserved(self):
        """Test that unicode content is preserved correctly."""
        text = "日本語テスト émojis 🎉 ñoño café"
        chunker = Chunker(max_tokens=512)
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_special_characters_preserved(self):
        """Test that special characters are preserved."""
        text = "Price: $100.50 (discount 10%) - use code: TEST#2024!"
        chunker = Chunker(max_tokens=512)
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
