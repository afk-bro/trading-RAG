"""Unit tests for metadata extractor service."""

import pytest

from app.services.extractor import (
    MetadataExtractor,
    ExtractedMetadata,
    VALID_TICKERS,
    EXCLUDED_WORDS,
)


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()

    # Symbol extraction tests
    def test_extract_symbols_with_dollar_sign(self):
        """Test extracting symbols with $ prefix."""
        text = "Looking at $AAPL and $GOOGL today"
        symbols = self.extractor.extract_symbols(text)
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    def test_extract_symbols_uppercase(self):
        """Test extracting uppercase symbols."""
        text = "MSFT and NVDA are performing well"
        symbols = self.extractor.extract_symbols(text)
        assert "MSFT" in symbols
        assert "NVDA" in symbols

    def test_extract_symbols_normalized_to_uppercase(self):
        """Test that symbols are normalized to uppercase."""
        text = "$aapl stock is up"
        symbols = self.extractor.extract_symbols(text)
        assert "AAPL" in symbols
        assert "aapl" not in symbols

    def test_extract_symbols_filters_excluded_words(self):
        """Test that common words are excluded."""
        text = "I AM looking at THE market today"
        symbols = self.extractor.extract_symbols(text)
        assert "I" not in symbols
        assert "AM" not in symbols
        assert "THE" not in symbols

    def test_extract_symbols_validates_against_allowlist(self):
        """Test that only valid tickers are extracted."""
        text = "XYZ is not a real ticker, but AAPL is"
        symbols = self.extractor.extract_symbols(text)
        assert "AAPL" in symbols
        assert "XYZ" not in symbols  # Not in allowlist

    def test_extract_symbols_empty_text(self):
        """Test extracting from empty text."""
        symbols = self.extractor.extract_symbols("")
        assert symbols == []

    def test_extract_symbols_no_tickers(self):
        """Test text with no tickers."""
        text = "This is a sentence without any stock tickers."
        symbols = self.extractor.extract_symbols(text)
        assert symbols == []

    # Entity extraction tests
    def test_extract_entities_federal_reserve(self):
        """Test extracting Fed-related entities."""
        text = "The Fed announced new policy. Powell spoke about rates."
        entities = self.extractor.extract_entities(text)
        assert "Federal Reserve" in entities
        assert "Jerome Powell" in entities

    def test_extract_entities_case_insensitive(self):
        """Test that entity extraction is case-insensitive."""
        text = "POWELL and the FED made announcements"
        entities = self.extractor.extract_entities(text)
        assert "Jerome Powell" in entities
        assert "Federal Reserve" in entities

    def test_extract_entities_fomc(self):
        """Test extracting FOMC entity."""
        text = "The FOMC meeting is scheduled for next week"
        entities = self.extractor.extract_entities(text)
        assert "FOMC" in entities

    def test_extract_entities_multiple(self):
        """Test extracting multiple entities."""
        text = "Yellen and Powell discussed the treasury market"
        entities = self.extractor.extract_entities(text)
        assert "Janet Yellen" in entities
        assert "Jerome Powell" in entities
        assert "US Treasury" in entities

    def test_extract_entities_empty_text(self):
        """Test extracting from empty text."""
        entities = self.extractor.extract_entities("")
        assert entities == []

    # Topic extraction tests
    def test_extract_topics_macro(self):
        """Test extracting macro topics."""
        text = "Inflation is rising and the economy is slowing"
        topics = self.extractor.extract_topics(text)
        assert "macro" in topics

    def test_extract_topics_rates(self):
        """Test extracting rates topics."""
        text = "Interest rate hike expected, treasury yields rising"
        topics = self.extractor.extract_topics(text)
        assert "rates" in topics

    def test_extract_topics_earnings(self):
        """Test extracting earnings topics."""
        text = "Q3 earnings beat expectations with strong revenue growth"
        topics = self.extractor.extract_topics(text)
        assert "earnings" in topics

    def test_extract_topics_tech(self):
        """Test extracting tech topics."""
        text = "AI and machine learning are transforming cloud computing"
        topics = self.extractor.extract_topics(text)
        assert "tech" in topics

    def test_extract_topics_crypto(self):
        """Test extracting crypto topics."""
        text = "Bitcoin and ethereum prices surge as crypto market rallies"
        topics = self.extractor.extract_topics(text)
        assert "crypto" in topics

    def test_extract_topics_options(self):
        """Test extracting options topics."""
        text = "Buying calls with high implied volatility before earnings"
        topics = self.extractor.extract_topics(text)
        assert "options" in topics

    def test_extract_topics_multiple(self):
        """Test extracting multiple topics."""
        text = "Tech earnings beat estimates as AI stocks rally amid rate cut hopes"
        topics = self.extractor.extract_topics(text)
        assert "tech" in topics
        assert "earnings" in topics
        assert "rates" in topics

    def test_extract_topics_empty_text(self):
        """Test extracting from empty text."""
        topics = self.extractor.extract_topics("")
        assert topics == []

    # Quality score tests
    def test_estimate_quality_empty_text(self):
        """Test quality score for empty text."""
        score = self.extractor.estimate_quality("")
        assert score == 0.0

    def test_estimate_quality_short_text(self):
        """Test quality score for very short text."""
        score = self.extractor.estimate_quality("Too short")
        assert score == 0.3

    def test_estimate_quality_normal_text(self):
        """Test quality score for normal text."""
        text = (
            "This is a well-structured paragraph with multiple sentences. "
            "It contains various words and has good diversity. "
            "The content is meaningful and well-formatted."
        )
        score = self.extractor.estimate_quality(text)
        assert 0.0 < score <= 1.0

    # Full extraction tests
    def test_extract_all_metadata(self):
        """Test extracting all metadata at once."""
        text = (
            "$AAPL and $MSFT reported strong earnings. "
            "Powell said the Fed will consider rate cuts if inflation eases."
        )
        metadata = self.extractor.extract(text)

        assert isinstance(metadata, ExtractedMetadata)
        assert "AAPL" in metadata.symbols
        assert "MSFT" in metadata.symbols
        assert "Jerome Powell" in metadata.entities
        assert "Federal Reserve" in metadata.entities
        assert "earnings" in metadata.topics
        assert "rates" in metadata.topics
        assert 0.0 <= metadata.quality_score <= 1.0

    # Speaker detection tests
    def test_detect_speaker_name_colon(self):
        """Test detecting speaker with Name: format."""
        text = "John Smith: The market is looking strong today"
        speaker = self.extractor.detect_speaker(text)
        assert speaker == "John Smith"

    def test_detect_speaker_not_found(self):
        """Test when no speaker is detected."""
        text = "The market is looking strong today"
        speaker = self.extractor.detect_speaker(text)
        assert speaker is None

    def test_extract_no_detectable_entities(self):
        """Test extracting from content with no detectable symbols, entities, or topics."""
        text = "The quick brown fox jumps over the lazy dog. This is a simple sentence."
        metadata = self.extractor.extract(text)

        # Should return empty arrays, not errors
        assert isinstance(metadata.symbols, list)
        assert isinstance(metadata.entities, list)
        assert isinstance(metadata.topics, list)
        assert len(metadata.symbols) == 0
        assert len(metadata.entities) == 0
        assert len(metadata.topics) == 0
        # Quality score should still be calculated
        assert 0.0 <= metadata.quality_score <= 1.0


class TestValidTickers:
    """Tests for valid tickers set."""

    def test_major_indices_included(self):
        """Test that major indices are in allowlist."""
        assert "SPY" in VALID_TICKERS
        assert "QQQ" in VALID_TICKERS
        assert "IWM" in VALID_TICKERS

    def test_mega_caps_included(self):
        """Test that mega caps are in allowlist."""
        assert "AAPL" in VALID_TICKERS
        assert "MSFT" in VALID_TICKERS
        assert "GOOGL" in VALID_TICKERS
        assert "AMZN" in VALID_TICKERS
        assert "META" in VALID_TICKERS
        assert "NVDA" in VALID_TICKERS
        assert "TSLA" in VALID_TICKERS


class TestExcludedWords:
    """Tests for excluded words set."""

    def test_common_words_excluded(self):
        """Test that common words are excluded."""
        assert "I" in EXCLUDED_WORDS
        assert "A" in EXCLUDED_WORDS
        assert "THE" in EXCLUDED_WORDS
        assert "AND" in EXCLUDED_WORDS

    def test_acronyms_excluded(self):
        """Test that common acronyms are excluded."""
        assert "CEO" in EXCLUDED_WORDS
        assert "CFO" in EXCLUDED_WORDS
        assert "IPO" in EXCLUDED_WORDS
        assert "GDP" in EXCLUDED_WORDS
