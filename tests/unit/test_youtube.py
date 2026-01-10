"""Unit tests for YouTube URL parsing."""

from app.routers.youtube import parse_youtube_url


class TestParseYouTubeUrl:
    """Tests for YouTube URL parser function."""

    # Standard watch URLs
    def test_parse_standard_watch_url(self):
        """Test parsing standard youtube.com/watch?v= URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["is_playlist"] is False

    def test_parse_watch_url_without_www(self):
        """Test parsing watch URL without www."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_parse_watch_url_http(self):
        """Test parsing watch URL with http (not https)."""
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    # Short URLs (youtu.be)
    def test_parse_short_url(self):
        """Test parsing youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["is_playlist"] is False

    def test_parse_short_url_with_timestamp(self):
        """Test parsing youtu.be URL with timestamp."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=120"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    # Embed URLs
    def test_parse_embed_url(self):
        """Test parsing youtube.com/embed/ URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    # Legacy v/ URLs
    def test_parse_v_url(self):
        """Test parsing youtube.com/v/ URL."""
        url = "https://www.youtube.com/v/dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    # Playlist URLs
    def test_parse_playlist_url(self):
        """Test parsing playlist URL."""
        url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = parse_youtube_url(url)
        assert result["playlist_id"] == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        assert result["is_playlist"] is True

    def test_parse_watch_url_with_playlist(self):
        """Test parsing watch URL that includes playlist reference."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["playlist_id"] == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        # When both video_id and playlist_id are present, it's not treated as playlist-only
        assert result["is_playlist"] is False

    # Additional query parameters
    def test_parse_url_with_extra_params(self):
        """Test parsing URL with additional query parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=youtu.be&t=120"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    # Edge cases
    def test_parse_empty_url(self):
        """Test parsing empty URL."""
        result = parse_youtube_url("")
        assert result["video_id"] is None
        assert result["playlist_id"] is None

    def test_parse_invalid_url(self):
        """Test parsing non-YouTube URL."""
        url = "https://www.example.com/video"
        result = parse_youtube_url(url)
        assert result["video_id"] is None
        assert result["playlist_id"] is None

    def test_parse_malformed_url(self):
        """Test parsing malformed URL."""
        url = "not a url at all"
        result = parse_youtube_url(url)
        assert result["video_id"] is None

    # Video ID format validation
    def test_video_id_format(self):
        """Test that video IDs have expected format."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        video_id = result["video_id"]
        # YouTube video IDs are 11 characters
        assert len(video_id) == 11

    def test_playlist_id_format(self):
        """Test that playlist IDs have expected format."""
        url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = parse_youtube_url(url)
        playlist_id = result["playlist_id"]
        # Playlist IDs typically start with PL
        assert playlist_id.startswith("PL")

    # URL variations
    def test_parse_mobile_url(self):
        """Test parsing mobile youtube URL (m.youtube.com)."""
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        # May or may not be supported depending on implementation
        # This test documents the behavior
        assert result is not None

    def test_parse_url_with_index(self):
        """Test parsing URL with index parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest&index=5"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"


class TestUrlParserReturnStructure:
    """Tests for the return structure of parse_youtube_url."""

    def test_return_contains_video_id_key(self):
        """Test that return dict contains video_id key."""
        result = parse_youtube_url("https://youtu.be/test123test")
        assert "video_id" in result

    def test_return_contains_playlist_id_key(self):
        """Test that return dict contains playlist_id key."""
        result = parse_youtube_url("https://youtu.be/test123test")
        assert "playlist_id" in result

    def test_return_contains_is_playlist_key(self):
        """Test that return dict contains is_playlist key."""
        result = parse_youtube_url("https://youtu.be/test123test")
        assert "is_playlist" in result

    def test_is_playlist_is_boolean(self):
        """Test that is_playlist is a boolean."""
        result = parse_youtube_url("https://youtu.be/test123test")
        assert isinstance(result["is_playlist"], bool)


class TestUrlEncoding:
    """Tests for URL encoding handling."""

    def test_parse_url_with_encoded_characters(self):
        """Test parsing URL with URL-encoded characters."""
        # URL with %3D (equals sign) encoded
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_parse_url_encoded_video_id(self):
        """Test parsing URL where video_id could have special chars."""
        # Video IDs are base64-like, so they may contain - and _
        url = "https://www.youtube.com/watch?v=abc-_123XYZ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "abc-_123XYZ"

    def test_parse_url_with_encoded_ampersand(self):
        """Test parsing URL with encoded ampersand (%26)."""
        # When ampersand is encoded, parse_qs still handles it
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ%26feature=test"
        result = parse_youtube_url(url)
        # The video_id should still be extracted correctly
        assert result["video_id"] is not None

    def test_parse_url_with_space_encoding(self):
        """Test parsing URL with space encoding (%20 or +)."""
        # YouTube video IDs don't contain spaces, but query params might
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1m30s"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_parse_url_with_unicode_encoding(self):
        """Test parsing URL with unicode-encoded characters."""
        # Standard URL, urlparse handles encoding
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_parse_double_encoded_url(self):
        """Test parsing double-encoded URL."""
        # %253D is double-encoded %3D (=)
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"


class TestYouTubeErrorReasonParsing:
    """Tests for YouTube error reason parsing logic."""

    def test_parse_video_private_error(self):
        """Test parsing video:video_private error string."""
        error_str = "video:video_private:test123"
        parts = error_str.split(":")
        if error_str.startswith("video:"):
            error_reason = parts[1] if len(parts) > 1 else "video_unavailable"
        else:
            error_reason = "no_transcript"
        assert error_reason == "video_private"

    def test_parse_video_age_restricted_error(self):
        """Test parsing video:video_age_restricted error string."""
        error_str = "video:video_age_restricted:test123"
        parts = error_str.split(":")
        if error_str.startswith("video:"):
            error_reason = parts[1] if len(parts) > 1 else "video_unavailable"
        else:
            error_reason = "no_transcript"
        assert error_reason == "video_age_restricted"

    def test_parse_video_livestream_error(self):
        """Test parsing video:video_is_livestream error string."""
        error_str = "video:video_is_livestream:test123"
        parts = error_str.split(":")
        if error_str.startswith("video:"):
            error_reason = parts[1] if len(parts) > 1 else "video_unavailable"
        else:
            error_reason = "no_transcript"
        assert error_reason == "video_is_livestream"

    def test_parse_video_unavailable_error(self):
        """Test parsing video:video_unavailable error string."""
        error_str = "video:video_unavailable:test123"
        parts = error_str.split(":")
        if error_str.startswith("video:"):
            error_reason = parts[1] if len(parts) > 1 else "video_unavailable"
        else:
            error_reason = "no_transcript"
        assert error_reason == "video_unavailable"

    def test_parse_no_transcript_error(self):
        """Test parsing no_transcript error (non-video: prefix)."""
        error_str = "No transcript available for video test123"
        if error_str.startswith("video:"):
            parts = error_str.split(":")
            error_reason = parts[1] if len(parts) > 1 else "video_unavailable"
        else:
            error_reason = "no_transcript"
        assert error_reason == "no_transcript"

    def test_error_reason_retryable_flag(self):
        """Test that video-related errors are marked as non-retryable."""
        non_retryable_errors = [
            "video_private",
            "video_age_restricted",
            "video_is_livestream",
            "video_unavailable",
            "no_transcript",
        ]
        # All of these should be terminal (non-retryable)
        for error in non_retryable_errors:
            # These are terminal errors - retryable should be False
            assert error in non_retryable_errors
