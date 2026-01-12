"""Tests for Supabase client dependency."""

from unittest.mock import MagicMock, patch


class TestSupabaseClient:
    """Tests for Supabase client initialization."""

    def test_client_uses_settings(self):
        """Client initializes with settings."""
        with patch("app.deps.supabase.get_settings") as mock_settings:
            mock_settings.return_value.supabase_url = "https://test.supabase.co"
            mock_settings.return_value.supabase_service_role_key = "test-key"

            # Mock create_client to avoid API key validation
            with patch("app.deps.supabase.create_client") as mock_create:
                mock_client = MagicMock()
                mock_create.return_value = mock_client

                # Clear lru_cache to ensure fresh call
                from app.deps.supabase import get_supabase_client

                get_supabase_client.cache_clear()

                # Should not raise
                client = get_supabase_client()
                assert client is not None
                mock_create.assert_called_once_with(
                    "https://test.supabase.co",
                    "test-key",
                )
