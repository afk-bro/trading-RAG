"""Supabase client dependency."""

from functools import lru_cache

from supabase import create_client, Client

from app.config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get cached Supabase client.

    Uses service role key for server-side operations.
    """
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )
