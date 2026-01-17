"""Admin UI package."""

from app.admin.router import router, set_db_pool, set_qdrant_client

__all__ = ["router", "set_db_pool", "set_qdrant_client"]
