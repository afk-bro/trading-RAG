"""Admin UI package."""

from app.admin.router import router, set_db_pool

__all__ = ["router", "set_db_pool"]
