"""Shared utilities for admin modules.

Consolidates common patterns used across admin endpoints:
- JSON serialization for API responses
- Database pool access helpers
- Pagination constants
- Common data transformations
"""

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from fastapi import HTTPException, status


# =============================================================================
# Pagination Constants
# =============================================================================


class PaginationDefaults:
    """Standard pagination limits for admin endpoints."""

    # List endpoints
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100

    # Detail endpoints (e.g., runs within a tune)
    DETAIL_DEFAULT_LIMIT = 50
    DETAIL_MAX_LIMIT = 200

    # Leaderboard
    LEADERBOARD_DEFAULT_LIMIT = 50
    LEADERBOARD_MAX_LIMIT = 100


# =============================================================================
# JSON Serialization
# =============================================================================


def json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format.

    Handles common types that json.dumps() can't serialize:
    - datetime/date → ISO format string
    - UUID → string
    - Decimal → float
    - bytes → decoded string
    - dict/list → recursive conversion
    - Objects with to_dict() → dict

    Args:
        obj: Any object to serialize

    Returns:
        JSON-serializable representation
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    if hasattr(obj, "to_dict"):
        return json_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return json_serializable(obj.__dict__)
    # Fallback: try string conversion
    return str(obj)


def parse_json_field(value: Any) -> Any:
    """Parse a potentially JSON-encoded field.

    Safely handles:
    - Already parsed dicts/lists (returns as-is)
    - JSON strings (parses them)
    - None values (returns None)
    - Invalid JSON (returns original value)

    Args:
        value: Value that might be a JSON string

    Returns:
        Parsed value or original if not JSON
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def parse_jsonb_fields(obj: dict, fields: list[str]) -> dict:
    """Parse multiple JSONB string fields in a dict.

    Modifies the dict in place for efficiency.

    Args:
        obj: Dictionary containing fields to parse
        fields: List of field names to parse as JSON

    Returns:
        The modified dict (same reference)
    """
    for field in fields:
        if field in obj and obj[field] is not None:
            obj[field] = parse_json_field(obj[field])
    return obj


# =============================================================================
# Database Pool Helpers
# =============================================================================


def require_db_pool(pool: Any, service_name: str = "Database") -> Any:
    """Validate that database pool is available.

    Args:
        pool: Database connection pool (may be None)
        service_name: Name for error message

    Returns:
        The pool if valid

    Raises:
        HTTPException: 503 if pool is None
    """
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service_name} connection not available",
        )
    return pool


# =============================================================================
# String/Bool Conversions
# =============================================================================


def parse_bool_param(value: Optional[str]) -> Optional[bool]:
    """Parse string query parameter to boolean.

    Args:
        value: String value ('true', 'false', or None)

    Returns:
        True, False, or None if value is None/empty
    """
    if value is None or value == "":
        return None
    return value.lower() == "true"


# =============================================================================
# Template Response Helpers
# =============================================================================


def prepare_for_template(obj: dict, uuid_fields: Optional[list[str]] = None) -> dict:
    """Prepare a dict for Jinja2 template rendering.

    Converts UUID and datetime fields to strings for template compatibility.

    Args:
        obj: Dictionary to prepare
        uuid_fields: List of field names containing UUIDs

    Returns:
        Modified dict with string conversions
    """
    uuid_fields = uuid_fields or ["id", "workspace_id", "strategy_entity_id"]

    for field in uuid_fields:
        if obj.get(field) is not None:
            obj[field] = str(obj[field])

    # Convert datetime fields
    for key, value in obj.items():
        if isinstance(value, (datetime, date)):
            obj[key] = value.isoformat()

    return obj


# =============================================================================
# Error Limit Constants
# =============================================================================

MAX_ERRORS_IN_RESPONSE = 10  # Limit error arrays in API responses


__all__ = [
    # Pagination
    "PaginationDefaults",
    # JSON
    "json_serializable",
    "parse_json_field",
    "parse_jsonb_fields",
    # Database
    "require_db_pool",
    # Conversions
    "parse_bool_param",
    "prepare_for_template",
    # Constants
    "MAX_ERRORS_IN_RESPONSE",
]
