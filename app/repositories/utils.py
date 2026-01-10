"""Utility functions for repository operations."""

import json
from typing import Any, Optional, Union


def ensure_json(value: Optional[Union[str, dict, list]]) -> Optional[Union[dict, list]]:
    """
    Normalize JSONB values from database to Python dict/list.

    asyncpg can return JSONB as:
    - dict/list (when codec is configured)
    - str (default behavior)
    - None (NULL)

    This helper ensures consistent handling regardless of asyncpg config.

    Args:
        value: The value from database (str, dict, list, or None)

    Returns:
        Parsed Python dict/list, or None

    Raises:
        TypeError: If value is an unexpected type
        json.JSONDecodeError: If string is not valid JSON
    """
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        return json.loads(value)

    raise TypeError(
        f"Expected str, dict, list, or None for JSONB value, got {type(value).__name__}"
    )


def parse_jsonb_fields(row_dict: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """
    Parse multiple JSONB fields in a row dict.

    Modifies the dict in-place and returns it for convenience.

    Args:
        row_dict: Dict from database row
        fields: List of field names to parse

    Returns:
        The modified dict
    """
    for field in fields:
        if field in row_dict and row_dict[field] is not None:
            row_dict[field] = ensure_json(row_dict[field])
    return row_dict
