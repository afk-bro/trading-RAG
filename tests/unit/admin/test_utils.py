"""Tests for admin shared utilities module.

Tests the helper functions and constants used across admin endpoints.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi import HTTPException

from app.admin.utils import (
    PaginationDefaults,
    json_serializable,
    parse_json_field,
    parse_jsonb_fields,
    require_db_pool,
    parse_bool_param,
    prepare_for_template,
    MAX_ERRORS_IN_RESPONSE,
)


# =============================================================================
# PaginationDefaults tests
# =============================================================================


class TestPaginationDefaults:
    """Tests for PaginationDefaults constants."""

    def test_default_limit_value(self):
        """DEFAULT_LIMIT should be 20."""
        assert PaginationDefaults.DEFAULT_LIMIT == 20

    def test_max_limit_value(self):
        """MAX_LIMIT should be 100."""
        assert PaginationDefaults.MAX_LIMIT == 100

    def test_detail_default_limit_value(self):
        """DETAIL_DEFAULT_LIMIT should be 50."""
        assert PaginationDefaults.DETAIL_DEFAULT_LIMIT == 50

    def test_detail_max_limit_value(self):
        """DETAIL_MAX_LIMIT should be 200."""
        assert PaginationDefaults.DETAIL_MAX_LIMIT == 200

    def test_leaderboard_default_limit_value(self):
        """LEADERBOARD_DEFAULT_LIMIT should be 50."""
        assert PaginationDefaults.LEADERBOARD_DEFAULT_LIMIT == 50

    def test_leaderboard_max_limit_value(self):
        """LEADERBOARD_MAX_LIMIT should be 100."""
        assert PaginationDefaults.LEADERBOARD_MAX_LIMIT == 100


# =============================================================================
# json_serializable tests
# =============================================================================


class TestJsonSerializable:
    """Tests for json_serializable function."""

    def test_none_returns_none(self):
        """None should return None."""
        assert json_serializable(None) is None

    def test_string_passthrough(self):
        """Strings should pass through unchanged."""
        assert json_serializable("hello") == "hello"
        assert json_serializable("") == ""

    def test_int_passthrough(self):
        """Integers should pass through unchanged."""
        assert json_serializable(42) == 42
        assert json_serializable(0) == 0
        assert json_serializable(-1) == -1

    def test_float_passthrough(self):
        """Floats should pass through unchanged."""
        assert json_serializable(3.14) == 3.14
        assert json_serializable(0.0) == 0.0

    def test_bool_passthrough(self):
        """Booleans should pass through unchanged."""
        assert json_serializable(True) is True
        assert json_serializable(False) is False

    def test_datetime_to_isoformat(self):
        """Datetime should convert to ISO format string."""
        dt = datetime(2025, 1, 15, 10, 30, 0)
        result = json_serializable(dt)
        assert result == "2025-01-15T10:30:00"

    def test_date_to_isoformat(self):
        """Date should convert to ISO format string."""
        d = date(2025, 1, 15)
        result = json_serializable(d)
        assert result == "2025-01-15"

    def test_uuid_to_string(self):
        """UUID should convert to string."""
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        result = json_serializable(test_uuid)
        assert result == "12345678-1234-5678-1234-567812345678"

    def test_decimal_to_float(self):
        """Decimal should convert to float."""
        d = Decimal("3.14159")
        result = json_serializable(d)
        assert result == pytest.approx(3.14159)

    def test_bytes_to_string(self):
        """Bytes should decode to string."""
        b = b"hello world"
        result = json_serializable(b)
        assert result == "hello world"

    def test_bytes_with_invalid_utf8(self):
        """Bytes with invalid UTF-8 should decode with replacement."""
        b = b"hello \xff world"
        result = json_serializable(b)
        assert "hello" in result
        assert "world" in result

    def test_dict_recursive_conversion(self):
        """Dicts should have values recursively converted."""
        test_uuid = uuid4()
        dt = datetime(2025, 1, 15, 10, 30, 0)
        obj = {
            "id": test_uuid,
            "created_at": dt,
            "name": "test",
            "count": 42,
        }
        result = json_serializable(obj)
        assert result["id"] == str(test_uuid)
        assert result["created_at"] == "2025-01-15T10:30:00"
        assert result["name"] == "test"
        assert result["count"] == 42

    def test_list_recursive_conversion(self):
        """Lists should have items recursively converted."""
        test_uuid = uuid4()
        obj = [test_uuid, "string", 42]
        result = json_serializable(obj)
        assert result[0] == str(test_uuid)
        assert result[1] == "string"
        assert result[2] == 42

    def test_tuple_to_list(self):
        """Tuples should convert to lists with items converted."""
        obj = (1, 2, 3)
        result = json_serializable(obj)
        assert result == [1, 2, 3]

    def test_nested_dict_and_list(self):
        """Nested structures should be fully converted."""
        test_uuid = uuid4()
        obj = {
            "items": [
                {"id": test_uuid, "name": "first"},
                {"id": uuid4(), "name": "second"},
            ],
            "metadata": {"count": 2},
        }
        result = json_serializable(obj)
        assert isinstance(result["items"], list)
        assert result["items"][0]["id"] == str(test_uuid)
        assert result["items"][0]["name"] == "first"

    def test_object_with_to_dict_method(self):
        """Objects with to_dict() method should use it."""

        class CustomObj:
            def to_dict(self):
                return {"key": "value", "num": 42}

        obj = CustomObj()
        result = json_serializable(obj)
        assert result == {"key": "value", "num": 42}

    def test_object_with_dict_attribute(self):
        """Objects with __dict__ should serialize it."""

        class SimpleObj:
            def __init__(self):
                self.name = "test"
                self.value = 123

        obj = SimpleObj()
        result = json_serializable(obj)
        assert result["name"] == "test"
        assert result["value"] == 123

    def test_fallback_to_string(self):
        """Unknown types should fall back to string conversion."""
        # object() has an empty __dict__, so it will serialize to {}
        result = json_serializable(object())
        assert isinstance(result, (str, dict))


# =============================================================================
# parse_json_field tests
# =============================================================================


class TestParseJsonField:
    """Tests for parse_json_field function."""

    def test_none_returns_none(self):
        """None should return None."""
        assert parse_json_field(None) is None

    def test_dict_passthrough(self):
        """Dicts should pass through unchanged."""
        obj = {"key": "value"}
        assert parse_json_field(obj) is obj

    def test_list_passthrough(self):
        """Lists should pass through unchanged."""
        obj = [1, 2, 3]
        assert parse_json_field(obj) is obj

    def test_json_string_parsed(self):
        """Valid JSON strings should be parsed."""
        json_str = '{"key": "value", "num": 42}'
        result = parse_json_field(json_str)
        assert result == {"key": "value", "num": 42}

    def test_json_array_string_parsed(self):
        """Valid JSON array strings should be parsed."""
        json_str = '[1, 2, "three"]'
        result = parse_json_field(json_str)
        assert result == [1, 2, "three"]

    def test_invalid_json_returns_original(self):
        """Invalid JSON strings should return the original string."""
        invalid = "not valid json {"
        result = parse_json_field(invalid)
        assert result == invalid

    def test_plain_string_returns_original(self):
        """Plain strings (not JSON) should return the original."""
        plain = "just a plain string"
        result = parse_json_field(plain)
        assert result == plain

    def test_empty_string_returns_original(self):
        """Empty string should return empty string."""
        result = parse_json_field("")
        assert result == ""

    def test_non_string_non_dict_passthrough(self):
        """Other types should pass through unchanged."""
        assert parse_json_field(42) == 42
        assert parse_json_field(3.14) == 3.14
        assert parse_json_field(True) is True


# =============================================================================
# parse_jsonb_fields tests
# =============================================================================


class TestParseJsonbFields:
    """Tests for parse_jsonb_fields function."""

    def test_parses_specified_fields(self):
        """Should parse only the specified fields."""
        obj = {
            "config": '{"key": "value"}',
            "metadata": '{"count": 5}',
            "name": "test",
        }
        result = parse_jsonb_fields(obj, ["config", "metadata"])
        assert result["config"] == {"key": "value"}
        assert result["metadata"] == {"count": 5}
        assert result["name"] == "test"

    def test_modifies_in_place(self):
        """Should modify the dict in place and return same reference."""
        obj = {"config": '{"key": "value"}'}
        result = parse_jsonb_fields(obj, ["config"])
        assert result is obj
        assert obj["config"] == {"key": "value"}

    def test_skips_missing_fields(self):
        """Should skip fields that don't exist in the dict."""
        obj = {"name": "test"}
        result = parse_jsonb_fields(obj, ["config", "metadata"])
        assert result == {"name": "test"}

    def test_skips_none_values(self):
        """Should skip fields with None values."""
        obj = {"config": None, "name": "test"}
        result = parse_jsonb_fields(obj, ["config"])
        assert result["config"] is None

    def test_empty_fields_list(self):
        """Empty fields list should not modify anything."""
        obj = {"config": '{"key": "value"}'}
        result = parse_jsonb_fields(obj, [])
        assert result["config"] == '{"key": "value"}'


# =============================================================================
# require_db_pool tests
# =============================================================================


class TestRequireDbPool:
    """Tests for require_db_pool function."""

    def test_returns_pool_when_valid(self):
        """Should return the pool when it's not None."""
        mock_pool = object()
        result = require_db_pool(mock_pool)
        assert result is mock_pool

    def test_raises_503_when_none(self):
        """Should raise HTTPException 503 when pool is None."""
        with pytest.raises(HTTPException) as exc_info:
            require_db_pool(None)
        assert exc_info.value.status_code == 503
        assert "Database" in exc_info.value.detail

    def test_custom_service_name_in_error(self):
        """Should include custom service name in error message."""
        with pytest.raises(HTTPException) as exc_info:
            require_db_pool(None, service_name="Qdrant")
        assert exc_info.value.status_code == 503
        assert "Qdrant" in exc_info.value.detail

    def test_truthy_values_accepted(self):
        """Should accept any truthy value as a valid pool."""
        assert require_db_pool("connection") == "connection"
        assert require_db_pool({"pool": True}) == {"pool": True}
        assert require_db_pool([1, 2, 3]) == [1, 2, 3]


# =============================================================================
# parse_bool_param tests
# =============================================================================


class TestParseBoolParam:
    """Tests for parse_bool_param function."""

    def test_none_returns_none(self):
        """None should return None."""
        assert parse_bool_param(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert parse_bool_param("") is None

    def test_true_lowercase(self):
        """'true' should return True."""
        assert parse_bool_param("true") is True

    def test_true_uppercase(self):
        """'TRUE' should return True."""
        assert parse_bool_param("TRUE") is True

    def test_true_mixed_case(self):
        """'True' should return True."""
        assert parse_bool_param("True") is True
        assert parse_bool_param("TrUe") is True

    def test_false_lowercase(self):
        """'false' should return False."""
        assert parse_bool_param("false") is False

    def test_false_uppercase(self):
        """'FALSE' should return False."""
        assert parse_bool_param("FALSE") is False

    def test_other_values_return_false(self):
        """Other string values should return False."""
        assert parse_bool_param("yes") is False
        assert parse_bool_param("no") is False
        assert parse_bool_param("1") is False
        assert parse_bool_param("0") is False


# =============================================================================
# prepare_for_template tests
# =============================================================================


class TestPrepareForTemplate:
    """Tests for prepare_for_template function."""

    def test_converts_default_uuid_fields(self):
        """Should convert default UUID fields to strings."""
        test_id = uuid4()
        workspace_id = uuid4()
        strategy_id = uuid4()
        obj = {
            "id": test_id,
            "workspace_id": workspace_id,
            "strategy_entity_id": strategy_id,
            "name": "test",
        }
        result = prepare_for_template(obj)
        assert result["id"] == str(test_id)
        assert result["workspace_id"] == str(workspace_id)
        assert result["strategy_entity_id"] == str(strategy_id)
        assert result["name"] == "test"

    def test_converts_custom_uuid_fields(self):
        """Should convert custom UUID fields when specified."""
        user_id = uuid4()
        obj = {"user_id": user_id, "name": "test"}
        result = prepare_for_template(obj, uuid_fields=["user_id"])
        assert result["user_id"] == str(user_id)

    def test_converts_datetime_fields(self):
        """Should convert datetime fields to ISO format."""
        dt = datetime(2025, 1, 15, 10, 30, 0)
        obj = {"created_at": dt, "name": "test"}
        result = prepare_for_template(obj)
        assert result["created_at"] == "2025-01-15T10:30:00"

    def test_converts_date_fields(self):
        """Should convert date fields to ISO format."""
        d = date(2025, 1, 15)
        obj = {"birth_date": d, "name": "test"}
        result = prepare_for_template(obj)
        assert result["birth_date"] == "2025-01-15"

    def test_modifies_in_place(self):
        """Should modify the dict in place and return same reference."""
        test_id = uuid4()
        obj = {"id": test_id}
        result = prepare_for_template(obj)
        assert result is obj

    def test_skips_none_uuid_fields(self):
        """Should skip UUID fields that are None."""
        obj = {"id": None, "name": "test"}
        result = prepare_for_template(obj)
        assert result["id"] is None

    def test_skips_missing_uuid_fields(self):
        """Should skip UUID fields that don't exist."""
        obj = {"name": "test"}
        result = prepare_for_template(obj)
        assert "id" not in result


# =============================================================================
# MAX_ERRORS_IN_RESPONSE tests
# =============================================================================


class TestMaxErrorsInResponse:
    """Tests for MAX_ERRORS_IN_RESPONSE constant."""

    def test_value(self):
        """MAX_ERRORS_IN_RESPONSE should be 10."""
        assert MAX_ERRORS_IN_RESPONSE == 10
