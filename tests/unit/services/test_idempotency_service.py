"""Unit tests for idempotency service."""

import pytest

from app.services.idempotency import (
    IdempotencyKeyTooLongError,
    IdempotencyKeyReusedError,
    IdempotencyRecord,
    MAX_IDEMPOTENCY_KEY_LENGTH,
    compute_request_hash,
    validate_idempotency_key,
    verify_hash_match,
    _normalize_float_fields,
)


class TestValidateIdempotencyKey:
    """Tests for validate_idempotency_key function."""

    def test_valid_key_short(self):
        """Short keys should be valid."""
        validate_idempotency_key("abc123")  # No exception

    def test_valid_key_max_length(self):
        """Key at exactly max length should be valid."""
        key = "a" * MAX_IDEMPOTENCY_KEY_LENGTH
        validate_idempotency_key(key)  # No exception

    def test_invalid_key_too_long(self):
        """Key exceeding max length should raise error."""
        key = "a" * (MAX_IDEMPOTENCY_KEY_LENGTH + 1)
        with pytest.raises(IdempotencyKeyTooLongError) as exc_info:
            validate_idempotency_key(key)
        assert exc_info.value.key_length == MAX_IDEMPOTENCY_KEY_LENGTH + 1

    def test_valid_key_empty(self):
        """Empty key should be valid (business logic may reject later)."""
        validate_idempotency_key("")  # No exception

    def test_valid_key_unicode(self):
        """Unicode keys should be valid."""
        validate_idempotency_key("日本語キー-123")  # No exception


class TestComputeRequestHash:
    """Tests for compute_request_hash function."""

    def test_deterministic_hash(self):
        """Same payload should produce same hash."""
        payload = {"a": 1, "b": "test", "c": [1, 2, 3]}
        hash1 = compute_request_hash(payload)
        hash2 = compute_request_hash(payload)
        assert hash1 == hash2

    def test_hash_is_64_chars(self):
        """SHA256 hex should be 64 characters."""
        payload = {"test": True}
        result = compute_request_hash(payload)
        assert len(result) == 64

    def test_key_order_independent(self):
        """Dict key order should not affect hash."""
        payload1 = {"b": 2, "a": 1}
        payload2 = {"a": 1, "b": 2}
        assert compute_request_hash(payload1) == compute_request_hash(payload2)

    def test_different_values_different_hash(self):
        """Different values should produce different hashes."""
        payload1 = {"x": 1}
        payload2 = {"x": 2}
        assert compute_request_hash(payload1) != compute_request_hash(payload2)

    def test_nested_dict_ordering(self):
        """Nested dicts should also be ordered."""
        payload1 = {"outer": {"b": 2, "a": 1}}
        payload2 = {"outer": {"a": 1, "b": 2}}
        assert compute_request_hash(payload1) == compute_request_hash(payload2)

    def test_unicode_support(self):
        """Unicode content should be handled correctly."""
        payload = {"text": "日本語テスト 🎉"}
        result = compute_request_hash(payload)
        assert len(result) == 64

    def test_float_normalization_specified_fields_only(self):
        """Floats should only be normalized in specified fields."""
        payload = {
            "cash": 10000.123456789012345,
            "name": "test",
            "other_float": 1.999999999999999,
        }

        # With float_fields specified
        hash_normalized = compute_request_hash(
            payload,
            float_fields=["cash"],
            float_precision=10,
        )

        # Payload with slightly different cash value that normalizes the same
        payload_similar = {
            "cash": 10000.1234567890,  # Same to 10 decimals
            "name": "test",
            "other_float": 1.999999999999999,
        }
        hash_similar = compute_request_hash(
            payload_similar,
            float_fields=["cash"],
            float_precision=10,
        )

        assert hash_normalized == hash_similar

    def test_float_not_normalized_if_not_specified(self):
        """Floats not in float_fields should not be normalized."""
        payload = {"cash": 10000.123456789012345}

        # Without normalization
        hash1 = compute_request_hash(payload)

        # With normalization for different field
        hash2 = compute_request_hash(payload, float_fields=["other_field"])

        # Same since neither actually normalized cash
        assert hash1 == hash2

    def test_null_values(self):
        """None values should be handled."""
        payload = {"a": None, "b": 1}
        result = compute_request_hash(payload)
        assert len(result) == 64


class TestNormalizeFloatFields:
    """Tests for _normalize_float_fields helper."""

    def test_normalizes_float_at_path(self):
        """Float at specified path should be rounded."""
        obj = {"cash": 10000.123456789012345}
        result = _normalize_float_fields(obj, ["cash"], precision=5)
        assert result["cash"] == 10000.12346  # Rounded to 5 decimals

    def test_nested_path_normalization(self):
        """Nested float fields should be normalized."""
        obj = {"params": {"threshold": 0.123456789}}
        result = _normalize_float_fields(obj, ["params.threshold"], precision=4)
        assert result["params"]["threshold"] == 0.1235

    def test_list_path_normalization(self):
        """Floats in arrays at specified paths should be normalized."""
        obj = {"values": [1.111111, 2.222222]}
        result = _normalize_float_fields(obj, ["values[]"], precision=2)
        assert result["values"] == [1.11, 2.22]

    def test_unspecified_float_unchanged(self):
        """Floats not in float_fields should remain unchanged."""
        obj = {"keep": 1.999999999999}
        result = _normalize_float_fields(obj, ["other"], precision=2)
        assert result["keep"] == 1.999999999999

    def test_non_float_unchanged(self):
        """Non-float values should remain unchanged."""
        obj = {"name": "test", "count": 42}
        result = _normalize_float_fields(obj, ["name", "count"], precision=2)
        assert result == obj


class TestVerifyHashMatch:
    """Tests for verify_hash_match function."""

    def test_matching_hash_passes(self):
        """Matching hash should not raise."""
        from uuid import uuid4

        record = IdempotencyRecord(
            id=uuid4(),
            workspace_id=uuid4(),
            idempotency_key="test-key",
            request_hash="abc123",
            endpoint="test.endpoint",
            http_method="POST",
            api_version=None,
            resource_id=None,
            response_json=None,
            error_code=None,
            error_json=None,
            status="pending",
        )

        verify_hash_match(record, "abc123")  # No exception

    def test_mismatched_hash_raises(self):
        """Mismatched hash should raise IdempotencyKeyReusedError."""
        from uuid import uuid4

        record = IdempotencyRecord(
            id=uuid4(),
            workspace_id=uuid4(),
            idempotency_key="test-key",
            request_hash="abc123",
            endpoint="test.endpoint",
            http_method="POST",
            api_version=None,
            resource_id=None,
            response_json=None,
            error_code=None,
            error_json=None,
            status="pending",
        )

        with pytest.raises(IdempotencyKeyReusedError) as exc_info:
            verify_hash_match(record, "xyz789")

        assert exc_info.value.expected_hash == "abc123"
        assert exc_info.value.actual_hash == "xyz789"


class TestIdempotencyRecord:
    """Tests for IdempotencyRecord dataclass."""

    def test_from_row(self):
        """Record should be created from database row dict."""
        from uuid import uuid4

        row_id = uuid4()
        workspace_id = uuid4()
        resource_id = uuid4()

        row = {
            "id": row_id,
            "workspace_id": workspace_id,
            "idempotency_key": "test-key-123",
            "request_hash": "a" * 64,
            "endpoint": "backtests.tune",
            "http_method": "POST",
            "api_version": "v1",
            "resource_id": resource_id,
            "response_json": {"tune_id": str(resource_id)},
            "error_code": None,
            "error_json": None,
            "status": "completed",
        }

        record = IdempotencyRecord.from_row(row)

        assert record.id == row_id
        assert record.workspace_id == workspace_id
        assert record.idempotency_key == "test-key-123"
        assert record.request_hash == "a" * 64
        assert record.endpoint == "backtests.tune"
        assert record.http_method == "POST"
        assert record.api_version == "v1"
        assert record.resource_id == resource_id
        assert record.response_json == {"tune_id": str(resource_id)}
        assert record.status == "completed"

    def test_from_row_optional_fields(self):
        """Record should handle missing optional fields."""
        from uuid import uuid4

        row = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "idempotency_key": "test",
            "request_hash": "b" * 64,
            "endpoint": "test.endpoint",
            "http_method": "POST",
            "status": "pending",
            # Optional fields not present
        }

        record = IdempotencyRecord.from_row(row)

        assert record.api_version is None
        assert record.resource_id is None
        assert record.response_json is None
        assert record.error_code is None
        assert record.error_json is None


class TestEdgeCases:
    """Edge case tests for idempotency service."""

    def test_empty_payload_hash(self):
        """Empty dict should produce valid hash."""
        result = compute_request_hash({})
        assert len(result) == 64

    def test_complex_nested_structure(self):
        """Complex nested structures should hash deterministically."""
        payload = {
            "level1": {
                "level2": {
                    "level3": [
                        {"a": 1},
                        {"b": 2},
                    ]
                }
            },
            "array": [1, 2, 3],
            "null": None,
            "bool": True,
        }

        hash1 = compute_request_hash(payload)
        hash2 = compute_request_hash(payload)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_float_precision_edge_cases(self):
        """Float precision edge cases should be handled."""
        # Very small floats
        payload = {"tiny": 0.00000000001}
        result = compute_request_hash(
            payload, float_fields=["tiny"], float_precision=10
        )
        assert len(result) == 64

        # Very large floats
        payload = {"huge": 999999999999.999999}
        result = compute_request_hash(payload, float_fields=["huge"], float_precision=5)
        assert len(result) == 64

    def test_special_characters_in_key(self):
        """Special characters in key should be valid."""
        special_keys = [
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key:with:colons",
            "key/with/slashes",
            "key@with@at",
            "key+with+plus",
            "key=with=equals",
        ]
        for key in special_keys:
            validate_idempotency_key(key)  # Should not raise
