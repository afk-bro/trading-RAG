"""Generic idempotency service for API endpoints.

Provides atomic claim semantics to prevent duplicate operations on client retries.
Keys are scoped to (workspace_id, endpoint) and expire after 7 days.

Usage:
    1. Client sends X-Idempotency-Key header
    2. Router calls claim_or_get() to atomically claim or retrieve existing
    3. If claimed (is_new=True), proceed with operation, then call complete()
    4. If exists (is_new=False), verify hash and replay response

Idempotency Contract:
    - Request hash: SHA256 of canonical JSON (sort_keys=True, separators=(",",":"))
    - Float normalization: Only for known float fields, rounded to 10 decimals
    - Key length: Max 200 characters
    - Expiry: 7 days from creation
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# Maximum allowed idempotency key length
MAX_IDEMPOTENCY_KEY_LENGTH = 200


class IdempotencyKeyTooLongError(ValueError):
    """Raised when idempotency key exceeds max length."""

    def __init__(self, key_length: int):
        self.key_length = key_length
        super().__init__(
            f"Idempotency key too long: {key_length} chars (max {MAX_IDEMPOTENCY_KEY_LENGTH})"
        )


class IdempotencyKeyReusedError(Exception):
    """Raised when idempotency key is reused with a different payload."""

    def __init__(self, expected_hash: str, actual_hash: str):
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__("Idempotency key reused with different payload")


class IdempotencyKeyInProgressError(Exception):
    """Raised when idempotency key is still being processed by another request."""

    def __init__(self, retry_after_seconds: int = 5):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Idempotency key in progress, retry after {retry_after_seconds}s"
        )


@dataclass
class IdempotencyRecord:
    """Record from the idempotency_keys table."""

    id: UUID
    workspace_id: UUID
    idempotency_key: str
    request_hash: str
    endpoint: str
    http_method: str
    api_version: str | None
    resource_id: UUID | None
    response_json: dict | None
    error_code: str | None
    error_json: dict | None
    status: str  # "pending", "completed", "failed"

    @classmethod
    def from_row(cls, row: dict) -> "IdempotencyRecord":
        """Create from database row."""
        return cls(
            id=row["id"],
            workspace_id=row["workspace_id"],
            idempotency_key=row["idempotency_key"],
            request_hash=row["request_hash"],
            endpoint=row["endpoint"],
            http_method=row["http_method"],
            api_version=row.get("api_version"),
            resource_id=row.get("resource_id"),
            response_json=row.get("response_json"),
            error_code=row.get("error_code"),
            error_json=row.get("error_json"),
            status=row["status"],
        )


def validate_idempotency_key(key: str) -> None:
    """Validate idempotency key length.

    Raises:
        IdempotencyKeyTooLongError: If key exceeds max length
    """
    if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise IdempotencyKeyTooLongError(len(key))


def compute_request_hash(
    payload: dict,
    float_fields: list[str] | None = None,
    float_precision: int = 10,
) -> str:
    """Compute SHA256 hash of canonical JSON payload.

    Canonicalization rules:
    - sort_keys=True (deterministic dict key order)
    - separators=(",", ":") (no whitespace)
    - ensure_ascii=False (unicode support)
    - Float normalization: ONLY for explicitly listed fields (to avoid accidental collisions)

    Args:
        payload: Request payload to hash
        float_fields: List of dotted paths to float fields that should be normalized
                     (e.g., ["objective_params.lambda", "objective_params.threshold"])
        float_precision: Decimal places for float normalization (default 10)

    Returns:
        64-character SHA256 hex string
    """
    # Normalize floats only in specified fields
    if float_fields:
        payload = _normalize_float_fields(payload, float_fields, float_precision)

    # Canonical JSON
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_float_fields(
    obj: Any,
    float_fields: list[str],
    precision: int,
    current_path: str = "",
) -> Any:
    """Normalize float fields at specified paths.

    Args:
        obj: Object to process
        float_fields: List of dotted paths to normalize
        precision: Decimal places to round to
        current_path: Current path in traversal (for matching)

    Returns:
        Object with normalized floats at specified paths
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            path = f"{current_path}.{k}" if current_path else k
            result[k] = _normalize_float_fields(v, float_fields, precision, path)
        return result
    elif isinstance(obj, list):
        return [
            _normalize_float_fields(item, float_fields, precision, f"{current_path}[]")
            for item in obj
        ]
    elif isinstance(obj, float) and current_path in float_fields:
        return round(obj, precision)
    return obj


def verify_hash_match(record: IdempotencyRecord, computed_hash: str) -> None:
    """Verify that the computed hash matches the stored hash.

    Raises:
        IdempotencyKeyReusedError: If hashes don't match
    """
    if record.request_hash != computed_hash:
        logger.warning(
            "idempotency_key_reused",
            idempotency_key=record.idempotency_key,
            endpoint=record.endpoint,
            expected_hash=record.request_hash[:16] + "...",
            actual_hash=computed_hash[:16] + "...",
        )
        raise IdempotencyKeyReusedError(record.request_hash, computed_hash)
