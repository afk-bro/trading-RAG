"""Contract tests for QueryMeta schema stability.

These tests ensure the QueryMeta shape remains consistent across refactors.
Any breaking change to the schema will fail these tests, alerting developers
to update consuming dashboards/observability tooling.

Run with: pytest tests/unit/test_query_meta_contract.py -v
"""

import pytest
from app.schemas import QueryMeta, RerankState


class TestQueryMetaContract:
    """Contract snapshot tests for QueryMeta schema."""

    # Expected field names and their types (for schema stability)
    EXPECTED_FIELDS = {
        # Timing fields (required ints, optional ints)
        "embed_ms": int,
        "search_ms": int,
        "rerank_ms": (int, type(None)),  # Optional[int]
        "expand_ms": (int, type(None)),
        "answer_ms": (int, type(None)),
        "total_ms": int,
        # Count fields
        "candidates_searched": int,
        "seeds_count": int,
        "chunks_after_expand": int,
        "neighbors_added": int,
        # Rerank state fields
        "rerank_state": str,  # Enum serializes to string
        "rerank_enabled": bool,
        "rerank_method": (str, type(None)),
        "rerank_model": (str, type(None)),
        "rerank_timeout": bool,
        "rerank_fallback": bool,
        # Neighbor fields
        "neighbor_enabled": bool,
    }

    def test_querymeta_has_expected_fields(self):
        """Verify QueryMeta has exactly the expected fields."""
        meta = QueryMeta(
            embed_ms=10,
            search_ms=20,
            total_ms=100,
            candidates_searched=50,
            seeds_count=10,
            chunks_after_expand=12,
        )

        actual_fields = set(meta.model_fields.keys())
        expected_fields = set(self.EXPECTED_FIELDS.keys())

        # Check for missing fields (schema regression)
        missing = expected_fields - actual_fields
        assert not missing, f"QueryMeta missing expected fields: {missing}"

        # Check for unexpected new fields (schema drift - update test if intentional)
        extra = actual_fields - expected_fields
        assert not extra, f"QueryMeta has unexpected fields: {extra}. Update test if intentional."

    def test_querymeta_field_types_match_contract(self):
        """Verify field types match the contract."""
        meta = QueryMeta(
            embed_ms=10,
            search_ms=20,
            total_ms=100,
            candidates_searched=50,
            seeds_count=10,
            chunks_after_expand=12,
        )

        # Serialize to dict for type checking
        data = meta.model_dump()

        for field_name, expected_type in self.EXPECTED_FIELDS.items():
            value = data[field_name]

            if isinstance(expected_type, tuple):
                # Optional field - can be one of multiple types
                assert isinstance(value, expected_type), (
                    f"Field '{field_name}' has type {type(value).__name__}, "
                    f"expected one of {expected_type}"
                )
            else:
                # Required field or None for optional
                if value is not None:
                    assert isinstance(value, expected_type), (
                        f"Field '{field_name}' has type {type(value).__name__}, "
                        f"expected {expected_type.__name__}"
                    )

    def test_rerank_disabled_nulls_rerank_fields(self):
        """When rerank disabled, rerank-specific fields should be null/default."""
        meta = QueryMeta(
            embed_ms=10,
            search_ms=20,
            total_ms=100,
            candidates_searched=50,
            seeds_count=10,
            chunks_after_expand=12,
            rerank_state=RerankState.DISABLED,
            rerank_enabled=False,
        )

        assert meta.rerank_state == RerankState.DISABLED
        assert meta.rerank_enabled is False
        assert meta.rerank_ms is None
        assert meta.rerank_method is None
        assert meta.rerank_model is None
        assert meta.rerank_timeout is False
        assert meta.rerank_fallback is False

    def test_rerank_ok_populates_rerank_fields(self):
        """When rerank OK, rerank-specific fields should be populated."""
        meta = QueryMeta(
            embed_ms=10,
            search_ms=20,
            rerank_ms=50,
            total_ms=100,
            candidates_searched=50,
            seeds_count=10,
            chunks_after_expand=12,
            rerank_state=RerankState.OK,
            rerank_enabled=True,
            rerank_method="cross_encoder",
            rerank_model="BAAI/bge-reranker-v2-m3",
        )

        assert meta.rerank_state == RerankState.OK
        assert meta.rerank_enabled is True
        assert meta.rerank_ms == 50
        assert meta.rerank_method == "cross_encoder"
        assert meta.rerank_model == "BAAI/bge-reranker-v2-m3"
        assert meta.rerank_timeout is False
        assert meta.rerank_fallback is False

    def test_querymeta_serializes_to_json_compatible_dict(self):
        """Verify QueryMeta serializes cleanly for JSON responses."""
        meta = QueryMeta(
            embed_ms=10,
            search_ms=20,
            total_ms=100,
            candidates_searched=50,
            seeds_count=10,
            chunks_after_expand=12,
            rerank_state=RerankState.TIMEOUT_FALLBACK,
            rerank_timeout=True,
            rerank_fallback=True,
        )

        data = meta.model_dump()

        # Enum should serialize to string value
        assert data["rerank_state"] == "timeout_fallback"
        assert isinstance(data["rerank_state"], str)

        # All values should be JSON-serializable types
        import json
        json_str = json.dumps(data)  # Should not raise
        assert json_str is not None


class TestRerankStateTransitions:
    """Matrix test for rerank state transitions and field expectations.

    Ensures observability consistency across all rerank states.
    """

    # State matrix: (state, expected_fields)
    STATE_MATRIX = [
        # (state, rerank_enabled, rerank_timeout, rerank_fallback, rerank_ms_is_none)
        (RerankState.DISABLED, False, False, False, True),
        (RerankState.OK, True, False, False, False),
        (RerankState.TIMEOUT_FALLBACK, True, True, True, False),
        (RerankState.ERROR_FALLBACK, True, False, True, False),
    ]

    @pytest.mark.parametrize(
        "state,enabled,timeout,fallback,ms_is_none",
        STATE_MATRIX,
        ids=["disabled", "ok", "timeout_fallback", "error_fallback"],
    )
    def test_rerank_state_field_consistency(
        self, state, enabled, timeout, fallback, ms_is_none
    ):
        """Verify each rerank state has consistent field values."""
        meta = QueryMeta(
            embed_ms=10,
            search_ms=20,
            rerank_ms=None if ms_is_none else 50,
            total_ms=100,
            candidates_searched=50,
            seeds_count=10,
            chunks_after_expand=12,
            rerank_state=state,
            rerank_enabled=enabled,
            rerank_timeout=timeout,
            rerank_fallback=fallback,
            rerank_method="cross_encoder" if enabled else None,
            rerank_model="mock-model" if enabled else None,
        )

        # Verify state string matches enum value
        assert meta.rerank_state == state
        assert meta.model_dump()["rerank_state"] == state.value

        # Verify field consistency
        assert meta.rerank_enabled == enabled
        assert meta.rerank_timeout == timeout
        assert meta.rerank_fallback == fallback
        assert (meta.rerank_ms is None) == ms_is_none

    def test_rerank_state_enum_values_are_strings(self):
        """Verify RerankState enum values are lowercase strings."""
        expected_values = {"disabled", "ok", "timeout_fallback", "error_fallback"}
        actual_values = {s.value for s in RerankState}

        assert actual_values == expected_values

    def test_all_rerank_states_covered(self):
        """Ensure test matrix covers all RerankState values."""
        tested_states = {row[0] for row in self.STATE_MATRIX}
        all_states = set(RerankState)

        missing = all_states - tested_states
        assert not missing, f"State matrix missing states: {missing}"
