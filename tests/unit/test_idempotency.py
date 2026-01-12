"""Tests for run plan idempotency utilities."""

import pytest
from uuid import UUID

from app.services.testing.idempotency import compute_request_hash


class TestComputeRequestHash:
    """Tests for request hash computation."""

    def test_deterministic_for_same_input(self):
        """Same input always produces same hash."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")
        plan = {"inputs": {"symbol": "BTC"}, "resolved": {}}

        hash1 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )
        hash2 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256 truncated to 32 chars

    def test_different_for_different_input(self):
        """Different inputs produce different hashes."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")
        plan = {"inputs": {"symbol": "BTC"}, "resolved": {}}

        hash1 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )
        hash2 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="calmar",  # Different objective
            plan=plan,
        )

        assert hash1 != hash2

    def test_key_order_does_not_affect_hash(self):
        """Dict key order doesn't change hash (canonical JSON)."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")

        plan1 = {"a": 1, "b": 2}
        plan2 = {"b": 2, "a": 1}

        hash1 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan1,
        )
        hash2 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan2,
        )

        assert hash1 == hash2

    def test_includes_strategy_entity_id_when_provided(self):
        """Strategy entity ID affects hash when provided."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")
        strategy_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        plan = {"inputs": {}}

        hash_without = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )
        hash_with = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
            objective_name="sharpe",
            plan=plan,
        )

        assert hash_without != hash_with
