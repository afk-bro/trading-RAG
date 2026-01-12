"""Idempotency utilities for run plans."""

import hashlib
import json
from typing import Any, Optional
from uuid import UUID


def compute_request_hash(
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID],
    objective_name: str,
    plan: dict[str, Any],
) -> str:
    """
    Compute canonical hash of run plan request for duplicate detection.

    Args:
        workspace_id: Workspace ID
        strategy_entity_id: Optional strategy entity ID
        objective_name: Objective function name
        plan: Full plan dict

    Returns:
        32-character hex hash (SHA256 truncated)
    """
    canonical = {
        "workspace_id": str(workspace_id),
        "strategy_entity_id": str(strategy_entity_id) if strategy_entity_id else None,
        "objective_name": objective_name,
        "plan": plan,
    }
    # Sort keys recursively for determinism
    json_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:32]
