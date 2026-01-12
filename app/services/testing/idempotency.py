"""Idempotency utilities for run plans."""

import hashlib
import json
from typing import Any, Optional, Protocol
from uuid import UUID

from fastapi import HTTPException


class RunPlansRepositoryProtocol(Protocol):
    """Protocol for run plans repository methods used by idempotency."""

    async def get_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[dict[str, Any]]: ...

    async def get_by_request_hash(
        self, request_hash: str
    ) -> Optional[dict[str, Any]]: ...

    async def create_run_plan(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        objective_name: str,
        n_variants: int,
        plan: dict[str, Any],
        status: str = "pending",
        idempotency_key: Optional[str] = None,
        request_hash: Optional[str] = None,
    ) -> UUID: ...


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


async def create_run_plan_with_idempotency(
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID],
    objective_name: str,
    plan: dict[str, Any],
    idempotency_key: Optional[str],
    repo: RunPlansRepositoryProtocol,
    n_variants: int = 0,
) -> dict[str, Any]:
    """
    Create run plan with idempotency handling.

    Args:
        workspace_id: Workspace ID
        strategy_entity_id: Optional strategy entity ID
        objective_name: Objective function name
        plan: Full plan dict
        idempotency_key: Optional client-provided key
        repo: Run plans repository
        n_variants: Number of variants

    Returns:
        Dict with id and status ("created" or "existing")

    Raises:
        HTTPException 409: If duplicate detected
    """
    # Check for existing by idempotency key
    if idempotency_key:
        existing = await repo.get_by_idempotency_key(idempotency_key)
        if existing:
            if existing["status"] != "pending":
                raise HTTPException(
                    status_code=409,
                    detail=f"Plan {existing['id']} already {existing['status']}",
                )
            return {"id": existing["id"], "status": "existing"}

    # Compute request hash
    request_hash = compute_request_hash(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        objective_name=objective_name,
        plan=plan,
    )

    # Check for existing by request hash
    existing_by_hash = await repo.get_by_request_hash(request_hash)
    if existing_by_hash:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate request (plan {existing_by_hash['id']})",
        )

    # Create new plan
    plan_id = await repo.create_run_plan(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        objective_name=objective_name,
        n_variants=n_variants,
        plan=plan,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )

    return {"id": plan_id, "status": "created"}
