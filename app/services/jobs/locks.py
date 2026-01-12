"""Advisory lock utilities for job execution."""

import hashlib
from uuid import UUID


def job_lock_key(job_name: str, workspace_id: UUID) -> int:
    """
    Generate stable 64-bit unsigned lock key for pg_try_advisory_lock.

    Args:
        job_name: Name of the job (e.g., "rollup_events")
        workspace_id: Workspace UUID for scoping

    Returns:
        Unsigned 64-bit integer suitable for advisory lock
    """
    raw = f"{job_name}:{workspace_id}"
    h = hashlib.sha256(raw.encode()).digest()[:8]
    return int.from_bytes(h, byteorder="big", signed=False)
