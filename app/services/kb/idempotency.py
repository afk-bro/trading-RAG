"""KB Trial Ingestion Idempotency Primitives.

Provides deterministic point ID generation and content hash computation
for idempotent trial ingestion via kb_trial_index.

This is Phase 4 of the trial ingestion design.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol
from uuid import UUID, uuid5

import structlog

logger = structlog.get_logger(__name__)

# Namespace for deterministic point IDs
# This is a fixed UUID that ensures consistent point IDs across ingestion runs
KB_NAMESPACE = UUID("c8f4e2a1-5b3d-4c7e-9f1a-2d8b6e0c3a5f")


class IngestAction(str, Enum):
    """Action taken during ingestion."""

    INSERTED = "inserted"  # New trial added to Qdrant
    UPDATED = "updated"  # Existing trial updated (content changed)
    SKIPPED = "skipped"  # No changes needed (hash matches)
    UNARCHIVED = "unarchived"  # Previously archived trial restored


@dataclass
class IngestResult:
    """Result of a single trial ingestion attempt.

    Attributes:
        source_type: Type of source (tune_run or test_variant)
        source_id: ID of the source record
        action: What action was taken
        point_id: Qdrant point ID
        content_hash: Content hash after ingestion
        error: Error message if failed
    """

    source_type: str
    source_id: UUID
    action: IngestAction
    point_id: UUID
    content_hash: Optional[str] = None
    error: Optional[str] = None


@dataclass
class IndexEntry:
    """Entry from kb_trial_index table.

    Represents the current state of a trial in the KB index.

    Attributes:
        id: Row ID in kb_trial_index
        workspace_id: Workspace ID
        source_type: Source type (tune_run or test_variant)
        source_id: Source record ID
        qdrant_point_id: Qdrant point ID
        content_hash: SHA256 content hash
        content_hash_algo: Hash algorithm version
        embed_model: Embedding model used
        collection_name: Qdrant collection name
        ingested_at: When first ingested
        archived_at: When archived (None if active)
        archived_reason: Why archived
        archived_by: Who archived
    """

    id: UUID
    workspace_id: UUID
    source_type: str
    source_id: UUID
    qdrant_point_id: UUID
    content_hash: str
    content_hash_algo: str
    embed_model: str
    collection_name: str
    ingested_at: datetime
    archived_at: Optional[datetime] = None
    archived_reason: Optional[str] = None
    archived_by: Optional[str] = None


def compute_point_id(workspace_id: UUID, source_type: str, source_id: UUID) -> UUID:
    """Compute deterministic Qdrant point ID.

    Uses UUID5 with KB namespace for reproducibility. This ensures
    the same trial always gets the same point ID, enabling idempotent
    upserts and preventing duplicates.

    Args:
        workspace_id: Workspace ID
        source_type: Source type (tune_run or test_variant)
        source_id: Source record ID

    Returns:
        Deterministic UUID for Qdrant point
    """
    name = f"{workspace_id}:{source_type}:{source_id}"
    return uuid5(KB_NAMESPACE, name)


def compute_content_hash(
    embed_text: str,
    collection_name: str,
    strategy_name: str,
    params: dict,
    sharpe_oos: Optional[float],
    return_frac_oos: Optional[float],
    max_dd_frac_oos: Optional[float],
    regime_schema_version: Optional[str],
    experiment_type: Optional[str] = None,
    kb_status: Optional[str] = None,
) -> str:
    """Compute content hash for change detection.

    Hash includes all fields that affect embedding or retrieval.
    Used to detect when a trial needs re-ingestion.

    Args:
        embed_text: Text that will be embedded
        collection_name: Qdrant collection name
        strategy_name: Strategy name
        params: Strategy parameters
        sharpe_oos: OOS Sharpe ratio
        return_frac_oos: OOS return fraction
        max_dd_frac_oos: OOS max drawdown fraction
        regime_schema_version: Regime schema version
        experiment_type: Experiment type (for kb_status consideration)
        kb_status: KB status (for retrieval filtering)

    Returns:
        SHA256 hash string (full 64 chars)
    """
    canonical = json.dumps(
        {
            "embed_text": embed_text,
            "collection": collection_name,
            "strategy_name": strategy_name,
            "params": params,
            "metrics": {
                "sharpe_oos": sharpe_oos,
                "return_frac_oos": return_frac_oos,
                "max_dd_frac_oos": max_dd_frac_oos,
            },
            "regime_schema_version": regime_schema_version,
            "experiment_type": experiment_type,
            "kb_status": kb_status,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_content_hash_from_trial(
    trial,  # TrialDoc
    collection_name: str,
    embed_text: str,
    experiment_type: Optional[str] = None,
    kb_status: Optional[str] = None,
) -> str:
    """Compute content hash from TrialDoc.

    Convenience wrapper around compute_content_hash.

    Args:
        trial: TrialDoc instance
        collection_name: Qdrant collection name
        embed_text: Text that will be embedded
        experiment_type: Experiment type override
        kb_status: KB status override

    Returns:
        SHA256 hash string
    """
    regime_version = None
    if hasattr(trial, "regime_oos") and trial.regime_oos:
        regime_version = trial.regime_oos.schema_version

    return compute_content_hash(
        embed_text=embed_text,
        collection_name=collection_name,
        strategy_name=trial.strategy_name,
        params=trial.params,
        sharpe_oos=trial.sharpe_oos,
        return_frac_oos=trial.return_frac_oos,
        max_dd_frac_oos=trial.max_dd_frac_oos,
        regime_schema_version=regime_version,
        experiment_type=experiment_type,
        kb_status=kb_status,
    )


class KBTrialIndexRepository(Protocol):
    """Protocol for kb_trial_index persistence operations.

    Implementations handle the actual database operations.
    """

    async def get_index_entry(
        self, workspace_id: UUID, source_type: str, source_id: UUID
    ) -> Optional[IndexEntry]:
        """Get index entry for a trial.

        Args:
            workspace_id: Workspace ID
            source_type: Source type (tune_run or test_variant)
            source_id: Source record ID

        Returns:
            IndexEntry or None if not found
        """
        ...

    async def insert_index_entry(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        qdrant_point_id: UUID,
        content_hash: str,
        content_hash_algo: str,
        regime_schema_version: Optional[str],
        embed_model: str,
        collection_name: str,
    ) -> UUID:
        """Insert a new index entry.

        Args:
            workspace_id: Workspace ID
            source_type: Source type
            source_id: Source record ID
            qdrant_point_id: Qdrant point ID
            content_hash: Content hash
            content_hash_algo: Hash algorithm (e.g., "sha256_v1")
            regime_schema_version: Regime schema version
            embed_model: Embedding model used
            collection_name: Qdrant collection name

        Returns:
            New row ID
        """
        ...

    async def update_index_hash(self, entry_id: UUID, content_hash: str) -> None:
        """Update content hash for existing entry.

        Args:
            entry_id: Row ID in kb_trial_index
            content_hash: New content hash
        """
        ...

    async def unarchive_entry(self, entry_id: UUID, content_hash: str) -> None:
        """Clear archived_at and update hash.

        Args:
            entry_id: Row ID in kb_trial_index
            content_hash: New content hash
        """
        ...

    async def archive_entry(
        self, entry_id: UUID, reason: str, actor: Optional[str]
    ) -> None:
        """Set archived_at, reason, and actor.

        Args:
            entry_id: Row ID in kb_trial_index
            reason: Reason for archiving
            actor: Who is archiving
        """
        ...

    async def get_archived_entries(
        self, workspace_id: UUID, limit: int = 100
    ) -> list[IndexEntry]:
        """Get archived entries for a workspace.

        Args:
            workspace_id: Workspace ID
            limit: Maximum entries to return

        Returns:
            List of archived IndexEntry
        """
        ...


@dataclass
class BatchIngestResult:
    """Result of batch ingestion.

    Attributes:
        total: Total trials processed
        inserted: Number inserted
        updated: Number updated
        skipped: Number skipped (no changes)
        unarchived: Number unarchived
        errors: Number of errors
        by_source: Counts by source type
        error_details: List of error messages
    """

    total: int
    inserted: int
    updated: int
    skipped: int
    unarchived: int
    errors: int
    by_source: dict[str, int]
    error_details: list[str]

    @classmethod
    def from_results(cls, results: list[IngestResult]) -> "BatchIngestResult":
        """Create summary from list of results."""
        by_source: dict[str, int] = {}
        error_details: list[str] = []
        inserted = updated = skipped = unarchived = errors = 0

        for r in results:
            by_source[r.source_type] = by_source.get(r.source_type, 0) + 1

            if r.error:
                errors += 1
                error_details.append(f"{r.source_type}:{r.source_id}: {r.error}")
            elif r.action == IngestAction.INSERTED:
                inserted += 1
            elif r.action == IngestAction.UPDATED:
                updated += 1
            elif r.action == IngestAction.SKIPPED:
                skipped += 1
            elif r.action == IngestAction.UNARCHIVED:
                unarchived += 1

        return cls(
            total=len(results),
            inserted=inserted,
            updated=updated,
            skipped=skipped,
            unarchived=unarchived,
            errors=errors,
            by_source=by_source,
            error_details=error_details,
        )

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "total": self.total,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "unarchived": self.unarchived,
            "errors": self.errors,
            "by_source": self.by_source,
        }
