"""KB Trial Ingestion Service.

Provides batch ingestion of eligible trials from kb_eligible_trials view
into Qdrant vector store with idempotency via kb_trial_index.

This is Phase 4 of the trial ingestion design.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol
from uuid import UUID

import structlog

from app.services.kb.constants import (
    KB_TRIALS_COLLECTION_NAME,
    REGIME_SCHEMA_VERSION,
)
from app.services.kb.embed import KBEmbeddingAdapter
from app.services.kb.idempotency import (
    BatchIngestResult,
    IndexEntry,
    IngestAction,
    IngestResult,
    KBTrialIndexRepository,
    compute_content_hash_from_trial,
    compute_point_id,
)
from app.services.kb.trial_doc import (
    TrialDoc,
    build_trial_doc_from_eligible_row,
    trial_to_metadata,
    trial_to_text,
)
from app.services.kb.types import RegimeSnapshot

logger = structlog.get_logger(__name__)


class EligibleTrialsRepository(Protocol):
    """Protocol for fetching eligible trials from kb_eligible_trials view."""

    async def get_eligible_trials(
        self,
        workspace_id: UUID,
        source_types: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch eligible trials from view.

        Args:
            workspace_id: Filter by workspace
            source_types: Filter by source types (tune_run, test_variant)
            since: Filter by created_at >= since
            limit: Maximum rows to return

        Returns:
            List of dicts from kb_eligible_trials view
        """
        ...


class QdrantAdapter(Protocol):
    """Protocol for Qdrant vector operations."""

    async def upsert_point(
        self,
        collection_name: str,
        point_id: UUID,
        vector: list[float],
        payload: dict,
    ) -> None:
        """Upsert a single point to collection."""
        ...

    async def delete_point(
        self,
        collection_name: str,
        point_id: UUID,
    ) -> None:
        """Delete a point from collection."""
        ...


@dataclass
class IngestConfig:
    """Configuration for ingestion."""

    collection_name: str = KB_TRIALS_COLLECTION_NAME
    embed_model: str = "nomic-embed-text"
    content_hash_algo: str = "sha256_v1"
    batch_size: int = 50
    dry_run: bool = False


class KBTrialIngester:
    """Ingests eligible trials into Qdrant with idempotency.

    Uses kb_trial_index to track what's been ingested and detect changes.
    Supports insert, update, skip, and unarchive operations.
    """

    def __init__(
        self,
        index_repo: KBTrialIndexRepository,
        eligible_repo: EligibleTrialsRepository,
        embedder: KBEmbeddingAdapter,
        qdrant: QdrantAdapter,
        config: Optional[IngestConfig] = None,
    ):
        """Initialize ingester.

        Args:
            index_repo: Repository for kb_trial_index operations
            eligible_repo: Repository for fetching eligible trials
            embedder: Embedding adapter for generating vectors
            qdrant: Qdrant adapter for vector operations
            config: Ingestion configuration
        """
        self._index_repo = index_repo
        self._eligible_repo = eligible_repo
        self._embedder = embedder
        self._qdrant = qdrant
        self._config = config or IngestConfig()

    async def ingest_workspace(
        self,
        workspace_id: UUID,
        source_types: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> BatchIngestResult:
        """Ingest all eligible trials for a workspace.

        Fetches from kb_eligible_trials view and processes each trial,
        tracking results for reporting.

        Args:
            workspace_id: Workspace to ingest for
            source_types: Filter by source types (default: all)
            since: Only ingest trials created after this time
            limit: Maximum trials to process

        Returns:
            BatchIngestResult with counts and errors
        """
        log = logger.bind(
            workspace_id=str(workspace_id),
            source_types=source_types,
            since=since.isoformat() if since else None,
        )
        log.info("Starting workspace ingestion")

        # Fetch eligible trials
        rows = await self._eligible_repo.get_eligible_trials(
            workspace_id=workspace_id,
            source_types=source_types,
            since=since,
            limit=limit,
        )
        log.info("Fetched eligible trials", count=len(rows))

        if not rows:
            return BatchIngestResult.from_results([])

        # Process each trial
        results: list[IngestResult] = []
        for row in rows:
            try:
                result = await self._ingest_row(workspace_id, row)
                results.append(result)
            except Exception as e:
                log.error(
                    "Failed to ingest trial",
                    source_type=row.get("source_type"),
                    source_id=str(row.get("source_id")),
                    error=str(e),
                )
                results.append(
                    IngestResult(
                        source_type=row.get("source_type", "unknown"),
                        source_id=row.get("source_id"),
                        action=IngestAction.SKIPPED,
                        point_id=compute_point_id(
                            workspace_id,
                            row.get("source_type", "unknown"),
                            row.get("source_id"),
                        ),
                        error=str(e),
                    )
                )

        batch_result = BatchIngestResult.from_results(results)
        log.info(
            "Completed workspace ingestion",
            total=batch_result.total,
            inserted=batch_result.inserted,
            updated=batch_result.updated,
            skipped=batch_result.skipped,
            unarchived=batch_result.unarchived,
            errors=batch_result.errors,
        )

        # Observability counters: emit one log per action type for aggregation
        for action, count in [
            ("inserted", batch_result.inserted),
            ("updated", batch_result.updated),
            ("skipped", batch_result.skipped),
            ("unarchived", batch_result.unarchived),
            ("error", batch_result.errors),
        ]:
            if count > 0:
                log.info(
                    "kb_ingest_action_total",
                    action=action,
                    count=count,
                    workspace_id=str(workspace_id),
                    collection_name=self._config.collection_name,
                )

        return batch_result

    async def _ingest_row(self, workspace_id: UUID, row: dict) -> IngestResult:
        """Ingest a single eligible trial row.

        Failure handling:
        - Qdrant upsert runs first, then DB index update
        - If Qdrant fails: exception raised, DB unchanged (safe)
        - If DB fails after Qdrant success: orphaned Qdrant point exists
          but idempotency allows safe re-run (content hash will match,
          triggering update which re-syncs the index)
        - Correlation: log messages include workspace_id, source_type, source_id

        Args:
            workspace_id: Workspace ID
            row: Row from kb_eligible_trials view

        Returns:
            IngestResult with action taken
        """
        source_type = row.get("source_type")
        source_id = row.get("source_id")

        log = logger.bind(
            workspace_id=str(workspace_id),
            source_type=source_type,
            source_id=str(source_id),
        )

        # Build TrialDoc from row
        trial = build_trial_doc_from_eligible_row(row)
        if trial is None:
            return IngestResult(
                source_type=source_type,
                source_id=source_id,
                action=IngestAction.SKIPPED,
                point_id=compute_point_id(workspace_id, source_type, source_id),
                error="Could not build TrialDoc from row",
            )

        # Compute deterministic point ID
        point_id = compute_point_id(workspace_id, source_type, source_id)

        # Generate embed text
        embed_text = trial_to_text(trial)

        # Compute content hash
        experiment_type = row.get("experiment_type")
        kb_status = row.get("kb_status")
        content_hash = compute_content_hash_from_trial(
            trial=trial,
            collection_name=self._config.collection_name,
            embed_text=embed_text,
            experiment_type=experiment_type,
            kb_status=kb_status,
        )

        # Get regime schema version
        regime_version = None
        if trial.regime_oos:
            regime_version = trial.regime_oos.schema_version
        elif trial.regime_is:
            regime_version = trial.regime_is.schema_version

        # Check existing index entry
        existing = await self._index_repo.get_index_entry(
            workspace_id=workspace_id,
            source_type=source_type,
            source_id=source_id,
        )

        # Determine action
        if existing is None:
            # New trial - insert
            action = IngestAction.INSERTED
            log.debug("Inserting new trial")
        elif existing.archived_at is not None:
            # Was archived - unarchive and re-upsert
            action = IngestAction.UNARCHIVED
            log.debug("Unarchiving trial")
        elif existing.content_hash == content_hash:
            # No changes - skip
            log.debug("Skipping unchanged trial")
            return IngestResult(
                source_type=source_type,
                source_id=source_id,
                action=IngestAction.SKIPPED,
                point_id=point_id,
                content_hash=content_hash,
            )
        else:
            # Content changed - update
            action = IngestAction.UPDATED
            log.debug(
                "Updating changed trial",
                old_hash=existing.content_hash[:16],
                new_hash=content_hash[:16],
            )

        # Dry run - don't actually do anything
        if self._config.dry_run:
            log.info("Dry run - would perform action", action=action.value)
            return IngestResult(
                source_type=source_type,
                source_id=source_id,
                action=action,
                point_id=point_id,
                content_hash=content_hash,
            )

        # Generate embedding
        embed_result = await self._embedder.embed([embed_text])
        if not embed_result.vectors or len(embed_result.vectors) == 0:
            return IngestResult(
                source_type=source_type,
                source_id=source_id,
                action=IngestAction.SKIPPED,
                point_id=point_id,
                error="Embedding returned no vectors",
            )

        vector = embed_result.vectors[0]

        # Build Qdrant payload
        payload = trial_to_metadata(
            trial,
            embedding_model_id=self._config.embed_model,
            vector_dim=len(vector),
        )
        # Add ingestion-specific metadata
        payload["source_type"] = source_type
        payload["experiment_type"] = experiment_type
        payload["kb_status"] = kb_status
        payload["kb_promoted_at"] = (
            row.get("kb_promoted_at").isoformat()
            if row.get("kb_promoted_at")
            else None
        )

        # Upsert to Qdrant
        await self._qdrant.upsert_point(
            collection_name=self._config.collection_name,
            point_id=point_id,
            vector=vector,
            payload=payload,
        )

        # Update index
        if action == IngestAction.INSERTED:
            await self._index_repo.insert_index_entry(
                workspace_id=workspace_id,
                source_type=source_type,
                source_id=source_id,
                qdrant_point_id=point_id,
                content_hash=content_hash,
                content_hash_algo=self._config.content_hash_algo,
                regime_schema_version=regime_version,
                embed_model=self._config.embed_model,
                collection_name=self._config.collection_name,
            )
        elif action == IngestAction.UNARCHIVED:
            await self._index_repo.unarchive_entry(
                entry_id=existing.id,
                content_hash=content_hash,
            )
        elif action == IngestAction.UPDATED:
            await self._index_repo.update_index_hash(
                entry_id=existing.id,
                content_hash=content_hash,
            )

        log.info("Ingested trial", action=action.value)

        return IngestResult(
            source_type=source_type,
            source_id=source_id,
            action=action,
            point_id=point_id,
            content_hash=content_hash,
        )

    async def ingest_single(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        row: Optional[dict] = None,
    ) -> IngestResult:
        """Ingest a single trial by ID.

        If row is not provided, fetches from eligible view.
        Useful for re-ingestion after promotion.

        Args:
            workspace_id: Workspace ID
            source_type: Source type (tune_run or test_variant)
            source_id: Source record ID
            row: Optional pre-fetched row data

        Returns:
            IngestResult
        """
        if row is None:
            # Fetch the specific trial from view
            # Note: This requires a different query method
            rows = await self._eligible_repo.get_eligible_trials(
                workspace_id=workspace_id,
                source_types=[source_type],
                limit=1,
            )
            # Filter to specific source_id
            matching = [r for r in rows if r.get("source_id") == source_id]
            if not matching:
                return IngestResult(
                    source_type=source_type,
                    source_id=source_id,
                    action=IngestAction.SKIPPED,
                    point_id=compute_point_id(workspace_id, source_type, source_id),
                    error="Trial not found in eligible view",
                )
            row = matching[0]

        return await self._ingest_row(workspace_id, row)

    async def archive_trial(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        reason: str,
        actor: Optional[str] = None,
    ) -> bool:
        """Archive a trial (remove from Qdrant, mark in index).

        Called when a trial is rejected or explicitly archived.

        Failure handling:
        - If Qdrant delete fails, we still mark the index entry archived
          with an error note. This prevents retry loops and allows manual
          cleanup via the runbook.
        - Re-running ingest on the same trial will unarchive and re-upsert,
          providing self-healing if Qdrant comes back.

        Args:
            workspace_id: Workspace ID
            source_type: Source type
            source_id: Source record ID
            reason: Reason for archiving
            actor: Who is archiving

        Returns:
            True if archived, False if not found
        """
        from uuid import uuid4

        correlation_id = str(uuid4())[:8]
        log = logger.bind(
            workspace_id=str(workspace_id),
            source_type=source_type,
            source_id=str(source_id),
            reason=reason,
            correlation_id=correlation_id,
        )

        # Find existing index entry
        existing = await self._index_repo.get_index_entry(
            workspace_id=workspace_id,
            source_type=source_type,
            source_id=source_id,
        )

        if existing is None:
            log.debug("Trial not in index, nothing to archive")
            return False

        if existing.archived_at is not None:
            log.debug("Trial already archived")
            return True

        # Delete from Qdrant (best-effort, log failures)
        qdrant_error = None
        try:
            await self._qdrant.delete_point(
                collection_name=existing.collection_name,
                point_id=existing.qdrant_point_id,
            )
        except Exception as e:
            qdrant_error = str(e)
            log.error(
                "kb_archive_qdrant_failed",
                error=qdrant_error,
                point_id=str(existing.qdrant_point_id),
                collection=existing.collection_name,
            )

        # Mark as archived in index (include error if Qdrant failed)
        archive_reason = reason
        if qdrant_error:
            archive_reason = f"{reason} [qdrant_delete_failed: {qdrant_error[:100]}]"

        await self._index_repo.archive_entry(
            entry_id=existing.id,
            reason=archive_reason,
            actor=actor,
        )

        if qdrant_error:
            log.warning(
                "Archived trial with Qdrant error - manual cleanup may be needed",
                qdrant_error=qdrant_error,
            )
        else:
            log.info("Archived trial")

        return True
