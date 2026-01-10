"""
KB ingestion pipeline for backtest trials.

Handles:
- Fetching tune_runs from database
- Converting to TrialDoc
- Embedding trial text
- Upserting to Qdrant
- Tracking ingestion progress

Features:
- Dry-run preview with sample diffs
- Advisory locking for concurrency safety
- Text hash tracking for drift detection
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

from app.services.kb.types import TrialDoc, utc_now_iso
from app.services.kb.trial_doc import (
    trial_to_text,
    trial_to_metadata,
    build_trial_doc_from_tune_run,
)
from app.services.kb.embed import (
    KBEmbeddingAdapter,
    EmbeddingError,
    get_kb_embedder,
)
from app.repositories.kb_trials import KBTrialRepository

logger = structlog.get_logger(__name__)

# Configuration
INGESTION_BATCH_SIZE = int(os.environ.get("KB_INGESTION_BATCH_SIZE", "50"))
INGESTION_EMBED_BATCH_SIZE = int(os.environ.get("KB_INGESTION_EMBED_BATCH_SIZE", "32"))

# Advisory lock ID for KB ingestion (arbitrary but unique)
KB_INGESTION_LOCK_ID = 0x4B42494E  # "KBIN" in hex


def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of trial text for drift detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class DryRunSample:
    """Sample item for dry-run preview."""

    tune_run_id: str
    strategy_name: str
    objective_type: str
    action: str  # "upsert" or "skip" or "reembed"
    reason: str  # Why this action


@dataclass
class DryRunPreview:
    """Preview of what would happen in a full run."""

    total_candidates: int
    would_upsert: int
    would_skip: int
    would_reembed: int  # Hash differs from stored
    samples: list[DryRunSample] = field(default_factory=list)


@dataclass
class IngestionStats:
    """Statistics from ingestion run."""

    total_fetched: int = 0
    total_skipped: int = 0  # Not completed, invalid, etc.
    total_embedded: int = 0
    total_upserted: int = 0
    total_failed: int = 0
    failed_tune_run_ids: list[str] = field(default_factory=list)
    embedding_failures: int = 0
    upsert_failures: int = 0
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0


@dataclass
class IngestionReport:
    """Report from ingestion run."""

    workspace_id: str
    collection_name: str
    model_id: str
    vector_dim: int
    stats: IngestionStats
    dry_run: bool = False
    preview: Optional[DryRunPreview] = None  # Populated in dry-run mode


class KBIngestionPipeline:
    """
    Pipeline for ingesting backtest trials into the Knowledge Base.

    Features:
    - Batch processing with configurable sizes
    - Idempotent (uses tune_run_id as point ID)
    - Resumable (tracks kb_ingested_at)
    - Supports dry run mode with preview
    - Advisory locking for concurrency safety
    - Text hash tracking for drift detection
    """

    def __init__(
        self,
        embedder: Optional[KBEmbeddingAdapter] = None,
        repository: Optional[KBTrialRepository] = None,
        tune_repo=None,  # BacktestTuneRepository
        batch_size: int = INGESTION_BATCH_SIZE,
        use_advisory_lock: bool = True,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            embedder: KB embedding adapter
            repository: KB trial repository (Qdrant)
            tune_repo: Backtest tune repository (database)
            batch_size: Batch size for processing
            use_advisory_lock: Whether to use advisory locks for concurrency
        """
        self._embedder = embedder
        self._repository = repository
        self._tune_repo = tune_repo
        self.batch_size = batch_size
        self.use_advisory_lock = use_advisory_lock

    @property
    def embedder(self) -> KBEmbeddingAdapter:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = get_kb_embedder()
        return self._embedder

    async def _acquire_workspace_lock(self, workspace_id: UUID) -> bool:
        """
        Try to acquire advisory lock for workspace ingestion.

        Uses pg_try_advisory_lock to prevent concurrent ingestion jobs
        for the same workspace.

        Args:
            workspace_id: Workspace to lock

        Returns:
            True if lock acquired, False if another job is running
        """
        if not self.use_advisory_lock or self._tune_repo is None:
            return True

        try:
            # Combine base lock ID with workspace hash for unique per-workspace lock
            workspace_hash = int(
                hashlib.md5(str(workspace_id).encode()).hexdigest()[:8], 16
            )
            lock_id = KB_INGESTION_LOCK_ID ^ workspace_hash

            result = await self._tune_repo.try_advisory_lock(lock_id)
            if not result:
                logger.warning(
                    "Could not acquire ingestion lock - another job may be running",
                    workspace_id=str(workspace_id),
                    lock_id=lock_id,
                )
            return result
        except AttributeError:
            # Repository doesn't support advisory locks
            logger.debug("Advisory locks not supported by repository")
            return True

    async def _release_workspace_lock(self, workspace_id: UUID) -> None:
        """Release advisory lock for workspace."""
        if not self.use_advisory_lock or self._tune_repo is None:
            return

        try:
            workspace_hash = int(
                hashlib.md5(str(workspace_id).encode()).hexdigest()[:8], 16
            )
            lock_id = KB_INGESTION_LOCK_ID ^ workspace_hash
            await self._tune_repo.release_advisory_lock(lock_id)
        except AttributeError:
            pass

    async def ingest_tune_runs(
        self,
        workspace_id: UUID,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        only_missing_vectors: bool = True,
        reembed: bool = False,
    ) -> IngestionReport:
        """
        Ingest tune_runs into the Knowledge Base.

        Args:
            workspace_id: Workspace to process
            since: Only process runs created after this time
            limit: Maximum number of runs to process
            dry_run: If True, don't actually upsert (generates preview)
            only_missing_vectors: Only process runs without vectors
            reembed: Force re-embedding even if already ingested

        Returns:
            IngestionReport with statistics (and preview if dry_run)
        """
        started_at = utc_now_iso()
        stats = IngestionStats(started_at=started_at)
        preview: Optional[DryRunPreview] = None

        # Ensure we have model info
        model_id = self.embedder.model_id
        vector_dim = await self.embedder.get_vector_dim()
        collection_name = self.embedder.get_collection_name()

        logger.info(
            "Starting KB ingestion",
            workspace_id=str(workspace_id),
            collection=collection_name,
            model_id=model_id,
            vector_dim=vector_dim,
            since=str(since) if since else None,
            limit=limit,
            dry_run=dry_run,
            reembed=reembed,
        )

        # Acquire advisory lock (skip for dry run)
        if not dry_run:
            if not await self._acquire_workspace_lock(workspace_id):
                raise RuntimeError(
                    f"Could not acquire ingestion lock for workspace {workspace_id}. "
                    "Another ingestion job may be running."
                )

        try:
            # Ensure collection exists (if not dry run)
            if not dry_run and self._repository:
                await self._repository.ensure_collection(
                    dimension=vector_dim,
                    recreate=False,
                )

            # Fetch tune_runs to process
            tune_runs = await self._fetch_tune_runs(
                workspace_id=workspace_id,
                since=since,
                limit=limit,
                only_missing=only_missing_vectors and not reembed,
            )
            stats.total_fetched = len(tune_runs)

            if not tune_runs:
                logger.info("No tune_runs to process")
                stats.completed_at = utc_now_iso()
                return IngestionReport(
                    workspace_id=str(workspace_id),
                    collection_name=collection_name,
                    model_id=model_id,
                    vector_dim=vector_dim,
                    stats=stats,
                    dry_run=dry_run,
                )

            # Generate dry-run preview if needed
            if dry_run:
                preview = await self._generate_dry_run_preview(
                    tune_runs=tune_runs,
                    workspace_id=workspace_id,
                    model_id=model_id,
                    reembed=reembed,
                )

            # Process in batches
            for batch_start in range(0, len(tune_runs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(tune_runs))
                batch = tune_runs[batch_start:batch_end]

                batch_stats = await self._process_batch(
                    batch=batch,
                    workspace_id=workspace_id,
                    model_id=model_id,
                    vector_dim=vector_dim,
                    dry_run=dry_run,
                )

                stats.total_skipped += batch_stats.total_skipped
                stats.total_embedded += batch_stats.total_embedded
                stats.total_upserted += batch_stats.total_upserted
                stats.total_failed += batch_stats.total_failed
                stats.failed_tune_run_ids.extend(batch_stats.failed_tune_run_ids)
                stats.embedding_failures += batch_stats.embedding_failures
                stats.upsert_failures += batch_stats.upsert_failures

                logger.debug(
                    "Processed batch",
                    batch_start=batch_start,
                    batch_size=len(batch),
                    embedded=batch_stats.total_embedded,
                    upserted=batch_stats.total_upserted,
                )
        finally:
            # Always release lock
            if not dry_run:
                await self._release_workspace_lock(workspace_id)

        stats.completed_at = utc_now_iso()
        stats.duration_seconds = (
            datetime.fromisoformat(stats.completed_at.replace("Z", "+00:00"))
            - datetime.fromisoformat(stats.started_at.replace("Z", "+00:00"))
        ).total_seconds()

        logger.info(
            "KB ingestion complete",
            workspace_id=str(workspace_id),
            total_fetched=stats.total_fetched,
            total_embedded=stats.total_embedded,
            total_upserted=stats.total_upserted,
            total_failed=stats.total_failed,
            duration_seconds=stats.duration_seconds,
        )

        return IngestionReport(
            workspace_id=str(workspace_id),
            collection_name=collection_name,
            model_id=model_id,
            vector_dim=vector_dim,
            stats=stats,
            dry_run=dry_run,
            preview=preview,
        )

    async def _generate_dry_run_preview(
        self,
        tune_runs: list[dict],
        workspace_id: UUID,
        model_id: str,
        reembed: bool = False,
        sample_count: int = 3,
    ) -> DryRunPreview:
        """
        Generate preview of what ingestion would do.

        Args:
            tune_runs: List of candidate tune_runs
            workspace_id: Workspace ID
            model_id: Current embedding model ID
            reembed: Whether this is a reembed operation
            sample_count: Number of samples to include

        Returns:
            DryRunPreview with counts and samples
        """
        would_upsert = 0
        would_skip = 0
        would_reembed = 0
        samples: list[DryRunSample] = []

        for run_data in tune_runs:
            tune_run = run_data.get("tune_run", run_data)
            tune = run_data.get("tune", {})

            tune_run_id = str(tune_run.get("id", "unknown"))
            strategy_name = tune.get("strategy_name", "unknown")
            objective_type = tune.get("objective_type", "unknown")
            status = tune_run.get("status", "unknown")

            # Determine action
            if status != "completed":
                action = "skip"
                reason = f"status={status}"
                would_skip += 1
            elif tune_run.get("kb_ingested_at") is None:
                action = "upsert"
                reason = "missing from KB"
                would_upsert += 1
            elif reembed:
                # Check for hash drift if hash is stored
                stored_hash = tune_run.get("kb_text_hash")
                if stored_hash:
                    # Build doc and compute current hash
                    doc = build_trial_doc_from_tune_run(
                        tune_run=tune_run,
                        tune=tune,
                        workspace_id=workspace_id,
                    )
                    if doc:
                        current_hash = compute_text_hash(trial_to_text(doc))
                        if stored_hash != current_hash:
                            action = "reembed"
                            reason = (
                                f"hash drift ({stored_hash[:8]}→{current_hash[:8]})"
                            )
                            would_reembed += 1
                        else:
                            action = "reembed"
                            reason = "forced reembed (hash unchanged)"
                            would_reembed += 1
                    else:
                        action = "skip"
                        reason = "failed to build doc"
                        would_skip += 1
                else:
                    action = "reembed"
                    reason = "forced reembed (no stored hash)"
                    would_reembed += 1
            else:
                action = "skip"
                reason = "already ingested"
                would_skip += 1

            # Collect samples
            if len(samples) < sample_count and action != "skip":
                samples.append(
                    DryRunSample(
                        tune_run_id=tune_run_id,
                        strategy_name=strategy_name,
                        objective_type=objective_type,
                        action=action,
                        reason=reason,
                    )
                )

        return DryRunPreview(
            total_candidates=len(tune_runs),
            would_upsert=would_upsert,
            would_skip=would_skip,
            would_reembed=would_reembed,
            samples=samples,
        )

    async def _fetch_tune_runs(
        self,
        workspace_id: UUID,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        only_missing: bool = True,
    ) -> list[dict]:
        """
        Fetch tune_runs from database.

        Args:
            workspace_id: Workspace ID
            since: Only fetch runs created after this time
            limit: Maximum number to fetch
            only_missing: Only fetch runs without kb_ingested_at

        Returns:
            List of tune_run dicts with their parent tune
        """
        if self._tune_repo is None:
            logger.warning("No tune repository configured, returning empty list")
            return []

        # This will be implemented to query the database
        # For now, we'll call a method that should be added to BacktestTuneRepository
        try:
            runs = await self._tune_repo.list_tune_runs_for_kb(
                workspace_id=workspace_id,
                since=since,
                limit=limit,
                only_missing_kb=only_missing,
            )
            return runs
        except AttributeError:
            logger.warning("list_tune_runs_for_kb not implemented in tune repository")
            return []

    async def _process_batch(
        self,
        batch: list[dict],
        workspace_id: UUID,
        model_id: str,
        vector_dim: int,
        dry_run: bool = False,
    ) -> IngestionStats:
        """
        Process a batch of tune_runs.

        Args:
            batch: List of tune_run dicts
            workspace_id: Workspace ID
            model_id: Embedding model ID
            vector_dim: Vector dimension
            dry_run: If True, don't actually upsert

        Returns:
            Batch statistics
        """
        stats = IngestionStats()

        # Convert to TrialDocs
        trial_docs: list[tuple[TrialDoc, dict]] = []
        for run_data in batch:
            tune_run = run_data.get("tune_run", run_data)
            tune = run_data.get("tune", {})

            doc = build_trial_doc_from_tune_run(
                tune_run=tune_run,
                tune=tune,
                workspace_id=workspace_id,
            )

            if doc is None:
                stats.total_skipped += 1
                continue

            trial_docs.append((doc, tune_run))

        if not trial_docs:
            return stats

        # Generate texts for embedding and compute hashes
        texts = [trial_to_text(doc) for doc, _ in trial_docs]
        text_hashes = [compute_text_hash(text) for text in texts]

        # Embed texts
        try:
            embed_result = await self.embedder.embed_texts(texts, skip_failures=True)
            stats.total_embedded = len(texts) - len(embed_result.failed_indices)
            stats.embedding_failures = len(embed_result.failed_indices)
        except EmbeddingError as e:
            logger.error("Batch embedding failed", error=e.message)
            stats.embedding_failures = len(texts)
            for doc, tune_run in trial_docs:
                stats.failed_tune_run_ids.append(str(doc.tune_run_id))
            stats.total_failed = len(trial_docs)
            return stats

        # Build points for upsert (with text hash for tracking)
        points = []
        for i, (doc, tune_run) in enumerate(trial_docs):
            if i in embed_result.failed_indices:
                stats.failed_tune_run_ids.append(str(doc.tune_run_id))
                stats.total_failed += 1
                continue

            vector = embed_result.vectors[i]
            if not vector:
                stats.failed_tune_run_ids.append(str(doc.tune_run_id))
                stats.total_failed += 1
                continue

            payload = trial_to_metadata(
                doc,
                embedding_model_id=model_id,
                vector_dim=vector_dim,
            )

            points.append(
                {
                    "id": str(doc.tune_run_id),
                    "vector": vector,
                    "payload": payload,
                    "text_hash": text_hashes[i],  # For DB tracking
                }
            )

        # Upsert to Qdrant (if not dry run)
        if not dry_run and points and self._repository:
            try:
                count = await self._repository.upsert_batch(points)
                stats.total_upserted = count
            except Exception as e:
                logger.error("Batch upsert failed", error=str(e))
                stats.upsert_failures = len(points)
                stats.total_failed += len(points)
                for point in points:
                    stats.failed_tune_run_ids.append(point["id"])
        elif dry_run:
            stats.total_upserted = len(points)  # Would have upserted

        # Mark as ingested in DB (if not dry run)
        if not dry_run and self._tune_repo:
            for point in points:
                try:
                    await self._mark_ingested(
                        tune_run_id=UUID(point["id"]),
                        model_id=model_id,
                        vector_dim=vector_dim,
                        text_hash=point["text_hash"],
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to mark tune_run as ingested",
                        tune_run_id=point["id"],
                        error=str(e),
                    )

        return stats

    async def _mark_ingested(
        self,
        tune_run_id: UUID,
        model_id: str,
        vector_dim: int,
        text_hash: Optional[str] = None,
    ) -> None:
        """
        Mark a tune_run as ingested in the database.

        Args:
            tune_run_id: Tune run ID
            model_id: Embedding model used
            vector_dim: Vector dimension
            text_hash: SHA256 hash of trial text (for drift detection)
        """
        if self._tune_repo is None:
            return

        try:
            await self._tune_repo.mark_kb_ingested(
                tune_run_id=tune_run_id,
                kb_ingested_at=utc_now_iso(),
                kb_embedding_model_id=model_id,
                kb_vector_dim=vector_dim,
                kb_text_hash=text_hash,
            )
        except AttributeError:
            logger.debug("mark_kb_ingested not implemented in tune repository")

    async def ingest_single(
        self,
        tune_run: dict,
        tune: dict,
        workspace_id: UUID,
        dry_run: bool = False,
    ) -> bool:
        """
        Ingest a single tune_run.

        Args:
            tune_run: Tune run dict
            tune: Parent tune dict
            workspace_id: Workspace ID
            dry_run: If True, don't actually upsert

        Returns:
            True if successful
        """
        model_id = self.embedder.model_id
        vector_dim = await self.embedder.get_vector_dim()

        doc = build_trial_doc_from_tune_run(
            tune_run=tune_run,
            tune=tune,
            workspace_id=workspace_id,
        )

        if doc is None:
            logger.debug("Skipping tune_run (not completed)")
            return False

        text = trial_to_text(doc)

        try:
            vector = await self.embedder.embed_single(text)
        except EmbeddingError as e:
            logger.error("Failed to embed trial", error=e.message)
            return False

        payload = trial_to_metadata(
            doc,
            embedding_model_id=model_id,
            vector_dim=vector_dim,
        )

        if not dry_run and self._repository:
            await self._repository.upsert_trial(
                point_id=doc.tune_run_id,
                vector=vector,
                payload=payload,
            )

            if self._tune_repo:
                await self._mark_ingested(
                    tune_run_id=doc.tune_run_id,
                    model_id=model_id,
                    vector_dim=vector_dim,
                )

        logger.debug(
            "Ingested trial",
            tune_run_id=str(doc.tune_run_id),
            strategy=doc.strategy_name,
        )

        return True


# Module-level pipeline instance
_pipeline: Optional[KBIngestionPipeline] = None


def get_ingestion_pipeline() -> KBIngestionPipeline:
    """Get or create ingestion pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = KBIngestionPipeline()
    return _pipeline
