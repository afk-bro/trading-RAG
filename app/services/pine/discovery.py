"""Pine script discovery service.

Scans filesystem for .pine files, parses them, compares fingerprints,
and emits SSE events for changes.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import structlog

from app.services.pine.adapters.filesystem import SourceFile, scan_pine_files
from app.services.pine.discovery_repository import (
    ArchiveResult,
    StrategyScript,
    StrategyScriptRepository,
    UpsertResult,
)
from app.services.pine.parser import ScriptType, parse_pine
from app.services.pine.spec_generator import generate_strategy_spec

logger = structlog.get_logger(__name__)


@dataclass
class DiscoveryResult:
    """Result of a discovery run."""

    scripts_scanned: int = 0
    scripts_new: int = 0
    scripts_updated: int = 0
    scripts_unchanged: int = 0
    specs_generated: int = 0
    scripts_ingested: int = 0
    scripts_ingest_failed: int = 0
    chunks_created: int = 0
    scripts_archived: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ScriptChange:
    """Represents a change to a script."""

    script: StrategyScript
    change_type: Literal["new", "updated", "unchanged"]
    changed_fields: list[str] = field(default_factory=list)


def normalize_rel_path(path: str) -> str:
    """
    Normalize path to POSIX style.

    - Replace backslashes with forward slashes
    - Remove leading slashes
    - Remove . and .. components
    """
    normalized = path.replace("\\", "/").lstrip("/")
    parts = [p for p in normalized.split("/") if p not in (".", "..")]
    return "/".join(parts)


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PineDiscoveryService:
    """
    Service for discovering and tracking Pine scripts.

    Workflow:
    1. Scan filesystem for .pine files
    2. Parse each file
    3. Compare SHA256 against stored fingerprints
    4. Upsert new/changed scripts
    5. Generate specs for strategies
    6. Auto-ingest to KB (optional)
    7. Emit SSE events
    """

    def __init__(self, pool, settings, qdrant_client=None):
        """Initialize with connection pool, settings, and optional qdrant client."""
        self._pool = pool
        self._settings = settings
        self._qdrant_client = qdrant_client
        self._repo = StrategyScriptRepository(pool)

    async def discover(
        self,
        workspace_id: UUID,
        scan_paths: Optional[list[str]] = None,
        generate_specs: bool = True,
        auto_ingest: bool = True,
        emit_events: bool = True,
        dry_run: bool = False,
        discovery_run_id: Optional[str] = None,
        archive_stale_days: Optional[int] = None,
    ) -> DiscoveryResult:
        """
        Discover Pine scripts in the specified paths.

        Args:
            workspace_id: Workspace to associate scripts with
            scan_paths: Paths to scan (defaults to DATA_DIR/pine)
            generate_specs: Whether to generate specs for strategies
            auto_ingest: Whether to auto-ingest new/changed scripts to KB
            emit_events: Whether to emit SSE events
            dry_run: If True, don't persist or emit events
            discovery_run_id: Optional ID for log correlation
            archive_stale_days: If set, archive scripts not seen in N days

        Returns:
            DiscoveryResult with counts and errors
        """
        run_id = discovery_run_id or f"disc-{uuid4().hex[:8]}"
        log = logger.bind(
            discovery_run_id=run_id,
            workspace_id=str(workspace_id),
            dry_run=dry_run,
        )

        # Default scan paths
        if not scan_paths:
            default_path = Path(self._settings.data_dir) / "pine"
            scan_paths = [str(default_path)]

        log.info("discovery_started", scan_paths=scan_paths)

        result = DiscoveryResult()

        # Phase 1: Scan and parse
        parsed_scripts: list[tuple[SourceFile, StrategyScript]] = []
        for scan_path in scan_paths:
            try:
                scripts = self._scan_and_parse(scan_path, workspace_id, result, log)
                parsed_scripts.extend(scripts)
            except Exception as e:
                log.warning("scan_path_error", scan_path=scan_path, error=str(e))
                result.errors.append(f"Failed to scan {scan_path}: {e}")

        result.scripts_scanned = len(parsed_scripts)

        if dry_run:
            # Dry run: classify without persisting
            for source, script in parsed_scripts:
                existing = await self._repo.get_by_path(
                    workspace_id, script.source_type, script.rel_path
                )
                if existing is None:
                    result.scripts_new += 1
                elif existing.sha256 != script.sha256:
                    result.scripts_updated += 1
                else:
                    result.scripts_unchanged += 1

            log.info(
                "discovery_dry_run_complete",
                scanned=result.scripts_scanned,
                new=result.scripts_new,
                updated=result.scripts_updated,
                unchanged=result.scripts_unchanged,
            )
            return result

        # Phase 2: Reconcile and persist
        changes: list[ScriptChange] = []
        for source, script in parsed_scripts:
            try:
                upsert_result = await self._repo.upsert(script)
                change_type = self._classify_change(upsert_result)

                if change_type == "new":
                    result.scripts_new += 1
                elif change_type == "updated":
                    result.scripts_updated += 1
                else:
                    result.scripts_unchanged += 1

                changes.append(
                    ScriptChange(
                        script=upsert_result.script,
                        change_type=change_type,
                        changed_fields=upsert_result.changed_fields,
                    )
                )
            except Exception as e:
                log.warning(
                    "script_upsert_error",
                    rel_path=script.rel_path,
                    error=str(e),
                )
                result.errors.append(f"Failed to upsert {script.rel_path}: {e}")

        # Phase 2 events: discovered/updated (AFTER commit)
        if emit_events:
            await self._emit_discovery_events(changes, workspace_id, log)

        # Phase 3: Generate specs for strategies
        if generate_specs:
            specs_generated = await self._generate_specs(
                changes, workspace_id, log, emit_events
            )
            result.specs_generated = specs_generated

        # Phase 4: Auto-ingest new/changed scripts to KB
        if auto_ingest:
            ingest_result = await self._auto_ingest_scripts(
                changes,
                workspace_id,
                scan_paths[0] if scan_paths else None,
                log,
                emit_events,
            )
            result.scripts_ingested = ingest_result["ingested"]
            result.scripts_ingest_failed = ingest_result["failed"]
            result.chunks_created = ingest_result["chunks"]

        # Phase 5: Archive stale scripts (if enabled)
        if archive_stale_days is not None:
            archive_result = await self._archive_stale(
                workspace_id, archive_stale_days, log, emit_events
            )
            result.scripts_archived = archive_result.archived_count

        log.info(
            "discovery_complete",
            scanned=result.scripts_scanned,
            new=result.scripts_new,
            updated=result.scripts_updated,
            unchanged=result.scripts_unchanged,
            specs_generated=result.specs_generated,
            ingested=result.scripts_ingested,
            ingest_failed=result.scripts_ingest_failed,
            chunks=result.chunks_created,
            archived=result.scripts_archived,
            errors=len(result.errors),
        )

        return result

    def _scan_and_parse(
        self,
        scan_path: str,
        workspace_id: UUID,
        result: DiscoveryResult,
        log,
    ) -> list[tuple[SourceFile, StrategyScript]]:
        """Scan directory and parse all .pine files."""
        path = Path(scan_path)
        if not path.exists():
            log.warning("scan_path_not_found", scan_path=scan_path)
            result.errors.append(f"Path not found: {scan_path}")
            return []

        parsed: list[tuple[SourceFile, StrategyScript]] = []

        try:
            source_files = scan_pine_files(path)
        except Exception as e:
            log.warning("scan_error", scan_path=scan_path, error=str(e))
            result.errors.append(f"Scan error for {scan_path}: {e}")
            return []

        for source in source_files:
            try:
                script = self._parse_source(source, workspace_id)
                parsed.append((source, script))
            except Exception as e:
                log.warning(
                    "parse_error",
                    rel_path=source.rel_path,
                    error=str(e),
                )
                result.errors.append(f"Parse error for {source.rel_path}: {e}")

        return parsed

    def _parse_source(self, source: SourceFile, workspace_id: UUID) -> StrategyScript:
        """Parse a source file into a StrategyScript model."""
        # Parse the Pine script
        parse_result = parse_pine(source)

        # Compute SHA256
        sha256 = compute_sha256(source.content)

        # Normalize path
        rel_path = normalize_rel_path(source.rel_path)

        # Determine script type
        script_type = None
        if parse_result.script_type:
            script_type = parse_result.script_type.value

        # Determine Pine version
        pine_version = None
        if parse_result.pine_version:
            pine_version = str(parse_result.pine_version.value)

        return StrategyScript(
            id=uuid4(),
            workspace_id=workspace_id,
            rel_path=rel_path,
            source_type="local",
            sha256=sha256,
            pine_version=pine_version,
            script_type=script_type,
            title=parse_result.title,
            status="discovered",
        )

    def _classify_change(
        self, upsert_result: UpsertResult
    ) -> Literal["new", "updated", "unchanged"]:
        """Classify the type of change from upsert result."""
        if upsert_result.is_new:
            return "new"
        elif upsert_result.changed_fields:
            return "updated"
        else:
            return "unchanged"

    async def _emit_discovery_events(
        self,
        changes: list[ScriptChange],
        workspace_id: UUID,
        log,
    ) -> None:
        """Emit SSE events for discovery changes."""
        from app.services.events import get_event_bus
        from app.services.events.schemas import (
            pine_script_discovered,
            pine_script_updated,
        )

        bus = get_event_bus()

        for change in changes:
            if change.change_type == "unchanged":
                continue

            try:
                if change.change_type == "new":
                    event = pine_script_discovered(
                        event_id="",
                        workspace_id=workspace_id,
                        script_id=change.script.id,
                        rel_path=change.script.rel_path,
                        sha256=change.script.sha256,
                        script_type=change.script.script_type or "unknown",
                        title=change.script.title,
                        status=change.script.status,
                    )
                else:
                    event = pine_script_updated(
                        event_id="",
                        workspace_id=workspace_id,
                        script_id=change.script.id,
                        rel_path=change.script.rel_path,
                        sha256=change.script.sha256,
                        status=change.script.status,
                        changes=change.changed_fields,
                    )

                await bus.publish(event)
            except Exception as e:
                log.warning(
                    "event_emit_error",
                    script_id=str(change.script.id),
                    change_type=change.change_type,
                    error=str(e),
                )

    async def _generate_specs(
        self,
        changes: list[ScriptChange],
        workspace_id: UUID,
        log,
        emit_events: bool = True,
    ) -> int:
        """Generate specs for strategy scripts."""
        from app.services.events import get_event_bus
        from app.services.events.schemas import pine_script_spec_generated
        from app.services.pine.models import PineScriptEntry, PineVersion, LintSummary

        bus = get_event_bus() if emit_events else None
        count = 0

        for change in changes:
            script = change.script

            # Only generate specs for strategies
            if script.script_type != "strategy":
                continue

            # Only for new or updated scripts
            if change.change_type == "unchanged":
                continue

            try:
                # Create a minimal PineScriptEntry for spec generation
                entry = PineScriptEntry(
                    rel_path=script.rel_path,
                    sha256=script.sha256,
                    pine_version=PineVersion(script.pine_version or "5"),
                    script_type=ScriptType.STRATEGY,
                    title=script.title,
                    inputs=[],  # Will be populated from parse if needed
                    imports=[],
                    features={},
                    lint=LintSummary(error_count=0, warning_count=0, info_count=0),
                )

                # Generate spec
                spec = generate_strategy_spec(entry)
                spec_dict = {
                    "name": spec.name,
                    "source_path": spec.source_path,
                    "pine_version": spec.pine_version,
                    "description": spec.description,
                    "sha256": spec.sha256,
                    "params": [
                        {
                            "name": p.name,
                            "display_name": p.display_name,
                            "type": p.type,
                            "default": p.default,
                            "min_value": p.min_value,
                            "max_value": p.max_value,
                            "step": p.step,
                            "options": p.options,
                            "sweepable": p.sweepable,
                            "priority": p.priority,
                        }
                        for p in spec.params
                    ],
                    "sweep_config": spec.sweep_config,
                }

                # Update in DB
                updated = await self._repo.update_spec(
                    script.id, spec_dict, script.lint_json
                )

                if updated:
                    count += 1

                    # Emit spec_generated event
                    if bus:
                        sweepable_count = len(spec.sweepable_params)
                        event = pine_script_spec_generated(
                            event_id="",
                            workspace_id=workspace_id,
                            script_id=script.id,
                            rel_path=script.rel_path,
                            sweepable_count=sweepable_count,
                        )
                        await bus.publish(event)

            except Exception as e:
                log.warning(
                    "spec_generation_error",
                    script_id=str(script.id),
                    rel_path=script.rel_path,
                    error=str(e),
                )

        return count

    async def _auto_ingest_scripts(
        self,
        changes: list[ScriptChange],
        workspace_id: UUID,
        source_root: Optional[str],
        log,
        emit_events: bool = True,
    ) -> dict:
        """
        Auto-ingest new or changed scripts to the KB.

        Only ingests scripts where:
        - Script is new or updated (content changed)
        - OR ingest_status is None (never ingested)
        - OR sha256 != last_ingested_sha (content changed since last ingest)

        Returns:
            Dict with counts: {"ingested": int, "failed": int, "chunks": int}
        """
        from pathlib import Path

        from app.routers.ingest import ingest_pipeline, set_db_pool, set_qdrant_client
        from app.schemas import SourceType
        from app.services.events import get_event_bus
        from app.services.events.schemas import pine_script_ingested
        from app.services.pine.formatting import (
            build_canonical_url,
            build_ingest_doc_id,
            build_pine_metadata,
            strategy_script_to_entry,
        )
        from app.services.pine.ingest import format_script_content

        # Set up the ingest pipeline's globals
        set_db_pool(self._pool)
        if self._qdrant_client:
            set_qdrant_client(self._qdrant_client)

        result = {"ingested": 0, "failed": 0, "chunks": 0}
        bus = get_event_bus() if emit_events else None

        # Determine source root for reading files
        root = None
        if source_root:
            root = Path(source_root)
        elif self._settings and hasattr(self._settings, "data_dir"):
            root = Path(self._settings.data_dir) / "pine"

        for change in changes:
            script = change.script

            # Skip if unchanged and already ingested with same sha
            if change.change_type == "unchanged":
                # Check if needs re-ingest (sha changed since last ingest)
                if not script.needs_ingest():
                    continue

            try:
                # Read source content
                source_content: Optional[str] = None
                if root:
                    source_path = root / script.rel_path
                    if source_path.exists():
                        try:
                            source_content = source_path.read_text(encoding="utf-8")
                        except Exception as e:
                            log.warning(
                                "source_read_error",
                                rel_path=script.rel_path,
                                error=str(e),
                            )

                # Convert to PineScriptEntry for formatting
                entry = strategy_script_to_entry(script, source_content=source_content)

                # Format content for embedding
                content = format_script_content(
                    entry=entry,
                    source_content=source_content,
                    include_source=True,
                    max_source_lines=100,
                )

                # Build canonical URL and metadata
                canonical_url = build_canonical_url(script.source_type, script.rel_path)
                pine_metadata = build_pine_metadata(script, entry)

                # Use stable doc_id for replace-in-place
                idempotency_key = build_ingest_doc_id(
                    script.source_type, script.rel_path
                )

                # Call ingest pipeline
                response = await ingest_pipeline(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=SourceType.PINE_SCRIPT,
                    source_url=None,
                    canonical_url=canonical_url,
                    idempotency_key=idempotency_key,
                    content_hash=script.sha256,
                    title=script.title or script.rel_path,
                    language="pine",
                    settings=self._settings,
                    update_existing=True,  # Replace-in-place on content change
                    pine_metadata=pine_metadata,
                )

                # Update ingest tracking in DB
                await self._repo.update_ingest_status(
                    script_id=script.id,
                    doc_id=response.doc_id,
                    status="ok",
                    error=None,
                )

                result["ingested"] += 1
                result["chunks"] += response.chunks_created

                # Emit event on success
                if bus and response.status in ("created", "indexed"):
                    try:
                        event = pine_script_ingested(
                            event_id="",
                            workspace_id=workspace_id,
                            script_id=script.id,
                            doc_id=response.doc_id,
                            rel_path=script.rel_path,
                            content_sha=script.sha256,
                            chunks_created=response.chunks_created,
                        )
                        await bus.publish(event)
                    except Exception as e:
                        log.warning(
                            "ingest_event_emit_error",
                            script_id=str(script.id),
                            error=str(e),
                        )

                log.debug(
                    "script_ingested",
                    script_id=str(script.id),
                    rel_path=script.rel_path,
                    doc_id=str(response.doc_id),
                    chunks=response.chunks_created,
                )

            except Exception as e:
                result["failed"] += 1

                # Update ingest tracking with error
                error_msg = str(e)
                try:
                    await self._repo.update_ingest_status(
                        script_id=script.id,
                        doc_id=None,
                        status="error",
                        error=error_msg,
                    )
                except Exception:
                    pass  # Best effort

                log.warning(
                    "script_ingest_error",
                    script_id=str(script.id),
                    rel_path=script.rel_path,
                    error=error_msg,
                )

        if result["ingested"] > 0 or result["failed"] > 0:
            log.info(
                "auto_ingest_complete",
                ingested=result["ingested"],
                failed=result["failed"],
                chunks=result["chunks"],
            )

        return result

    async def _archive_stale(
        self,
        workspace_id: UUID,
        older_than_days: int,
        log,
        emit_events: bool = True,
    ) -> ArchiveResult:
        """Archive scripts not seen in N days and emit events."""
        from app.services.events import get_event_bus
        from app.services.events.schemas import pine_script_archived

        try:
            archive_result = await self._repo.mark_archived(
                workspace_id, older_than_days
            )

            if archive_result.archived_count > 0:
                log.info(
                    "scripts_archived",
                    count=archive_result.archived_count,
                    older_than_days=older_than_days,
                )

                # Emit archived events
                if emit_events:
                    bus = get_event_bus()
                    for script in archive_result.archived_scripts:
                        try:
                            last_seen_str = None
                            if script.last_seen_at:
                                last_seen_str = script.last_seen_at.isoformat()

                            event = pine_script_archived(
                                event_id="",
                                workspace_id=workspace_id,
                                script_id=script.id,
                                rel_path=script.rel_path,
                                last_seen_at=last_seen_str,
                            )
                            await bus.publish(event)
                        except Exception as e:
                            log.warning(
                                "archived_event_emit_error",
                                script_id=str(script.id),
                                error=str(e),
                            )

            return archive_result

        except Exception as e:
            log.warning(
                "archive_stale_error",
                older_than_days=older_than_days,
                error=str(e),
            )
            return ArchiveResult(archived_count=0)

    # =========================================================================
    # GitHub Repository Discovery
    # =========================================================================

    async def discover_repo(
        self,
        workspace_id: UUID,
        repo_id: UUID,
        trigger: Literal["manual", "poll", "webhook"] = "manual",
        force_full_scan: bool = False,
        generate_specs: bool = True,
        auto_ingest: bool = True,
        emit_events: bool = True,
    ) -> "RepoDiscoveryResult":
        """
        Discover Pine scripts from a GitHub repository.

        Workflow:
        1. Load repo config from DB
        2. Ensure clone exists (clone if first time)
        3. git fetch origin <branch>
        4. Compute changes (diff old..new or full scan)
        5. For each changed file:
           - A/M: Read via git show, parse, compute sha256, upsert
           - D: Mark script as deleted
        6. Update repo.last_seen_commit
        7. Generate specs, auto-ingest, emit events

        Args:
            workspace_id: Workspace owning the repo
            repo_id: Repository ID to scan
            trigger: What triggered this discovery (manual, poll, webhook)
            force_full_scan: Force full scan even if we have last_seen_commit
            generate_specs: Generate specs for strategies
            auto_ingest: Auto-ingest to KB
            emit_events: Emit SSE events

        Returns:
            RepoDiscoveryResult with counts and status
        """
        from pathlib import Path

        from app.services.pine.adapters.git import (
            BranchNotFoundError,
            GitAdapter,
            GitRepo,
            build_github_blob_url,
        )
        from app.services.pine.repo_registry import PineRepoRepository

        run_id = f"repo-{uuid4().hex[:8]}"
        log = logger.bind(
            discovery_run_id=run_id,
            workspace_id=str(workspace_id),
            repo_id=str(repo_id),
            trigger=trigger,
        )

        # Initialize result
        result = RepoDiscoveryResult()

        # Load repo config
        repo_registry = PineRepoRepository(self._pool)
        repo = await repo_registry.get(repo_id)

        if repo is None:
            result.status = "error"
            result.errors.append(f"Repository not found: {repo_id}")
            return result

        if repo.workspace_id != workspace_id:
            result.status = "error"
            result.errors.append("Repository does not belong to workspace")
            return result

        if not repo.enabled:
            result.status = "error"
            result.errors.append("Repository is disabled")
            return result

        log = log.bind(repo_slug=repo.repo_slug, branch=repo.branch)
        log.info("repo_discovery_started")

        # Initialize git adapter
        data_dir = Path(self._settings.data_dir) if self._settings else Path("/data")
        git_adapter = GitAdapter(data_dir)

        try:
            # Step 1: Ensure clone exists
            clone_path = await git_adapter.ensure_clone(repo.repo_url, repo.repo_slug)
            await repo_registry.update_clone_path(repo_id, str(clone_path))

            # Step 2: Fetch and diff
            git_repo = GitRepo(
                repo_id=repo.id,
                repo_slug=repo.repo_slug,
                clone_path=clone_path,
                branch=repo.branch,
                last_seen_commit=None if force_full_scan else repo.last_seen_commit,
                scan_globs=repo.scan_globs,
            )

            try:
                scan_result = await git_adapter.fetch_and_diff(git_repo)
                await repo_registry.update_pull_result(repo_id, ok=True, error=None)
            except BranchNotFoundError as e:
                error_msg = str(e)
                await repo_registry.update_pull_result(
                    repo_id, ok=False, error=error_msg
                )
                await repo_registry.update_scan_result(
                    repo_id,
                    commit="",
                    ok=False,
                    error=error_msg,
                    scripts_count=repo.scripts_count,
                )
                result.status = "error"
                result.errors.append(error_msg)
                log.error("repo_branch_not_found", branch=repo.branch)
                return result

            result.commit_before = repo.last_seen_commit
            result.commit_after = scan_result.current_commit
            result.is_full_scan = scan_result.is_full_scan

            if not scan_result.changes:
                # No changes
                await repo_registry.update_scan_result(
                    repo_id,
                    commit=scan_result.current_commit,
                    ok=True,
                    error=None,
                    scripts_count=repo.scripts_count,
                )
                result.status = "success"
                log.info("repo_discovery_no_changes")
                return result

            # Step 3: Process each change
            changes: list[ScriptChange] = []

            for change in scan_result.changes:
                result.scripts_scanned += 1

                try:
                    if change.status == "D":
                        # Deletion: find and mark script as deleted
                        existing = await self._repo.get_by_path(
                            workspace_id, "github", change.path
                        )
                        if existing:
                            await self._repo.mark_deleted(
                                existing.id,
                                scan_commit=scan_result.current_commit,
                            )
                            result.scripts_deleted += 1
                            log.debug("script_deleted", rel_path=change.path)
                    else:
                        # Add or Modify: read, parse, upsert
                        content = await git_adapter.read_file_at(
                            clone_path,
                            scan_result.current_commit,
                            change.path,
                        )

                        # Parse and create script model
                        script = self._parse_github_source(
                            content=content,
                            rel_path=change.path,
                            workspace_id=workspace_id,
                            repo_slug=repo.repo_slug,
                        )

                        # Build source URL
                        source_url = build_github_blob_url(
                            repo.repo_slug,
                            scan_result.current_commit,
                            change.path,
                        )

                        # Upsert with GitHub tracking
                        upsert_result = await self._repo.upsert_github_script(
                            script=script,
                            repo_id=repo_id,
                            scan_commit=scan_result.current_commit,
                            source_url=source_url,
                        )

                        change_type = self._classify_change(upsert_result)

                        if change_type == "new":
                            result.scripts_new += 1
                        elif change_type == "updated":
                            result.scripts_updated += 1
                        else:
                            result.scripts_unchanged += 1

                        changes.append(
                            ScriptChange(
                                script=upsert_result.script,
                                change_type=change_type,
                                changed_fields=upsert_result.changed_fields,
                            )
                        )

                except Exception as e:
                    log.warning(
                        "script_process_error",
                        rel_path=change.path,
                        status=change.status,
                        error=str(e),
                    )
                    result.errors.append(f"Failed to process {change.path}: {e}")

            # Step 4: Emit discovery events
            if emit_events and changes:
                await self._emit_discovery_events(changes, workspace_id, log)

            # Step 5: Generate specs for strategies
            if generate_specs and changes:
                result.specs_generated = await self._generate_specs(
                    changes, workspace_id, log, emit_events
                )

            # Step 6: Auto-ingest
            if auto_ingest and changes:
                ingest_result = await self._auto_ingest_github_scripts(
                    changes=changes,
                    workspace_id=workspace_id,
                    repo_slug=repo.repo_slug,
                    clone_path=clone_path,
                    commit=scan_result.current_commit,
                    git_adapter=git_adapter,
                    log=log,
                    emit_events=emit_events,
                )
                result.scripts_ingested = ingest_result["ingested"]
                result.scripts_ingest_failed = ingest_result["failed"]
                result.chunks_created = ingest_result["chunks"]

            # Step 7: Update repo state
            scripts_count = await self._count_repo_scripts(repo_id)
            await repo_registry.update_scan_result(
                repo_id,
                commit=scan_result.current_commit,
                ok=len(result.errors) == 0,
                error=result.errors[0] if result.errors else None,
                scripts_count=scripts_count,
            )

            result.status = "success" if not result.errors else "partial"

            log.info(
                "repo_discovery_complete",
                scanned=result.scripts_scanned,
                new=result.scripts_new,
                updated=result.scripts_updated,
                deleted=result.scripts_deleted,
                unchanged=result.scripts_unchanged,
                specs_generated=result.specs_generated,
                ingested=result.scripts_ingested,
                errors=len(result.errors),
            )

        except Exception as e:
            error_msg = str(e)[:1000]
            result.status = "error"
            result.errors.append(error_msg)

            # Update repo state on failure
            try:
                await repo_registry.update_scan_result(
                    repo_id,
                    commit="",
                    ok=False,
                    error=error_msg,
                    scripts_count=repo.scripts_count,
                )
            except Exception:
                pass  # Best effort

            log.error("repo_discovery_error", error=error_msg)

        return result

    def _parse_github_source(
        self,
        content: str,
        rel_path: str,
        workspace_id: UUID,
        repo_slug: str,
    ) -> StrategyScript:
        """Parse content from GitHub into a StrategyScript model."""
        from app.services.pine.adapters.filesystem import SourceFile

        # Create a synthetic SourceFile
        source = SourceFile(
            rel_path=rel_path,
            content=content,
            abs_path=None,
            source_id=None,
            mtime=None,
        )

        # Parse the Pine script
        parse_result = parse_pine(source)

        # Compute SHA256
        sha256 = compute_sha256(content)

        # Normalize path
        normalized_path = normalize_rel_path(rel_path)

        # Determine script type
        script_type = None
        if parse_result.script_type:
            script_type = parse_result.script_type.value

        # Determine Pine version
        pine_version = None
        if parse_result.pine_version:
            pine_version = str(parse_result.pine_version.value)

        return StrategyScript(
            id=uuid4(),
            workspace_id=workspace_id,
            rel_path=normalized_path,
            source_type="github",
            sha256=sha256,
            pine_version=pine_version,
            script_type=script_type,
            title=parse_result.title,
            status="discovered",
        )

    async def _auto_ingest_github_scripts(
        self,
        changes: list[ScriptChange],
        workspace_id: UUID,
        repo_slug: str,
        clone_path: Path,
        commit: str,
        git_adapter,
        log,
        emit_events: bool = True,
    ) -> dict:
        """
        Auto-ingest GitHub scripts to the KB.

        Similar to _auto_ingest_scripts but reads content via git show.

        Returns:
            Dict with counts: {"ingested": int, "failed": int, "chunks": int}
        """
        from app.routers.ingest import ingest_pipeline, set_db_pool, set_qdrant_client
        from app.schemas import SourceType
        from app.services.events import get_event_bus
        from app.services.events.schemas import pine_script_ingested
        from app.services.pine.formatting import (
            build_canonical_url,
            build_ingest_doc_id,
            build_pine_metadata,
            strategy_script_to_entry,
        )
        from app.services.pine.ingest import format_script_content

        # Set up the ingest pipeline's globals
        set_db_pool(self._pool)
        if self._qdrant_client:
            set_qdrant_client(self._qdrant_client)

        result = {"ingested": 0, "failed": 0, "chunks": 0}
        bus = get_event_bus() if emit_events else None

        for change in changes:
            script = change.script

            # Skip if unchanged and already ingested with same sha
            if change.change_type == "unchanged":
                if not script.needs_ingest():
                    continue

            try:
                # Read source content via git show
                source_content = await git_adapter.read_file_at(
                    clone_path, commit, script.rel_path
                )

                # Convert to PineScriptEntry for formatting
                entry = strategy_script_to_entry(script, source_content=source_content)

                # Format content for embedding
                content = format_script_content(
                    entry=entry,
                    source_content=source_content,
                    include_source=True,
                    max_source_lines=100,
                )

                # Build canonical URL with repo_slug
                canonical_url = build_canonical_url(
                    script.source_type,
                    script.rel_path,
                    repo_slug=repo_slug,
                )
                pine_metadata = build_pine_metadata(script, entry)

                # Use stable doc_id for replace-in-place (includes repo_slug)
                idempotency_key = build_ingest_doc_id(
                    script.source_type,
                    script.rel_path,
                    repo_slug=repo_slug,
                )

                # Call ingest pipeline
                response = await ingest_pipeline(
                    workspace_id=workspace_id,
                    content=content,
                    source_type=SourceType.PINE_SCRIPT,
                    source_url=script.source_url,
                    canonical_url=canonical_url,
                    idempotency_key=idempotency_key,
                    content_hash=script.sha256,
                    title=script.title or script.rel_path,
                    language="pine",
                    settings=self._settings,
                    update_existing=True,
                    pine_metadata=pine_metadata,
                )

                # Update ingest tracking in DB
                await self._repo.update_ingest_status(
                    script_id=script.id,
                    doc_id=response.doc_id,
                    status="ok",
                    error=None,
                )

                result["ingested"] += 1
                result["chunks"] += response.chunks_created

                # Emit event on success
                if bus and response.status in ("created", "indexed"):
                    try:
                        event = pine_script_ingested(
                            event_id="",
                            workspace_id=workspace_id,
                            script_id=script.id,
                            doc_id=response.doc_id,
                            rel_path=script.rel_path,
                            content_sha=script.sha256,
                            chunks_created=response.chunks_created,
                        )
                        await bus.publish(event)
                    except Exception as e:
                        log.warning(
                            "ingest_event_emit_error",
                            script_id=str(script.id),
                            error=str(e),
                        )

                log.debug(
                    "github_script_ingested",
                    script_id=str(script.id),
                    rel_path=script.rel_path,
                    doc_id=str(response.doc_id),
                    chunks=response.chunks_created,
                )

            except Exception as e:
                result["failed"] += 1

                # Update ingest tracking with error
                error_msg = str(e)
                try:
                    await self._repo.update_ingest_status(
                        script_id=script.id,
                        doc_id=None,
                        status="error",
                        error=error_msg,
                    )
                except Exception:
                    pass  # Best effort

                log.warning(
                    "github_script_ingest_error",
                    script_id=str(script.id),
                    rel_path=script.rel_path,
                    error=error_msg,
                )

        if result["ingested"] > 0 or result["failed"] > 0:
            log.info(
                "github_auto_ingest_complete",
                ingested=result["ingested"],
                failed=result["failed"],
                chunks=result["chunks"],
            )

        return result

    async def _count_repo_scripts(self, repo_id: UUID) -> int:
        """Count active scripts for a repository."""
        scripts = await self._repo.get_by_repo(repo_id, include_deleted=False)
        return len(scripts)


@dataclass
class RepoDiscoveryResult:
    """Result of a GitHub repository discovery run."""

    status: Literal["success", "partial", "error"] = "success"
    scripts_scanned: int = 0
    scripts_new: int = 0
    scripts_updated: int = 0
    scripts_deleted: int = 0
    scripts_unchanged: int = 0
    specs_generated: int = 0
    scripts_ingested: int = 0
    scripts_ingest_failed: int = 0
    chunks_created: int = 0
    commit_before: Optional[str] = None
    commit_after: Optional[str] = None
    is_full_scan: bool = False
    errors: list[str] = field(default_factory=list)
