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
    6. Emit SSE events
    """

    def __init__(self, pool, settings):
        """Initialize with connection pool and settings."""
        self._pool = pool
        self._settings = settings
        self._repo = StrategyScriptRepository(pool)

    async def discover(
        self,
        workspace_id: UUID,
        scan_paths: Optional[list[str]] = None,
        generate_specs: bool = True,
        dry_run: bool = False,
        discovery_run_id: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Discover Pine scripts in the specified paths.

        Args:
            workspace_id: Workspace to associate scripts with
            scan_paths: Paths to scan (defaults to DATA_DIR/pine)
            generate_specs: Whether to generate specs for strategies
            dry_run: If True, don't persist or emit events
            discovery_run_id: Optional ID for log correlation

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
        await self._emit_discovery_events(changes, workspace_id, log)

        # Phase 3: Generate specs for strategies
        if generate_specs:
            specs_generated = await self._generate_specs(changes, workspace_id, log)
            result.specs_generated = specs_generated

        log.info(
            "discovery_complete",
            scanned=result.scripts_scanned,
            new=result.scripts_new,
            updated=result.scripts_updated,
            unchanged=result.scripts_unchanged,
            specs_generated=result.specs_generated,
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

    def _classify_change(self, upsert_result: UpsertResult) -> str:
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
    ) -> int:
        """Generate specs for strategy scripts."""
        from app.services.events import get_event_bus
        from app.services.events.schemas import pine_script_spec_generated
        from app.services.pine.models import PineScriptEntry, PineVersion, LintSummary

        bus = get_event_bus()
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
