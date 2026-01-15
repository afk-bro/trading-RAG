"""
Pine Script ingestion service.

Ingests Pine Script files from a registry into the RAG system.

Usage:
    from app.services.pine.ingest import PineIngestService

    service = PineIngestService(db_pool, qdrant_client, settings)
    result = await service.ingest_from_registry(
        workspace_id=workspace_id,
        registry_path="data/pine_registry.json",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

from app.config import Settings
from app.schemas import SourceType
from app.services.pine.models import PineScriptEntry
from app.services.pine.registry import load_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ScriptIngestResult:
    """Result of ingesting a single script."""

    rel_path: str
    success: bool
    doc_id: Optional[UUID] = None
    chunks_created: int = 0
    status: str = "pending"  # indexed, exists, failed, skipped
    error: Optional[str] = None


@dataclass
class PineIngestResult:
    """Result of ingesting Pine scripts from registry."""

    scripts_processed: int = 0
    scripts_indexed: int = 0
    scripts_skipped: int = 0
    scripts_failed: int = 0
    total_chunks: int = 0
    results: list[ScriptIngestResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """All scripts processed without failures."""
        return self.scripts_failed == 0


# =============================================================================
# Content Formatting
# =============================================================================


def format_script_content(
    entry: PineScriptEntry,
    source_content: Optional[str] = None,
    include_source: bool = True,
    max_source_lines: int = 100,
) -> str:
    """
    Format a Pine Script entry for RAG embedding.

    Creates a structured text representation that includes:
    - Script metadata (title, type, version)
    - Input parameters with descriptions
    - Detected features
    - Lint summary
    - Optionally, the source code (truncated if large)
    """
    lines = []

    # Header
    title = entry.title or entry.rel_path
    lines.append(f"# {title}")
    lines.append("")

    # Metadata section
    lines.append("## Metadata")
    lines.append(f"- **Type**: {entry.script_type.value}")
    lines.append(f"- **Pine Version**: {entry.pine_version.value}")
    lines.append(f"- **File**: {entry.rel_path}")
    if entry.short_title:
        lines.append(f"- **Short Title**: {entry.short_title}")
    if entry.overlay is not None:
        lines.append(f"- **Overlay**: {entry.overlay}")
    lines.append("")

    # Inputs section
    if entry.inputs:
        lines.append("## Inputs")
        for inp in entry.inputs:
            default_str = ""
            if inp.default is not None:
                default_str = f" (default: {inp.default})"
            elif inp.default_expr:
                default_str = f" (default: {inp.default_expr})"

            type_str = inp.type.value if inp.type else "unknown"
            lines.append(f"- **{inp.name}** ({type_str}){default_str}")

            if inp.tooltip:
                lines.append(f"  - {inp.tooltip}")
            if inp.options:
                opts = ", ".join(str(o) for o in inp.options[:5])
                if len(inp.options) > 5:
                    opts += f"... (+{len(inp.options) - 5} more)"
                lines.append(f"  - Options: {opts}")
        lines.append("")

    # Imports section
    if entry.imports:
        lines.append("## Imports")
        for imp in entry.imports:
            alias_str = f" as {imp.alias}" if imp.alias else ""
            lines.append(f"- {imp.path}{alias_str}")
        lines.append("")

    # Features section
    active_features = [k for k, v in entry.features.items() if v]
    if active_features:
        lines.append("## Features")
        for feat in sorted(active_features):
            # Convert feature key to readable label
            label = (
                feat.replace("uses_", "Uses ").replace("is_", "Is ").replace("_", " ")
            )
            lines.append(f"- {label}")
        lines.append("")

    # Lint summary
    if entry.lint and (
        entry.lint.error_count > 0
        or entry.lint.warning_count > 0
        or entry.lint.info_count > 0
    ):
        lines.append("## Lint Summary")
        if entry.lint.error_count > 0:
            lines.append(f"- Errors: {entry.lint.error_count}")
        if entry.lint.warning_count > 0:
            lines.append(f"- Warnings: {entry.lint.warning_count}")
        if entry.lint.info_count > 0:
            lines.append(f"- Info: {entry.lint.info_count}")
        lines.append("")

    # Source code section
    if include_source and source_content:
        lines.append("## Source Code")
        lines.append("```pine")

        source_lines = source_content.splitlines()
        if len(source_lines) > max_source_lines:
            # Include first portion with truncation notice
            lines.extend(source_lines[:max_source_lines])
            lines.append(f"// ... ({len(source_lines) - max_source_lines} more lines)")
        else:
            lines.extend(source_lines)

        lines.append("```")

    return "\n".join(lines)


def extract_symbols_from_script(entry: PineScriptEntry, content: str) -> list[str]:
    """
    Extract potential ticker symbols from Pine Script.

    Looks for common patterns like:
    - syminfo.tickerid references
    - Explicit ticker strings in quotes
    - Common index/forex symbols
    """
    import re

    symbols = set()

    # Look for ticker strings in quotes (e.g., "AAPL", "BTCUSD")
    ticker_pattern = re.compile(r'"([A-Z]{1,5}(?:USD|USDT|BTC|ETH)?)"')
    for match in ticker_pattern.finditer(content):
        candidate = match.group(1)
        # Filter out common non-ticker strings
        if candidate not in {"TRUE", "FALSE", "NONE", "NULL", "NA"}:
            if len(candidate) >= 2:
                symbols.add(candidate)

    # Look for common index symbols
    index_pattern = re.compile(r"\b(SPX|SPY|QQQ|DJI|IXIC|VIX|DXY)\b")
    for match in index_pattern.finditer(content):
        symbols.add(match.group(1))

    return sorted(symbols)[:10]  # Limit to 10 symbols


# =============================================================================
# Ingest Service
# =============================================================================


class PineIngestService:
    """
    Service for ingesting Pine Script files into the RAG system.

    Reads a pine_registry.json, optionally reads source files,
    and ingests each script through the standard RAG pipeline.
    """

    def __init__(
        self,
        db_pool,
        qdrant_client,
        settings: Optional[Settings] = None,
    ):
        self._db_pool = db_pool
        self._qdrant_client = qdrant_client
        self._settings = settings

    async def ingest_from_registry(
        self,
        workspace_id: UUID,
        registry_path: Path | str,
        source_root: Optional[Path | str] = None,
        include_source: bool = True,
        max_source_lines: int = 100,
        skip_lint_errors: bool = False,
        update_existing: bool = False,
    ) -> PineIngestResult:
        """
        Ingest Pine scripts from a registry file.

        Args:
            workspace_id: Target workspace for ingestion
            registry_path: Path to pine_registry.json
            source_root: Root directory containing .pine files
                         (defaults to registry's root field)
            include_source: Include source code in embedded content
            max_source_lines: Max lines of source to include
            skip_lint_errors: Skip scripts with lint errors
            update_existing: Update existing documents (vs skip)

        Returns:
            PineIngestResult with per-script results
        """
        # Lazy import to avoid circular dependencies
        from app.routers.ingest import ingest_pipeline, set_db_pool, set_qdrant_client

        # Set up the ingest pipeline's globals
        set_db_pool(self._db_pool)
        set_qdrant_client(self._qdrant_client)

        # Load registry
        registry_path = Path(registry_path)
        logger.info(f"Loading registry from {registry_path}")
        registry = load_registry(registry_path)

        # Determine source root
        if source_root is not None:
            root = Path(source_root).resolve()
        elif registry.root is not None:
            root = Path(registry.root).resolve()
        else:
            # Fall back to registry file's parent directory
            root = registry_path.parent.resolve()

        logger.info(
            f"Ingesting {len(registry.scripts)} scripts from {root} "
            f"into workspace {workspace_id}"
        )

        result = PineIngestResult()
        result.results = []

        for rel_path, entry in registry.scripts.items():
            script_result = await self._ingest_script(
                workspace_id=workspace_id,
                entry=entry,
                root=root,
                include_source=include_source,
                max_source_lines=max_source_lines,
                skip_lint_errors=skip_lint_errors,
                update_existing=update_existing,
                ingest_fn=ingest_pipeline,
            )

            result.results.append(script_result)
            result.scripts_processed += 1

            if script_result.success:
                if script_result.status == "indexed":
                    result.scripts_indexed += 1
                    result.total_chunks += script_result.chunks_created
                elif script_result.status in ("exists", "skipped"):
                    result.scripts_skipped += 1
            else:
                result.scripts_failed += 1

        logger.info(
            f"Ingestion complete: {result.scripts_indexed} indexed, "
            f"{result.scripts_skipped} skipped, {result.scripts_failed} failed"
        )

        return result

    async def _ingest_script(
        self,
        workspace_id: UUID,
        entry: PineScriptEntry,
        root: Path,
        include_source: bool,
        max_source_lines: int,
        skip_lint_errors: bool,
        update_existing: bool,
        ingest_fn,
    ) -> ScriptIngestResult:
        """Ingest a single Pine Script."""
        rel_path = entry.rel_path

        # Skip if has lint errors and skip_lint_errors is True
        if skip_lint_errors and entry.lint and entry.lint.has_errors:
            logger.debug(f"Skipping {rel_path}: has lint errors")
            return ScriptIngestResult(
                rel_path=rel_path,
                success=True,
                status="skipped",
            )

        # Read source file if requested
        source_content: Optional[str] = None
        if include_source:
            source_path = root / rel_path
            if source_path.exists():
                try:
                    source_content = source_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to read source for {rel_path}: {e}")
            else:
                logger.debug(f"Source file not found: {source_path}")

        # Format content for embedding
        content = format_script_content(
            entry=entry,
            source_content=source_content,
            include_source=include_source,
            max_source_lines=max_source_lines,
        )

        # Build canonical URL from rel_path
        # Format: pine://{source_id or 'local'}/{rel_path}
        source_id = entry.source_id or "local"
        canonical_url = f"pine://{source_id}/{rel_path}"

        try:
            response = await ingest_fn(
                workspace_id=workspace_id,
                content=content,
                source_type=SourceType.PINE_SCRIPT,
                source_url=None,
                canonical_url=canonical_url,
                idempotency_key=entry.sha256,  # Use content hash as idempotency
                content_hash=entry.sha256,
                title=entry.title or entry.rel_path,
                author=None,  # Could be extracted from script comments
                published_at=entry.source_mtime,
                language="pine",
                settings=self._settings,
                update_existing=update_existing,
            )

            return ScriptIngestResult(
                rel_path=rel_path,
                success=True,
                doc_id=response.doc_id,
                chunks_created=response.chunks_created,
                status=response.status,
            )

        except Exception as e:
            logger.error(f"Failed to ingest {rel_path}: {e}")
            return ScriptIngestResult(
                rel_path=rel_path,
                success=False,
                status="failed",
                error=str(e),
            )


# =============================================================================
# Convenience Functions
# =============================================================================


async def ingest_pine_registry(
    db_pool,
    qdrant_client,
    workspace_id: UUID,
    registry_path: Path | str,
    source_root: Optional[Path | str] = None,
    settings: Optional[Settings] = None,
    **kwargs,
) -> PineIngestResult:
    """
    Convenience function to ingest a Pine Script registry.

    Args:
        db_pool: Database connection pool
        qdrant_client: Qdrant client
        workspace_id: Target workspace
        registry_path: Path to pine_registry.json
        source_root: Optional source directory override
        settings: Optional settings override
        **kwargs: Additional arguments passed to ingest_from_registry

    Returns:
        PineIngestResult
    """
    service = PineIngestService(db_pool, qdrant_client, settings)
    return await service.ingest_from_registry(
        workspace_id=workspace_id,
        registry_path=registry_path,
        source_root=source_root,
        **kwargs,
    )
