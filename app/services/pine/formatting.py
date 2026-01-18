"""
Shared formatting utilities for Pine Script documents.

Provides common functions for building document content and metadata
from either PineScriptEntry (registry) or StrategyScript (discovery).
"""

from pathlib import Path
from typing import Any, Optional

from app.services.pine.discovery_repository import StrategyScript
from app.services.pine.models import (
    LintFinding,
    LintSummary,
    PineScriptEntry,
    PineVersion,
    ScriptType,
)
from app.services.pine.parser import parse_pine
from app.services.pine.adapters.filesystem import SourceFile


def strategy_script_to_entry(
    script: StrategyScript,
    source_content: Optional[str] = None,
    source_root: Optional[Path] = None,
) -> PineScriptEntry:
    """
    Convert a StrategyScript (from discovery DB) to PineScriptEntry.

    If source_content is not provided and source_root is set,
    attempts to read and parse the source file.

    Args:
        script: StrategyScript from discovery database
        source_content: Optional raw source code
        source_root: Optional root directory for source files

    Returns:
        PineScriptEntry suitable for format_script_content()
    """
    # If we have source content, parse it to get full entry info
    if source_content:
        source_file = SourceFile(rel_path=script.rel_path, content=source_content)
        parse_result = parse_pine(source_file)

        return PineScriptEntry(
            rel_path=script.rel_path,
            sha256=script.sha256,
            pine_version=parse_result.pine_version
            or PineVersion(script.pine_version or "5"),
            script_type=parse_result.script_type
            or ScriptType(script.script_type or "strategy"),
            title=parse_result.title or script.title,
            short_title=parse_result.short_title,
            overlay=parse_result.overlay,
            inputs=parse_result.inputs,
            imports=parse_result.imports,
            features=parse_result.features,
            lint=LintSummary(
                error_count=(
                    script.lint_json.get("errors", 0) if script.lint_json else 0
                ),
                warning_count=(
                    script.lint_json.get("warnings", 0) if script.lint_json else 0
                ),
                info_count=script.lint_json.get("info", 0) if script.lint_json else 0,
            ),
        )

    # If we have source_root, try to read the file
    if source_root:
        source_path = source_root / script.rel_path
        if source_path.exists():
            try:
                content = source_path.read_text(encoding="utf-8")
                return strategy_script_to_entry(script, source_content=content)
            except Exception:
                pass  # Fall through to minimal entry

    # Minimal entry from DB fields only (no parse info)
    return PineScriptEntry(
        rel_path=script.rel_path,
        sha256=script.sha256,
        pine_version=PineVersion(script.pine_version or "5"),
        script_type=ScriptType(script.script_type or "strategy"),
        title=script.title,
        inputs=[],
        imports=[],
        features={},
        lint=LintSummary(
            error_count=script.lint_json.get("errors", 0) if script.lint_json else 0,
            warning_count=(
                script.lint_json.get("warnings", 0) if script.lint_json else 0
            ),
            info_count=script.lint_json.get("info", 0) if script.lint_json else 0,
        ),
    )


def build_canonical_url(
    source_type: str,
    rel_path: str,
    repo_slug: Optional[str] = None,
) -> str:
    """
    Build canonical URL for a Pine script.

    Format: pine://{source_type}/{rel_path}
    For github: pine://github/{repo_slug}/{rel_path}

    Args:
        source_type: Source type (local, github, etc.)
        rel_path: Relative path to script
        repo_slug: Repository identifier for github sources (e.g., "owner/repo").
                   Phase B3.1: Required for github sources to ensure uniqueness.

    Returns:
        Canonical URL string

    Examples:
        >>> build_canonical_url("local", "strategies/breakout.pine")
        'pine://local/strategies/breakout.pine'

        >>> build_canonical_url("github", "strategies/rsi.pine", "acme/trading-scripts")
        'pine://github/acme/trading-scripts/strategies/rsi.pine'
    """
    # Normalize path
    normalized = rel_path.replace("\\", "/").lstrip("/")
    normalized = "/".join(p for p in normalized.split("/") if p not in (".", ".."))

    # Phase B3.1: Include repo_slug for github sources
    if source_type == "github" and repo_slug:
        return f"pine://github/{repo_slug}/{normalized}"

    return f"pine://{source_type}/{normalized}"


def build_pine_metadata(
    script: StrategyScript,
    entry: Optional[PineScriptEntry] = None,
    lint_findings: Optional[list[LintFinding]] = None,
) -> dict[str, Any]:
    """
    Build pine_metadata JSONB for document storage.

    Args:
        script: StrategyScript from discovery
        entry: Optional PineScriptEntry with parsed info
        lint_findings: Optional lint findings list

    Returns:
        Dict suitable for pine_metadata column
    """
    # Build findings list (capped at 200)
    findings_list: Optional[list[dict]] = None
    if lint_findings:
        findings_list = [f.to_dict() for f in lint_findings[:200]]

    # Use entry if available, otherwise fall back to script fields
    if entry:
        return {
            "schema_version": "pine_meta_v1",
            "script_type": entry.script_type.value if entry.script_type else None,
            "pine_version": entry.pine_version.value if entry.pine_version else None,
            "rel_path": entry.rel_path,
            "inputs": [inp.to_dict() for inp in entry.inputs] if entry.inputs else [],
            "imports": (
                [imp.to_dict() for imp in entry.imports] if entry.imports else []
            ),
            "features": entry.features or {},
            "lint_summary": {
                "errors": entry.lint.error_count if entry.lint else 0,
                "warnings": entry.lint.warning_count if entry.lint else 0,
                "info": entry.lint.info_count if entry.lint else 0,
            },
            "lint_available": entry.lint is not None,
            "lint_findings": findings_list,
            # Discovery metadata
            "discovery_id": str(script.id),
            "content_sha": script.sha256,
        }

    # Minimal metadata from script only
    lint_json = script.lint_json or {}
    return {
        "schema_version": "pine_meta_v1",
        "script_type": script.script_type,
        "pine_version": script.pine_version,
        "rel_path": script.rel_path,
        "inputs": [],
        "imports": [],
        "features": {},
        "lint_summary": {
            "errors": lint_json.get("errors", 0),
            "warnings": lint_json.get("warnings", 0),
            "info": lint_json.get("info", 0),
        },
        "lint_available": bool(script.lint_json),
        "lint_findings": findings_list,
        # Discovery metadata
        "discovery_id": str(script.id),
        "content_sha": script.sha256,
    }


def build_ingest_doc_id(
    source_type: str,
    rel_path: str,
    repo_slug: Optional[str] = None,
) -> str:
    """
    Build a stable document ID for ingest.

    Uses source identity only (not content hash) so we replace-in-place
    when content changes, rather than creating multiple versions.

    Args:
        source_type: Source type (local, github, etc.)
        rel_path: Relative path to script
        repo_slug: Repository identifier for github sources (e.g., "owner/repo").
                   Phase B3.1: Required for github sources to avoid collisions
                   between scripts with same rel_path in different repos.

    Returns:
        Stable document ID string

    Examples:
        >>> build_ingest_doc_id("local", "strategies/breakout.pine")
        'pine:local:strategies/breakout.pine'

        >>> build_ingest_doc_id("github", "strategies/rsi.pine", "acme/trading-scripts")
        'pine:github:acme/trading-scripts:strategies/rsi.pine'
    """
    normalized = rel_path.replace("\\", "/").lstrip("/")
    normalized = "/".join(p for p in normalized.split("/") if p not in (".", ".."))

    # Phase B3.1: Include repo_slug for github sources to avoid doc_id collisions
    # between scripts with same rel_path in different repositories.
    if source_type == "github" and repo_slug:
        return f"pine:github:{repo_slug}:{normalized}"

    return f"pine:{source_type}:{normalized}"
