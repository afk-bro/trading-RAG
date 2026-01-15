"""
Pine Script Registry builder.

Orchestrates adapter → parser → linter to produce JSON artifacts.

Outputs:
- pine_registry.json: Script metadata with lint summaries
- pine_lint_report.json: Full lint findings per script

Design choices:
- Best-effort: Parser errors recorded as E_INTERNAL_PARSE, build continues
- Deterministic: Sorted keys, consistent JSON formatting
- Fingerprinted: SHA256 from raw content for change detection
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.services.pine.adapters.filesystem import scan_pine_files
from app.services.pine.constants import (
    LINTER_VERSION,
    PARSER_VERSION,
    PINE_EXTENSIONS,
    REGISTRY_SCHEMA_VERSION,
    LINT_REPORT_SCHEMA_VERSION,
)
from app.services.pine.linter import LinterConfig, LintResult, lint_pine
from app.services.pine.models import (
    LintFinding,
    LintSeverity,
    PineLintReport,
    PineRegistry,
    PineScriptEntry,
    PineVersion,
    ScriptLintResult,
    ScriptType,
    SourceFile,
    utc_now,
)
from app.services.pine.parser import ParseResult, parse_pine

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Synthetic lint code for internal parse errors
LINT_E_INTERNAL_PARSE = "E999"

# Default output filenames
DEFAULT_REGISTRY_FILENAME = "pine_registry.json"
DEFAULT_LINT_REPORT_FILENAME = "pine_lint_report.json"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RegistryConfig:
    """Configuration for registry builder."""

    # File scanning
    extensions: tuple[str, ...] = PINE_EXTENSIONS
    max_file_size: Optional[int] = 1 * 1024 * 1024  # 1MB

    # Output
    output_dir: Optional[Path] = None
    registry_filename: str = DEFAULT_REGISTRY_FILENAME
    lint_report_filename: str = DEFAULT_LINT_REPORT_FILENAME

    # JSON formatting
    json_indent: int = 2
    json_sort_keys: bool = True
    json_ensure_ascii: bool = False

    # Behavior
    fail_on_parse_error: bool = False  # If True, raise on parse errors


# =============================================================================
# Build Result
# =============================================================================


@dataclass
class RegistryBuildResult:
    """Result of building the registry."""

    registry: PineRegistry
    lint_report: PineLintReport

    # Build metadata
    files_scanned: int = 0
    files_parsed: int = 0
    parse_errors: int = 0

    # Output paths (if written)
    registry_path: Optional[Path] = None
    lint_report_path: Optional[Path] = None

    @property
    def total_errors(self) -> int:
        return self.registry.total_errors

    @property
    def total_warnings(self) -> int:
        return self.registry.total_warnings

    @property
    def success(self) -> bool:
        """Build succeeded (no parse errors, though lint errors are OK)."""
        return self.parse_errors == 0


# =============================================================================
# Internal: Process Single File
# =============================================================================


@dataclass
class _FileProcessResult:
    """Result of processing a single file."""

    entry: PineScriptEntry
    lint_result: LintResult
    parse_error: Optional[str] = None


def _process_file(
    source: SourceFile,
    linter_config: Optional[LinterConfig],
) -> _FileProcessResult:
    """
    Process a single Pine Script file.

    Best-effort: If parsing fails, create entry with UNKNOWN values
    and synthetic E_INTERNAL_PARSE lint finding.
    """
    # Compute content hash
    sha256 = hashlib.sha256(source.content.encode("utf-8")).hexdigest()

    parse_error: Optional[str] = None
    parse_result: Optional[ParseResult] = None

    # Try to parse
    try:
        parse_result = parse_pine(source)
    except Exception as e:
        # Record error, continue with fallback
        parse_error = f"{type(e).__name__}: {e}"
        logger.warning(
            f"Parse error for {source.rel_path}: {parse_error}\n"
            f"{traceback.format_exc()}"
        )

    # Build entry
    if parse_result is not None:
        entry = PineScriptEntry(
            rel_path=source.rel_path,
            sha256=sha256,
            source_id=source.source_id,
            pine_version=parse_result.pine_version,
            script_type=parse_result.script_type,
            title=parse_result.title,
            short_title=parse_result.short_title,
            overlay=parse_result.overlay,
            imports=list(parse_result.imports),
            inputs=list(parse_result.inputs),
            features=dict(parse_result.features),
            parsed_at=utc_now(),
            source_mtime=source.mtime,
        )
    else:
        # Fallback entry for parse errors
        entry = PineScriptEntry(
            rel_path=source.rel_path,
            sha256=sha256,
            source_id=source.source_id,
            pine_version=PineVersion.UNKNOWN,
            script_type=ScriptType.UNKNOWN,
            parsed_at=utc_now(),
            source_mtime=source.mtime,
        )

    # Lint the file
    if parse_result is not None:
        lint_result = lint_pine(parse_result, source.content, linter_config)
    else:
        # Create synthetic lint result for parse error
        lint_result = LintResult(
            findings=[
                LintFinding(
                    severity=LintSeverity.ERROR,
                    code=LINT_E_INTERNAL_PARSE,
                    message=f"Internal parse error: {parse_error}",
                    line=None,
                )
            ]
        )

    # Update entry's lint summary
    entry.lint = lint_result.to_summary()

    return _FileProcessResult(
        entry=entry,
        lint_result=lint_result,
        parse_error=parse_error,
    )


# =============================================================================
# Main Builder
# =============================================================================


def build_registry(
    root: Path | str,
    config: Optional[RegistryConfig] = None,
    linter_config: Optional[LinterConfig] = None,
) -> RegistryBuildResult:
    """
    Build Pine Script registry from a directory.

    Args:
        root: Root directory to scan for .pine files
        config: Registry configuration
        linter_config: Linter configuration (passed through)

    Returns:
        RegistryBuildResult with registry, lint report, and metadata

    Raises:
        FileNotFoundError: If root doesn't exist
        NotADirectoryError: If root is not a directory
        FileTooLargeError: If a file exceeds max_file_size (fail-fast)
        FileReadError: If a file can't be read (fail-fast)
    """
    if config is None:
        config = RegistryConfig()

    root = Path(root).resolve()

    # Scan files
    logger.info(f"Scanning {root} for Pine Script files...")
    files = scan_pine_files(
        root,
        extensions=config.extensions,
        max_file_size=config.max_file_size,
    )
    logger.info(f"Found {len(files)} files")

    # Initialize artifacts
    now = utc_now()

    registry = PineRegistry(
        schema_version=REGISTRY_SCHEMA_VERSION,
        generated_at=now,
        root=str(root),
        root_kind="filesystem",
        parser_version=PARSER_VERSION,
        linter_version=LINTER_VERSION,
    )

    lint_report = PineLintReport(
        schema_version=LINT_REPORT_SCHEMA_VERSION,
        generated_at=now,
        linter_version=LINTER_VERSION,
    )

    # Process each file
    files_parsed = 0
    parse_errors = 0

    for source in files:
        result = _process_file(source, linter_config)

        # Add to registry
        registry.scripts[source.rel_path] = result.entry

        # Add to lint report
        lint_report.results[source.rel_path] = ScriptLintResult(
            rel_path=source.rel_path,
            sha256=result.entry.sha256,
            findings=list(result.lint_result.findings),
        )

        if result.parse_error:
            parse_errors += 1
            if config.fail_on_parse_error:
                raise RuntimeError(
                    f"Parse error in {source.rel_path}: {result.parse_error}"
                )
        else:
            files_parsed += 1

    logger.info(
        f"Processed {len(files)} files: "
        f"{files_parsed} parsed, {parse_errors} errors, "
        f"{registry.total_errors} lint errors, {registry.total_warnings} lint warnings"
    )

    return RegistryBuildResult(
        registry=registry,
        lint_report=lint_report,
        files_scanned=len(files),
        files_parsed=files_parsed,
        parse_errors=parse_errors,
    )


def build_and_write_registry(
    root: Path | str,
    output_dir: Optional[Path | str] = None,
    config: Optional[RegistryConfig] = None,
    linter_config: Optional[LinterConfig] = None,
) -> RegistryBuildResult:
    """
    Build registry and write JSON artifacts to disk.

    Args:
        root: Root directory to scan
        output_dir: Output directory (defaults to config.output_dir or root/data/)
        config: Registry configuration
        linter_config: Linter configuration

    Returns:
        RegistryBuildResult with output paths set
    """
    if config is None:
        config = RegistryConfig()

    # Determine output directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    elif config.output_dir is not None:
        out_dir = config.output_dir
    else:
        out_dir = Path(root) / "data"

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build registry
    result = build_registry(root, config, linter_config)

    # Write registry JSON
    registry_path = out_dir / config.registry_filename
    _write_json(
        registry_path,
        result.registry.to_dict(),
        indent=config.json_indent,
        sort_keys=config.json_sort_keys,
        ensure_ascii=config.json_ensure_ascii,
    )
    result.registry_path = registry_path
    logger.info(f"Wrote registry to {registry_path}")

    # Write lint report JSON
    lint_report_path = out_dir / config.lint_report_filename
    _write_json(
        lint_report_path,
        result.lint_report.to_dict(),
        indent=config.json_indent,
        sort_keys=config.json_sort_keys,
        ensure_ascii=config.json_ensure_ascii,
    )
    result.lint_report_path = lint_report_path
    logger.info(f"Wrote lint report to {lint_report_path}")

    return result


def _write_json(
    path: Path,
    data: dict,
    indent: int = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON with deterministic formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )
        f.write("\n")  # Trailing newline


# =============================================================================
# Convenience: Load Registry
# =============================================================================


def load_registry(path: Path | str) -> PineRegistry:
    """Load a PineRegistry from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PineRegistry.from_dict(data)


def load_lint_report(path: Path | str) -> PineLintReport:
    """Load a PineLintReport from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PineLintReport.from_dict(data)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """
    CLI entry point for Pine Script registry builder.

    Usage:
        python -m app.services.pine.registry --build ./scripts
        python -m app.services.pine.registry --build ./scripts --output ./data
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Pine Script Registry Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build",
        metavar="ROOT",
        help="Build registry from ROOT directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        help="Output directory (default: ROOT/data/)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pine"],
        help="File extensions to scan (default: .pine)",
    )
    parser.add_argument(
        "--fail-on-parse-error",
        action="store_true",
        help="Exit with error if any file fails to parse",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress info logging",
    )

    args = parser.parse_args()

    if not args.build:
        parser.print_help()
        return 1

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    root = Path(args.build)
    if not root.exists():
        print(f"Error: Directory not found: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Error: Not a directory: {root}", file=sys.stderr)
        return 1

    # Build config
    config = RegistryConfig(
        extensions=tuple(args.extensions),
        fail_on_parse_error=args.fail_on_parse_error,
    )

    output_dir = Path(args.output) if args.output else None

    try:
        result = build_and_write_registry(root, output_dir=output_dir, config=config)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(f"\nRegistry built: {result.registry_path}")
    print(f"Lint report: {result.lint_report_path}")
    print(f"Files scanned: {result.files_scanned}")
    print(f"Files parsed: {result.files_parsed}")
    if result.parse_errors:
        print(f"Parse errors: {result.parse_errors}")
    print(f"Lint errors: {result.total_errors}")
    print(f"Lint warnings: {result.total_warnings}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
