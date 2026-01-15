"""
Filesystem adapter for Pine Script files.

Scans a directory tree and returns SourceFile objects for processing.
Designed for determinism and fail-fast behavior.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.services.pine.constants import PINE_EXTENSIONS
from app.services.pine.models import SourceFile


# =============================================================================
# Configuration
# =============================================================================

# Default max file size (1 MB) - Pine scripts should be small
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024

# Directories to always skip
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
    }
)


# =============================================================================
# Exceptions
# =============================================================================


class FilesystemAdapterError(Exception):
    """Base exception for filesystem adapter errors."""

    pass


class FileTooLargeError(FilesystemAdapterError):
    """Raised when a file exceeds the size limit."""

    def __init__(self, path: Path, size: int, max_size: int):
        self.path = path
        self.size = size
        self.max_size = max_size
        super().__init__(f"File too large: {path} ({size:,} bytes > {max_size:,} max)")


class FileReadError(FilesystemAdapterError):
    """Raised when a file cannot be read."""

    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot read file: {path} ({reason})")


# =============================================================================
# Core Function
# =============================================================================


def scan_pine_files(
    root: Path | str,
    *,
    extensions: tuple[str, ...] = PINE_EXTENSIONS,
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS,
    max_file_size: Optional[int] = DEFAULT_MAX_FILE_SIZE,
) -> list[SourceFile]:
    """
    Scan a directory tree for Pine Script files.

    Args:
        root: Root directory to scan
        extensions: File extensions to include (default: .pine, .pinescript)
        excluded_dirs: Directory names to skip (default: .git, node_modules, etc.)
        max_file_size: Maximum file size in bytes (None to disable)

    Returns:
        List of SourceFile objects, sorted by rel_path (POSIX-normalized)

    Raises:
        FileNotFoundError: If root doesn't exist
        NotADirectoryError: If root is not a directory
        FileTooLargeError: If a file exceeds max_file_size
        FileReadError: If a file cannot be read (permissions, encoding)
    """
    root = Path(root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    results: list[SourceFile] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories (modifying dirnames in-place)
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

        for filename in filenames:
            # Check extension
            if not filename.lower().endswith(extensions):
                continue

            abs_path = Path(dirpath) / filename

            # Check file size
            if max_file_size is not None:
                try:
                    size = abs_path.stat().st_size
                except OSError as e:
                    raise FileReadError(abs_path, str(e)) from e

                if size > max_file_size:
                    raise FileTooLargeError(abs_path, size, max_file_size)

            # Read content
            try:
                content = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise FileReadError(abs_path, f"encoding error: {e}") from e
            except OSError as e:
                raise FileReadError(abs_path, str(e)) from e

            # Get mtime as UTC datetime
            try:
                stat = abs_path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            except OSError:
                mtime = None

            # Compute relative path with POSIX separators
            rel_path = abs_path.relative_to(root).as_posix()

            results.append(
                SourceFile(
                    rel_path=rel_path,
                    content=content,
                    abs_path=str(abs_path),
                    source_id=None,  # Filesystem has no external ID
                    mtime=mtime,
                )
            )

    # Sort by rel_path for deterministic ordering
    results.sort(key=lambda sf: sf.rel_path)

    return results


def scan_single_file(
    file_path: Path | str,
    root: Optional[Path | str] = None,
    *,
    max_file_size: Optional[int] = DEFAULT_MAX_FILE_SIZE,
) -> SourceFile:
    """
    Read a single Pine Script file.

    Args:
        file_path: Path to the file
        root: Optional root for computing rel_path (defaults to file's parent)
        max_file_size: Maximum file size in bytes (None to disable)

    Returns:
        SourceFile object

    Raises:
        FileNotFoundError: If file doesn't exist
        FileTooLargeError: If file exceeds max_file_size
        FileReadError: If file cannot be read
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise FileReadError(file_path, "not a regular file")

    # Determine root for relative path
    if root is not None:
        root = Path(root).resolve()
    else:
        root = file_path.parent

    # Check file size
    if max_file_size is not None:
        try:
            size = file_path.stat().st_size
        except OSError as e:
            raise FileReadError(file_path, str(e)) from e

        if size > max_file_size:
            raise FileTooLargeError(file_path, size, max_file_size)

    # Read content
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise FileReadError(file_path, f"encoding error: {e}") from e
    except OSError as e:
        raise FileReadError(file_path, str(e)) from e

    # Get mtime
    try:
        stat = file_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    except OSError:
        mtime = None

    # Compute relative path
    try:
        rel_path = file_path.relative_to(root).as_posix()
    except ValueError:
        # file_path is not under root
        rel_path = file_path.name

    return SourceFile(
        rel_path=rel_path,
        content=content,
        abs_path=str(file_path),
        source_id=None,
        mtime=mtime,
    )
