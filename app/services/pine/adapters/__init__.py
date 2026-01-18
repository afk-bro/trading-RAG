"""
Pine Script source adapters.

Adapters provide a uniform interface for reading Pine Script files
from different sources (filesystem, GitHub, etc.).

All adapters return list[SourceFile] for processing.
"""

from app.services.pine.adapters.filesystem import (
    FilesystemAdapterError,
    FileReadError,
    FileTooLargeError,
    scan_pine_files,
    scan_single_file,
)
from app.services.pine.adapters.git import (
    BranchNotFoundError,
    FileChange,
    GitAdapter,
    GitAdapterError,
    GitCommandError,
    GitRepo,
    GitScanResult,
    InvalidRepoSlugError,
    InvalidRepoUrlError,
    build_github_blob_url,
    extract_slug_from_url,
    validate_repo_slug,
)
from app.services.pine.models import SourceFile

__all__ = [
    # Core type
    "SourceFile",
    # Filesystem adapter
    "scan_pine_files",
    "scan_single_file",
    # Git adapter
    "GitAdapter",
    "GitRepo",
    "GitScanResult",
    "FileChange",
    "validate_repo_slug",
    "extract_slug_from_url",
    "build_github_blob_url",
    # Exceptions
    "FilesystemAdapterError",
    "FileTooLargeError",
    "FileReadError",
    "GitAdapterError",
    "GitCommandError",
    "InvalidRepoSlugError",
    "InvalidRepoUrlError",
    "BranchNotFoundError",
]
