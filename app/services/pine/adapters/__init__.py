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
from app.services.pine.models import SourceFile

__all__ = [
    # Core type
    "SourceFile",
    # Filesystem adapter
    "scan_pine_files",
    "scan_single_file",
    # Exceptions
    "FilesystemAdapterError",
    "FileTooLargeError",
    "FileReadError",
]
