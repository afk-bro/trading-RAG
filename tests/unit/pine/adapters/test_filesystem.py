"""
Unit tests for Pine Script filesystem adapter.

Tests scan_pine_files() and scan_single_file() functions.
Uses pytest tmp_path fixture for isolated filesystem tests.
"""

import pytest
from datetime import timezone

from app.services.pine.adapters.filesystem import (
    scan_pine_files,
    scan_single_file,
    FileTooLargeError,
)


class TestScanPineFiles:
    """Tests for scan_pine_files()."""

    def test_scan_returns_sorted_rel_paths(self, tmp_path):
        """Results are sorted by rel_path alphabetically."""
        # Create files in reverse order
        (tmp_path / "zebra.pine").write_text("//@version=5")
        (tmp_path / "alpha.pine").write_text("//@version=5")
        (tmp_path / "middle.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 3
        assert [sf.rel_path for sf in results] == [
            "alpha.pine",
            "middle.pine",
            "zebra.pine",
        ]

    def test_scan_returns_nested_paths_sorted(self, tmp_path):
        """Nested paths are sorted correctly with POSIX separators."""
        (tmp_path / "strategies").mkdir()
        (tmp_path / "indicators").mkdir()
        (tmp_path / "strategies" / "breakout.pine").write_text("//@version=5")
        (tmp_path / "indicators" / "rsi.pine").write_text("//@version=5")
        (tmp_path / "main.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 3
        # POSIX paths, sorted alphabetically
        assert [sf.rel_path for sf in results] == [
            "indicators/rsi.pine",
            "main.pine",
            "strategies/breakout.pine",
        ]

    def test_scan_skips_excluded_dirs(self, tmp_path):
        """Excluded directories like .git are skipped."""
        (tmp_path / "valid.pine").write_text("//@version=5")

        # Create excluded dirs with .pine files inside
        for excluded in [".git", "node_modules", "__pycache__"]:
            excluded_dir = tmp_path / excluded
            excluded_dir.mkdir()
            (excluded_dir / "ignored.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 1
        assert results[0].rel_path == "valid.pine"

    def test_scan_reads_content(self, tmp_path):
        """File content is read correctly."""
        content = """//@version=5
indicator("Test", overlay=true)
plot(close)
"""
        (tmp_path / "test.pine").write_text(content)

        results = scan_pine_files(tmp_path)

        assert len(results) == 1
        assert results[0].content == content

    def test_scan_sets_mtime_utc(self, tmp_path):
        """mtime is set as UTC timezone-aware datetime."""
        (tmp_path / "test.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 1
        assert results[0].mtime is not None
        assert results[0].mtime.tzinfo == timezone.utc

    def test_scan_sets_abs_path(self, tmp_path):
        """abs_path is set to the full resolved path."""
        (tmp_path / "test.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 1
        assert results[0].abs_path is not None
        assert results[0].abs_path.endswith("test.pine")
        # Should be absolute
        assert results[0].abs_path.startswith("/")

    def test_scan_source_id_is_none(self, tmp_path):
        """source_id is None for filesystem adapter."""
        (tmp_path / "test.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 1
        assert results[0].source_id is None

    def test_scan_rejects_large_file(self, tmp_path):
        """Files exceeding max_file_size raise FileTooLargeError."""
        # Create a file larger than limit
        large_content = "x" * 1000
        (tmp_path / "large.pine").write_text(large_content)

        with pytest.raises(FileTooLargeError) as exc_info:
            scan_pine_files(tmp_path, max_file_size=100)

        assert exc_info.value.size == 1000
        assert exc_info.value.max_size == 100
        assert "large.pine" in str(exc_info.value.path)

    def test_scan_allows_large_file_when_disabled(self, tmp_path):
        """Large files are allowed when max_file_size=None."""
        large_content = "x" * 10000
        (tmp_path / "large.pine").write_text(large_content)

        results = scan_pine_files(tmp_path, max_file_size=None)

        assert len(results) == 1
        assert len(results[0].content) == 10000

    def test_scan_filters_by_extension(self, tmp_path):
        """Only files with matching extensions are included."""
        (tmp_path / "valid.pine").write_text("//@version=5")
        (tmp_path / "also_valid.pinescript").write_text("//@version=5")
        (tmp_path / "not_pine.txt").write_text("not pine")
        (tmp_path / "not_pine.py").write_text("# python")

        results = scan_pine_files(tmp_path)

        assert len(results) == 2
        rel_paths = {sf.rel_path for sf in results}
        assert rel_paths == {"valid.pine", "also_valid.pinescript"}

    def test_scan_custom_extensions(self, tmp_path):
        """Custom extensions can be specified."""
        (tmp_path / "custom.pine.txt").write_text("//@version=5")
        (tmp_path / "normal.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path, extensions=(".pine.txt",))

        assert len(results) == 1
        assert results[0].rel_path == "custom.pine.txt"

    def test_scan_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        results = scan_pine_files(tmp_path)
        assert results == []

    def test_scan_nonexistent_root_raises(self, tmp_path):
        """Nonexistent root raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            scan_pine_files(tmp_path / "nonexistent")

    def test_scan_file_as_root_raises(self, tmp_path):
        """File (not directory) as root raises NotADirectoryError."""
        file_path = tmp_path / "file.pine"
        file_path.write_text("//@version=5")

        with pytest.raises(NotADirectoryError):
            scan_pine_files(file_path)

    def test_scan_uses_posix_paths(self, tmp_path):
        """rel_path always uses forward slashes (POSIX style)."""
        nested = tmp_path / "deeply" / "nested" / "dir"
        nested.mkdir(parents=True)
        (nested / "script.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path)

        assert len(results) == 1
        assert "\\" not in results[0].rel_path
        assert results[0].rel_path == "deeply/nested/dir/script.pine"

    def test_scan_custom_excluded_dirs(self, tmp_path):
        """Custom excluded_dirs can be specified."""
        (tmp_path / "vendor").mkdir()
        (tmp_path / "vendor" / "lib.pine").write_text("//@version=5")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.pine").write_text("//@version=5")

        results = scan_pine_files(tmp_path, excluded_dirs=frozenset({"vendor"}))

        assert len(results) == 1
        assert results[0].rel_path == "src/main.pine"


class TestScanSingleFile:
    """Tests for scan_single_file()."""

    def test_reads_single_file(self, tmp_path):
        """Single file is read correctly."""
        content = "//@version=5\nplot(close)"
        file_path = tmp_path / "test.pine"
        file_path.write_text(content)

        result = scan_single_file(file_path)

        assert result.content == content
        assert result.rel_path == "test.pine"

    def test_respects_root_for_rel_path(self, tmp_path):
        """rel_path is computed relative to provided root."""
        nested = tmp_path / "strategies" / "breakout"
        nested.mkdir(parents=True)
        file_path = nested / "bb.pine"
        file_path.write_text("//@version=5")

        result = scan_single_file(file_path, root=tmp_path)

        assert result.rel_path == "strategies/breakout/bb.pine"

    def test_defaults_to_parent_for_rel_path(self, tmp_path):
        """Without root, rel_path is just filename."""
        file_path = tmp_path / "test.pine"
        file_path.write_text("//@version=5")

        result = scan_single_file(file_path)

        assert result.rel_path == "test.pine"

    def test_nonexistent_file_raises(self, tmp_path):
        """Nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            scan_single_file(tmp_path / "nonexistent.pine")

    def test_large_file_raises(self, tmp_path):
        """Large file raises FileTooLargeError."""
        file_path = tmp_path / "large.pine"
        file_path.write_text("x" * 1000)

        with pytest.raises(FileTooLargeError):
            scan_single_file(file_path, max_file_size=100)


class TestSourceFileImmutability:
    """Tests that SourceFile is properly frozen."""

    def test_source_file_is_frozen(self, tmp_path):
        """SourceFile cannot be modified after creation."""
        (tmp_path / "test.pine").write_text("//@version=5")
        results = scan_pine_files(tmp_path)
        sf = results[0]

        with pytest.raises(AttributeError):
            sf.rel_path = "modified.pine"  # type: ignore

        with pytest.raises(AttributeError):
            sf.content = "modified"  # type: ignore
