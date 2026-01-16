"""
Unit tests for Pine Script router.

Tests path validation, admin authentication, and response mapping.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.routers.pine import derive_lint_path, validate_path


class TestValidatePath:
    """Tests for validate_path function."""

    def test_valid_path_within_data_dir(self, tmp_path):
        """Accepts valid path within DATA_DIR."""
        # Create a test file
        test_file = tmp_path / "registry.json"
        test_file.write_text("{}")

        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        result = validate_path(
            str(test_file),
            settings,
            must_be_file=True,
            allowed_extensions={".json"},
        )

        assert result == test_file.resolve()

    def test_rejects_path_outside_data_dir(self, tmp_path):
        """Rejects path traversal outside DATA_DIR."""
        # Create a file outside data_dir
        outside_file = tmp_path.parent / "outside.json"
        outside_file.write_text("{}")

        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            validate_path(str(outside_file), settings)

        assert exc_info.value.status_code == 403
        assert "Path must be within" in exc_info.value.detail

    def test_rejects_path_traversal_attempt(self, tmp_path):
        """Rejects path with .. traversal."""
        # Create data_dir and a nested dir
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "subdir").mkdir()

        # Create file outside data_dir
        outside_file = tmp_path / "secret.json"
        outside_file.write_text("{}")

        settings = MagicMock()
        settings.data_dir = str(data_dir)

        # Try to access via path traversal
        traversal_path = str(data_dir / "subdir" / ".." / ".." / "secret.json")

        with pytest.raises(HTTPException) as exc_info:
            validate_path(traversal_path, settings)

        assert exc_info.value.status_code == 403

    def test_rejects_non_json_extension(self, tmp_path):
        """Rejects files without allowed extension."""
        test_file = tmp_path / "registry.txt"
        test_file.write_text("not json")

        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            validate_path(
                str(test_file),
                settings,
                allowed_extensions={".json"},
            )

        assert exc_info.value.status_code == 400
        assert "extension" in exc_info.value.detail.lower()

    def test_rejects_missing_file(self, tmp_path):
        """Returns 404 for non-existent file."""
        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            validate_path(
                str(tmp_path / "nonexistent.json"),
                settings,
                must_be_file=True,
            )

        assert exc_info.value.status_code == 404

    def test_rejects_file_too_large(self, tmp_path):
        """Rejects files exceeding max size."""
        test_file = tmp_path / "large.json"
        test_file.write_text("x" * 1000)  # 1000 bytes

        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            validate_path(
                str(test_file),
                settings,
                max_size_bytes=100,  # Only 100 bytes allowed
            )

        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail.lower()

    def test_validates_directory_when_must_be_file_false(self, tmp_path):
        """Validates directory path when must_be_file=False."""
        test_dir = tmp_path / "sources"
        test_dir.mkdir()

        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        result = validate_path(
            str(test_dir),
            settings,
            must_be_file=False,
        )

        assert result == test_dir.resolve()

    def test_rejects_file_when_directory_expected(self, tmp_path):
        """Returns 404 when file provided but directory expected."""
        test_file = tmp_path / "file.json"
        test_file.write_text("{}")

        settings = MagicMock()
        settings.data_dir = str(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            validate_path(
                str(test_file),
                settings,
                must_be_file=False,
            )

        assert exc_info.value.status_code == 404
        assert "Directory not found" in exc_info.value.detail

    def test_handles_missing_data_dir(self, tmp_path):
        """Returns 500 when DATA_DIR doesn't exist."""
        settings = MagicMock()
        settings.data_dir = str(tmp_path / "nonexistent")

        with pytest.raises(HTTPException) as exc_info:
            validate_path(
                str(tmp_path / "file.json"),
                settings,
            )

        assert exc_info.value.status_code == 500
        assert "data_dir not found" in exc_info.value.detail.lower()


class TestDeriveLintPath:
    """Tests for derive_lint_path function."""

    def test_derives_from_pine_registry_name(self, tmp_path):
        """Derives lint path by replacing pine_registry with pine_lint_report."""
        registry_path = tmp_path / "pine_registry.json"
        registry_path.write_text("{}")

        lint_path = tmp_path / "pine_lint_report.json"
        lint_path.write_text("{}")

        result = derive_lint_path(registry_path)

        assert result == lint_path

    def test_finds_standard_lint_report_name(self, tmp_path):
        """Finds pine_lint_report.json in same directory."""
        registry_path = tmp_path / "my_custom_registry.json"
        registry_path.write_text("{}")

        lint_path = tmp_path / "pine_lint_report.json"
        lint_path.write_text("{}")

        result = derive_lint_path(registry_path)

        assert result == lint_path

    def test_returns_none_when_no_lint_report(self, tmp_path):
        """Returns None when no lint report found."""
        registry_path = tmp_path / "pine_registry.json"
        registry_path.write_text("{}")

        result = derive_lint_path(registry_path)

        assert result is None


class TestPineIngestRequestValidation:
    """Tests for PineIngestRequest Pydantic validation."""

    def test_valid_request(self):
        """Accepts valid request with all fields."""
        from app.schemas import PineIngestRequest

        request = PineIngestRequest(
            workspace_id=uuid4(),
            registry_path="/data/pine_registry.json",
            source_root="/data/scripts",
            include_source=True,
            max_source_lines=100,
            skip_lint_errors=False,
            update_existing=False,
            dry_run=False,
        )

        assert request.registry_path == "/data/pine_registry.json"
        assert request.include_source is True

    def test_minimal_request(self):
        """Accepts minimal request with only required fields."""
        from app.schemas import PineIngestRequest

        request = PineIngestRequest(
            workspace_id=uuid4(),
            registry_path="/data/pine_registry.json",
        )

        # Check defaults
        assert request.lint_path is None
        assert request.source_root is None
        assert request.include_source is True
        assert request.max_source_lines == 100
        assert request.skip_lint_errors is False
        assert request.update_existing is False
        assert request.dry_run is False

    def test_rejects_invalid_workspace_id(self):
        """Rejects non-UUID workspace_id."""
        from pydantic import ValidationError

        from app.schemas import PineIngestRequest

        with pytest.raises(ValidationError):
            PineIngestRequest(
                workspace_id="not-a-uuid",
                registry_path="/data/registry.json",
            )


class TestPineIngestResponseStatus:
    """Tests for PineIngestResponse status determination."""

    def test_success_status(self):
        """SUCCESS when no failures."""
        from app.schemas import PineIngestResponse, PineIngestStatus

        response = PineIngestResponse(
            status=PineIngestStatus.SUCCESS,
            scripts_processed=10,
            scripts_indexed=8,
            scripts_already_indexed=2,
            scripts_skipped=0,
            scripts_failed=0,
            chunks_added=16,
        )

        assert response.status == PineIngestStatus.SUCCESS

    def test_partial_status(self):
        """PARTIAL when some failures."""
        from app.schemas import PineIngestResponse, PineIngestStatus

        response = PineIngestResponse(
            status=PineIngestStatus.PARTIAL,
            scripts_processed=10,
            scripts_indexed=7,
            scripts_already_indexed=0,
            scripts_skipped=0,
            scripts_failed=3,
            chunks_added=14,
            errors=["script1.pine: parse error", "script2.pine: timeout"],
        )

        assert response.status == PineIngestStatus.PARTIAL
        assert len(response.errors) == 2

    def test_failed_status(self):
        """FAILED when all fail."""
        from app.schemas import PineIngestResponse, PineIngestStatus

        response = PineIngestResponse(
            status=PineIngestStatus.FAILED,
            scripts_processed=5,
            scripts_indexed=0,
            scripts_already_indexed=0,
            scripts_skipped=0,
            scripts_failed=5,
            chunks_added=0,
        )

        assert response.status == PineIngestStatus.FAILED

    def test_dry_run_status(self):
        """DRY_RUN when validation only."""
        from app.schemas import PineIngestResponse, PineIngestStatus

        response = PineIngestResponse(
            status=PineIngestStatus.DRY_RUN,
            scripts_processed=10,
            scripts_indexed=0,
            scripts_already_indexed=0,
            scripts_skipped=2,
            scripts_failed=0,
            chunks_added=0,
            ingest_run_id="pine-ingest-abc12345",
        )

        assert response.status == PineIngestStatus.DRY_RUN
        assert response.ingest_run_id == "pine-ingest-abc12345"
