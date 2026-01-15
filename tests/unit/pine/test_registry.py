"""
Unit tests for Pine Script registry builder.

Tests the orchestration layer that produces JSON artifacts.
Focused on the core build → write → load flow.
"""

import hashlib

from app.services.pine.constants import (
    LINT_REPORT_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
)
from app.services.pine.models import LintSeverity, PineVersion, ScriptType
from app.services.pine.registry import (
    LINT_E_INTERNAL_PARSE,
    RegistryConfig,
    build_and_write_registry,
    build_registry,
    load_lint_report,
    load_registry,
)


class TestBuildRegistry:
    """Core build tests."""

    def test_build_writes_both_files(self, tmp_path):
        """build_and_write_registry creates both JSON artifacts."""
        (tmp_path / "test.pine").write_text("//@version=5\nindicator('Test')")

        output_dir = tmp_path / "output"
        result = build_and_write_registry(tmp_path, output_dir=output_dir)

        # Both files exist
        assert result.registry_path is not None
        assert result.lint_report_path is not None
        assert result.registry_path.exists()
        assert result.lint_report_path.exists()

        # Correct filenames
        assert result.registry_path.name == "pine_registry.json"
        assert result.lint_report_path.name == "pine_lint_report.json"

        # Files have content
        assert result.registry_path.stat().st_size > 0
        assert result.lint_report_path.stat().st_size > 0

    def test_build_is_deterministic_ordering(self, tmp_path):
        """Scripts are ordered deterministically by rel_path."""
        # Create in reverse order
        (tmp_path / "zebra.pine").write_text("//@version=5\nindicator('Z')")
        (tmp_path / "alpha.pine").write_text("//@version=5\nindicator('A')")
        (tmp_path / "middle.pine").write_text("//@version=5\nindicator('M')")

        output_dir = tmp_path / "output"
        result = build_and_write_registry(tmp_path, output_dir=output_dir)

        # Registry keys are sorted
        keys = list(result.registry.scripts.keys())
        assert keys == sorted(keys)
        assert keys == ["alpha.pine", "middle.pine", "zebra.pine"]

        # Lint report keys are sorted
        lint_keys = list(result.lint_report.results.keys())
        assert lint_keys == sorted(lint_keys)

        # JSON file has sorted order (check raw content)
        with open(result.registry_path) as f:
            content = f.read()
        alpha_pos = content.find('"alpha.pine"')
        zebra_pos = content.find('"zebra.pine"')
        assert alpha_pos < zebra_pos

    def test_build_includes_sha256_and_mtime(self, tmp_path):
        """Entries include SHA256 fingerprint and mtime."""
        content = "//@version=5\nindicator('Test')\nplot(close)\n"
        pine_file = tmp_path / "test.pine"
        pine_file.write_text(content)

        result = build_registry(tmp_path)

        entry = result.registry.scripts["test.pine"]

        # SHA256 is correct
        expected_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert entry.sha256 == expected_sha
        assert len(entry.sha256) == 64

        # mtime is set
        assert entry.source_mtime is not None
        assert entry.source_mtime.tzinfo is not None  # UTC

        # parsed_at is set
        assert entry.parsed_at is not None

    def test_build_includes_lint_summary_counts(self, tmp_path):
        """Registry entries include lint summary with counts."""
        # File with lint errors (missing version and declaration)
        (tmp_path / "bad.pine").write_text("plot(close)")
        # Valid file
        (tmp_path / "good.pine").write_text("//@version=5\nindicator('Good')")

        result = build_registry(tmp_path)

        # Bad file has errors
        bad_entry = result.registry.scripts["bad.pine"]
        assert bad_entry.lint.error_count >= 2  # E001 + E003
        assert bad_entry.lint.has_errors is True

        # Good file has no errors
        good_entry = result.registry.scripts["good.pine"]
        assert good_entry.lint.error_count == 0
        assert good_entry.lint.has_errors is False

        # Totals aggregate
        assert result.registry.total_errors >= 2

    def test_lint_report_contains_findings(self, tmp_path):
        """Lint report contains full findings with codes and messages."""
        # File with lint issues
        (tmp_path / "test.pine").write_text("plot(close)")

        result = build_registry(tmp_path)

        lint_result = result.lint_report.results["test.pine"]

        # Has findings
        assert len(lint_result.findings) >= 2

        # Findings have required fields
        for finding in lint_result.findings:
            assert finding.severity in LintSeverity
            assert finding.code  # non-empty
            assert finding.message  # non-empty

        # Check specific error codes
        codes = {f.code for f in lint_result.findings}
        assert "E001" in codes  # Missing version
        assert "E003" in codes  # Missing declaration


class TestLoadRoundTrip:
    """Tests for JSON serialization round-trips."""

    def test_load_round_trip_registry(self, tmp_path):
        """Registry survives write → load round-trip."""
        (tmp_path / "test.pine").write_text(
            "//@version=5\nindicator('Test', overlay=true)"
        )

        output_dir = tmp_path / "output"
        result = build_and_write_registry(tmp_path, output_dir=output_dir)

        # Load from disk
        loaded = load_registry(result.registry_path)

        # Metadata preserved
        assert loaded.schema_version == REGISTRY_SCHEMA_VERSION
        assert loaded.root == str(tmp_path)
        assert loaded.root_kind == "filesystem"
        assert loaded.generated_at is not None

        # Entry preserved
        assert "test.pine" in loaded.scripts
        entry = loaded.scripts["test.pine"]
        assert entry.pine_version == PineVersion.V5
        assert entry.script_type == ScriptType.INDICATOR
        assert entry.title == "Test"
        assert entry.overlay is True
        assert entry.sha256 == result.registry.scripts["test.pine"].sha256

    def test_load_round_trip_lint_report(self, tmp_path):
        """Lint report survives write → load round-trip."""
        # File with lint errors
        (tmp_path / "test.pine").write_text("plot(close)")

        output_dir = tmp_path / "output"
        result = build_and_write_registry(tmp_path, output_dir=output_dir)

        # Load from disk
        loaded = load_lint_report(result.lint_report_path)

        # Metadata preserved
        assert loaded.schema_version == LINT_REPORT_SCHEMA_VERSION
        assert loaded.generated_at is not None

        # Findings preserved
        assert "test.pine" in loaded.results
        loaded_findings = loaded.results["test.pine"].findings
        original_findings = result.lint_report.results["test.pine"].findings

        assert len(loaded_findings) == len(original_findings)
        for orig, load in zip(original_findings, loaded_findings):
            assert load.severity == orig.severity
            assert load.code == orig.code
            assert load.message == orig.message


class TestParseErrorHandling:
    """Tests for graceful handling of parse errors."""

    def test_build_handles_parse_exception_as_internal_error(
        self, tmp_path, monkeypatch
    ):
        """Parse exceptions create E999 synthetic error, build continues."""
        (tmp_path / "good.pine").write_text("//@version=5\nindicator('Good')")
        (tmp_path / "bad.pine").write_text("//@version=5\nindicator('Bad')")

        # Monkeypatch parse_pine to fail for bad.pine
        original_parse = __import__(
            "app.services.pine.parser", fromlist=["parse_pine"]
        ).parse_pine

        def mock_parse(source):
            if "bad.pine" in source.rel_path:
                raise ValueError("Simulated parse explosion")
            return original_parse(source)

        monkeypatch.setattr(
            "app.services.pine.registry.parse_pine",
            mock_parse,
        )

        result = build_registry(tmp_path)

        # Build completed (best-effort)
        assert len(result.registry.scripts) == 2
        assert result.parse_errors == 1
        assert result.success is False

        # Good file parsed normally
        good_entry = result.registry.scripts["good.pine"]
        assert good_entry.pine_version == PineVersion.V5
        assert good_entry.script_type == ScriptType.INDICATOR

        # Bad file has fallback entry
        bad_entry = result.registry.scripts["bad.pine"]
        assert bad_entry.pine_version == PineVersion.UNKNOWN
        assert bad_entry.script_type == ScriptType.UNKNOWN
        assert bad_entry.lint.error_count >= 1

        # Lint report has synthetic E999 error
        bad_lint = result.lint_report.results["bad.pine"]
        error_codes = [f.code for f in bad_lint.findings]
        assert LINT_E_INTERNAL_PARSE in error_codes

        # Error message contains exception info
        e999_finding = next(
            f for f in bad_lint.findings if f.code == LINT_E_INTERNAL_PARSE
        )
        assert "Simulated parse explosion" in e999_finding.message


class TestRegistryBuildResultProperties:
    """Tests for RegistryBuildResult convenience properties."""

    def test_success_true_when_no_parse_errors(self, tmp_path):
        """success=True even with lint errors (parse succeeded)."""
        (tmp_path / "bad.pine").write_text("plot(close)")  # Lint errors

        result = build_registry(tmp_path)

        assert result.success is True
        assert result.parse_errors == 0
        assert result.total_errors > 0  # Has lint errors, that's OK

    def test_aggregates_totals(self, tmp_path):
        """total_errors/warnings aggregate across all scripts."""
        (tmp_path / "a.pine").write_text("plot(close)")  # E001 + E003
        (tmp_path / "b.pine").write_text("plot(open)")  # E001 + E003

        result = build_registry(tmp_path)

        # At least 4 errors total (2 per file)
        assert result.total_errors >= 4
        assert result.files_scanned == 2
        assert result.files_parsed == 2


class TestRegistryConfig:
    """Tests for RegistryConfig options."""

    def test_custom_extensions(self, tmp_path):
        """Custom extensions are respected."""
        (tmp_path / "a.pine").write_text("//@version=5\nindicator('A')")
        (tmp_path / "b.txt").write_text("//@version=5\nindicator('B')")

        # Default: only .pine
        result1 = build_registry(tmp_path)
        assert result1.files_scanned == 1

        # Custom: include .txt
        config = RegistryConfig(extensions=(".pine", ".txt"))
        result2 = build_registry(tmp_path, config=config)
        assert result2.files_scanned == 2

    def test_output_directory_created(self, tmp_path):
        """Output directory is created if missing."""
        (tmp_path / "test.pine").write_text("//@version=5\nindicator('Test')")

        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        result = build_and_write_registry(tmp_path, output_dir=output_dir)

        assert output_dir.exists()
        assert result.registry_path.parent == output_dir
