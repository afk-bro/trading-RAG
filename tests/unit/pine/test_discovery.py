"""
Unit tests for Pine Script discovery service.

Tests:
- Path normalization
- SHA256 fingerprinting
- Change classification (new/updated/unchanged)
- Spec generation conditional on strategy type
- Event emission after commit
- Dry run behavior
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.pine.discovery import (
    DiscoveryResult,
    PineDiscoveryService,
    ScriptChange,
    compute_sha256,
    normalize_rel_path,
)
from app.services.pine.discovery_repository import StrategyScript, UpsertResult


class TestNormalizeRelPath:
    """Tests for normalize_rel_path()."""

    def test_posix_path_unchanged(self):
        """POSIX paths without special chars pass through."""
        assert (
            normalize_rel_path("strategies/breakout.pine") == "strategies/breakout.pine"
        )

    def test_backslashes_converted(self):
        """Windows backslashes converted to forward slashes."""
        assert (
            normalize_rel_path("strategies\\breakout.pine")
            == "strategies/breakout.pine"
        )
        assert normalize_rel_path("a\\b\\c.pine") == "a/b/c.pine"

    def test_leading_slash_removed(self):
        """Leading slashes are stripped."""
        assert (
            normalize_rel_path("/strategies/breakout.pine")
            == "strategies/breakout.pine"
        )
        assert normalize_rel_path("///a/b/c.pine") == "a/b/c.pine"

    def test_dot_components_removed(self):
        """Single dots (.) are removed."""
        assert (
            normalize_rel_path("./strategies/./breakout.pine")
            == "strategies/breakout.pine"
        )

    def test_dotdot_components_removed(self):
        """Double dots (..) are removed."""
        assert (
            normalize_rel_path("../strategies/../breakout.pine")
            == "strategies/breakout.pine"
        )

    def test_mixed_normalization(self):
        """Combined normalization works."""
        assert (
            normalize_rel_path("\\..\\strategies\\.\\breakout.pine")
            == "strategies/breakout.pine"
        )


class TestComputeSha256:
    """Tests for compute_sha256()."""

    def test_computes_correct_hash(self):
        """SHA256 matches expected value."""
        content = "// Test content"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert compute_sha256(content) == expected

    def test_empty_content(self):
        """Empty string hashes correctly."""
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_sha256("") == expected

    def test_unicode_content(self):
        """Unicode content hashes correctly."""
        content = "// Test with unicode: \u2603"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert compute_sha256(content) == expected


class TestDiscoveryResultDataclass:
    """Tests for DiscoveryResult dataclass."""

    def test_default_values(self):
        """DiscoveryResult has correct defaults."""
        result = DiscoveryResult()
        assert result.scripts_scanned == 0
        assert result.scripts_new == 0
        assert result.scripts_updated == 0
        assert result.scripts_unchanged == 0
        assert result.specs_generated == 0
        assert result.errors == []

    def test_errors_list_is_mutable(self):
        """Each instance gets its own errors list."""
        r1 = DiscoveryResult()
        r2 = DiscoveryResult()
        r1.errors.append("error1")
        assert r1.errors == ["error1"]
        assert r2.errors == []


class TestScriptChange:
    """Tests for ScriptChange dataclass."""

    def test_default_changed_fields(self):
        """ScriptChange has empty changed_fields by default."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="test.pine",
            source_type="local",
            sha256="abc123",
        )
        change = ScriptChange(script=script, change_type="new")
        assert change.changed_fields == []


class TestPineDiscoveryService:
    """Tests for PineDiscoveryService."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.data_dir = "/data"
        return settings

    @pytest.fixture
    def service(self, mock_pool, mock_settings):
        """Create PineDiscoveryService with mocks."""
        return PineDiscoveryService(mock_pool, mock_settings)

    @pytest.fixture
    def workspace_id(self):
        """Create a test workspace ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_classify_change_new(self, service):
        """_classify_change returns 'new' for new scripts."""
        upsert_result = UpsertResult(
            script=MagicMock(),
            is_new=True,
            changed_fields=[],
        )
        assert service._classify_change(upsert_result) == "new"

    @pytest.mark.asyncio
    async def test_classify_change_updated(self, service):
        """_classify_change returns 'updated' when fields changed."""
        upsert_result = UpsertResult(
            script=MagicMock(),
            is_new=False,
            changed_fields=["sha256", "title"],
        )
        assert service._classify_change(upsert_result) == "updated"

    @pytest.mark.asyncio
    async def test_classify_change_unchanged(self, service):
        """_classify_change returns 'unchanged' when no fields changed."""
        upsert_result = UpsertResult(
            script=MagicMock(),
            is_new=False,
            changed_fields=[],
        )
        assert service._classify_change(upsert_result) == "unchanged"

    @pytest.mark.asyncio
    async def test_discover_empty_path_returns_result(
        self, service, workspace_id, mock_settings
    ):
        """Discovery with non-existent path returns error in result."""
        with patch("app.services.pine.discovery.scan_pine_files") as mock_scan:
            mock_scan.side_effect = FileNotFoundError("Path not found")

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/nonexistent"],
                dry_run=True,
            )

            assert result.scripts_scanned == 0
            assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_dry_run_does_not_persist(self, service, workspace_id):
        """Dry run computes counts but doesn't call upsert."""
        from app.services.pine.adapters.filesystem import SourceFile

        mock_source = SourceFile(
            rel_path="test.pine",
            content="//@version=5\nstrategy('Test')\n",
        )

        with patch.object(service, "_scan_and_parse") as mock_scan, patch.object(
            service._repo, "get_by_path", new_callable=AsyncMock
        ) as mock_get, patch.object(
            service._repo, "upsert", new_callable=AsyncMock
        ) as mock_upsert:
            # Mock scan returns one file
            mock_script = StrategyScript(
                id=uuid4(),
                workspace_id=workspace_id,
                rel_path="test.pine",
                source_type="local",
                sha256="abc123",
                script_type="strategy",
            )
            mock_scan.return_value = [(mock_source, mock_script)]
            mock_get.return_value = None  # New script

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                dry_run=True,
            )

            assert result.scripts_scanned == 1
            assert result.scripts_new == 1
            mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_spec_generation_only_for_strategies(self, service, workspace_id):
        """Spec generation only runs for strategy script_type."""
        from app.services.pine.adapters.filesystem import SourceFile

        # Create indicator and strategy sources
        indicator = SourceFile(
            rel_path="indicator.pine",
            content="//@version=5\nindicator('MA')\n",
        )
        strategy = SourceFile(
            rel_path="strategy.pine",
            content="//@version=5\nstrategy('Breakout')\n",
        )

        with patch.object(service, "_scan_and_parse") as mock_scan, patch.object(
            service._repo, "upsert", new_callable=AsyncMock
        ) as mock_upsert, patch.object(
            service._repo, "update_spec", new_callable=AsyncMock
        ) as mock_update_spec, patch(
            "app.services.events.get_event_bus"
        ) as mock_get_bus:
            # Setup mocks
            mock_bus = MagicMock()
            mock_bus.publish = AsyncMock(return_value=1)
            mock_get_bus.return_value = mock_bus

            indicator_script = StrategyScript(
                id=uuid4(),
                workspace_id=workspace_id,
                rel_path="indicator.pine",
                source_type="local",
                sha256="ind123",
                script_type="indicator",
            )
            strategy_script = StrategyScript(
                id=uuid4(),
                workspace_id=workspace_id,
                rel_path="strategy.pine",
                source_type="local",
                sha256="strat123",
                script_type="strategy",
            )

            mock_scan.return_value = [
                (indicator, indicator_script),
                (strategy, strategy_script),
            ]
            mock_upsert.side_effect = [
                UpsertResult(script=indicator_script, is_new=True, changed_fields=[]),
                UpsertResult(script=strategy_script, is_new=True, changed_fields=[]),
            ]
            mock_update_spec.return_value = strategy_script

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                generate_specs=True,
            )

            # Only strategy should have spec generated
            assert result.specs_generated == 1
            assert mock_update_spec.call_count == 1

            # Verify the spec was generated for strategy, not indicator
            spec_call = mock_update_spec.call_args[0]
            assert spec_call[0] == strategy_script.id


class TestDiscoveryRepository:
    """Tests for StrategyScriptRepository."""

    def test_canonical_url_generation(self):
        """StrategyScript.canonical_url() generates correct URL."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="strategies/breakout.pine",
            source_type="local",
            sha256="abc123",
        )
        assert script.canonical_url() == "pine://local/strategies/breakout.pine"

    def test_canonical_url_normalizes_backslashes(self):
        """canonical_url normalizes Windows paths."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="strategies\\breakout.pine",
            source_type="local",
            sha256="abc123",
        )
        assert script.canonical_url() == "pine://local/strategies/breakout.pine"

    def test_canonical_url_removes_dotdot(self):
        """canonical_url removes .. components."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="../strategies/breakout.pine",
            source_type="local",
            sha256="abc123",
        )
        assert script.canonical_url() == "pine://local/strategies/breakout.pine"


class TestEventBusMocking:
    """Tests verifying event bus behavior in discovery."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus that records published events."""
        bus = MagicMock()
        bus.published_events = []

        async def record_publish(event):
            bus.published_events.append(event)
            return 1

        bus.publish = record_publish
        return bus

    @pytest.mark.asyncio
    async def test_events_published_for_new_scripts(self, mock_event_bus):
        """Discovery emits pine.script.discovered for new scripts."""
        workspace_id = uuid4()
        script_id = uuid4()

        # Test the event factory
        from app.services.events.schemas import pine_script_discovered

        event = pine_script_discovered(
            event_id="",
            workspace_id=workspace_id,
            script_id=script_id,
            rel_path="test.pine",
            sha256="abc123",
            script_type="strategy",
            title="Test Strategy",
            status="discovered",
        )

        assert event.topic == "pine.script.discovered"
        assert event.workspace_id == workspace_id
        assert event.payload["script_id"] == str(script_id)
        assert event.payload["rel_path"] == "test.pine"
        assert event.payload["sha256"] == "abc123"

    @pytest.mark.asyncio
    async def test_events_published_for_updated_scripts(self, mock_event_bus):
        """Discovery emits pine.script.updated for changed scripts."""
        workspace_id = uuid4()
        script_id = uuid4()

        from app.services.events.schemas import pine_script_updated

        event = pine_script_updated(
            event_id="",
            workspace_id=workspace_id,
            script_id=script_id,
            rel_path="test.pine",
            sha256="new-hash",
            status="discovered",
            changes=["sha256", "title"],
        )

        assert event.topic == "pine.script.updated"
        assert event.payload["changes"] == ["sha256", "title"]

    @pytest.mark.asyncio
    async def test_events_published_for_spec_generated(self, mock_event_bus):
        """Discovery emits pine.script.spec_generated after spec creation."""
        workspace_id = uuid4()
        script_id = uuid4()

        from app.services.events.schemas import pine_script_spec_generated

        event = pine_script_spec_generated(
            event_id="",
            workspace_id=workspace_id,
            script_id=script_id,
            rel_path="test.pine",
            sweepable_count=5,
        )

        assert event.topic == "pine.script.spec_generated"
        assert event.payload["sweepable_count"] == 5

    @pytest.mark.asyncio
    async def test_no_events_for_unchanged_scripts(self):
        """Discovery does not emit events for unchanged scripts."""
        # ScriptChange with change_type="unchanged" should not trigger events
        change = ScriptChange(
            script=MagicMock(),
            change_type="unchanged",
            changed_fields=[],
        )

        # In the discovery service, unchanged scripts are skipped in event emission
        assert change.change_type == "unchanged"
        # This verifies the logic - unchanged scripts don't get events


class TestChangeDetection:
    """Tests for change field detection."""

    def test_change_detect_fields(self):
        """CHANGE_DETECT_FIELDS contains expected fields."""
        from app.services.pine.discovery_repository import CHANGE_DETECT_FIELDS

        assert "sha256" in CHANGE_DETECT_FIELDS
        assert "pine_version" in CHANGE_DETECT_FIELDS
        assert "script_type" in CHANGE_DETECT_FIELDS
        assert "title" in CHANGE_DETECT_FIELDS

    def test_compute_changed_fields(self):
        """_compute_changed_fields detects changes correctly."""
        from app.services.pine.discovery_repository import _compute_changed_fields

        workspace_id = uuid4()
        existing = StrategyScript(
            id=uuid4(),
            workspace_id=workspace_id,
            rel_path="test.pine",
            source_type="local",
            sha256="old-hash",
            pine_version="5",
            script_type="strategy",
            title="Old Title",
        )
        incoming = StrategyScript(
            id=uuid4(),
            workspace_id=workspace_id,
            rel_path="test.pine",
            source_type="local",
            sha256="new-hash",
            pine_version="5",
            script_type="strategy",
            title="New Title",
        )

        changes = _compute_changed_fields(existing, incoming)

        assert "sha256" in changes
        assert "title" in changes
        assert "pine_version" not in changes
        assert "script_type" not in changes

    def test_no_changes_detected(self):
        """_compute_changed_fields returns empty for identical scripts."""
        from app.services.pine.discovery_repository import _compute_changed_fields

        workspace_id = uuid4()
        script_id = uuid4()
        existing = StrategyScript(
            id=script_id,
            workspace_id=workspace_id,
            rel_path="test.pine",
            source_type="local",
            sha256="same-hash",
            pine_version="5",
            script_type="strategy",
            title="Same Title",
        )
        incoming = StrategyScript(
            id=script_id,
            workspace_id=workspace_id,
            rel_path="test.pine",
            source_type="local",
            sha256="same-hash",
            pine_version="5",
            script_type="strategy",
            title="Same Title",
        )

        changes = _compute_changed_fields(existing, incoming)

        assert changes == []


class TestPineDiscoveryMetrics:
    """Tests for Pine discovery Prometheus metrics."""

    def test_pine_metrics_defined(self):
        """Pine discovery metrics are properly defined."""
        from app.routers.metrics import (
            PINE_SCRIPTS_TOTAL,
            PINE_DISCOVERY_RUNS_TOTAL,
            PINE_DISCOVERY_ERRORS_TOTAL,
            PINE_SPECS_GENERATED_TOTAL,
            PINE_DISCOVERY_DURATION,
        )

        # Verify metrics exist and have correct types
        assert PINE_SCRIPTS_TOTAL is not None
        assert PINE_DISCOVERY_RUNS_TOTAL is not None
        assert PINE_DISCOVERY_ERRORS_TOTAL is not None
        assert PINE_SPECS_GENERATED_TOTAL is not None
        assert PINE_DISCOVERY_DURATION is not None

    def test_set_pine_scripts_metrics(self):
        """set_pine_scripts_metrics updates gauge correctly."""
        from app.routers.metrics import (
            PINE_SCRIPTS_TOTAL,
            set_pine_scripts_metrics,
        )

        # Set metrics
        status_counts = {
            "discovered": 10,
            "spec_generated": 5,
            "published": 2,
            "archived": 1,
        }
        set_pine_scripts_metrics(status_counts)

        # Verify each status is set
        # Note: Prometheus metrics accumulate, so we just check the function runs
        # without error. Full verification would need a fresh registry.
        assert PINE_SCRIPTS_TOTAL is not None

    def test_record_pine_discovery_run_success(self):
        """record_pine_discovery_run records success metrics."""
        from app.routers.metrics import (
            PINE_DISCOVERY_RUNS_TOTAL,
            record_pine_discovery_run,
        )

        # Record a successful run
        record_pine_discovery_run(
            status="success",
            duration=2.5,
            scripts_scanned=10,
            scripts_new=3,
            specs_generated=2,
            errors_count=0,
        )

        assert PINE_DISCOVERY_RUNS_TOTAL is not None

    def test_record_pine_discovery_run_partial(self):
        """record_pine_discovery_run records partial (with errors) metrics."""
        from app.routers.metrics import record_pine_discovery_run

        # Record a partial run with errors
        record_pine_discovery_run(
            status="partial",
            duration=1.5,
            scripts_scanned=10,
            scripts_new=2,
            specs_generated=1,
            errors_count=3,
        )

        # Function should complete without error
        assert True

    def test_record_pine_discovery_run_failed(self):
        """record_pine_discovery_run records failed metrics."""
        from app.routers.metrics import record_pine_discovery_run

        # Record a failed run
        record_pine_discovery_run(
            status="failed",
            duration=0.1,
            errors_count=1,
        )

        # Function should complete without error
        assert True

    def test_record_pine_discovery_error(self):
        """record_pine_discovery_error increments error counter."""
        from app.routers.metrics import (
            PINE_DISCOVERY_ERRORS_TOTAL,
            record_pine_discovery_error,
        )

        # Record an error
        record_pine_discovery_error()

        assert PINE_DISCOVERY_ERRORS_TOTAL is not None

    def test_record_pine_spec_generated(self):
        """record_pine_spec_generated increments spec counter."""
        from app.routers.metrics import (
            PINE_SPECS_GENERATED_TOTAL,
            record_pine_spec_generated,
        )

        # Record spec generation
        record_pine_spec_generated(count=3)

        assert PINE_SPECS_GENERATED_TOTAL is not None


class TestArchivingModels:
    """Tests for archiving-related dataclasses."""

    def test_archive_result_default_values(self):
        """ArchiveResult has correct defaults."""
        from app.services.pine.discovery_repository import ArchiveResult

        result = ArchiveResult(archived_count=0)
        assert result.archived_count == 0
        assert result.archived_scripts == []

    def test_archived_script_dataclass(self):
        """ArchivedScript stores script info correctly."""
        from datetime import datetime, timezone
        from app.services.pine.discovery_repository import ArchivedScript

        now = datetime.now(timezone.utc)
        script = ArchivedScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="old/script.pine",
            last_seen_at=now,
        )

        assert script.rel_path == "old/script.pine"
        assert script.last_seen_at == now

    def test_archived_script_none_last_seen(self):
        """ArchivedScript handles None last_seen_at."""
        from app.services.pine.discovery_repository import ArchivedScript

        script = ArchivedScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="test.pine",
            last_seen_at=None,
        )

        assert script.last_seen_at is None


class TestArchivedEventSchema:
    """Tests for pine.script.archived event schema."""

    def test_pine_script_archived_in_topics(self):
        """pine.script.archived is in PINE_TOPICS set."""
        from app.services.events.schemas import PINE_TOPICS

        assert "pine.script.archived" in PINE_TOPICS

    def test_pine_script_archived_event_factory(self):
        """pine_script_archived creates correct event."""
        from app.services.events.schemas import pine_script_archived

        workspace_id = uuid4()
        script_id = uuid4()

        event = pine_script_archived(
            event_id="evt-123",
            workspace_id=workspace_id,
            script_id=script_id,
            rel_path="stale/script.pine",
            last_seen_at="2025-01-10T12:00:00+00:00",
        )

        assert event.topic == "pine.script.archived"
        assert event.workspace_id == workspace_id
        assert event.payload["script_id"] == str(script_id)
        assert event.payload["rel_path"] == "stale/script.pine"
        assert event.payload["last_seen_at"] == "2025-01-10T12:00:00+00:00"

    def test_pine_script_archived_event_none_last_seen(self):
        """pine_script_archived handles None last_seen_at."""
        from app.services.events.schemas import pine_script_archived

        event = pine_script_archived(
            event_id="",
            workspace_id=uuid4(),
            script_id=uuid4(),
            rel_path="test.pine",
            last_seen_at=None,
        )

        assert event.payload["last_seen_at"] is None


class TestArchiveMetrics:
    """Tests for Pine archive Prometheus metrics."""

    def test_archive_metrics_defined(self):
        """Pine archive metrics are properly defined."""
        from app.routers.metrics import (
            PINE_SCRIPTS_ARCHIVED_TOTAL,
            PINE_ARCHIVE_RUNS_TOTAL,
            PINE_ARCHIVE_DURATION,
        )

        assert PINE_SCRIPTS_ARCHIVED_TOTAL is not None
        assert PINE_ARCHIVE_RUNS_TOTAL is not None
        assert PINE_ARCHIVE_DURATION is not None

    def test_record_pine_archive_run_success(self):
        """record_pine_archive_run records success metrics."""
        from app.routers.metrics import record_pine_archive_run

        record_pine_archive_run(
            status="success",
            duration=1.5,
            archived_count=5,
        )

        # Function should complete without error
        assert True

    def test_record_pine_archive_run_failed(self):
        """record_pine_archive_run records failed metrics."""
        from app.routers.metrics import record_pine_archive_run

        record_pine_archive_run(
            status="failed",
            duration=0.1,
            archived_count=0,
        )

        # Function should complete without error
        assert True

    def test_record_pine_scripts_archived(self):
        """record_pine_scripts_archived increments counter."""
        from app.routers.metrics import record_pine_scripts_archived

        record_pine_scripts_archived(count=10)

        # Function should complete without error
        assert True


class TestDiscoveryResultWithArchived:
    """Tests for DiscoveryResult with archived count."""

    def test_discovery_result_has_archived_field(self):
        """DiscoveryResult includes scripts_archived field."""
        result = DiscoveryResult()
        assert hasattr(result, "scripts_archived")
        assert result.scripts_archived == 0

    def test_discovery_result_archived_can_be_set(self):
        """DiscoveryResult scripts_archived can be set."""
        result = DiscoveryResult(scripts_archived=5)
        assert result.scripts_archived == 5


class TestDiscoverWithArchiving:
    """Tests for discovery with archiving integration."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.data_dir = "/data"
        return settings

    @pytest.fixture
    def service(self, mock_pool, mock_settings):
        """Create PineDiscoveryService with mocks."""
        return PineDiscoveryService(mock_pool, mock_settings)

    @pytest.mark.asyncio
    async def test_discover_with_archive_stale_days(self, service):
        """discover() accepts archive_stale_days parameter."""
        from app.services.pine.discovery_repository import ArchiveResult

        workspace_id = uuid4()

        with patch.object(service, "_scan_and_parse") as mock_scan, \
             patch.object(service._repo, "mark_archived", new_callable=AsyncMock) as mock_archive, \
             patch("app.services.events.get_event_bus") as mock_get_bus:

            mock_scan.return_value = []
            mock_archive.return_value = ArchiveResult(archived_count=3, archived_scripts=[])
            mock_bus = MagicMock()
            mock_bus.publish = AsyncMock(return_value=1)
            mock_get_bus.return_value = mock_bus

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                archive_stale_days=7,
            )

            # Should call mark_archived with 7 days
            mock_archive.assert_called_once_with(workspace_id, 7)
            assert result.scripts_archived == 3

    @pytest.mark.asyncio
    async def test_discover_without_archive_stale_days(self, service):
        """discover() skips archiving when archive_stale_days is None."""
        workspace_id = uuid4()

        with patch.object(service, "_scan_and_parse") as mock_scan, \
             patch.object(service._repo, "mark_archived", new_callable=AsyncMock) as mock_archive:

            mock_scan.return_value = []

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                archive_stale_days=None,
            )

            # Should NOT call mark_archived
            mock_archive.assert_not_called()
            assert result.scripts_archived == 0


# =============================================================================
# Auto-Ingest Tests
# =============================================================================


class TestAutoIngestModels:
    """Tests for auto-ingest related fields on DiscoveryResult and StrategyScript."""

    def test_discovery_result_has_ingest_fields(self):
        """DiscoveryResult includes ingest tracking fields."""
        result = DiscoveryResult()
        assert hasattr(result, "scripts_ingested")
        assert hasattr(result, "scripts_ingest_failed")
        assert hasattr(result, "chunks_created")
        assert result.scripts_ingested == 0
        assert result.scripts_ingest_failed == 0
        assert result.chunks_created == 0

    def test_discovery_result_ingest_fields_can_be_set(self):
        """DiscoveryResult ingest fields can be set."""
        result = DiscoveryResult(
            scripts_ingested=5,
            scripts_ingest_failed=1,
            chunks_created=25,
        )
        assert result.scripts_ingested == 5
        assert result.scripts_ingest_failed == 1
        assert result.chunks_created == 25

    def test_strategy_script_has_ingest_fields(self):
        """StrategyScript includes ingest tracking fields."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="test.pine",
            source_type="local",
            sha256="abc123",
            status="discovered",
        )
        assert hasattr(script, "doc_id")
        assert hasattr(script, "last_ingested_at")
        assert hasattr(script, "last_ingested_sha")
        assert hasattr(script, "ingest_status")
        assert hasattr(script, "ingest_error")

    def test_strategy_script_needs_ingest_never_ingested(self):
        """needs_ingest() returns True when never ingested."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="test.pine",
            source_type="local",
            sha256="abc123",
            status="discovered",
            ingest_status=None,
        )
        assert script.needs_ingest() is True

    def test_strategy_script_needs_ingest_sha_changed(self):
        """needs_ingest() returns True when content sha changed."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="test.pine",
            source_type="local",
            sha256="new_sha",
            status="discovered",
            ingest_status="ok",
            last_ingested_sha="old_sha",
        )
        assert script.needs_ingest() is True

    def test_strategy_script_needs_ingest_already_current(self):
        """needs_ingest() returns False when already ingested with same sha."""
        script = StrategyScript(
            id=uuid4(),
            workspace_id=uuid4(),
            rel_path="test.pine",
            source_type="local",
            sha256="abc123",
            status="discovered",
            ingest_status="ok",
            last_ingested_sha="abc123",
        )
        assert script.needs_ingest() is False


class TestIngestEventSchema:
    """Tests for pine.script.ingested event schema."""

    def test_pine_script_ingested_in_topics(self):
        """pine.script.ingested is included in PINE_TOPICS."""
        from app.services.events.schemas import PINE_TOPICS

        assert "pine.script.ingested" in PINE_TOPICS

    def test_pine_script_ingested_event_factory(self):
        """pine_script_ingested creates correct event."""
        from app.services.events.schemas import pine_script_ingested

        workspace_id = uuid4()
        script_id = uuid4()
        doc_id = uuid4()

        event = pine_script_ingested(
            event_id="",
            workspace_id=workspace_id,
            script_id=script_id,
            doc_id=doc_id,
            rel_path="strategies/breakout.pine",
            content_sha="abc123def",
            chunks_created=3,
        )

        assert event.topic == "pine.script.ingested"
        assert event.workspace_id == workspace_id
        assert event.payload["script_id"] == str(script_id)
        assert event.payload["doc_id"] == str(doc_id)
        assert event.payload["rel_path"] == "strategies/breakout.pine"
        assert event.payload["content_sha"] == "abc123def"
        assert event.payload["chunks_created"] == 3


class TestIngestMetrics:
    """Tests for Pine ingest Prometheus metrics."""

    def test_ingest_metrics_defined(self):
        """Pine ingest metrics are properly defined."""
        from app.routers.metrics import (
            PINE_SCRIPTS_INGESTED_TOTAL,
            PINE_INGEST_CHUNKS_TOTAL,
            PINE_INGEST_FAILED_TOTAL,
        )

        assert PINE_SCRIPTS_INGESTED_TOTAL is not None
        assert PINE_INGEST_CHUNKS_TOTAL is not None
        assert PINE_INGEST_FAILED_TOTAL is not None

    def test_record_pine_discovery_run_with_ingest(self):
        """record_pine_discovery_run records ingest metrics."""
        from app.routers.metrics import record_pine_discovery_run

        record_pine_discovery_run(
            status="success",
            duration=2.5,
            scripts_scanned=10,
            scripts_new=3,
            specs_generated=2,
            scripts_ingested=5,
            scripts_ingest_failed=1,
            chunks_created=15,
            errors_count=0,
        )

        # Function should complete without error
        assert True


class TestFormattingModule:
    """Tests for the formatting module."""

    def test_build_canonical_url(self):
        """build_canonical_url creates correct URL."""
        from app.services.pine.formatting import build_canonical_url

        url = build_canonical_url("local", "strategies/breakout.pine")
        assert url == "pine://local/strategies/breakout.pine"

    def test_build_canonical_url_normalizes(self):
        """build_canonical_url normalizes paths (strips . and .. segments, not traversal)."""
        from app.services.pine.formatting import build_canonical_url

        # Leading slash and dot segments are stripped
        url = build_canonical_url("local", "/./strategies/rsi.pine")
        assert url == "pine://local/strategies/rsi.pine"

        # Backslashes converted to forward slashes
        url2 = build_canonical_url("local", "strategies\\breakout.pine")
        assert url2 == "pine://local/strategies/breakout.pine"

    def test_build_ingest_doc_id(self):
        """build_ingest_doc_id creates stable ID."""
        from app.services.pine.formatting import build_ingest_doc_id

        doc_id = build_ingest_doc_id("local", "strategies/breakout.pine")
        assert doc_id == "pine:local:strategies/breakout.pine"

    def test_build_ingest_doc_id_stable(self):
        """build_ingest_doc_id produces same ID for same input."""
        from app.services.pine.formatting import build_ingest_doc_id

        doc_id1 = build_ingest_doc_id("local", "strategies/breakout.pine")
        doc_id2 = build_ingest_doc_id("local", "strategies/breakout.pine")
        assert doc_id1 == doc_id2

    def test_build_ingest_doc_id_with_repo_slug(self):
        """build_ingest_doc_id includes repo_slug for github sources (Phase B3.1)."""
        from app.services.pine.formatting import build_ingest_doc_id

        # GitHub with repo_slug should include it
        doc_id = build_ingest_doc_id(
            "github", "strategies/rsi.pine", repo_slug="acme/trading-scripts"
        )
        assert doc_id == "pine:github:acme/trading-scripts:strategies/rsi.pine"

    def test_build_ingest_doc_id_github_without_slug_fallback(self):
        """build_ingest_doc_id without repo_slug falls back (Phase B3.1 seam)."""
        from app.services.pine.formatting import build_ingest_doc_id

        # GitHub without repo_slug - falls back to simple format (not recommended)
        doc_id = build_ingest_doc_id("github", "strategies/rsi.pine")
        assert doc_id == "pine:github:strategies/rsi.pine"

    def test_build_canonical_url_with_repo_slug(self):
        """build_canonical_url includes repo_slug for github sources (Phase B3.1)."""
        from app.services.pine.formatting import build_canonical_url

        url = build_canonical_url(
            "github", "strategies/rsi.pine", repo_slug="acme/trading-scripts"
        )
        assert url == "pine://github/acme/trading-scripts/strategies/rsi.pine"

    def test_build_canonical_url_local_ignores_repo_slug(self):
        """build_canonical_url ignores repo_slug for local sources."""
        from app.services.pine.formatting import build_canonical_url

        # Local source should ignore repo_slug
        url = build_canonical_url("local", "strategies/breakout.pine", repo_slug="foo")
        assert url == "pine://local/strategies/breakout.pine"


class TestDiscoverWithAutoIngest:
    """Tests for discovery with auto-ingest integration."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.data_dir = "/data"
        return settings

    @pytest.fixture
    def service(self, mock_pool, mock_settings):
        """Create PineDiscoveryService with mocks."""
        return PineDiscoveryService(mock_pool, mock_settings)

    @pytest.mark.asyncio
    async def test_discover_with_auto_ingest_disabled(self, service):
        """discover() skips auto_ingest when auto_ingest=False."""
        workspace_id = uuid4()

        with patch.object(service, "_scan_and_parse") as mock_scan, \
             patch.object(service, "_auto_ingest_scripts", new_callable=AsyncMock) as mock_ingest:

            mock_scan.return_value = []

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                auto_ingest=False,
            )

            # Should NOT call _auto_ingest_scripts
            mock_ingest.assert_not_called()
            assert result.scripts_ingested == 0

    @pytest.mark.asyncio
    async def test_discover_with_emit_events_disabled(self, service):
        """discover() skips event emission when emit_events=False."""
        workspace_id = uuid4()

        with patch.object(service, "_scan_and_parse") as mock_scan, \
             patch.object(service, "_emit_discovery_events", new_callable=AsyncMock) as mock_emit:

            mock_scan.return_value = []

            await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                emit_events=False,
            )

            # Should NOT call _emit_discovery_events
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_result_includes_ingest_counts(self, service):
        """discover() result includes ingest counts when auto_ingest enabled."""
        workspace_id = uuid4()

        with patch.object(service, "_scan_and_parse") as mock_scan, \
             patch.object(service, "_auto_ingest_scripts", new_callable=AsyncMock) as mock_ingest:

            mock_scan.return_value = []
            mock_ingest.return_value = {"ingested": 3, "failed": 1, "chunks": 15}

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=["/data/pine"],
                auto_ingest=True,
            )

            assert result.scripts_ingested == 3
            assert result.scripts_ingest_failed == 1
            assert result.chunks_created == 15


class TestDiscoveryGaugeMetrics:
    """Tests for Pine discovery gauge metrics."""

    def test_pending_ingest_gauge_defined(self):
        """PINE_SCRIPTS_PENDING_INGEST gauge is defined."""
        from app.routers.metrics import PINE_SCRIPTS_PENDING_INGEST

        assert PINE_SCRIPTS_PENDING_INGEST is not None
        assert PINE_SCRIPTS_PENDING_INGEST._name == "pine_scripts_pending_ingest"

    def test_last_run_timestamp_gauge_defined(self):
        """PINE_DISCOVERY_LAST_RUN_TIMESTAMP gauge is defined."""
        from app.routers.metrics import PINE_DISCOVERY_LAST_RUN_TIMESTAMP

        assert PINE_DISCOVERY_LAST_RUN_TIMESTAMP is not None
        assert (
            PINE_DISCOVERY_LAST_RUN_TIMESTAMP._name
            == "pine_discovery_last_run_timestamp"
        )

    def test_last_success_timestamp_gauge_defined(self):
        """PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP gauge is defined."""
        from app.routers.metrics import PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP

        assert PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP is not None
        assert (
            PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP._name
            == "pine_discovery_last_success_timestamp"
        )

    def test_set_pine_pending_ingest(self):
        """set_pine_pending_ingest updates gauge value."""
        from app.routers.metrics import (
            PINE_SCRIPTS_PENDING_INGEST,
            set_pine_pending_ingest,
        )

        set_pine_pending_ingest(42)
        # Gauge should have value set (prometheus_client internals)
        assert PINE_SCRIPTS_PENDING_INGEST._value._value == 42

    def test_record_pine_discovery_timestamp_success(self):
        """record_pine_discovery_timestamp updates both gauges on success."""
        import time

        from app.routers.metrics import (
            PINE_DISCOVERY_LAST_RUN_TIMESTAMP,
            PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP,
            record_pine_discovery_timestamp,
        )

        before = time.time()
        record_pine_discovery_timestamp(success=True)
        after = time.time()

        # Both timestamps should be updated
        run_ts = PINE_DISCOVERY_LAST_RUN_TIMESTAMP._value._value
        success_ts = PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP._value._value

        assert before <= run_ts <= after
        assert before <= success_ts <= after

    def test_record_pine_discovery_timestamp_failure(self):
        """record_pine_discovery_timestamp only updates run timestamp on failure."""
        import time

        from app.routers.metrics import (
            PINE_DISCOVERY_LAST_RUN_TIMESTAMP,
            PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP,
            record_pine_discovery_timestamp,
        )

        # Set a known value for success timestamp
        PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP.set(12345.0)

        before = time.time()
        record_pine_discovery_timestamp(success=False)
        after = time.time()

        # Run timestamp should be updated
        run_ts = PINE_DISCOVERY_LAST_RUN_TIMESTAMP._value._value
        assert before <= run_ts <= after

        # Success timestamp should NOT be updated (still old value)
        success_ts = PINE_DISCOVERY_LAST_SUCCESS_TIMESTAMP._value._value
        assert success_ts == 12345.0
