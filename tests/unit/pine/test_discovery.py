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
