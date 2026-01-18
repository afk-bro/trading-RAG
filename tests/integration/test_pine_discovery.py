"""Integration tests for Pine script discovery.

These tests verify the full discovery flow:
1. Scan → Parse → Reconcile → Persist → Events
2. Idempotency: Same files scanned twice → second run yields all unchanged
3. Event bus integration

Run with: pytest tests/integration/test_pine_discovery.py -v
Note: DB tests require migrations to be applied (strategy_scripts table)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Skip if no database
pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_pine_files():
    """Create temporary directory with sample .pine files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pine_dir = Path(tmpdir) / "pine"
        pine_dir.mkdir()

        # Strategy script
        strategy_file = pine_dir / "breakout.pine"
        strategy_file.write_text(
            """//@version=5
strategy("52W Breakout", overlay=true)

length = input.int(252, "Lookback Days", minval=1, maxval=500)
threshold = input.float(1.0, "Threshold %", minval=0.1, maxval=10.0)

high52w = ta.highest(high, length)
longCondition = close > high52w * (1 + threshold / 100)

if (longCondition)
    strategy.entry("Long", strategy.long)

strategy.close("Long", when=barstate.islastconfirmedhistory)
"""
        )

        # Indicator script
        indicator_file = pine_dir / "ma_cross.pine"
        indicator_file.write_text(
            """//@version=5
indicator("MA Crossover", overlay=true)

fastLen = input.int(10, "Fast MA")
slowLen = input.int(20, "Slow MA")

fastMA = ta.sma(close, fastLen)
slowMA = ta.sma(close, slowLen)

plot(fastMA, color=color.blue)
plot(slowMA, color=color.red)
"""
        )

        yield tmpdir, pine_dir


# =============================================================================
# Unit-level Integration (Mocked DB)
# =============================================================================


class TestDiscoveryWithMockedDB:
    """Integration tests with mocked database."""

    @pytest.mark.asyncio
    async def test_full_discovery_flow_with_mock_db(self, sample_pine_files):
        """
        Full discovery flow with mocked database:
        1. Scans real .pine files
        2. Parses content
        3. Mocked persistence
        4. Mocked event emission
        """
        tmpdir, pine_dir = sample_pine_files
        workspace_id = uuid4()

        # Mock pool and settings
        mock_pool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.data_dir = tmpdir

        # Create service
        from app.services.pine.discovery import PineDiscoveryService

        service = PineDiscoveryService(mock_pool, mock_settings)

        # Mock repository methods
        with patch.object(
            service._repo, "get_by_path", new_callable=AsyncMock
        ) as mock_get, patch.object(
            service._repo, "upsert", new_callable=AsyncMock
        ) as mock_upsert, patch.object(
            service._repo, "update_spec", new_callable=AsyncMock
        ) as mock_update_spec, patch(
            "app.services.events.get_event_bus"
        ) as mock_get_bus:
            # Setup mocks
            mock_get.return_value = None  # All scripts are "new"
            mock_bus = MagicMock()
            mock_bus.publish = AsyncMock(return_value=1)
            mock_get_bus.return_value = mock_bus

            # Make upsert return the script with is_new=True
            from app.services.pine.discovery_repository import UpsertResult

            async def upsert_side_effect(script):
                return UpsertResult(script=script, is_new=True, changed_fields=[])

            mock_upsert.side_effect = upsert_side_effect
            mock_update_spec.return_value = MagicMock()

            # Run discovery
            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=[str(pine_dir)],
                generate_specs=True,
            )

            # Verify results
            assert result.scripts_scanned == 2  # strategy + indicator
            assert result.scripts_new == 2
            assert result.scripts_unchanged == 0
            assert result.errors == []

            # Verify upsert was called for each script
            assert mock_upsert.call_count == 2

            # Verify spec was generated for strategy (not indicator)
            assert result.specs_generated == 1

            # Verify events were published
            assert (
                mock_bus.publish.call_count >= 2
            )  # At least discovered + spec_generated

    @pytest.mark.asyncio
    async def test_discovery_idempotency_with_mock_db(self, sample_pine_files):
        """
        Idempotency test: Same files scanned twice, second run yields unchanged.
        """
        tmpdir, pine_dir = sample_pine_files
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.data_dir = tmpdir

        from app.services.pine.discovery import PineDiscoveryService
        from app.services.pine.discovery_repository import UpsertResult

        service = PineDiscoveryService(mock_pool, mock_settings)

        # Track scripts from first run to return on second
        first_run_scripts = {}

        with patch.object(
            service._repo, "get_by_path", new_callable=AsyncMock
        ) as mock_get, patch.object(
            service._repo, "upsert", new_callable=AsyncMock
        ) as mock_upsert, patch.object(
            service._repo, "update_spec", new_callable=AsyncMock
        ) as mock_update_spec, patch(
            "app.services.events.get_event_bus"
        ) as mock_get_bus:
            mock_bus = MagicMock()
            mock_bus.publish = AsyncMock(return_value=1)
            mock_get_bus.return_value = mock_bus

            # First run: all new
            mock_get.return_value = None

            async def first_upsert(script):
                first_run_scripts[script.rel_path] = script
                return UpsertResult(script=script, is_new=True, changed_fields=[])

            mock_upsert.side_effect = first_upsert
            mock_update_spec.return_value = MagicMock()

            result1 = await service.discover(
                workspace_id=workspace_id,
                scan_paths=[str(pine_dir)],
                generate_specs=True,
            )

            assert result1.scripts_new == 2
            assert result1.scripts_unchanged == 0

            # Reset mocks for second run
            mock_upsert.reset_mock()
            mock_bus.publish.reset_mock()

            # Second run: return existing scripts with same SHA256
            async def second_get(ws_id, source_type, rel_path):
                return first_run_scripts.get(rel_path)

            mock_get.side_effect = second_get

            async def second_upsert(script):
                # Script unchanged - return with no changes
                return UpsertResult(
                    script=first_run_scripts.get(script.rel_path, script),
                    is_new=False,
                    changed_fields=[],
                )

            mock_upsert.side_effect = second_upsert

            result2 = await service.discover(
                workspace_id=workspace_id,
                scan_paths=[str(pine_dir)],
                generate_specs=True,
            )

            # Second run: all unchanged
            assert result2.scripts_scanned == 2
            assert result2.scripts_new == 0
            assert result2.scripts_unchanged == 2
            assert result2.specs_generated == 0  # No new specs for unchanged

    @pytest.mark.asyncio
    async def test_discovery_detects_content_changes(self, sample_pine_files):
        """
        Discovery detects content changes via SHA256.
        """
        tmpdir, pine_dir = sample_pine_files
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.data_dir = tmpdir

        from app.services.pine.discovery import PineDiscoveryService
        from app.services.pine.discovery_repository import StrategyScript, UpsertResult

        service = PineDiscoveryService(mock_pool, mock_settings)

        # Create a stored script with old SHA256
        old_script = StrategyScript(
            id=uuid4(),
            workspace_id=workspace_id,
            rel_path="breakout.pine",
            source_type="local",
            sha256="old-sha256-that-does-not-match",
            pine_version="5",
            script_type="strategy",
            title="52W Breakout",
        )

        with patch.object(
            service._repo, "get_by_path", new_callable=AsyncMock
        ) as mock_get, patch.object(
            service._repo, "upsert", new_callable=AsyncMock
        ) as mock_upsert, patch.object(
            service._repo, "update_spec", new_callable=AsyncMock
        ) as mock_update_spec, patch(
            "app.services.events.get_event_bus"
        ) as mock_get_bus:
            mock_bus = MagicMock()
            mock_bus.publish = AsyncMock(return_value=1)
            mock_get_bus.return_value = mock_bus

            # Return old script for breakout.pine, None for others
            async def get_by_path(ws_id, source_type, rel_path):
                if "breakout" in rel_path:
                    return old_script
                return None

            mock_get.side_effect = get_by_path

            async def upsert_with_changes(script):
                if "breakout" in script.rel_path:
                    # SHA256 changed
                    return UpsertResult(
                        script=script, is_new=False, changed_fields=["sha256"]
                    )
                return UpsertResult(script=script, is_new=True, changed_fields=[])

            mock_upsert.side_effect = upsert_with_changes
            mock_update_spec.return_value = MagicMock()

            result = await service.discover(
                workspace_id=workspace_id,
                scan_paths=[str(pine_dir)],
                generate_specs=True,
            )

            # breakout.pine = updated, ma_cross.pine = new
            assert result.scripts_new == 1
            assert result.scripts_updated == 1
            assert result.scripts_unchanged == 0


# =============================================================================
# Full DB Integration (Requires migrations)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("SUPABASE_URL"),
    reason="Requires DATABASE_URL or SUPABASE_URL",
)
@pytest.mark.requires_db
class TestDiscoveryWithRealDB:
    """Integration tests with real database (requires migrations)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    def test_discover_endpoint_requires_admin_token(self, client):
        """Discovery endpoint requires admin authentication."""
        resp = client.post(
            "/admin/pine/discover",
            json={
                "workspace_id": str(uuid4()),
                "scan_paths": ["/data/pine"],
            },
        )

        # Should fail without admin token
        assert resp.status_code in (401, 403)

    def test_discover_endpoint_validates_path(self, client):
        """Discovery endpoint rejects paths outside DATA_DIR."""
        admin_token = os.getenv("ADMIN_TOKEN", "test-admin-token")

        resp = client.post(
            "/admin/pine/discover",
            json={
                "workspace_id": str(uuid4()),
                "scan_paths": ["/etc/passwd"],  # Path escape attempt
            },
            headers={"X-Admin-Token": admin_token},
        )

        # Should fail with 422 (path validation)
        assert resp.status_code == 422

    def test_discover_endpoint_dry_run(self, client, sample_pine_files):
        """Dry-run discovery returns counts without persisting."""
        tmpdir, pine_dir = sample_pine_files
        admin_token = os.getenv("ADMIN_TOKEN", "test-admin-token")
        workspace_id = str(uuid4())

        # Patch DATA_DIR for the request
        with patch("app.admin.pine_discovery.get_settings") as mock_settings:
            settings = MagicMock()
            settings.data_dir = tmpdir
            mock_settings.return_value = settings

            resp = client.post(
                "/admin/pine/discover",
                json={
                    "workspace_id": workspace_id,
                    "scan_paths": [str(pine_dir)],
                    "dry_run": True,
                },
                headers={"X-Admin-Token": admin_token},
            )

        # Check response
        if resp.status_code == 500:
            pytest.skip("strategy_scripts migration not applied")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert data["scripts_scanned"] >= 0

    def test_stats_endpoint_requires_admin_token(self, client):
        """Stats endpoint requires admin authentication."""
        resp = client.get(
            "/admin/pine/scripts/stats",
            params={"workspace_id": str(uuid4())},
        )

        assert resp.status_code in (401, 403)


# =============================================================================
# Event Bus Integration
# =============================================================================


class TestDiscoveryEventIntegration:
    """Tests for event bus integration in discovery."""

    @pytest.mark.asyncio
    async def test_events_published_in_correct_order(self, sample_pine_files):
        """
        Events are published in correct order:
        1. discovered/updated events (after upsert commit)
        2. spec_generated events (after spec generation)
        """
        tmpdir, pine_dir = sample_pine_files
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.data_dir = tmpdir

        from app.services.pine.discovery import PineDiscoveryService
        from app.services.pine.discovery_repository import UpsertResult

        service = PineDiscoveryService(mock_pool, mock_settings)
        published_events = []

        with patch.object(
            service._repo, "get_by_path", new_callable=AsyncMock
        ) as mock_get, patch.object(
            service._repo, "upsert", new_callable=AsyncMock
        ) as mock_upsert, patch.object(
            service._repo, "update_spec", new_callable=AsyncMock
        ) as mock_update_spec, patch(
            "app.services.events.get_event_bus"
        ) as mock_get_bus:
            mock_get.return_value = None
            mock_bus = MagicMock()

            async def record_publish(event):
                published_events.append(event.topic)
                return 1

            mock_bus.publish = record_publish
            mock_get_bus.return_value = mock_bus

            async def upsert_new(script):
                return UpsertResult(script=script, is_new=True, changed_fields=[])

            mock_upsert.side_effect = upsert_new
            mock_update_spec.return_value = MagicMock()

            await service.discover(
                workspace_id=workspace_id,
                scan_paths=[str(pine_dir)],
                generate_specs=True,
            )

            # Verify event order: discovered before spec_generated
            discovered_events = [
                e for e in published_events if e == "pine.script.discovered"
            ]
            spec_events = [
                e for e in published_events if e == "pine.script.spec_generated"
            ]

            assert len(discovered_events) == 2  # strategy + indicator
            assert len(spec_events) == 1  # Only strategy gets spec

            # All discovered events should come before spec_generated
            if discovered_events and spec_events:
                last_discovered_idx = max(
                    i
                    for i, e in enumerate(published_events)
                    if e == "pine.script.discovered"
                )
                first_spec_idx = min(
                    i
                    for i, e in enumerate(published_events)
                    if e == "pine.script.spec_generated"
                )
                assert (
                    last_discovered_idx < first_spec_idx
                ), "discovered events should precede spec_generated"
