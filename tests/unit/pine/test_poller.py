"""Unit tests for PineRepoPoller."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.services.pine.poller import (
    PineRepoPoller,
    PollRunResult,
    PollerHealth,
    get_poller,
    set_poller,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    return MagicMock()


@pytest.fixture
def mock_settings():
    """Create a mock settings object with polling config."""
    settings = MagicMock()
    settings.pine_repo_poll_enabled = True
    settings.pine_repo_poll_interval_minutes = 15
    settings.pine_repo_poll_tick_seconds = 60
    settings.pine_repo_poll_max_concurrency = 2
    settings.pine_repo_poll_max_repos_per_tick = 10
    settings.pine_repo_poll_backoff_max_multiplier = 16
    return settings


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant client."""
    return MagicMock()


@pytest.fixture
def sample_repo():
    """Create a sample PineRepo for testing."""
    repo = MagicMock()
    repo.id = UUID("12345678-1234-5678-1234-567812345678")
    repo.workspace_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    repo.repo_slug = "owner/test-repo"
    return repo


@pytest.fixture
def poller(mock_pool, mock_settings, mock_qdrant):
    """Create a PineRepoPoller instance."""
    return PineRepoPoller(mock_pool, mock_settings, mock_qdrant)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestPollRunResult:
    """Tests for PollRunResult dataclass."""

    def test_default_values(self):
        """Test default field values."""
        result = PollRunResult()

        assert result.repos_scanned == 0
        assert result.repos_succeeded == 0
        assert result.repos_failed == 0
        assert result.repos_skipped == 0
        assert result.errors == []
        assert result.duration_ms == 0

    def test_custom_values(self):
        """Test custom field values."""
        result = PollRunResult(
            repos_scanned=5,
            repos_succeeded=3,
            repos_failed=1,
            repos_skipped=1,
            errors=["error1", "error2"],
            duration_ms=1234,
        )

        assert result.repos_scanned == 5
        assert result.repos_succeeded == 3
        assert result.repos_failed == 1
        assert result.repos_skipped == 1
        assert result.errors == ["error1", "error2"]
        assert result.duration_ms == 1234


class TestPollerHealth:
    """Tests for PollerHealth dataclass."""

    def test_default_values(self):
        """Test default field values."""
        health = PollerHealth()

        assert health.enabled is False
        assert health.running is False
        assert health.last_run_at is None
        assert health.last_run_repos_scanned == 0
        assert health.last_run_errors == 0
        assert health.repos_due_count == 0
        assert health.poll_interval_minutes == 15
        assert health.poll_max_concurrency == 2
        assert health.poll_tick_seconds == 60

    def test_custom_values(self):
        """Test custom field values."""
        now = datetime.now(timezone.utc)
        health = PollerHealth(
            enabled=True,
            running=True,
            last_run_at=now,
            last_run_repos_scanned=10,
            last_run_errors=2,
            repos_due_count=5,
            poll_interval_minutes=30,
            poll_max_concurrency=4,
            poll_tick_seconds=120,
        )

        assert health.enabled is True
        assert health.running is True
        assert health.last_run_at == now
        assert health.last_run_repos_scanned == 10
        assert health.last_run_errors == 2
        assert health.repos_due_count == 5
        assert health.poll_interval_minutes == 30
        assert health.poll_max_concurrency == 4
        assert health.poll_tick_seconds == 120


# =============================================================================
# PineRepoPoller Tests
# =============================================================================


class TestPollerInitialization:
    """Tests for PineRepoPoller initialization."""

    def test_initial_state(self, poller, mock_settings):
        """Test initial poller state."""
        assert poller.is_running is False
        assert poller._task is None
        assert poller._last_run_at is None
        assert poller._last_run_result is None
        assert poller._inflight == 0

    def test_semaphore_initialization(self, poller, mock_settings):
        """Test semaphore is initialized with correct concurrency."""
        # Access the internal semaphore value
        assert poller._semaphore._value == mock_settings.pine_repo_poll_max_concurrency


class TestPollerStartStop:
    """Tests for start/stop methods."""

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_pool, mock_qdrant):
        """Test start does nothing when polling is disabled."""
        settings = MagicMock()
        settings.pine_repo_poll_enabled = False
        settings.pine_repo_poll_max_concurrency = 2

        poller = PineRepoPoller(mock_pool, settings, mock_qdrant)
        await poller.start()

        assert poller.is_running is False
        assert poller._task is None

    @pytest.mark.asyncio
    async def test_start_already_running(self, poller):
        """Test start does nothing if already running."""
        poller._running = True

        await poller.start()

        # Should not create a new task
        assert poller._task is None

    @pytest.mark.asyncio
    async def test_start_creates_task(self, poller):
        """Test start creates background task."""
        with patch.object(poller, "_poll_loop", new_callable=AsyncMock):
            await poller.start()

            assert poller.is_running is True
            assert poller._task is not None

            # Clean up
            poller._stop_event.set()
            await poller._task

    @pytest.mark.asyncio
    async def test_stop_not_running(self, poller):
        """Test stop does nothing if not running."""
        await poller.stop()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_stop_graceful(self, poller):
        """Test graceful stop with running task."""
        # Start the poller
        with patch.object(poller, "_poll_loop") as mock_loop:
            # Make loop wait for stop event
            async def wait_for_stop():
                await poller._stop_event.wait()

            mock_loop.side_effect = wait_for_stop

            await poller.start()
            assert poller.is_running is True

            # Stop should set event and wait
            await poller.stop(timeout=5.0)

            assert poller.is_running is False
            assert poller._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_stop_timeout_cancels(self, poller):
        """Test stop cancels task on timeout."""

        # Create a task that won't respond to stop event
        async def infinite_loop():
            while True:
                await asyncio.sleep(10)

        poller._running = True
        poller._task = asyncio.create_task(infinite_loop())

        # Stop with short timeout
        await poller.stop(timeout=0.1)

        assert poller.is_running is False
        assert poller._task.cancelled() or poller._task.done()


class TestPollerRunOnce:
    """Tests for run_once method."""

    @pytest.mark.asyncio
    async def test_run_once_delegates(self, poller):
        """Test run_once delegates to _do_poll_tick."""
        expected_result = PollRunResult(repos_scanned=5, repos_succeeded=5)

        with patch.object(
            poller,
            "_do_poll_tick",
            new_callable=AsyncMock,
            return_value=expected_result,
        ):
            result = await poller.run_once()

            assert result == expected_result
            poller._do_poll_tick.assert_called_once()


class TestGetHealth:
    """Tests for get_health method."""

    @pytest.mark.asyncio
    async def test_get_health_no_previous_run(self, poller, mock_settings):
        """Test health when no previous run."""
        with patch.object(
            poller._repo_registry,
            "count_due_for_poll",
            new_callable=AsyncMock,
            return_value=5,
        ):
            health = await poller.get_health()

            assert health.enabled is True
            assert health.running is False
            assert health.last_run_at is None
            assert health.last_run_repos_scanned == 0
            assert health.last_run_errors == 0
            assert health.repos_due_count == 5
            assert (
                health.poll_interval_minutes
                == mock_settings.pine_repo_poll_interval_minutes
            )
            assert (
                health.poll_max_concurrency
                == mock_settings.pine_repo_poll_max_concurrency
            )
            assert health.poll_tick_seconds == mock_settings.pine_repo_poll_tick_seconds

    @pytest.mark.asyncio
    async def test_get_health_with_previous_run(self, poller):
        """Test health with previous run result."""
        now = datetime.now(timezone.utc)
        poller._running = True
        poller._last_run_at = now
        poller._last_run_result = PollRunResult(
            repos_scanned=10,
            repos_failed=2,
        )

        with patch.object(
            poller._repo_registry,
            "count_due_for_poll",
            new_callable=AsyncMock,
            return_value=3,
        ):
            health = await poller.get_health()

            assert health.running is True
            assert health.last_run_at == now
            assert health.last_run_repos_scanned == 10
            assert health.last_run_errors == 2
            assert health.repos_due_count == 3


class TestDoPollTick:
    """Tests for _do_poll_tick method."""

    @pytest.mark.asyncio
    async def test_no_repos_due(self, poller):
        """Test tick when no repos are due."""
        with patch.object(
            poller._repo_registry,
            "list_due_for_poll",
            new_callable=AsyncMock,
            return_value=[],
        ), patch.object(
            poller._repo_registry,
            "count_due_for_poll",
            new_callable=AsyncMock,
            return_value=0,
        ):
            result = await poller._do_poll_tick()

            assert result.repos_scanned == 0
            assert result.repos_succeeded == 0
            assert result.repos_failed == 0
            assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_all_repos_succeed(self, poller, sample_repo):
        """Test tick when all repos succeed."""
        repos = [sample_repo]

        with patch.object(
            poller._repo_registry,
            "list_due_for_poll",
            new_callable=AsyncMock,
            return_value=repos,
        ), patch.object(
            poller._repo_registry,
            "count_due_for_poll",
            new_callable=AsyncMock,
            return_value=1,
        ), patch.object(
            poller,
            "_scan_repo_with_semaphore",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await poller._do_poll_tick()

            assert result.repos_scanned == 1
            assert result.repos_succeeded == 1
            assert result.repos_failed == 0
            assert result.repos_skipped == 0
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_some_repos_fail(self, poller, sample_repo):
        """Test tick when some repos fail."""
        repo2 = MagicMock()
        repo2.id = UUID("22222222-2222-2222-2222-222222222222")
        repo2.workspace_id = sample_repo.workspace_id
        repo2.repo_slug = "owner/fail-repo"

        repos = [sample_repo, repo2]

        with patch.object(
            poller._repo_registry,
            "list_due_for_poll",
            new_callable=AsyncMock,
            return_value=repos,
        ), patch.object(
            poller._repo_registry,
            "count_due_for_poll",
            new_callable=AsyncMock,
            return_value=2,
        ), patch.object(
            poller, "_scan_repo_with_semaphore", new_callable=AsyncMock
        ) as mock_scan:
            # First repo succeeds, second fails
            mock_scan.side_effect = [True, False]

            result = await poller._do_poll_tick()

            assert result.repos_scanned == 2
            assert result.repos_succeeded == 1
            assert result.repos_skipped == 1  # False return counts as skipped

    @pytest.mark.asyncio
    async def test_exception_during_scan(self, poller, sample_repo):
        """Test tick handles exceptions during scan."""
        repos = [sample_repo]

        with patch.object(
            poller._repo_registry,
            "list_due_for_poll",
            new_callable=AsyncMock,
            return_value=repos,
        ), patch.object(
            poller._repo_registry,
            "count_due_for_poll",
            new_callable=AsyncMock,
            return_value=1,
        ), patch.object(
            poller, "_scan_repo_with_semaphore", new_callable=AsyncMock
        ) as mock_scan:
            mock_scan.side_effect = Exception("Connection failed")

            result = await poller._do_poll_tick()

            assert result.repos_scanned == 1
            assert result.repos_succeeded == 0
            assert result.repos_failed == 1
            assert "owner/test-repo: Connection failed" in result.errors[0]


class TestScanRepoWithSemaphore:
    """Tests for _scan_repo_with_semaphore method."""

    @pytest.mark.asyncio
    async def test_updates_inflight_count(self, poller, sample_repo):
        """Test inflight count is updated during scan."""
        inflight_during_scan = None

        async def capture_inflight(repo):
            nonlocal inflight_during_scan
            inflight_during_scan = poller._inflight
            return True

        with patch.object(poller, "_scan_repo", side_effect=capture_inflight):
            initial_inflight = poller._inflight
            await poller._scan_repo_with_semaphore(sample_repo)
            final_inflight = poller._inflight

        assert initial_inflight == 0
        assert inflight_during_scan == 1
        assert final_inflight == 0

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, poller, sample_repo):
        """Test semaphore limits concurrent scans."""
        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrent(repo):
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
            await asyncio.sleep(0.05)  # Simulate work
            async with lock:
                current_concurrent -= 1
            return True

        # Create 5 repos but semaphore is 2
        repos = [MagicMock() for _ in range(5)]

        with patch.object(poller, "_scan_repo", side_effect=track_concurrent):
            tasks = [poller._scan_repo_with_semaphore(repo) for repo in repos]
            await asyncio.gather(*tasks)

        # Should never exceed semaphore limit
        assert max_concurrent_seen <= poller._semaphore._value


class TestScanRepo:
    """Tests for _scan_repo method."""

    @pytest.mark.asyncio
    async def test_scan_success(self, poller, sample_repo):
        """Test successful scan updates poll state."""
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.scripts_new = 5
        mock_result.scripts_updated = 2

        with patch(
            "app.services.pine.discovery.PineDiscoveryService"
        ) as MockDiscovery, patch.object(
            poller._repo_registry, "update_poll_success", new_callable=AsyncMock
        ) as mock_update:
            mock_discovery = AsyncMock()
            mock_discovery.discover_repo.return_value = mock_result
            MockDiscovery.return_value = mock_discovery

            result = await poller._scan_repo(sample_repo)

            assert result is True
            mock_update.assert_called_once_with(
                repo_id=sample_repo.id,
                interval_minutes=poller._settings.pine_repo_poll_interval_minutes,
            )

    @pytest.mark.asyncio
    async def test_scan_partial_success(self, poller, sample_repo):
        """Test partial scan success updates poll state."""
        mock_result = MagicMock()
        mock_result.status = "partial"

        with patch(
            "app.services.pine.discovery.PineDiscoveryService"
        ) as MockDiscovery, patch.object(
            poller._repo_registry, "update_poll_success", new_callable=AsyncMock
        ) as mock_update:
            mock_discovery = AsyncMock()
            mock_discovery.discover_repo.return_value = mock_result
            MockDiscovery.return_value = mock_discovery

            result = await poller._scan_repo(sample_repo)

            assert result is True
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_error_status(self, poller, sample_repo):
        """Test error status updates poll failure state."""
        mock_result = MagicMock()
        mock_result.status = "error"
        mock_result.errors = ["Error 1", "Error 2"]

        with patch(
            "app.services.pine.discovery.PineDiscoveryService"
        ) as MockDiscovery, patch.object(
            poller._repo_registry, "update_poll_failure", new_callable=AsyncMock
        ) as mock_update:
            mock_discovery = AsyncMock()
            mock_discovery.discover_repo.return_value = mock_result
            MockDiscovery.return_value = mock_discovery

            result = await poller._scan_repo(sample_repo)

            assert result is False
            mock_update.assert_called_once_with(
                repo_id=sample_repo.id,
                base_interval_minutes=poller._settings.pine_repo_poll_interval_minutes,
                max_backoff_multiplier=poller._settings.pine_repo_poll_backoff_max_multiplier,
            )

    @pytest.mark.asyncio
    async def test_scan_exception(self, poller, sample_repo):
        """Test exception during scan updates poll failure state."""
        with patch(
            "app.services.pine.discovery.PineDiscoveryService"
        ) as MockDiscovery, patch.object(
            poller._repo_registry, "update_poll_failure", new_callable=AsyncMock
        ) as mock_update:
            mock_discovery = AsyncMock()
            mock_discovery.discover_repo.side_effect = Exception("Clone failed")
            MockDiscovery.return_value = mock_discovery

            result = await poller._scan_repo(sample_repo)

            assert result is False
            mock_update.assert_called_once()


# =============================================================================
# Module-Level Singleton Tests
# =============================================================================


class TestPollerSingleton:
    """Tests for module-level get_poller/set_poller."""

    def test_get_set_poller(self, poller):
        """Test get/set poller."""
        # Clear any existing poller
        set_poller(None)
        assert get_poller() is None

        # Set poller
        set_poller(poller)
        assert get_poller() is poller

        # Clear poller
        set_poller(None)
        assert get_poller() is None


# =============================================================================
# Poll Loop Tests
# =============================================================================


class TestPollLoop:
    """Tests for _poll_loop method."""

    @pytest.mark.asyncio
    async def test_poll_loop_respects_stop_event(self, poller):
        """Test poll loop stops when stop event is set."""
        tick_count = 0

        async def mock_tick():
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 2:
                poller._stop_event.set()
            return PollRunResult()

        poller._settings.pine_repo_poll_tick_seconds = 0.01  # Fast ticks for testing

        with patch.object(poller, "_do_poll_tick", side_effect=mock_tick):
            await poller._poll_loop()

        assert tick_count >= 2

    @pytest.mark.asyncio
    async def test_poll_loop_handles_tick_errors(self, poller):
        """Test poll loop continues after tick errors."""
        tick_count = 0

        async def mock_tick():
            nonlocal tick_count
            tick_count += 1
            if tick_count == 1:
                raise Exception("Tick error")
            poller._stop_event.set()
            return PollRunResult()

        poller._settings.pine_repo_poll_tick_seconds = 0.01

        with patch.object(poller, "_do_poll_tick", side_effect=mock_tick):
            await poller._poll_loop()

        # Should have continued after error
        assert tick_count >= 2

    @pytest.mark.asyncio
    async def test_poll_loop_updates_state(self, poller):
        """Test poll loop updates last_run state."""
        result = PollRunResult(repos_scanned=5, repos_succeeded=5)

        async def mock_tick():
            poller._stop_event.set()
            return result

        with patch.object(poller, "_do_poll_tick", side_effect=mock_tick):
            await poller._poll_loop()

        assert poller._last_run_result == result
        assert poller._last_run_at is not None
