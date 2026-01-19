"""Unit tests for health_checks service functions.

Tests focus on:
- Individual check functions handle errors gracefully
- collect_system_health keeps going after exceptions
- Output shape matches model contracts
- Overall status logic (OK/degraded/error)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.admin.services.health_checks import (
    check_database,
    collect_system_health,
)
from app.admin.services.health_models import (
    DBHealth,
    SystemHealthSnapshot,
)


class TestCheckDatabase:
    """Tests for check_database function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_pool_none(self):
        """Database check returns error status when pool is None."""
        result = await check_database(None)

        assert isinstance(result, DBHealth)
        assert result.status == "error"
        assert result.error is not None
        assert "pool" in result.error.lower() or "not" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_error_on_connection_failure(self):
        """Database check captures connection errors gracefully."""
        # Create a mock that raises on acquire
        mock_pool = MagicMock()
        mock_pool.acquire.side_effect = Exception("Connection refused")
        mock_pool.get_size.return_value = 0
        mock_pool.get_idle_size.return_value = 0

        result = await check_database(mock_pool)

        assert isinstance(result, DBHealth)
        assert result.status == "error"
        assert result.error is not None


class TestCollectSystemHealth:
    """Tests for collect_system_health aggregator."""

    @pytest.mark.asyncio
    async def test_returns_complete_snapshot_shape(self):
        """collect_system_health returns all required fields."""
        mock_pool = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.qdrant_collection = "test"
        mock_settings.redis_url = None
        mock_settings.git_sha = "abc123"

        # Patch all check functions to return mock results
        with patch("app.admin.services.health_checks.check_database") as mock_db, patch(
            "app.admin.services.health_checks.check_qdrant"
        ) as mock_qd, patch(
            "app.admin.services.health_checks.check_llm"
        ) as mock_llm, patch(
            "app.admin.services.health_checks.check_ingestion"
        ) as mock_ing, patch(
            "app.admin.services.health_checks.check_sse"
        ) as mock_sse, patch(
            "app.admin.services.health_checks.check_redis"
        ) as mock_redis, patch(
            "app.admin.services.health_checks.check_retention"
        ) as mock_ret, patch(
            "app.admin.services.health_checks.check_idempotency"
        ) as mock_idem, patch(
            "app.admin.services.health_checks.check_tunes"
        ) as mock_tunes, patch(
            "app.admin.services.health_checks.check_pine_repos"
        ) as mock_repos, patch(
            "app.admin.services.health_checks.check_pine_discovery"
        ) as mock_disc, patch(
            "app.admin.services.health_checks.check_pine_poller"
        ) as mock_poll:
            from app.admin.services.health_models import (
                DBHealth,
                IdempotencyHealth,
                IngestionHealth,
                LLMHealth,
                PineDiscoveryHealth,
                PinePollerHealth,
                PineReposHealth,
                QdrantHealth,
                RetentionHealth,
                SSEHealth,
                TuneHealth,
            )

            # All checks return OK
            mock_db.return_value = DBHealth(status="ok")
            mock_qd.return_value = QdrantHealth(status="ok")
            mock_llm.return_value = LLMHealth(status="ok", provider_configured=True)
            mock_ing.return_value = IngestionHealth(status="ok")
            mock_sse.return_value = SSEHealth(
                status="ok", bus_type="memory", subscribers=0
            )
            mock_redis.return_value = None  # Not configured
            mock_ret.return_value = RetentionHealth(status="ok")
            mock_idem.return_value = IdempotencyHealth(status="ok")
            mock_tunes.return_value = TuneHealth(status="ok")
            mock_repos.return_value = PineReposHealth(status="ok")
            mock_disc.return_value = PineDiscoveryHealth(status="ok")
            mock_poll.return_value = PinePollerHealth(status="ok", enabled=False)

            result = await collect_system_health(mock_settings, mock_pool)

        assert isinstance(result, SystemHealthSnapshot)
        assert result.overall_status == "ok"
        assert result.database is not None
        assert result.qdrant is not None
        assert result.llm is not None
        assert result.ingestion is not None
        assert result.sse is not None
        assert result.retention is not None
        assert result.idempotency is not None
        assert result.tunes is not None
        assert result.pine_repos is not None
        assert result.pine_discovery is not None
        assert result.pine_poller is not None
        assert result.timestamp is not None
        assert result.version is not None

    @pytest.mark.asyncio
    async def test_continues_after_check_exception(self):
        """collect_system_health captures exceptions and continues."""
        mock_pool = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.qdrant_collection = "test"
        mock_settings.redis_url = None
        mock_settings.git_sha = "abc123"

        with patch("app.admin.services.health_checks.check_database") as mock_db, patch(
            "app.admin.services.health_checks.check_qdrant"
        ) as mock_qd, patch(
            "app.admin.services.health_checks.check_llm"
        ) as mock_llm, patch(
            "app.admin.services.health_checks.check_ingestion"
        ) as mock_ing, patch(
            "app.admin.services.health_checks.check_sse"
        ) as mock_sse, patch(
            "app.admin.services.health_checks.check_redis"
        ) as mock_redis, patch(
            "app.admin.services.health_checks.check_retention"
        ) as mock_ret, patch(
            "app.admin.services.health_checks.check_idempotency"
        ) as mock_idem, patch(
            "app.admin.services.health_checks.check_tunes"
        ) as mock_tunes, patch(
            "app.admin.services.health_checks.check_pine_repos"
        ) as mock_repos, patch(
            "app.admin.services.health_checks.check_pine_discovery"
        ) as mock_disc, patch(
            "app.admin.services.health_checks.check_pine_poller"
        ) as mock_poll:
            from app.admin.services.health_models import (
                IdempotencyHealth,
                IngestionHealth,
                LLMHealth,
                PineDiscoveryHealth,
                PinePollerHealth,
                PineReposHealth,
                QdrantHealth,
                RetentionHealth,
                SSEHealth,
                TuneHealth,
            )

            # Database check raises exception
            mock_db.side_effect = Exception("Connection timeout")

            # All other checks return OK
            mock_qd.return_value = QdrantHealth(status="ok")
            mock_llm.return_value = LLMHealth(status="ok", provider_configured=True)
            mock_ing.return_value = IngestionHealth(status="ok")
            mock_sse.return_value = SSEHealth(
                status="ok", bus_type="memory", subscribers=0
            )
            mock_redis.return_value = None
            mock_ret.return_value = RetentionHealth(status="ok")
            mock_idem.return_value = IdempotencyHealth(status="ok")
            mock_tunes.return_value = TuneHealth(status="ok")
            mock_repos.return_value = PineReposHealth(status="ok")
            mock_disc.return_value = PineDiscoveryHealth(status="ok")
            mock_poll.return_value = PinePollerHealth(status="ok", enabled=False)

            result = await collect_system_health(mock_settings, mock_pool)

        # Should still return a complete snapshot
        assert isinstance(result, SystemHealthSnapshot)
        # Database should have error status from exception
        assert result.database.status == "error"
        assert "Connection timeout" in result.database.error
        # Overall should be error due to database failure
        assert result.overall_status == "error"
        # Other checks should still be present
        assert result.qdrant.status == "ok"

    @pytest.mark.asyncio
    async def test_overall_status_degraded_when_one_degraded(self):
        """Overall status is degraded when any check is degraded."""
        mock_pool = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.qdrant_collection = "test"
        mock_settings.redis_url = None
        mock_settings.git_sha = "abc123"

        with patch("app.admin.services.health_checks.check_database") as mock_db, patch(
            "app.admin.services.health_checks.check_qdrant"
        ) as mock_qd, patch(
            "app.admin.services.health_checks.check_llm"
        ) as mock_llm, patch(
            "app.admin.services.health_checks.check_ingestion"
        ) as mock_ing, patch(
            "app.admin.services.health_checks.check_sse"
        ) as mock_sse, patch(
            "app.admin.services.health_checks.check_redis"
        ) as mock_redis, patch(
            "app.admin.services.health_checks.check_retention"
        ) as mock_ret, patch(
            "app.admin.services.health_checks.check_idempotency"
        ) as mock_idem, patch(
            "app.admin.services.health_checks.check_tunes"
        ) as mock_tunes, patch(
            "app.admin.services.health_checks.check_pine_repos"
        ) as mock_repos, patch(
            "app.admin.services.health_checks.check_pine_discovery"
        ) as mock_disc, patch(
            "app.admin.services.health_checks.check_pine_poller"
        ) as mock_poll:
            from app.admin.services.health_models import (
                DBHealth,
                IdempotencyHealth,
                IngestionHealth,
                LLMHealth,
                PineDiscoveryHealth,
                PinePollerHealth,
                PineReposHealth,
                QdrantHealth,
                RetentionHealth,
                SSEHealth,
                TuneHealth,
            )

            mock_db.return_value = DBHealth(status="ok")
            mock_qd.return_value = QdrantHealth(status="ok")
            # LLM is degraded
            mock_llm.return_value = LLMHealth(
                status="degraded",
                provider_configured=True,
                degraded_count_1h=5,
            )
            mock_ing.return_value = IngestionHealth(status="ok")
            mock_sse.return_value = SSEHealth(
                status="ok", bus_type="memory", subscribers=0
            )
            mock_redis.return_value = None
            mock_ret.return_value = RetentionHealth(status="ok")
            mock_idem.return_value = IdempotencyHealth(status="ok")
            mock_tunes.return_value = TuneHealth(status="ok")
            mock_repos.return_value = PineReposHealth(status="ok")
            mock_disc.return_value = PineDiscoveryHealth(status="ok")
            mock_poll.return_value = PinePollerHealth(status="ok", enabled=False)

            result = await collect_system_health(mock_settings, mock_pool)

        assert result.overall_status == "degraded"
        assert result.components_degraded == 1
        assert result.components_ok >= 8  # Most are OK

    @pytest.mark.asyncio
    async def test_issues_list_populated_for_failing_checks(self):
        """Issues list contains descriptions of failing checks."""
        mock_pool = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.qdrant_collection = "test"
        mock_settings.redis_url = None
        mock_settings.git_sha = "abc123"

        with patch("app.admin.services.health_checks.check_database") as mock_db, patch(
            "app.admin.services.health_checks.check_qdrant"
        ) as mock_qd, patch(
            "app.admin.services.health_checks.check_llm"
        ) as mock_llm, patch(
            "app.admin.services.health_checks.check_ingestion"
        ) as mock_ing, patch(
            "app.admin.services.health_checks.check_sse"
        ) as mock_sse, patch(
            "app.admin.services.health_checks.check_redis"
        ) as mock_redis, patch(
            "app.admin.services.health_checks.check_retention"
        ) as mock_ret, patch(
            "app.admin.services.health_checks.check_idempotency"
        ) as mock_idem, patch(
            "app.admin.services.health_checks.check_tunes"
        ) as mock_tunes, patch(
            "app.admin.services.health_checks.check_pine_repos"
        ) as mock_repos, patch(
            "app.admin.services.health_checks.check_pine_discovery"
        ) as mock_disc, patch(
            "app.admin.services.health_checks.check_pine_poller"
        ) as mock_poll:
            from app.admin.services.health_models import (
                DBHealth,
                IdempotencyHealth,
                IngestionHealth,
                LLMHealth,
                PineDiscoveryHealth,
                PinePollerHealth,
                PineReposHealth,
                QdrantHealth,
                RetentionHealth,
                SSEHealth,
                TuneHealth,
            )

            # Database has error
            mock_db.return_value = DBHealth(status="error", error="Connection refused")
            # Qdrant has error
            mock_qd.return_value = QdrantHealth(status="error", error="Unreachable")
            mock_llm.return_value = LLMHealth(status="ok", provider_configured=True)
            mock_ing.return_value = IngestionHealth(status="ok")
            mock_sse.return_value = SSEHealth(
                status="ok", bus_type="memory", subscribers=0
            )
            mock_redis.return_value = None
            mock_ret.return_value = RetentionHealth(status="ok")
            mock_idem.return_value = IdempotencyHealth(status="ok")
            mock_tunes.return_value = TuneHealth(status="ok")
            mock_repos.return_value = PineReposHealth(status="ok")
            mock_disc.return_value = PineDiscoveryHealth(status="ok")
            mock_poll.return_value = PinePollerHealth(status="ok", enabled=False)

            result = await collect_system_health(mock_settings, mock_pool)

        assert result.overall_status == "error"
        assert len(result.issues) >= 2
        # Check that issues contain relevant info
        issues_text = " ".join(result.issues)
        assert "Database" in issues_text
        assert "Qdrant" in issues_text

    @pytest.mark.asyncio
    async def test_error_takes_precedence_over_degraded(self):
        """Overall status is error even if some checks are only degraded."""
        mock_pool = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        mock_settings.qdrant_collection = "test"
        mock_settings.redis_url = None
        mock_settings.git_sha = "abc123"

        with patch("app.admin.services.health_checks.check_database") as mock_db, patch(
            "app.admin.services.health_checks.check_qdrant"
        ) as mock_qd, patch(
            "app.admin.services.health_checks.check_llm"
        ) as mock_llm, patch(
            "app.admin.services.health_checks.check_ingestion"
        ) as mock_ing, patch(
            "app.admin.services.health_checks.check_sse"
        ) as mock_sse, patch(
            "app.admin.services.health_checks.check_redis"
        ) as mock_redis, patch(
            "app.admin.services.health_checks.check_retention"
        ) as mock_ret, patch(
            "app.admin.services.health_checks.check_idempotency"
        ) as mock_idem, patch(
            "app.admin.services.health_checks.check_tunes"
        ) as mock_tunes, patch(
            "app.admin.services.health_checks.check_pine_repos"
        ) as mock_repos, patch(
            "app.admin.services.health_checks.check_pine_discovery"
        ) as mock_disc, patch(
            "app.admin.services.health_checks.check_pine_poller"
        ) as mock_poll:
            from app.admin.services.health_models import (
                DBHealth,
                IdempotencyHealth,
                IngestionHealth,
                LLMHealth,
                PineDiscoveryHealth,
                PinePollerHealth,
                PineReposHealth,
                QdrantHealth,
                RetentionHealth,
                SSEHealth,
                TuneHealth,
            )

            # Database is degraded
            mock_db.return_value = DBHealth(status="degraded")
            # Qdrant has error (more severe)
            mock_qd.return_value = QdrantHealth(status="error", error="Down")
            mock_llm.return_value = LLMHealth(status="ok", provider_configured=True)
            mock_ing.return_value = IngestionHealth(status="ok")
            mock_sse.return_value = SSEHealth(
                status="ok", bus_type="memory", subscribers=0
            )
            mock_redis.return_value = None
            mock_ret.return_value = RetentionHealth(status="ok")
            mock_idem.return_value = IdempotencyHealth(status="ok")
            mock_tunes.return_value = TuneHealth(status="ok")
            mock_repos.return_value = PineReposHealth(status="ok")
            mock_disc.return_value = PineDiscoveryHealth(status="ok")
            mock_poll.return_value = PinePollerHealth(status="ok", enabled=False)

            result = await collect_system_health(mock_settings, mock_pool)

        # Error should take precedence
        assert result.overall_status == "error"
        assert result.components_error == 1
        assert result.components_degraded == 1
