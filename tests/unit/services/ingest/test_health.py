"""Unit tests for source health validation service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.services.ingest.health import (
    HealthCheckResult,
    HealthResult,
    HealthStatus,
    get_source_health_summary,
    validate_source_health,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def workspace_id():
    """Fixed workspace ID for testing."""
    return UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def source_id():
    """Fixed source ID for testing."""
    return UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


def create_mock_conn(doc_row=None, chunk_count=0, vector_count=0):
    """Create a mock connection with configurable return values."""
    conn = MagicMock()

    async def mock_fetchrow(query, *args):
        if "FROM documents" in query:
            return doc_row
        elif "FROM chunks" in query:
            return {"count": chunk_count}
        elif "FROM chunk_vectors" in query:
            return {"count": vector_count}
        return None

    conn.fetchrow = AsyncMock(side_effect=mock_fetchrow)
    return conn


# =============================================================================
# Test validate_source_health
# =============================================================================


class TestValidateSourceHealth:
    """Tests for validate_source_health function."""

    @pytest.mark.asyncio
    async def test_document_not_found_returns_failed(
        self, mock_pool, workspace_id, source_id
    ):
        """When document doesn't exist, return FAILED status."""
        mock_conn = create_mock_conn(doc_row=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(mock_pool, workspace_id, source_id)

        assert result.status == HealthStatus.FAILED
        assert result.error_code == "document_not_found"
        assert not result.is_healthy
        assert len(result.checks) == 1
        assert result.checks[0].name == "document_exists"
        assert not result.checks[0].passed

    @pytest.mark.asyncio
    async def test_no_chunks_returns_failed(self, mock_pool, workspace_id, source_id):
        """When document has no chunks, return FAILED status."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=0, vector_count=0)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(mock_pool, workspace_id, source_id)

        assert result.status == HealthStatus.FAILED
        assert not result.chunk_count_ok
        # Find chunk_count check
        chunk_check = next(c for c in result.checks if c.name == "chunk_count")
        assert not chunk_check.passed
        assert "No chunks found" in chunk_check.message

    @pytest.mark.asyncio
    async def test_missing_embeddings_returns_failed_when_required(
        self, mock_pool, workspace_id, source_id
    ):
        """When embeddings are missing and required, return FAILED status."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=5, vector_count=0)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(
            mock_pool, workspace_id, source_id, require_embeddings=True
        )

        assert result.status == HealthStatus.FAILED
        assert not result.embeddings_ok
        # Find embeddings check
        embed_check = next(c for c in result.checks if c.name == "embeddings_parity")
        assert not embed_check.passed

    @pytest.mark.asyncio
    async def test_missing_embeddings_returns_degraded_when_not_required(
        self, mock_pool, workspace_id, source_id
    ):
        """When embeddings are missing but not required, return DEGRADED status."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=5, vector_count=0)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(
            mock_pool, workspace_id, source_id, require_embeddings=False
        )

        assert result.status == HealthStatus.DEGRADED
        assert not result.embeddings_ok

    @pytest.mark.asyncio
    async def test_all_checks_pass_returns_ok(self, mock_pool, workspace_id, source_id):
        """When all checks pass, return OK status."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=10, vector_count=10)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(mock_pool, workspace_id, source_id)

        assert result.status == HealthStatus.OK
        assert result.is_healthy
        assert result.chunk_count_ok
        assert result.embeddings_ok
        assert all(c.passed for c in result.checks)

    @pytest.mark.asyncio
    async def test_expected_chunk_count_mismatch_returns_failed(
        self, mock_pool, workspace_id, source_id
    ):
        """When actual chunk count doesn't match expected, fail."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=5, vector_count=5)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(
            mock_pool, workspace_id, source_id, expected_chunk_count=10
        )

        assert result.status == HealthStatus.FAILED
        assert not result.chunk_count_ok
        chunk_check = next(c for c in result.checks if c.name == "chunk_count")
        assert not chunk_check.passed
        assert "mismatch" in chunk_check.message.lower()
        assert chunk_check.expected == 10
        assert chunk_check.actual == 5

    @pytest.mark.asyncio
    async def test_expected_vector_count_mismatch_returns_failed(
        self, mock_pool, workspace_id, source_id
    ):
        """When actual vector count doesn't match expected, fail."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=10, vector_count=5)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(
            mock_pool,
            workspace_id,
            source_id,
            expected_vector_count=10,
            require_embeddings=True,
        )

        assert result.status == HealthStatus.FAILED
        assert not result.embeddings_ok
        embed_check = next(c for c in result.checks if c.name == "embeddings_parity")
        assert not embed_check.passed
        assert "mismatch" in embed_check.message.lower()

    @pytest.mark.asyncio
    async def test_missing_indexed_timestamp_returns_degraded(
        self, mock_pool, workspace_id, source_id
    ):
        """When last_indexed_at is not set, return DEGRADED."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": None,  # Not indexed yet
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=10, vector_count=10)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await validate_source_health(mock_pool, workspace_id, source_id)

        # Should be DEGRADED because indexed_timestamp failed but it's non-critical
        assert result.status == HealthStatus.DEGRADED
        indexed_check = next(c for c in result.checks if c.name == "indexed_timestamp")
        assert not indexed_check.passed


# =============================================================================
# Test HealthResult properties
# =============================================================================


class TestHealthResultProperties:
    """Tests for HealthResult dataclass properties."""

    def test_is_healthy_true_when_ok(self, workspace_id, source_id):
        """is_healthy returns True when status is OK."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.OK,
            checks=[],
        )
        assert result.is_healthy

    def test_is_healthy_false_when_degraded(self, workspace_id, source_id):
        """is_healthy returns False when status is DEGRADED."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.DEGRADED,
            checks=[],
        )
        assert not result.is_healthy

    def test_is_healthy_false_when_failed(self, workspace_id, source_id):
        """is_healthy returns False when status is FAILED."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.FAILED,
            checks=[],
        )
        assert not result.is_healthy

    def test_chunk_count_ok_true_when_check_passes(self, workspace_id, source_id):
        """chunk_count_ok returns True when chunk_count check passes."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.OK,
            checks=[
                HealthCheckResult(name="chunk_count", passed=True, message="OK"),
            ],
        )
        assert result.chunk_count_ok

    def test_chunk_count_ok_false_when_check_fails(self, workspace_id, source_id):
        """chunk_count_ok returns False when chunk_count check fails."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.FAILED,
            checks=[
                HealthCheckResult(
                    name="chunk_count", passed=False, message="No chunks"
                ),
            ],
        )
        assert not result.chunk_count_ok

    def test_embeddings_ok_true_when_check_passes(self, workspace_id, source_id):
        """embeddings_ok returns True when embeddings_parity check passes."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.OK,
            checks=[
                HealthCheckResult(name="embeddings_parity", passed=True, message="OK"),
            ],
        )
        assert result.embeddings_ok

    def test_embeddings_ok_false_when_no_check(self, workspace_id, source_id):
        """embeddings_ok returns False when no embeddings_parity check exists."""
        result = HealthResult(
            source_id=source_id,
            workspace_id=workspace_id,
            status=HealthStatus.OK,
            checks=[],
        )
        assert not result.embeddings_ok


# =============================================================================
# Test get_source_health_summary
# =============================================================================


class TestGetSourceHealthSummary:
    """Tests for get_source_health_summary function."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_expected_keys(
        self, mock_pool, workspace_id, source_id
    ):
        """Summary returns dict with status, chunk_count_ok, embeddings_ok, checks."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=5, vector_count=5)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        summary = await get_source_health_summary(mock_pool, workspace_id, source_id)

        assert "status" in summary
        assert "chunk_count_ok" in summary
        assert "embeddings_ok" in summary
        assert "checks" in summary
        assert isinstance(summary["checks"], list)

    @pytest.mark.asyncio
    async def test_summary_checks_have_expected_fields(
        self, mock_pool, workspace_id, source_id
    ):
        """Each check in summary has name, passed, message fields."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=5, vector_count=5)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        summary = await get_source_health_summary(mock_pool, workspace_id, source_id)

        for check in summary["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "message" in check

    @pytest.mark.asyncio
    async def test_summary_uses_require_embeddings_false(
        self, mock_pool, workspace_id, source_id
    ):
        """Summary uses require_embeddings=False (report, don't fail)."""
        doc_row = {
            "id": source_id,
            "status": "active",
            "last_indexed_at": datetime.now(timezone.utc),
        }
        # Missing embeddings - should be DEGRADED not FAILED
        mock_conn = create_mock_conn(doc_row=doc_row, chunk_count=5, vector_count=0)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        summary = await get_source_health_summary(mock_pool, workspace_id, source_id)

        # Should be degraded, not failed, because summary uses require_embeddings=False
        assert summary["status"] == "degraded"
        assert not summary["embeddings_ok"]


# =============================================================================
# Test HealthCheckResult
# =============================================================================


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_create_basic_check(self):
        """Can create a basic check with required fields."""
        check = HealthCheckResult(
            name="test_check",
            passed=True,
            message="Test passed",
        )
        assert check.name == "test_check"
        assert check.passed is True
        assert check.message == "Test passed"
        assert check.expected is None
        assert check.actual is None

    def test_create_check_with_counts(self):
        """Can create a check with expected/actual counts."""
        check = HealthCheckResult(
            name="chunk_count",
            passed=False,
            message="Mismatch",
            expected=10,
            actual=5,
        )
        assert check.expected == 10
        assert check.actual == 5
