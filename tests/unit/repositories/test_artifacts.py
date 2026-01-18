"""Tests for artifact repository."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

import pytest

from app.repositories.artifacts import ArtifactRepository, Artifact


class TestArtifact:
    """Tests for Artifact dataclass."""

    def test_artifact_creation(self):
        """Test creating an Artifact instance with all fields."""
        artifact_id = uuid4()
        workspace_id = uuid4()
        run_id = uuid4()
        job_id = uuid4()
        created_at = datetime.now(timezone.utc)

        artifact = Artifact(
            id=artifact_id,
            workspace_id=workspace_id,
            run_id=run_id,
            job_type="tune",
            artifact_kind="tune_json",
            artifact_path="artifacts/tunes/abc123/tune.json",
            job_id=job_id,
            file_size_bytes=1024,
            data_revision={"checksum": "abc123", "row_count": 100},
            is_pinned=False,
            pinned_at=None,
            pinned_by=None,
            created_at=created_at,
        )

        assert artifact.id == artifact_id
        assert artifact.workspace_id == workspace_id
        assert artifact.run_id == run_id
        assert artifact.job_type == "tune"
        assert artifact.artifact_kind == "tune_json"
        assert artifact.artifact_path == "artifacts/tunes/abc123/tune.json"
        assert artifact.job_id == job_id
        assert artifact.file_size_bytes == 1024
        assert artifact.data_revision == {"checksum": "abc123", "row_count": 100}
        assert artifact.is_pinned is False
        assert artifact.pinned_at is None
        assert artifact.pinned_by is None
        assert artifact.created_at == created_at

    def test_artifact_optional_fields(self):
        """Test Artifact with minimal required fields only."""
        artifact = Artifact(
            id=uuid4(),
            workspace_id=uuid4(),
            run_id=uuid4(),
            job_type="wfo",
            artifact_kind="wfo_json",
            artifact_path="artifacts/wfo/def456/wfo.json",
        )

        assert artifact.job_id is None
        assert artifact.file_size_bytes is None
        assert artifact.data_revision is None
        assert artifact.is_pinned is False
        assert artifact.pinned_at is None
        assert artifact.pinned_by is None
        assert artifact.created_at is None

    def test_artifact_pinned_fields(self):
        """Test Artifact with pinned metadata."""
        pinned_at = datetime.now(timezone.utc)
        artifact = Artifact(
            id=uuid4(),
            workspace_id=uuid4(),
            run_id=uuid4(),
            job_type="tune",
            artifact_kind="trials_csv",
            artifact_path="artifacts/tunes/abc123/trials.csv",
            is_pinned=True,
            pinned_at=pinned_at,
            pinned_by="admin@example.com",
        )

        assert artifact.is_pinned is True
        assert artifact.pinned_at == pinned_at
        assert artifact.pinned_by == "admin@example.com"


class TestArtifactRepository:
    """Tests for ArtifactRepository."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        return MagicMock()

    def test_repository_creation(self, mock_pool):
        """Test creating a repository instance."""
        repo = ArtifactRepository(mock_pool)
        assert repo._pool == mock_pool

    @pytest.mark.asyncio
    async def test_create_artifact(self, mock_pool):
        """Test creating a new artifact record."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        artifact_id = uuid4()
        workspace_id = uuid4()
        run_id = uuid4()
        job_id = uuid4()
        created_at = datetime.now(timezone.utc)

        fake_row = {
            "id": artifact_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "job_type": "tune",
            "artifact_kind": "tune_json",
            "artifact_path": "artifacts/tunes/abc123/tune.json",
            "job_id": job_id,
            "file_size_bytes": 2048,
            "data_revision": {"checksum": "xyz789"},
            "is_pinned": False,
            "pinned_at": None,
            "pinned_by": None,
            "created_at": created_at,
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = ArtifactRepository(mock_pool)
        result = await repo.create(
            workspace_id=workspace_id,
            run_id=run_id,
            job_type="tune",
            artifact_kind="tune_json",
            artifact_path="artifacts/tunes/abc123/tune.json",
            job_id=job_id,
            file_size_bytes=2048,
            data_revision={"checksum": "xyz789"},
        )

        assert result.id == artifact_id
        assert result.workspace_id == workspace_id
        assert result.run_id == run_id
        assert result.job_type == "tune"
        assert result.artifact_kind == "tune_json"
        assert result.file_size_bytes == 2048
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_artifact_minimal(self, mock_pool):
        """Test creating artifact with minimal required fields."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        artifact_id = uuid4()
        workspace_id = uuid4()
        run_id = uuid4()

        fake_row = {
            "id": artifact_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "job_type": "wfo",
            "artifact_kind": "wfo_json",
            "artifact_path": "artifacts/wfo/xyz/wfo.json",
            "job_id": None,
            "file_size_bytes": None,
            "data_revision": None,
            "is_pinned": False,
            "pinned_at": None,
            "pinned_by": None,
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = ArtifactRepository(mock_pool)
        result = await repo.create(
            workspace_id=workspace_id,
            run_id=run_id,
            job_type="wfo",
            artifact_kind="wfo_json",
            artifact_path="artifacts/wfo/xyz/wfo.json",
        )

        assert result.id == artifact_id
        assert result.job_id is None
        assert result.file_size_bytes is None
        assert result.data_revision is None
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_run(self, mock_pool):
        """Test getting all artifacts for a run."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run_id = uuid4()
        workspace_id = uuid4()

        fake_rows = [
            {
                "id": uuid4(),
                "workspace_id": workspace_id,
                "run_id": run_id,
                "job_type": "tune",
                "artifact_kind": "tune_json",
                "artifact_path": "artifacts/tunes/abc123/tune.json",
                "job_id": None,
                "file_size_bytes": 1024,
                "data_revision": None,
                "is_pinned": False,
                "pinned_at": None,
                "pinned_by": None,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": uuid4(),
                "workspace_id": workspace_id,
                "run_id": run_id,
                "job_type": "tune",
                "artifact_kind": "trials_csv",
                "artifact_path": "artifacts/tunes/abc123/trials.csv",
                "job_id": None,
                "file_size_bytes": 4096,
                "data_revision": None,
                "is_pinned": True,
                "pinned_at": datetime.now(timezone.utc),
                "pinned_by": "user@example.com",
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_conn.fetch.return_value = fake_rows

        repo = ArtifactRepository(mock_pool)
        results = await repo.get_by_run(run_id)

        assert len(results) == 2
        assert results[0].artifact_kind == "tune_json"
        assert results[1].artifact_kind == "trials_csv"
        assert results[1].is_pinned is True
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_run_empty(self, mock_pool):
        """Test getting artifacts for a run with no results."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = []

        repo = ArtifactRepository(mock_pool)
        results = await repo.get_by_run(uuid4())

        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_kind(self, mock_pool):
        """Test getting specific artifact by kind."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run_id = uuid4()
        artifact_id = uuid4()

        fake_row = {
            "id": artifact_id,
            "workspace_id": uuid4(),
            "run_id": run_id,
            "job_type": "tune",
            "artifact_kind": "equity_csv",
            "artifact_path": "artifacts/tunes/abc123/equity.csv",
            "job_id": None,
            "file_size_bytes": 2048,
            "data_revision": None,
            "is_pinned": False,
            "pinned_at": None,
            "pinned_by": None,
            "created_at": datetime.now(timezone.utc),
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = ArtifactRepository(mock_pool)
        result = await repo.get_by_kind(run_id, "equity_csv")

        assert result is not None
        assert result.id == artifact_id
        assert result.artifact_kind == "equity_csv"
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_kind_not_found(self, mock_pool):
        """Test getting artifact by kind when not found."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo = ArtifactRepository(mock_pool)
        result = await repo.get_by_kind(uuid4(), "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_pin_artifact(self, mock_pool):
        """Test pinning an artifact."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = True  # Row was updated

        repo = ArtifactRepository(mock_pool)
        result = await repo.pin(uuid4(), pinned_by="admin@example.com")

        assert result is True
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_pin_artifact_not_found(self, mock_pool):
        """Test pinning a nonexistent artifact."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = False  # No row updated

        repo = ArtifactRepository(mock_pool)
        result = await repo.pin(uuid4(), pinned_by="admin@example.com")

        assert result is False

    @pytest.mark.asyncio
    async def test_unpin_artifact(self, mock_pool):
        """Test unpinning an artifact."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = True  # Row was updated

        repo = ArtifactRepository(mock_pool)
        result = await repo.unpin(uuid4())

        assert result is True
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_unpin_artifact_not_found(self, mock_pool):
        """Test unpinning a nonexistent artifact."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = False  # No row updated

        repo = ArtifactRepository(mock_pool)
        result = await repo.unpin(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_list_unpinned_older_than(self, mock_pool):
        """Test listing unpinned artifacts older than N days."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        fake_rows = [
            {
                "id": uuid4(),
                "workspace_id": uuid4(),
                "run_id": uuid4(),
                "job_type": "tune",
                "artifact_kind": "tune_json",
                "artifact_path": "artifacts/tunes/old1/tune.json",
                "job_id": None,
                "file_size_bytes": 1024,
                "data_revision": None,
                "is_pinned": False,
                "pinned_at": None,
                "pinned_by": None,
                "created_at": old_date,
            },
            {
                "id": uuid4(),
                "workspace_id": uuid4(),
                "run_id": uuid4(),
                "job_type": "wfo",
                "artifact_kind": "wfo_json",
                "artifact_path": "artifacts/wfo/old2/wfo.json",
                "job_id": None,
                "file_size_bytes": 2048,
                "data_revision": None,
                "is_pinned": False,
                "pinned_at": None,
                "pinned_by": None,
                "created_at": old_date - timedelta(days=10),
            },
        ]
        mock_conn.fetch.return_value = fake_rows

        repo = ArtifactRepository(mock_pool)
        results = await repo.list_unpinned_older_than(days=30)

        assert len(results) == 2
        assert results[0].is_pinned is False
        assert results[1].is_pinned is False
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_unpinned_older_than_empty(self, mock_pool):
        """Test listing unpinned artifacts when none qualify."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = []

        repo = ArtifactRepository(mock_pool)
        results = await repo.list_unpinned_older_than(days=30)

        assert results == []

    @pytest.mark.asyncio
    async def test_delete_artifact(self, mock_pool):
        """Test deleting an artifact record."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = True  # Row was deleted

        repo = ArtifactRepository(mock_pool)
        result = await repo.delete(uuid4())

        assert result is True
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_artifact_not_found(self, mock_pool):
        """Test deleting a nonexistent artifact."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = False  # No row deleted

        repo = ArtifactRepository(mock_pool)
        result = await repo.delete(uuid4())

        assert result is False
