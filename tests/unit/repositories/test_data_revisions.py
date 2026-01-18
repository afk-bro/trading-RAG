"""Tests for data revisions repository."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from app.repositories.data_revisions import DataRevisionRepository, DataRevision


class TestDataRevision:
    """Tests for DataRevision dataclass."""

    def test_datarevision_creation(self):
        """Test creating a DataRevision instance."""
        revision = DataRevision(
            id=1,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
            row_count=4320,
            checksum="a1b2c3d4e5f6",
            computed_at=datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc),
            job_id="job-123",
        )
        assert revision.exchange_id == "kucoin"
        assert revision.symbol == "BTC-USDT"
        assert revision.timeframe == "1h"
        assert revision.row_count == 4320
        assert revision.checksum == "a1b2c3d4e5f6"
        assert revision.job_id == "job-123"

    def test_datarevision_optional_job_id(self):
        """Test DataRevision with no job_id."""
        revision = DataRevision(
            id=1,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
            row_count=100,
            checksum="checksum123",
            computed_at=datetime.now(timezone.utc),
        )
        assert revision.job_id is None


class TestDataRevisionRepository:
    """Tests for DataRevisionRepository."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        return MagicMock()

    def test_repository_creation(self, mock_pool):
        """Test creating a repository instance."""
        repo = DataRevisionRepository(mock_pool)
        assert repo._pool == mock_pool

    @pytest.mark.asyncio
    async def test_upsert(self, mock_pool):
        """Test upserting a data revision."""
        # Setup mock
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        fake_row = {
            "id": 1,
            "exchange_id": "kucoin",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "row_count": 4320,
            "checksum": "a1b2c3d4e5f6",
            "computed_at": datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc),
            "job_id": "job-123",
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = DataRevisionRepository(mock_pool)
        result = await repo.upsert(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
            row_count=4320,
            checksum="a1b2c3d4e5f6",
            job_id="job-123",
        )

        assert result.exchange_id == "kucoin"
        assert result.checksum == "a1b2c3d4e5f6"
        assert result.row_count == 4320
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_found(self, mock_pool):
        """Test getting an existing data revision."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        fake_row = {
            "id": 1,
            "exchange_id": "kucoin",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "row_count": 4320,
            "checksum": "a1b2c3d4e5f6",
            "computed_at": datetime.now(timezone.utc),
            "job_id": None,
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = DataRevisionRepository(mock_pool)
        result = await repo.get(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        assert result is not None
        assert result.checksum == "a1b2c3d4e5f6"

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_pool):
        """Test getting a non-existent data revision."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo = DataRevisionRepository(mock_pool)
        result = await repo.get(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_has_changed_no_existing(self, mock_pool):
        """Test has_changed when no revision exists."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo = DataRevisionRepository(mock_pool)
        result = await repo.has_changed(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
            checksum="newchecksum",
        )

        assert result is True  # New data = changed

    @pytest.mark.asyncio
    async def test_has_changed_same_checksum(self, mock_pool):
        """Test has_changed with matching checksum."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        fake_row = {
            "id": 1,
            "exchange_id": "kucoin",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "row_count": 4320,
            "checksum": "samechecksum",
            "computed_at": datetime.now(timezone.utc),
            "job_id": None,
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = DataRevisionRepository(mock_pool)
        result = await repo.has_changed(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
            checksum="samechecksum",
        )

        assert result is False  # Same checksum = not changed

    @pytest.mark.asyncio
    async def test_has_changed_different_checksum(self, mock_pool):
        """Test has_changed with different checksum."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        fake_row = {
            "id": 1,
            "exchange_id": "kucoin",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "row_count": 4320,
            "checksum": "oldchecksum",
            "computed_at": datetime.now(timezone.utc),
            "job_id": None,
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = DataRevisionRepository(mock_pool)
        result = await repo.has_changed(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 7, 1, tzinfo=timezone.utc),
            checksum="newchecksum",
        )

        assert result is True  # Different checksum = changed

    @pytest.mark.asyncio
    async def test_get_latest(self, mock_pool):
        """Test getting the most recent revision."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        fake_row = {
            "id": 5,
            "exchange_id": "kucoin",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "start_ts": datetime(2024, 6, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "row_count": 720,
            "checksum": "latestchecksum",
            "computed_at": datetime(2024, 7, 15, tzinfo=timezone.utc),
            "job_id": None,
        }
        mock_conn.fetchrow.return_value = fake_row

        repo = DataRevisionRepository(mock_pool)
        result = await repo.get_latest(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
        )

        assert result is not None
        assert result.id == 5
        assert result.checksum == "latestchecksum"

    @pytest.mark.asyncio
    async def test_list_for_symbol(self, mock_pool):
        """Test listing revisions for a symbol."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        fake_rows = [
            {
                "id": 2,
                "exchange_id": "kucoin",
                "symbol": "BTC-USDT",
                "timeframe": "1h",
                "start_ts": datetime(2024, 6, 1, tzinfo=timezone.utc),
                "end_ts": datetime(2024, 7, 1, tzinfo=timezone.utc),
                "row_count": 720,
                "checksum": "checksum2",
                "computed_at": datetime(2024, 7, 10, tzinfo=timezone.utc),
                "job_id": None,
            },
            {
                "id": 1,
                "exchange_id": "kucoin",
                "symbol": "BTC-USDT",
                "timeframe": "1h",
                "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "end_ts": datetime(2024, 6, 1, tzinfo=timezone.utc),
                "row_count": 3600,
                "checksum": "checksum1",
                "computed_at": datetime(2024, 6, 1, tzinfo=timezone.utc),
                "job_id": None,
            },
        ]
        mock_conn.fetch.return_value = fake_rows

        repo = DataRevisionRepository(mock_pool)
        results = await repo.list_for_symbol(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            limit=10,
        )

        assert len(results) == 2
        assert results[0].id == 2
        assert results[1].id == 1
