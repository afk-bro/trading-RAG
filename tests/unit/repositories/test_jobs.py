"""Tests for job repository."""

from unittest.mock import MagicMock

from app.repositories.jobs import JobRepository


class TestJobRepository:
    def test_repository_creation(self):
        mock_pool = MagicMock()
        repo = JobRepository(mock_pool)
        assert repo._pool == mock_pool

    def test_backoff_calculation(self):
        repo = JobRepository(MagicMock())
        # Formula: min(300, 2^attempt * 5) + jitter
        # jitter = random.randint(0, min(10, base // 2))

        # First retry (attempt=1): 2^1 * 5 = 10, jitter up to 5
        b1 = repo._calculate_backoff(1)
        assert 10 <= b1 <= 15  # 10 + jitter(0-5)

        # Second retry (attempt=2): 2^2 * 5 = 20, jitter up to 10
        b2 = repo._calculate_backoff(2)
        assert 20 <= b2 <= 30  # 20 + jitter(0-10)

        # Max backoff (attempt=10): capped at 300, jitter up to 10
        b_max = repo._calculate_backoff(10)
        assert 300 <= b_max <= 310  # 300 + jitter(0-10)
