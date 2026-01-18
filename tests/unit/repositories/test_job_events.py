"""Tests for job events repository."""

from unittest.mock import MagicMock

from app.repositories.job_events import JobEventsRepository


class TestJobEventsRepository:
    def test_repository_creation(self):
        mock_pool = MagicMock()
        repo = JobEventsRepository(mock_pool)
        assert repo._pool == mock_pool
