"""Tests for job advisory lock utilities."""

from uuid import uuid4

from app.services.jobs.locks import job_lock_key


class TestJobLockKey:
    """Tests for lock key generation."""

    def test_deterministic_key(self):
        """Same inputs produce same key."""
        workspace_id = uuid4()
        key1 = job_lock_key("rollup_events", workspace_id)
        key2 = job_lock_key("rollup_events", workspace_id)
        assert key1 == key2

    def test_different_jobs_different_keys(self):
        """Different job names produce different keys."""
        workspace_id = uuid4()
        key1 = job_lock_key("rollup_events", workspace_id)
        key2 = job_lock_key("cleanup_events", workspace_id)
        assert key1 != key2

    def test_different_workspaces_different_keys(self):
        """Different workspaces produce different keys."""
        key1 = job_lock_key("rollup_events", uuid4())
        key2 = job_lock_key("rollup_events", uuid4())
        assert key1 != key2

    def test_key_is_unsigned_int(self):
        """Key is unsigned 64-bit integer."""
        key = job_lock_key("test", uuid4())
        assert isinstance(key, int)
        assert key >= 0
