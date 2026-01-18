"""Tests for job worker."""
import socket
import os
from unittest.mock import MagicMock

from app.jobs.worker import WorkerRunner, generate_worker_id


class TestGenerateWorkerId:
    def test_worker_id_format(self):
        worker_id = generate_worker_id()
        # Format: hostname:pid
        assert ":" in worker_id
        hostname, pid = worker_id.split(":")
        assert hostname == socket.gethostname()
        assert pid == str(os.getpid())


class TestWorkerRunner:
    def test_worker_creation(self):
        mock_pool = MagicMock()
        worker = WorkerRunner(mock_pool)
        assert worker._pool == mock_pool
        assert worker._running is False
