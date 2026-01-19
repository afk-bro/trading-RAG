"""Tests for job registry."""

import pytest
from app.jobs.registry import JobRegistry
from app.jobs.types import JobType


class TestJobRegistry:
    def test_register_handler(self):
        registry = JobRegistry()

        async def dummy_handler(job, ctx):
            return {"ok": True}

        registry.register(JobType.DATA_FETCH, dummy_handler)
        assert registry.get_handler(JobType.DATA_FETCH) == dummy_handler

    def test_get_unregistered_handler_raises(self):
        registry = JobRegistry()
        with pytest.raises(KeyError):
            registry.get_handler(JobType.DATA_FETCH)

    def test_decorator_registration(self):
        registry = JobRegistry()

        @registry.handler(JobType.TUNE)
        async def tune_handler(job, ctx):
            pass

        assert registry.get_handler(JobType.TUNE) == tune_handler
