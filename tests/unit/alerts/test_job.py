"""Tests for alert evaluator job."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.alerts.job import AlertEvaluatorJob


class TestAlertEvaluatorJob:
    """Tests for scheduled alert evaluation."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.fixture
    def mock_conn(self):
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=True)  # Lock acquired
        conn.fetch = AsyncMock(return_value=[])
        return conn

    @pytest.mark.asyncio
    async def test_job_acquires_lock(self, mock_pool, mock_conn):
        """Job acquires advisory lock."""
        workspace_id = uuid4()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        job = AlertEvaluatorJob(mock_pool)
        result = await job.run(workspace_id=workspace_id)

        assert result["lock_acquired"] is True
        mock_conn.fetchval.assert_called()

    @pytest.mark.asyncio
    async def test_job_returns_metrics(self, mock_pool, mock_conn):
        """Job returns evaluation metrics."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": rule_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": None,
                    "regime_key": None,
                    "timeframe": "1h",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                    "cooldown_minutes": 60,
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        job = AlertEvaluatorJob(mock_pool)

        with patch.object(job, "_fetch_buckets", return_value=[]):
            result = await job.run(workspace_id=workspace_id)

        assert "rules_loaded" in result["metrics"]
        assert result["metrics"]["rules_loaded"] == 1

    @pytest.mark.asyncio
    async def test_job_skips_insufficient_data(self, mock_pool, mock_conn):
        """Job counts skipped evaluations."""
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": rule_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": strategy_id,
                    "regime_key": "high_vol",
                    "timeframe": "1h",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30, "consecutive_buckets": 5},
                    "cooldown_minutes": 60,
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        job = AlertEvaluatorJob(mock_pool)

        # Return only 2 buckets, but rule needs 5
        with patch.object(
            job,
            "_fetch_buckets",
            return_value=[
                MagicMock(drift_score=0.35, avg_confidence=0.7),
                MagicMock(drift_score=0.35, avg_confidence=0.7),
            ],
        ):
            result = await job.run(workspace_id=workspace_id)

        assert result["metrics"]["tuples_skipped_insufficient_data"] >= 1
