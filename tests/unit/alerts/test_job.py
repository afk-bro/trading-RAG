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

    @pytest.mark.asyncio
    async def test_job_continues_after_rule_error(self, mock_pool, mock_conn):
        """Job continues processing after one rule fails."""
        workspace_id = uuid4()
        rule_id_1 = uuid4()
        rule_id_2 = uuid4()
        strategy_id = uuid4()

        # Two rules: first will fail, second should still be processed
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": rule_id_1,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": strategy_id,
                    "regime_key": "high_vol",
                    "timeframe": "1h",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                    "cooldown_minutes": 60,
                },
                {
                    "id": rule_id_2,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": strategy_id,
                    "regime_key": "low_vol",
                    "timeframe": "1h",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                    "cooldown_minutes": 60,
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        job = AlertEvaluatorJob(mock_pool)

        call_count = 0

        async def failing_fetch_buckets(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated failure for first rule")
            return []

        with patch.object(job, "_fetch_buckets", side_effect=failing_fetch_buckets):
            result = await job.run(workspace_id=workspace_id)

        # Job should complete successfully
        assert result["status"] == "completed"
        # Both rules were loaded
        assert result["metrics"]["rules_loaded"] == 2
        # One error was recorded
        assert result["metrics"]["evaluation_errors"] == 1
        # Second rule was still evaluated (call_count == 2)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_job_handles_lock_release_error(self, mock_pool, mock_conn):
        """Job handles lock release failure gracefully."""
        workspace_id = uuid4()

        # Track calls to fetchval
        call_count = 0

        async def fetchval_with_failing_unlock(query, *args):
            nonlocal call_count
            call_count += 1
            if "pg_advisory_unlock" in query:
                raise RuntimeError("Simulated lock release failure")
            # For pg_try_advisory_lock, return True
            return True

        mock_conn.fetchval = AsyncMock(side_effect=fetchval_with_failing_unlock)
        mock_conn.fetch = AsyncMock(return_value=[])  # No rules
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        job = AlertEvaluatorJob(mock_pool)

        # Should not raise - lock release error is caught and logged
        result = await job.run(workspace_id=workspace_id)

        # Job should still complete
        assert result["status"] == "completed"
        assert result["lock_acquired"] is True

    @pytest.mark.asyncio
    async def test_job_tracks_evaluation_errors_metric(self, mock_pool, mock_conn):
        """Job tracks evaluation_errors metric correctly."""
        workspace_id = uuid4()
        strategy_id = uuid4()

        # Three rules, all will fail
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": strategy_id,
                    "regime_key": f"regime_{i}",
                    "timeframe": "1h",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                    "cooldown_minutes": 60,
                }
                for i in range(3)
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        job = AlertEvaluatorJob(mock_pool)

        async def always_fail(*args, **kwargs):
            raise Exception("Always fails")

        with patch.object(job, "_fetch_buckets", side_effect=always_fail):
            result = await job.run(workspace_id=workspace_id)

        assert result["status"] == "completed"
        assert result["metrics"]["rules_loaded"] == 3
        assert result["metrics"]["evaluation_errors"] == 3
