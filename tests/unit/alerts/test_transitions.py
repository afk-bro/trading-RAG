"""Tests for alert transition layer."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from app.services.alerts.transitions import AlertTransitionManager
from app.services.alerts.models import EvalResult, RuleType


class TestAlertTransitionManager:
    """Tests for transition layer."""

    @pytest.fixture
    def mock_repo(self):
        return AsyncMock()

    @pytest.fixture
    def manager(self, mock_repo):
        return AlertTransitionManager(mock_repo)

    @pytest.mark.asyncio
    async def test_process_activation_new_event(self, manager, mock_repo):
        """New activation creates event."""
        mock_repo.get_existing_event = AsyncMock(return_value=None)
        mock_repo.upsert_activate = AsyncMock(return_value={"id": uuid4()})

        eval_result = EvalResult(condition_met=True, trigger_value=0.35)

        result = await manager.process_evaluation(
            eval_result=eval_result,
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            fingerprint="v1:test",
            cooldown_minutes=60,
        )

        assert result["action"] == "activated"
        mock_repo.upsert_activate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_still_active_updates_last_seen(self, manager, mock_repo):
        """Still active updates last_seen only."""
        existing = {
            "id": uuid4(),
            "status": "active",
            "activated_at": datetime.now(timezone.utc),
            "last_seen": datetime.now(timezone.utc),
        }
        mock_repo.get_existing_event = AsyncMock(return_value=existing)
        mock_repo.update_last_seen = AsyncMock(return_value=True)

        eval_result = EvalResult(condition_met=True, trigger_value=0.35)

        result = await manager.process_evaluation(
            eval_result=eval_result,
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            fingerprint="v1:test",
            cooldown_minutes=60,
        )

        assert result["action"] == "updated_last_seen"
        mock_repo.update_last_seen.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_reactivation_within_cooldown_suppressed(
        self, manager, mock_repo
    ):
        """Reactivation within cooldown is suppressed."""
        existing = {
            "id": uuid4(),
            "status": "resolved",
            "activated_at": datetime.now(timezone.utc) - timedelta(minutes=30),
            "last_seen": datetime.now(timezone.utc),
        }
        mock_repo.get_existing_event = AsyncMock(return_value=existing)

        eval_result = EvalResult(condition_met=True, trigger_value=0.35)

        result = await manager.process_evaluation(
            eval_result=eval_result,
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            fingerprint="v1:test",
            cooldown_minutes=60,  # 60 min cooldown, only 30 min since last
        )

        assert result["action"] == "suppressed_cooldown"
        mock_repo.upsert_activate.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_resolve_active_event(self, manager, mock_repo):
        """Resolves active event when condition clears."""
        existing = {
            "id": uuid4(),
            "status": "active",
            "activated_at": datetime.now(timezone.utc),
            "last_seen": datetime.now(timezone.utc),
        }
        mock_repo.get_existing_event = AsyncMock(return_value=existing)
        mock_repo.resolve = AsyncMock(return_value=True)

        eval_result = EvalResult(condition_clear=True, trigger_value=0.20)

        result = await manager.process_evaluation(
            eval_result=eval_result,
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            fingerprint="v1:test",
            cooldown_minutes=60,
        )

        assert result["action"] == "resolved"
        mock_repo.resolve.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_insufficient_data_no_change(self, manager, mock_repo):
        """Insufficient data results in no action."""
        eval_result = EvalResult(insufficient_data=True)

        result = await manager.process_evaluation(
            eval_result=eval_result,
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            fingerprint="v1:test",
            cooldown_minutes=60,
        )

        assert result["action"] == "no_change"
        mock_repo.get_existing_event.assert_not_called()
