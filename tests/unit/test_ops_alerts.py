"""Unit tests for operational alerts."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.ops_alerts.models import (
    DedupePeriod,
    EvalContext,
    OpsRuleType,
    Severity,
    get_all_rules,
    get_rule,
)
from app.services.ops_alerts.evaluator import OpsAlertEvaluator
from app.services.ops_alerts.telegram import TelegramNotifier
from app.repositories.ops_alerts import OpsAlertsRepository, OpsAlert, UpsertResult


class TestOpsAlertModels:
    """Test ops alert model definitions."""

    def test_all_rules_defined(self):
        """All expected rules are defined."""
        rules = get_all_rules()
        rule_types = {r.rule_type for r in rules}

        assert OpsRuleType.HEALTH_DEGRADED in rule_types
        assert OpsRuleType.WEAK_COVERAGE_P1 in rule_types
        assert OpsRuleType.WEAK_COVERAGE_P2 in rule_types
        assert OpsRuleType.DRIFT_SPIKE in rule_types
        assert OpsRuleType.CONFIDENCE_DROP in rule_types

    def test_rule_dedupe_key_daily(self):
        """Daily rules build correct dedupe keys."""
        rule = get_rule(OpsRuleType.HEALTH_DEGRADED)
        assert rule.dedupe_period == DedupePeriod.DAILY

        key = rule.build_dedupe_key("2026-01-19")
        assert key == "health_degraded:2026-01-19"

    def test_rule_dedupe_key_hourly(self):
        """Hourly rules build correct dedupe keys."""
        rule = get_rule(OpsRuleType.DRIFT_SPIKE)
        assert rule.dedupe_period == DedupePeriod.HOURLY

        key = rule.build_dedupe_key("2026-01-19-14")
        assert key == "drift_spike:2026-01-19-14"

    def test_bucket_key_daily(self):
        """Daily rules generate correct bucket keys."""
        rule = get_rule(OpsRuleType.WEAK_COVERAGE_P1)
        now = datetime(2026, 1, 19, 14, 30, 0, tzinfo=timezone.utc)

        bucket = rule.get_bucket_key(now)
        assert bucket == "2026-01-19"

    def test_bucket_key_hourly(self):
        """Hourly rules generate correct bucket keys."""
        rule = get_rule(OpsRuleType.CONFIDENCE_DROP)
        now = datetime(2026, 1, 19, 14, 30, 0, tzinfo=timezone.utc)

        bucket = rule.get_bucket_key(now)
        assert bucket == "2026-01-19-14"


class TestOpsAlertEvaluator:
    """Test alert condition evaluation."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.upsert.return_value = UpsertResult(
            id=uuid4(),
            is_new=True,
            previous_severity=None,
            current_severity="high",
            escalated=False,
        )
        repo.resolve_by_dedupe_key.return_value = None
        return repo

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock()
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool

    @pytest.fixture
    def evaluator(self, mock_repo, mock_pool):
        """Create evaluator with mocks."""
        return OpsAlertEvaluator(mock_repo, mock_pool)

    def test_eval_health_degraded_triggers_on_degraded(self, evaluator):
        """Health rule triggers when overall_status is degraded."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )

        # Mock health snapshot
        health = MagicMock()
        health.overall_status = "degraded"
        health.issues = ["db: slow queries", "qdrant: high latency"]
        health.components_error = 0
        health.components_degraded = 2
        ctx.health_snapshot = health

        rule = get_rule(OpsRuleType.HEALTH_DEGRADED)
        condition = evaluator._eval_health_degraded(rule, ctx)

        assert condition.triggered is True
        assert condition.severity == Severity.HIGH
        assert "degraded" in condition.payload["overall_status"]
        assert condition.dedupe_key == "health_degraded:2026-01-19"

    def test_eval_health_degraded_critical_on_error(self, evaluator):
        """Health rule escalates to critical when overall_status is error."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )

        health = MagicMock()
        health.overall_status = "error"
        health.issues = ["db: connection refused"]
        health.components_error = 1
        health.components_degraded = 0
        ctx.health_snapshot = health

        rule = get_rule(OpsRuleType.HEALTH_DEGRADED)
        condition = evaluator._eval_health_degraded(rule, ctx)

        assert condition.triggered is True
        assert condition.severity == Severity.CRITICAL

    def test_eval_health_degraded_no_trigger_on_ok(self, evaluator):
        """Health rule does not trigger when overall_status is ok."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )

        health = MagicMock()
        health.overall_status = "ok"
        health.issues = []
        health.components_error = 0
        health.components_degraded = 0
        ctx.health_snapshot = health

        rule = get_rule(OpsRuleType.HEALTH_DEGRADED)
        condition = evaluator._eval_health_degraded(rule, ctx)

        assert condition.triggered is False

    def test_eval_health_degraded_skipped_when_unavailable(self, evaluator):
        """Health rule is skipped when health snapshot unavailable."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
            health_snapshot=None,
        )

        rule = get_rule(OpsRuleType.HEALTH_DEGRADED)
        condition = evaluator._eval_health_degraded(rule, ctx)

        assert condition.triggered is False
        assert condition.skip_reason == "health_unavailable"

    def test_eval_weak_coverage_p1_triggers(self, evaluator):
        """Coverage P1 rule triggers when P1 gaps exist."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )
        ctx.coverage_stats = {
            "p1_open": 2,
            "p2_open": 5,
            "total_open": 7,
            "worst_score": 0.12,
            "worst_run_id": uuid4(),
        }

        rule = get_rule(OpsRuleType.WEAK_COVERAGE_P1)
        condition = evaluator._eval_weak_coverage_p1(rule, ctx)

        assert condition.triggered is True
        assert condition.severity == Severity.HIGH
        assert condition.payload["count"] == 2

    def test_eval_weak_coverage_p1_no_trigger_when_zero(self, evaluator):
        """Coverage P1 rule does not trigger when no P1 gaps."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )
        ctx.coverage_stats = {
            "p1_open": 0,
            "p2_open": 5,
            "total_open": 5,
            "worst_score": 0.35,
            "worst_run_id": None,
        }

        rule = get_rule(OpsRuleType.WEAK_COVERAGE_P1)
        condition = evaluator._eval_weak_coverage_p1(rule, ctx)

        assert condition.triggered is False

    def test_eval_drift_spike_triggers_on_weak_rate_spike(self, evaluator):
        """Drift rule triggers when weak_rate doubles."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )
        ctx.match_run_stats = {
            "count_15m": 60,
            "weak_rate_15m": 0.40,  # 40% weak
            "avg_score_15m": 0.55,
            "count_24h": 1000,
            "weak_rate_24h": 0.15,  # 15% baseline -> 40% is >2x
            "avg_score_24h": 0.60,
        }

        rule = get_rule(OpsRuleType.DRIFT_SPIKE)
        condition = evaluator._eval_drift_spike(rule, ctx)

        assert condition.triggered is True
        assert condition.payload["trigger_reason"] == "weak_rate"

    def test_eval_drift_spike_volume_gated(self, evaluator):
        """Drift rule is gated when volume is insufficient."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )
        ctx.match_run_stats = {
            "count_15m": 10,  # Below min_sample_count of 50
            "weak_rate_15m": 0.80,
            "avg_score_15m": 0.30,
            "count_24h": 100,
            "weak_rate_24h": 0.10,
            "avg_score_24h": 0.60,
        }

        rule = get_rule(OpsRuleType.DRIFT_SPIKE)
        condition = evaluator._eval_drift_spike(rule, ctx)

        assert condition.triggered is False
        assert "insufficient_volume" in condition.skip_reason

    def test_eval_confidence_drop_triggers_on_floor(self, evaluator):
        """Confidence rule triggers when score drops below floor."""
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc),
        )
        ctx.match_run_stats = {
            "count_15m": 60,
            "weak_rate_15m": 0.50,
            "avg_score_15m": 0.40,  # Below floor of 0.45
            "count_24h": 1000,
            "weak_rate_24h": 0.20,
            "avg_score_24h": 0.55,
        }

        rule = get_rule(OpsRuleType.CONFIDENCE_DROP)
        condition = evaluator._eval_confidence_drop(rule, ctx)

        assert condition.triggered is True
        assert condition.payload["trigger_reason"] == "floor"


class TestTelegramNotifier:
    """Test Telegram notification formatting."""

    @pytest.fixture
    def notifier(self):
        """Create notifier instance."""
        return TelegramNotifier(
            bot_token="test_token",
            chat_id="-100123456789",
            base_url="https://rag.example.com",
            enabled=True,
        )

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        return OpsAlert(
            id=uuid4(),
            workspace_id=uuid4(),
            rule_type="health_degraded",
            severity="critical",
            status="active",
            rule_version="v1",
            dedupe_key="health_degraded:2026-01-19",
            payload={
                "overall_status": "error",
                "issues": ["db: connection timeout"],
                "components_error": 1,
            },
            source="alert_evaluator",
            job_run_id=uuid4(),
            created_at=datetime.now(timezone.utc),
            last_seen_at=datetime.now(timezone.utc),
            resolved_at=None,
            acknowledged_at=None,
            acknowledged_by=None,
        )

    def test_format_message_alert(self, notifier, sample_alert):
        """Alert message contains expected elements."""
        message = notifier._format_message(
            sample_alert, is_recovery=False, is_escalation=False
        )

        assert "ALERT" in message
        assert "Health Degraded" in message
        assert "critical" in message
        assert "error" in message
        assert "connection timeout" in message
        assert sample_alert.job_run_id is not None
        assert str(sample_alert.id)[:8] in message

    def test_format_message_recovery(self, notifier, sample_alert):
        """Recovery message has correct formatting."""
        sample_alert.status = "resolved"
        message = notifier._format_message(
            sample_alert, is_recovery=True, is_escalation=False
        )

        assert "RECOVERED" in message
        assert "🟢" in message
        assert "Condition cleared" in message

    def test_format_message_escalation(self, notifier, sample_alert):
        """Escalation message is identified."""
        message = notifier._format_message(
            sample_alert, is_recovery=False, is_escalation=True
        )

        assert "ESCALATED" in message

    def test_escape_html(self, notifier):
        """HTML characters are escaped."""
        assert notifier._escape_html("<script>") == "&lt;script&gt;"
        assert notifier._escape_html("&test") == "&amp;test"

    @pytest.mark.asyncio
    async def test_disabled_notifier_returns_false(self):
        """Disabled notifier returns False without sending."""
        notifier = TelegramNotifier(
            bot_token="test",
            chat_id="123",
            enabled=False,
        )

        # This would fail if it tried to actually send
        result = await notifier.send_alert(MagicMock(), is_recovery=False)

        assert result is False


class TestUpsertResult:
    """Test UpsertResult dataclass."""

    def test_is_new_detection(self):
        """UpsertResult correctly reports new alerts."""
        result = UpsertResult(
            id=uuid4(),
            is_new=True,
            previous_severity=None,
            current_severity="high",
            escalated=False,
        )
        assert result.is_new is True
        assert result.escalated is False

    def test_escalation_detection(self):
        """UpsertResult correctly reports escalation."""
        result = UpsertResult(
            id=uuid4(),
            is_new=False,
            previous_severity="medium",
            current_severity="critical",
            escalated=True,
        )
        assert result.is_new is False
        assert result.escalated is True
