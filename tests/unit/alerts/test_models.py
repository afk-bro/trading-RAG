"""Tests for alert models."""

import pytest
from uuid import uuid4

from app.services.alerts.models import (
    AlertRule,
    AlertEvent,
    EvalResult,
    RuleType,
    Severity,
    AlertStatus,
    DriftSpikeConfig,
    ConfidenceDropConfig,
)


class TestAlertRule:
    """Tests for AlertRule model."""

    def test_create_drift_spike_rule(self):
        """Create drift spike rule with valid config."""
        rule = AlertRule(
            id=uuid4(),
            workspace_id=uuid4(),
            rule_type=RuleType.DRIFT_SPIKE,
            config={"drift_threshold": 0.30, "consecutive_buckets": 2},
        )
        assert rule.rule_type == RuleType.DRIFT_SPIKE
        assert rule.enabled is True
        assert rule.cooldown_minutes == 60

    def test_drift_spike_config_validation(self):
        """DriftSpikeConfig validates fields."""
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        assert config.drift_threshold == 0.30
        assert config.hysteresis == 0.05  # default

    def test_drift_spike_config_invalid_threshold(self):
        """DriftSpikeConfig rejects invalid threshold."""
        with pytest.raises(ValueError):
            DriftSpikeConfig(drift_threshold=-0.1, consecutive_buckets=2)

    def test_drift_spike_config_resolve_n_default(self):
        """DriftSpikeConfig resolve_n defaults to consecutive_buckets."""
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=3)
        assert config.resolve_n == 3

    def test_drift_spike_config_resolve_n_explicit(self):
        """DriftSpikeConfig resolve_n uses explicit value when set."""
        config = DriftSpikeConfig(
            drift_threshold=0.30,
            consecutive_buckets=2,
            resolve_consecutive_buckets=4,
        )
        assert config.resolve_n == 4

    def test_confidence_drop_config_validation(self):
        """ConfidenceDropConfig validates fields."""
        config = ConfidenceDropConfig(trend_threshold=0.05)
        assert config.trend_threshold == 0.05
        assert config.hysteresis == 0.02  # default

    def test_confidence_drop_config_invalid_threshold(self):
        """ConfidenceDropConfig rejects invalid threshold."""
        with pytest.raises(ValueError):
            ConfidenceDropConfig(trend_threshold=1.5)


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_eval_result_active(self):
        """EvalResult for active condition."""
        result = EvalResult(
            condition_met=True,
            condition_clear=False,
            trigger_value=0.35,
            context={"threshold": 0.30},
        )
        assert result.condition_met is True
        assert result.insufficient_data is False

    def test_eval_result_insufficient_data(self):
        """EvalResult with insufficient data."""
        result = EvalResult(insufficient_data=True)
        assert result.insufficient_data is True
        assert result.condition_met is False

    def test_eval_result_defaults(self):
        """EvalResult has sensible defaults."""
        result = EvalResult()
        assert result.condition_met is False
        assert result.condition_clear is False
        assert result.trigger_value == 0.0
        assert result.context == {}
        assert result.insufficient_data is False


class TestAlertEvent:
    """Tests for AlertEvent model."""

    def test_create_alert_event(self):
        """Create alert event."""
        event = AlertEvent(
            id=uuid4(),
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            status=AlertStatus.ACTIVE,
            severity=Severity.MEDIUM,
            context_json={"drift_threshold": 0.30},
            fingerprint="v1:high_vol/uptrend:1h",
        )
        assert event.status == AlertStatus.ACTIVE
        assert event.acknowledged is False

    def test_alert_event_defaults(self):
        """AlertEvent has correct defaults."""
        event = AlertEvent(
            id=uuid4(),
            workspace_id=uuid4(),
            rule_id=uuid4(),
            strategy_entity_id=uuid4(),
            regime_key="low_vol/sideways",
            timeframe="4h",
            rule_type=RuleType.CONFIDENCE_DROP,
            fingerprint="v1:low_vol/sideways:4h",
        )
        assert event.status == AlertStatus.ACTIVE
        assert event.severity == Severity.MEDIUM
        assert event.acknowledged is False
        assert event.acknowledged_at is None
        assert event.context_json == {}


class TestEnums:
    """Tests for enum values."""

    def test_rule_type_values(self):
        """RuleType has expected values."""
        assert RuleType.DRIFT_SPIKE.value == "drift_spike"
        assert RuleType.CONFIDENCE_DROP.value == "confidence_drop"
        assert RuleType.COMBO.value == "combo"

    def test_severity_values(self):
        """Severity has expected values."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"

    def test_alert_status_values(self):
        """AlertStatus has expected values."""
        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.RESOLVED.value == "resolved"
