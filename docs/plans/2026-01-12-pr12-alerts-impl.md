# PR12: In-App Alerts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement analytics alerting for drift spikes, confidence drops, and combo degradations with in-app delivery.

**Architecture:** Sink-based alert system: AlertRule defines thresholds, RuleEvaluator computes state, TransitionLayer handles DB upserts, AlertEvaluatorJob runs on schedule via JobRunner.

**Tech Stack:** PostgreSQL, asyncpg, FastAPI, Pydantic, Jinja2, PR11 JobRunner

---

## Task 1: Database Migrations

**Files:**
- Create: `migrations/044_alert_rules.sql`
- Create: `migrations/045_alert_events.sql`

**Step 1: Create alert_rules migration**

Create file `migrations/044_alert_rules.sql`:

```sql
-- Alert rules (definitions)
CREATE TABLE alert_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    rule_type TEXT NOT NULL CHECK (rule_type IN ('drift_spike', 'confidence_drop', 'combo')),
    strategy_entity_id UUID,
    regime_key TEXT,
    timeframe TEXT,
    enabled BOOLEAN DEFAULT true,
    config JSONB NOT NULL DEFAULT '{}',
    cooldown_minutes INT DEFAULT 60,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alert_rules_workspace ON alert_rules(workspace_id, enabled);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_alert_rules_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_alert_rules_updated_at
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_alert_rules_updated_at();
```

**Step 2: Create alert_events migration**

Create file `migrations/045_alert_events.sql`:

```sql
-- Alert events (occurrences)
CREATE TABLE alert_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    rule_id UUID NOT NULL REFERENCES alert_rules(id),
    strategy_entity_id UUID NOT NULL,
    regime_key TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('drift_spike', 'confidence_drop', 'combo')),

    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved')),
    severity TEXT NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high')),

    acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,

    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,

    context_json JSONB NOT NULL DEFAULT '{}',
    fingerprint TEXT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(workspace_id, strategy_entity_id, regime_key, timeframe, rule_type, fingerprint)
);

-- Consistency constraints
ALTER TABLE alert_events ADD CONSTRAINT chk_ack_consistency CHECK (
    (acknowledged = false AND acknowledged_at IS NULL AND acknowledged_by IS NULL)
    OR (acknowledged = true AND acknowledged_at IS NOT NULL)
);

ALTER TABLE alert_events ADD CONSTRAINT chk_resolved_consistency CHECK (
    (status = 'active' AND resolved_at IS NULL)
    OR (status = 'resolved' AND resolved_at IS NOT NULL)
);

-- Indexes
CREATE INDEX idx_alert_events_active ON alert_events(workspace_id, status) WHERE status = 'active';
CREATE INDEX idx_alert_events_list ON alert_events(workspace_id, last_seen DESC);
CREATE INDEX idx_alert_events_filtered ON alert_events(workspace_id, status, severity, last_seen DESC);
CREATE INDEX idx_alert_events_needs_attention ON alert_events(workspace_id, last_seen DESC)
    WHERE status = 'active' AND acknowledged = false;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_alert_events_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_alert_events_updated_at
    BEFORE UPDATE ON alert_events
    FOR EACH ROW
    EXECUTE FUNCTION update_alert_events_updated_at();
```

**Step 3: Commit**

```bash
git add migrations/044_alert_rules.sql migrations/045_alert_events.sql
git commit -m "feat(db): add alert_rules and alert_events tables"
```

---

## Task 2: Pydantic Models & Schemas

**Files:**
- Create: `app/services/alerts/__init__.py`
- Create: `app/services/alerts/models.py`
- Test: `tests/unit/alerts/__init__.py`
- Test: `tests/unit/alerts/test_models.py`

**Step 1: Create directory structure**

```bash
mkdir -p app/services/alerts
mkdir -p tests/unit/alerts
touch app/services/alerts/__init__.py
touch tests/unit/alerts/__init__.py
```

**Step 2: Write the failing test for models**

Create `tests/unit/alerts/test_models.py`:

```python
"""Tests for alert models."""

import pytest
from datetime import datetime
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
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/unit/alerts/test_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.services.alerts.models'"

**Step 4: Write minimal implementation**

Create `app/services/alerts/models.py`:

```python
"""Alert models and schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class RuleType(str, Enum):
    """Alert rule types."""

    DRIFT_SPIKE = "drift_spike"
    CONFIDENCE_DROP = "confidence_drop"
    COMBO = "combo"


class Severity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AlertStatus(str, Enum):
    """Alert event status."""

    ACTIVE = "active"
    RESOLVED = "resolved"


class DriftSpikeConfig(BaseModel):
    """Configuration for drift spike rule."""

    drift_threshold: float = Field(0.30, ge=0.0, le=1.0)
    consecutive_buckets: int = Field(2, ge=1)
    resolve_consecutive_buckets: Optional[int] = None
    hysteresis: float = Field(0.05, ge=0.0)

    @property
    def resolve_n(self) -> int:
        return self.resolve_consecutive_buckets or self.consecutive_buckets


class ConfidenceDropConfig(BaseModel):
    """Configuration for confidence drop rule."""

    trend_threshold: float = Field(0.05, ge=0.0, le=1.0)
    hysteresis: float = Field(0.02, ge=0.0)


class ComboConfig(BaseModel):
    """Configuration for combo rule (uses underlying configs)."""

    pass  # Inherits from drift_spike and confidence_drop defaults


class AlertRule(BaseModel):
    """Alert rule definition."""

    id: UUID
    workspace_id: UUID
    rule_type: RuleType
    strategy_entity_id: Optional[UUID] = None
    regime_key: Optional[str] = None
    timeframe: Optional[str] = None
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    cooldown_minutes: int = 60
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AlertEvent(BaseModel):
    """Alert event occurrence."""

    id: UUID
    workspace_id: UUID
    rule_id: UUID
    strategy_entity_id: UUID
    regime_key: str
    timeframe: str
    rule_type: RuleType
    status: AlertStatus = AlertStatus.ACTIVE
    severity: Severity = Severity.MEDIUM
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    first_seen: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    context_json: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class EvalResult:
    """Result of rule evaluation."""

    condition_met: bool = False
    condition_clear: bool = False
    trigger_value: float = 0.0
    context: dict = field(default_factory=dict)
    insufficient_data: bool = False
```

Update `app/services/alerts/__init__.py`:

```python
"""Alert services."""

from app.services.alerts.models import (
    AlertEvent,
    AlertRule,
    AlertStatus,
    ConfidenceDropConfig,
    DriftSpikeConfig,
    EvalResult,
    RuleType,
    Severity,
)

__all__ = [
    "AlertEvent",
    "AlertRule",
    "AlertStatus",
    "ConfidenceDropConfig",
    "DriftSpikeConfig",
    "EvalResult",
    "RuleType",
    "Severity",
]
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/unit/alerts/test_models.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add app/services/alerts/ tests/unit/alerts/
git commit -m "feat(alerts): add alert models and schemas"
```

---

## Task 3: Rule Evaluators

**Files:**
- Create: `app/services/alerts/evaluators.py`
- Test: `tests/unit/alerts/test_evaluators.py`

**Step 1: Write the failing test**

Create `tests/unit/alerts/test_evaluators.py`:

```python
"""Tests for alert rule evaluators."""

import pytest
from dataclasses import dataclass

from app.services.alerts.evaluators import RuleEvaluator
from app.services.alerts.models import DriftSpikeConfig, ConfidenceDropConfig


@dataclass
class MockBucket:
    """Mock bucket for testing."""

    drift_score: float = 0.0
    avg_confidence: float = 0.0


class TestDriftSpikeEvaluator:
    """Tests for drift spike evaluation."""

    def test_condition_met_consecutive_buckets(self):
        """Condition met when N consecutive buckets exceed threshold."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        buckets = [
            MockBucket(drift_score=0.25),
            MockBucket(drift_score=0.32),
            MockBucket(drift_score=0.35),
        ]

        result = evaluator.evaluate_drift_spike(buckets, config)

        assert result.condition_met is True
        assert result.condition_clear is False
        assert result.trigger_value == 0.35

    def test_condition_not_met_single_bucket(self):
        """Condition not met with only one bucket above threshold."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        buckets = [
            MockBucket(drift_score=0.25),
            MockBucket(drift_score=0.28),
            MockBucket(drift_score=0.35),
        ]

        result = evaluator.evaluate_drift_spike(buckets, config)

        assert result.condition_met is False

    def test_condition_clear_with_hysteresis(self):
        """Condition clears when below threshold minus hysteresis."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(
            drift_threshold=0.30, consecutive_buckets=2, hysteresis=0.05
        )
        buckets = [
            MockBucket(drift_score=0.28),
            MockBucket(drift_score=0.24),
            MockBucket(drift_score=0.20),
        ]

        result = evaluator.evaluate_drift_spike(buckets, config)

        assert result.condition_clear is True

    def test_insufficient_data(self):
        """Returns insufficient_data when not enough buckets."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=3)
        buckets = [MockBucket(drift_score=0.35), MockBucket(drift_score=0.35)]

        result = evaluator.evaluate_drift_spike(buckets, config)

        assert result.insufficient_data is True

    def test_tie_break_prioritizes_alerting(self):
        """When both condition_met and condition_clear possible, prioritize alerting."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(
            drift_threshold=0.30, consecutive_buckets=2, hysteresis=0.0
        )
        # Edge case: exactly at threshold
        buckets = [MockBucket(drift_score=0.30), MockBucket(drift_score=0.30)]

        result = evaluator.evaluate_drift_spike(buckets, config)

        assert result.condition_met is True
        assert result.condition_clear is False


class TestConfidenceDropEvaluator:
    """Tests for confidence drop evaluation."""

    def test_condition_met_trend_below_threshold(self):
        """Condition met when trend delta exceeds threshold."""
        evaluator = RuleEvaluator()
        config = ConfidenceDropConfig(trend_threshold=0.05)
        # First half avg: 0.70, Second half avg: 0.60 → delta = -0.10
        buckets = [
            MockBucket(avg_confidence=0.72),
            MockBucket(avg_confidence=0.68),
            MockBucket(avg_confidence=0.62),
            MockBucket(avg_confidence=0.58),
        ]

        result = evaluator.evaluate_confidence_drop(buckets, config)

        assert result.condition_met is True
        assert result.trigger_value < 0

    def test_condition_not_met_stable_confidence(self):
        """Condition not met when confidence is stable."""
        evaluator = RuleEvaluator()
        config = ConfidenceDropConfig(trend_threshold=0.05)
        buckets = [
            MockBucket(avg_confidence=0.70),
            MockBucket(avg_confidence=0.71),
            MockBucket(avg_confidence=0.69),
            MockBucket(avg_confidence=0.70),
        ]

        result = evaluator.evaluate_confidence_drop(buckets, config)

        assert result.condition_met is False


class TestComboEvaluator:
    """Tests for combo rule evaluation."""

    def test_condition_met_both_active(self):
        """Combo condition met when both drift and confidence conditions met."""
        evaluator = RuleEvaluator()
        drift_config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        confidence_config = ConfidenceDropConfig(trend_threshold=0.05)

        # High drift + dropping confidence
        buckets = [
            MockBucket(drift_score=0.25, avg_confidence=0.75),
            MockBucket(drift_score=0.35, avg_confidence=0.70),
            MockBucket(drift_score=0.40, avg_confidence=0.62),
            MockBucket(drift_score=0.38, avg_confidence=0.58),
        ]

        result = evaluator.evaluate_combo(buckets, drift_config, confidence_config)

        assert result.condition_met is True
        assert "drift" in result.context
        assert "confidence" in result.context

    def test_condition_clear_either_clears(self):
        """Combo clears when either underlying condition clears."""
        evaluator = RuleEvaluator()
        drift_config = DriftSpikeConfig(
            drift_threshold=0.30, consecutive_buckets=2, hysteresis=0.05
        )
        confidence_config = ConfidenceDropConfig(trend_threshold=0.05)

        # Drift cleared, confidence still dropping
        buckets = [
            MockBucket(drift_score=0.20, avg_confidence=0.75),
            MockBucket(drift_score=0.18, avg_confidence=0.70),
            MockBucket(drift_score=0.15, avg_confidence=0.62),
            MockBucket(drift_score=0.12, avg_confidence=0.58),
        ]

        result = evaluator.evaluate_combo(buckets, drift_config, confidence_config)

        assert result.condition_clear is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/alerts/test_evaluators.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.services.alerts.evaluators'"

**Step 3: Write minimal implementation**

Create `app/services/alerts/evaluators.py`:

```python
"""Alert rule evaluators - pure, stateless evaluation logic."""

from typing import Protocol, Sequence

from app.services.alerts.models import (
    ConfidenceDropConfig,
    DriftSpikeConfig,
    EvalResult,
)


class BucketProtocol(Protocol):
    """Protocol for drift/confidence buckets."""

    drift_score: float
    avg_confidence: float


class RuleEvaluator:
    """Evaluates alert rules against bucket data."""

    def evaluate_drift_spike(
        self,
        buckets: Sequence[BucketProtocol],
        config: DriftSpikeConfig,
    ) -> EvalResult:
        """
        Evaluate drift spike condition.

        Returns EvalResult with condition_met=True if drift_score >= threshold
        for N consecutive buckets.
        """
        activate_n = config.consecutive_buckets
        resolve_n = config.resolve_n
        threshold = config.drift_threshold
        hysteresis = config.hysteresis

        if len(buckets) < max(activate_n, resolve_n):
            return EvalResult(insufficient_data=True)

        recent_activate = buckets[-activate_n:]
        recent_resolve = buckets[-resolve_n:]

        condition_met = all(b.drift_score >= threshold for b in recent_activate)
        condition_clear = all(
            b.drift_score < (threshold - hysteresis) for b in recent_resolve
        )

        # Tie-break: prioritize alerting
        if condition_met:
            condition_clear = False

        return EvalResult(
            condition_met=condition_met,
            condition_clear=condition_clear,
            trigger_value=buckets[-1].drift_score,
            context={
                "threshold": threshold,
                "consecutive_buckets": activate_n,
                "hysteresis": hysteresis,
                "current_drift": buckets[-1].drift_score,
                "recent_drifts": [b.drift_score for b in recent_activate],
            },
        )

    def evaluate_confidence_drop(
        self,
        buckets: Sequence[BucketProtocol],
        config: ConfidenceDropConfig,
    ) -> EvalResult:
        """
        Evaluate confidence drop condition.

        Returns EvalResult with condition_met=True if first-half vs second-half
        confidence delta exceeds threshold.
        """
        if len(buckets) < 2:
            return EvalResult(insufficient_data=True)

        mid = len(buckets) // 2
        first_half = buckets[:mid] if mid > 0 else buckets[:1]
        second_half = buckets[mid:] if mid < len(buckets) else buckets[-1:]

        first_half_avg = sum(b.avg_confidence for b in first_half) / len(first_half)
        second_half_avg = sum(b.avg_confidence for b in second_half) / len(second_half)
        trend_delta = second_half_avg - first_half_avg

        threshold = config.trend_threshold
        hysteresis = config.hysteresis

        condition_met = trend_delta <= -threshold
        condition_clear = trend_delta >= (-threshold + hysteresis)

        # Tie-break: prioritize alerting
        if condition_met:
            condition_clear = False

        return EvalResult(
            condition_met=condition_met,
            condition_clear=condition_clear,
            trigger_value=trend_delta,
            context={
                "trend_threshold": threshold,
                "trend_delta": round(trend_delta, 4),
                "first_half_avg": round(first_half_avg, 4),
                "second_half_avg": round(second_half_avg, 4),
            },
        )

    def evaluate_combo(
        self,
        buckets: Sequence[BucketProtocol],
        drift_config: DriftSpikeConfig,
        confidence_config: ConfidenceDropConfig,
    ) -> EvalResult:
        """
        Evaluate combo condition (drift spike + confidence drop).

        Returns EvalResult with condition_met=True if both underlying conditions met.
        Clears if either underlying condition clears (OR logic).
        """
        drift_result = self.evaluate_drift_spike(buckets, drift_config)
        confidence_result = self.evaluate_confidence_drop(buckets, confidence_config)

        # Handle insufficient data
        if drift_result.insufficient_data or confidence_result.insufficient_data:
            return EvalResult(insufficient_data=True)

        condition_met = drift_result.condition_met and confidence_result.condition_met
        condition_clear = drift_result.condition_clear or confidence_result.condition_clear

        # Tie-break: prioritize alerting
        if condition_met:
            condition_clear = False

        return EvalResult(
            condition_met=condition_met,
            condition_clear=condition_clear,
            trigger_value=drift_result.trigger_value,  # Use drift as primary
            context={
                "drift": drift_result.context,
                "confidence": confidence_result.context,
            },
        )
```

Update `app/services/alerts/__init__.py` to export:

```python
"""Alert services."""

from app.services.alerts.evaluators import RuleEvaluator
from app.services.alerts.models import (
    AlertEvent,
    AlertRule,
    AlertStatus,
    ConfidenceDropConfig,
    DriftSpikeConfig,
    EvalResult,
    RuleType,
    Severity,
)

__all__ = [
    "AlertEvent",
    "AlertRule",
    "AlertStatus",
    "ConfidenceDropConfig",
    "DriftSpikeConfig",
    "EvalResult",
    "RuleEvaluator",
    "RuleType",
    "Severity",
]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/alerts/test_evaluators.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/alerts/evaluators.py tests/unit/alerts/test_evaluators.py
git commit -m "feat(alerts): add rule evaluators for drift, confidence, combo"
```

---

## Task 4: Alert Repository

**Files:**
- Create: `app/repositories/alerts.py`
- Test: `tests/unit/test_alerts_repo.py`

**Step 1: Write the failing test**

Create `tests/unit/test_alerts_repo.py`:

```python
"""Tests for alerts repository."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.repositories.alerts import AlertsRepository
from app.services.alerts.models import RuleType, Severity, AlertStatus


class TestAlertRulesRepository:
    """Tests for alert rules operations."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.mark.asyncio
    async def test_list_rules_for_workspace(self, mock_pool):
        """List enabled rules for workspace."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_conn = AsyncMock()
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
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        rules = await repo.list_rules(workspace_id, enabled_only=True)

        assert len(rules) == 1
        assert rules[0]["rule_type"] == "drift_spike"
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_rule(self, mock_pool):
        """Create new alert rule."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "enabled": True,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        result = await repo.create_rule(
            workspace_id=workspace_id,
            rule_type=RuleType.DRIFT_SPIKE,
            config={"drift_threshold": 0.30},
        )

        assert result["id"] == rule_id
        mock_conn.fetchrow.assert_called_once()


class TestAlertEventsRepository:
    """Tests for alert events operations."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.mark.asyncio
    async def test_list_events_with_filters(self, mock_pool):
        """List events with status and severity filters."""
        workspace_id = uuid4()
        event_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": event_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "status": "active",
                    "severity": "medium",
                    "acknowledged": False,
                    "last_seen": datetime.now(timezone.utc),
                }
            ]
        )
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        events, total = await repo.list_events(
            workspace_id=workspace_id,
            status=AlertStatus.ACTIVE,
            acknowledged=False,
        )

        assert len(events) == 1
        assert events[0]["status"] == "active"
        assert total == 1

    @pytest.mark.asyncio
    async def test_upsert_activate(self, mock_pool):
        """Upsert activates alert event."""
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()
        event_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": event_id})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        result = await repo.upsert_activate(
            workspace_id=workspace_id,
            rule_id=rule_id,
            strategy_entity_id=strategy_id,
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            severity=Severity.MEDIUM,
            context_json={"threshold": 0.30},
            fingerprint="v1:high_vol/uptrend:1h",
        )

        assert result["id"] == event_id

    @pytest.mark.asyncio
    async def test_acknowledge_event(self, mock_pool):
        """Acknowledge alert event."""
        event_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        success = await repo.acknowledge(event_id, acknowledged_by="admin")

        assert success is True
        mock_conn.execute.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_alerts_repo.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `app/repositories/alerts.py`:

```python
"""Repository for alert rules and events."""

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.alerts.models import AlertStatus, RuleType, Severity

logger = structlog.get_logger(__name__)


class AlertsRepository:
    """Repository for alert rules and events queries."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    # =========================================================================
    # Alert Rules
    # =========================================================================

    async def list_rules(
        self,
        workspace_id: UUID,
        enabled_only: bool = False,
    ) -> list[dict]:
        """List alert rules for workspace."""
        query = """
            SELECT id, workspace_id, rule_type, strategy_entity_id, regime_key,
                   timeframe, enabled, config, cooldown_minutes, created_at, updated_at
            FROM alert_rules
            WHERE workspace_id = $1
        """
        params: list[Any] = [workspace_id]

        if enabled_only:
            query += " AND enabled = true"

        query += " ORDER BY created_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(r) for r in rows]

    async def get_rule(self, rule_id: UUID) -> Optional[dict]:
        """Get alert rule by ID."""
        query = """
            SELECT id, workspace_id, rule_type, strategy_entity_id, regime_key,
                   timeframe, enabled, config, cooldown_minutes, created_at, updated_at
            FROM alert_rules
            WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, rule_id)

        return dict(row) if row else None

    async def create_rule(
        self,
        workspace_id: UUID,
        rule_type: RuleType,
        config: dict,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
        timeframe: Optional[str] = None,
        cooldown_minutes: int = 60,
    ) -> dict:
        """Create new alert rule."""
        query = """
            INSERT INTO alert_rules (
                workspace_id, rule_type, strategy_entity_id, regime_key,
                timeframe, config, cooldown_minutes
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, workspace_id, rule_type, strategy_entity_id, regime_key,
                      timeframe, enabled, config, cooldown_minutes, created_at, updated_at
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                rule_type.value,
                strategy_entity_id,
                regime_key,
                timeframe,
                json.dumps(config),
                cooldown_minutes,
            )

        return dict(row)

    async def update_rule(
        self,
        rule_id: UUID,
        enabled: Optional[bool] = None,
        config: Optional[dict] = None,
        cooldown_minutes: Optional[int] = None,
    ) -> Optional[dict]:
        """Update alert rule."""
        updates = []
        params: list[Any] = []
        param_idx = 1

        if enabled is not None:
            updates.append(f"enabled = ${param_idx}")
            params.append(enabled)
            param_idx += 1

        if config is not None:
            updates.append(f"config = ${param_idx}")
            params.append(json.dumps(config))
            param_idx += 1

        if cooldown_minutes is not None:
            updates.append(f"cooldown_minutes = ${param_idx}")
            params.append(cooldown_minutes)
            param_idx += 1

        if not updates:
            return await self.get_rule(rule_id)

        params.append(rule_id)
        query = f"""
            UPDATE alert_rules
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, workspace_id, rule_type, strategy_entity_id, regime_key,
                      timeframe, enabled, config, cooldown_minutes, created_at, updated_at
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        return dict(row) if row else None

    async def delete_rule(self, rule_id: UUID) -> bool:
        """Delete alert rule."""
        query = "DELETE FROM alert_rules WHERE id = $1"
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, rule_id)
        return result == "DELETE 1"

    # =========================================================================
    # Alert Events
    # =========================================================================

    async def list_events(
        self,
        workspace_id: UUID,
        status: Optional[AlertStatus] = None,
        severity: Optional[Severity] = None,
        acknowledged: Optional[bool] = None,
        rule_type: Optional[RuleType] = None,
        strategy_entity_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List alert events with filters."""
        conditions = ["workspace_id = $1"]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1

        if severity:
            conditions.append(f"severity = ${param_idx}")
            params.append(severity.value)
            param_idx += 1

        if acknowledged is not None:
            conditions.append(f"acknowledged = ${param_idx}")
            params.append(acknowledged)
            param_idx += 1

        if rule_type:
            conditions.append(f"rule_type = ${param_idx}")
            params.append(rule_type.value)
            param_idx += 1

        if strategy_entity_id:
            conditions.append(f"strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT id, workspace_id, rule_id, strategy_entity_id, regime_key,
                   timeframe, rule_type, status, severity, acknowledged,
                   acknowledged_at, acknowledged_by, first_seen, activated_at,
                   last_seen, resolved_at, context_json, fingerprint,
                   created_at, updated_at
            FROM alert_events
            WHERE {where_clause}
            ORDER BY last_seen DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        count_query = f"SELECT COUNT(*) FROM alert_events WHERE {where_clause}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            total = await conn.fetchval(count_query, *params[:-2])

        return [dict(r) for r in rows], total or 0

    async def get_event(self, event_id: UUID) -> Optional[dict]:
        """Get alert event by ID."""
        query = """
            SELECT id, workspace_id, rule_id, strategy_entity_id, regime_key,
                   timeframe, rule_type, status, severity, acknowledged,
                   acknowledged_at, acknowledged_by, first_seen, activated_at,
                   last_seen, resolved_at, context_json, fingerprint,
                   created_at, updated_at
            FROM alert_events
            WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, event_id)

        return dict(row) if row else None

    async def get_existing_event(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
        rule_type: RuleType,
        fingerprint: str,
    ) -> Optional[dict]:
        """Get existing event by unique key."""
        query = """
            SELECT id, status, activated_at, last_seen
            FROM alert_events
            WHERE workspace_id = $1
              AND strategy_entity_id = $2
              AND regime_key = $3
              AND timeframe = $4
              AND rule_type = $5
              AND fingerprint = $6
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                strategy_entity_id,
                regime_key,
                timeframe,
                rule_type.value,
                fingerprint,
            )

        return dict(row) if row else None

    async def upsert_activate(
        self,
        workspace_id: UUID,
        rule_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
        rule_type: RuleType,
        severity: Severity,
        context_json: dict,
        fingerprint: str,
    ) -> dict:
        """Upsert alert event as active."""
        now = datetime.now(timezone.utc)
        query = """
            INSERT INTO alert_events (
                workspace_id, rule_id, strategy_entity_id, regime_key, timeframe,
                rule_type, status, severity, context_json, fingerprint,
                first_seen, activated_at, last_seen, acknowledged
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'active', $7, $8, $9, $10, $10, $10, false)
            ON CONFLICT (workspace_id, strategy_entity_id, regime_key, timeframe, rule_type, fingerprint)
            DO UPDATE SET
                status = 'active',
                severity = $7,
                context_json = $8,
                activated_at = CASE
                    WHEN alert_events.status = 'resolved' THEN $10
                    ELSE alert_events.activated_at
                END,
                last_seen = $10,
                resolved_at = NULL,
                acknowledged = CASE
                    WHEN alert_events.status = 'resolved' THEN false
                    ELSE alert_events.acknowledged
                END,
                acknowledged_at = CASE
                    WHEN alert_events.status = 'resolved' THEN NULL
                    ELSE alert_events.acknowledged_at
                END,
                acknowledged_by = CASE
                    WHEN alert_events.status = 'resolved' THEN NULL
                    ELSE alert_events.acknowledged_by
                END
            RETURNING id, workspace_id, status, activated_at, last_seen
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                rule_id,
                strategy_entity_id,
                regime_key,
                timeframe,
                rule_type.value,
                severity.value,
                json.dumps(context_json),
                fingerprint,
                now,
            )

        return dict(row)

    async def update_last_seen(self, event_id: UUID) -> bool:
        """Update last_seen timestamp for active event."""
        query = """
            UPDATE alert_events
            SET last_seen = NOW()
            WHERE id = $1 AND status = 'active'
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id)
        return result == "UPDATE 1"

    async def resolve(self, event_id: UUID) -> bool:
        """Resolve active alert event."""
        query = """
            UPDATE alert_events
            SET status = 'resolved', resolved_at = NOW()
            WHERE id = $1 AND status = 'active'
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id)
        return result == "UPDATE 1"

    async def acknowledge(
        self, event_id: UUID, acknowledged_by: Optional[str] = None
    ) -> bool:
        """Acknowledge alert event."""
        query = """
            UPDATE alert_events
            SET acknowledged = true, acknowledged_at = NOW(), acknowledged_by = $2
            WHERE id = $1 AND acknowledged = false
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id, acknowledged_by)
        return result == "UPDATE 1"

    async def unacknowledge(self, event_id: UUID) -> bool:
        """Unacknowledge alert event."""
        query = """
            UPDATE alert_events
            SET acknowledged = false, acknowledged_at = NULL, acknowledged_by = NULL
            WHERE id = $1 AND acknowledged = true
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, event_id)
        return result == "UPDATE 1"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_alerts_repo.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/repositories/alerts.py tests/unit/test_alerts_repo.py
git commit -m "feat(alerts): add alerts repository for rules and events"
```

---

## Task 5: Transition Layer

**Files:**
- Create: `app/services/alerts/transitions.py`
- Test: `tests/unit/alerts/test_transitions.py`

**Step 1: Write the failing test**

Create `tests/unit/alerts/test_transitions.py`:

```python
"""Tests for alert transition layer."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.services.alerts.transitions import AlertTransitionManager
from app.services.alerts.models import (
    EvalResult,
    RuleType,
    Severity,
    AlertStatus,
)


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/alerts/test_transitions.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/services/alerts/transitions.py`:

```python
"""Alert transition layer - handles state changes and DB operations."""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.alerts.models import EvalResult, RuleType, Severity

logger = structlog.get_logger(__name__)

SEVERITY_MAP = {
    RuleType.DRIFT_SPIKE: Severity.MEDIUM,
    RuleType.CONFIDENCE_DROP: Severity.MEDIUM,
    RuleType.COMBO: Severity.HIGH,
}


class AlertTransitionManager:
    """Manages alert state transitions."""

    def __init__(self, repo):
        """Initialize with alerts repository."""
        self.repo = repo

    async def process_evaluation(
        self,
        eval_result: EvalResult,
        workspace_id: UUID,
        rule_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
        rule_type: RuleType,
        fingerprint: str,
        cooldown_minutes: int,
    ) -> dict[str, Any]:
        """
        Process evaluation result and update DB state.

        Returns dict with action taken and details.
        """
        now = datetime.now(timezone.utc)

        # Insufficient data: no action
        if eval_result.insufficient_data:
            return {"action": "no_change", "reason": "insufficient_data"}

        # Get existing event
        existing = await self.repo.get_existing_event(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            timeframe=timeframe,
            rule_type=rule_type,
            fingerprint=fingerprint,
        )

        # Condition met: activate or update
        if eval_result.condition_met:
            if existing and existing["status"] == "active":
                # Still active: just update last_seen
                await self.repo.update_last_seen(existing["id"])
                return {"action": "updated_last_seen", "event_id": existing["id"]}

            # Potential activation (new or reactivation)
            if existing:
                activated_at = existing["activated_at"]
                if isinstance(activated_at, str):
                    activated_at = datetime.fromisoformat(activated_at)
                elapsed = (now - activated_at).total_seconds()
                if elapsed < cooldown_minutes * 60:
                    return {
                        "action": "suppressed_cooldown",
                        "reason": f"elapsed {elapsed:.0f}s < cooldown {cooldown_minutes * 60}s",
                    }

            # Activate
            severity = SEVERITY_MAP.get(rule_type, Severity.MEDIUM)
            context_json = {
                **eval_result.context,
                "deep_link": {
                    "strategy_entity_id": str(strategy_entity_id),
                    "timeframe": timeframe,
                    "regime_key": regime_key,
                },
            }

            result = await self.repo.upsert_activate(
                workspace_id=workspace_id,
                rule_id=rule_id,
                strategy_entity_id=strategy_entity_id,
                regime_key=regime_key,
                timeframe=timeframe,
                rule_type=rule_type,
                severity=severity,
                context_json=context_json,
                fingerprint=fingerprint,
            )

            return {"action": "activated", "event_id": result["id"]}

        # Condition clear: resolve if active
        elif eval_result.condition_clear:
            if existing and existing["status"] == "active":
                await self.repo.resolve(existing["id"])
                return {"action": "resolved", "event_id": existing["id"]}

        return {"action": "no_change", "reason": "unchanged"}
```

Update `app/services/alerts/__init__.py`:

```python
"""Alert services."""

from app.services.alerts.evaluators import RuleEvaluator
from app.services.alerts.models import (
    AlertEvent,
    AlertRule,
    AlertStatus,
    ConfidenceDropConfig,
    DriftSpikeConfig,
    EvalResult,
    RuleType,
    Severity,
)
from app.services.alerts.transitions import AlertTransitionManager

__all__ = [
    "AlertEvent",
    "AlertRule",
    "AlertStatus",
    "AlertTransitionManager",
    "ConfidenceDropConfig",
    "DriftSpikeConfig",
    "EvalResult",
    "RuleEvaluator",
    "RuleType",
    "Severity",
]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/alerts/test_transitions.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/alerts/transitions.py tests/unit/alerts/test_transitions.py
git commit -m "feat(alerts): add transition layer for state management"
```

---

## Task 6: Alert Evaluator Job

**Files:**
- Create: `app/services/alerts/job.py`
- Test: `tests/unit/alerts/test_job.py`

**Step 1: Write the failing test**

Create `tests/unit/alerts/test_job.py`:

```python
"""Tests for alert evaluator job."""

import pytest
from datetime import datetime, timezone
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
        mock_conn.fetchval.assert_called()  # Lock query

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

        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": rule_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": uuid4(),
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/alerts/test_job.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/services/alerts/job.py`:

```python
"""Alert evaluator job - scheduled evaluation of alert rules."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from app.repositories.alerts import AlertsRepository
from app.services.alerts.evaluators import RuleEvaluator
from app.services.alerts.models import (
    ConfidenceDropConfig,
    DriftSpikeConfig,
    RuleType,
)
from app.services.alerts.transitions import AlertTransitionManager

logger = structlog.get_logger(__name__)


class AlertEvaluatorJob:
    """Job for evaluating alert rules on schedule."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def run(
        self,
        workspace_id: UUID,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run alert evaluation for workspace.

        Returns job result with metrics.
        """
        metrics = {
            "rules_loaded": 0,
            "tuples_evaluated": 0,
            "tuples_skipped_insufficient_data": 0,
            "activations_suppressed_cooldown": 0,
            "alerts_activated": 0,
            "alerts_resolved": 0,
            "db_upserts": 0,
            "db_updates": 0,
        }

        async with self.pool.acquire() as conn:
            # Acquire advisory lock
            lock_key = hash(f"evaluate_alerts:{workspace_id}") % (2**31)
            lock_acquired = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", lock_key
            )

            if not lock_acquired:
                return {
                    "lock_acquired": False,
                    "status": "already_running",
                    "metrics": metrics,
                }

            try:
                # Load enabled rules
                repo = AlertsRepository(self.pool)
                rules = await repo.list_rules(workspace_id, enabled_only=True)
                metrics["rules_loaded"] = len(rules)

                if not rules:
                    return {
                        "lock_acquired": True,
                        "status": "completed",
                        "metrics": metrics,
                    }

                # Initialize components
                evaluator = RuleEvaluator()
                transition_mgr = AlertTransitionManager(repo)

                # Process each rule
                for rule in rules:
                    await self._process_rule(
                        rule=rule,
                        evaluator=evaluator,
                        transition_mgr=transition_mgr,
                        metrics=metrics,
                        dry_run=dry_run,
                    )

            finally:
                # Release lock
                await conn.fetchval("SELECT pg_advisory_unlock($1)", lock_key)

        return {
            "lock_acquired": True,
            "status": "completed",
            "metrics": metrics,
        }

    async def _process_rule(
        self,
        rule: dict,
        evaluator: RuleEvaluator,
        transition_mgr: AlertTransitionManager,
        metrics: dict,
        dry_run: bool,
    ) -> None:
        """Process a single alert rule."""
        rule_type = RuleType(rule["rule_type"])
        workspace_id = rule["workspace_id"]
        strategy_entity_id = rule.get("strategy_entity_id")
        regime_key = rule.get("regime_key")
        timeframe = rule.get("timeframe") or "1h"
        config = rule.get("config") or {}
        cooldown_minutes = rule.get("cooldown_minutes", 60)

        # Expand scope if strategy/regime is NULL (evaluate all)
        # For v1, we require explicit strategy_entity_id
        if not strategy_entity_id:
            # TODO: Enumerate strategies for workspace
            logger.info("Skipping rule with NULL strategy_entity_id", rule_id=rule["id"])
            return

        if not regime_key:
            # TODO: Enumerate regimes for strategy
            logger.info("Skipping rule with NULL regime_key", rule_id=rule["id"])
            return

        # Fetch bucket data
        buckets = await self._fetch_buckets(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            timeframe=timeframe,
        )

        metrics["tuples_evaluated"] += 1

        # Build fingerprint
        fingerprint = f"v1:{regime_key}:{timeframe}"

        # Evaluate based on rule type
        if rule_type == RuleType.DRIFT_SPIKE:
            drift_config = DriftSpikeConfig(**config)
            eval_result = evaluator.evaluate_drift_spike(buckets, drift_config)

        elif rule_type == RuleType.CONFIDENCE_DROP:
            confidence_config = ConfidenceDropConfig(**config)
            eval_result = evaluator.evaluate_confidence_drop(buckets, confidence_config)

        elif rule_type == RuleType.COMBO:
            drift_config = DriftSpikeConfig(**config.get("drift", {}))
            confidence_config = ConfidenceDropConfig(**config.get("confidence", {}))
            eval_result = evaluator.evaluate_combo(
                buckets, drift_config, confidence_config
            )
        else:
            return

        # Track insufficient data
        if eval_result.insufficient_data:
            metrics["tuples_skipped_insufficient_data"] += 1
            return

        if dry_run:
            return

        # Process transition
        result = await transition_mgr.process_evaluation(
            eval_result=eval_result,
            workspace_id=workspace_id,
            rule_id=rule["id"],
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            timeframe=timeframe,
            rule_type=rule_type,
            fingerprint=fingerprint,
            cooldown_minutes=cooldown_minutes,
        )

        # Update metrics
        action = result.get("action", "no_change")
        if action == "activated":
            metrics["alerts_activated"] += 1
            metrics["db_upserts"] += 1
        elif action == "resolved":
            metrics["alerts_resolved"] += 1
            metrics["db_updates"] += 1
        elif action == "updated_last_seen":
            metrics["db_updates"] += 1
        elif action == "suppressed_cooldown":
            metrics["activations_suppressed_cooldown"] += 1

    async def _fetch_buckets(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
    ) -> list:
        """
        Fetch drift/confidence bucket data.

        TODO: Integrate with existing analytics queries.
        """
        # Placeholder - integrate with existing regime drift queries
        return []
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/alerts/test_job.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/alerts/job.py tests/unit/alerts/test_job.py
git commit -m "feat(alerts): add alert evaluator job"
```

---

## Task 7: API Endpoints - Rules CRUD

**Files:**
- Create: `app/admin/alerts.py`
- Test: `tests/unit/admin/test_alerts_endpoints.py`

**Step 1: Write the failing test**

Create `tests/unit/admin/test_alerts_endpoints.py`:

```python
"""Tests for alert admin endpoints."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return MagicMock()


class TestAlertRulesEndpoints:
    """Tests for /admin/alerts/rules endpoints."""

    def test_list_rules_requires_admin_token(self, client):
        """List rules requires admin auth."""
        workspace_id = uuid4()
        response = client.get(f"/admin/alerts/rules?workspace_id={workspace_id}")
        assert response.status_code in (401, 403)

    def test_list_rules_success(self, client, mock_db_pool):
        """List rules returns rules for workspace."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_rules = AsyncMock(
            return_value=[
                {
                    "id": rule_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                }
            ]
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "rules" in data
        assert len(data["rules"]) == 1

    def test_create_rule_success(self, client, mock_db_pool):
        """Create rule returns new rule."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.create_rule = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "enabled": True,
            }
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                json={
                    "rule_type": "drift_spike",
                    "config": {"drift_threshold": 0.30},
                },
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["rule_type"] == "drift_spike"


class TestAlertEventsEndpoints:
    """Tests for /admin/alerts endpoints."""

    def test_list_events_requires_admin_token(self, client):
        """List events requires admin auth."""
        workspace_id = uuid4()
        response = client.get(f"/admin/alerts?workspace_id={workspace_id}")
        assert response.status_code in (401, 403)

    def test_list_events_success(self, client, mock_db_pool):
        """List events returns events for workspace."""
        workspace_id = uuid4()
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_events = AsyncMock(
            return_value=(
                [
                    {
                        "id": event_id,
                        "workspace_id": workspace_id,
                        "rule_type": "drift_spike",
                        "status": "active",
                        "severity": "medium",
                    }
                ],
                1,
            )
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] == 1

    def test_acknowledge_event_success(self, client, mock_db_pool):
        """Acknowledge event returns success."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.acknowledge = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/acknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        assert response.json()["acknowledged"] is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/admin/test_alerts_endpoints.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/admin/alerts.py`:

```python
"""Admin endpoints for alerts."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.repositories.alerts import AlertsRepository
from app.services.alerts.models import AlertStatus, RuleType, Severity

router = APIRouter(prefix="/alerts", tags=["admin-alerts"])
logger = structlog.get_logger(__name__)

_db_pool = None


def set_db_pool(pool):
    """Set database pool."""
    global _db_pool
    _db_pool = pool


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateRuleRequest(BaseModel):
    """Request to create alert rule."""

    rule_type: RuleType
    strategy_entity_id: Optional[UUID] = None
    regime_key: Optional[str] = None
    timeframe: Optional[str] = None
    config: dict = Field(default_factory=dict)
    cooldown_minutes: int = 60


class UpdateRuleRequest(BaseModel):
    """Request to update alert rule."""

    enabled: Optional[bool] = None
    config: Optional[dict] = None
    cooldown_minutes: Optional[int] = None


# =============================================================================
# Alert Rules Endpoints
# =============================================================================


@router.get("/rules")
async def list_rules(
    workspace_id: UUID = Query(...),
    enabled_only: bool = Query(False),
    _: str = Depends(require_admin_token),
):
    """List alert rules for workspace."""
    repo = AlertsRepository(_db_pool)
    rules = await repo.list_rules(workspace_id, enabled_only=enabled_only)
    return {"rules": rules}


@router.post("/rules", status_code=status.HTTP_201_CREATED)
async def create_rule(
    request: CreateRuleRequest,
    workspace_id: UUID = Query(...),
    _: str = Depends(require_admin_token),
):
    """Create new alert rule."""
    repo = AlertsRepository(_db_pool)
    rule = await repo.create_rule(
        workspace_id=workspace_id,
        rule_type=request.rule_type,
        config=request.config,
        strategy_entity_id=request.strategy_entity_id,
        regime_key=request.regime_key,
        timeframe=request.timeframe,
        cooldown_minutes=request.cooldown_minutes,
    )
    return rule


@router.get("/rules/{rule_id}")
async def get_rule(
    rule_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Get alert rule by ID."""
    repo = AlertsRepository(_db_pool)
    rule = await repo.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule


@router.patch("/rules/{rule_id}")
async def update_rule(
    rule_id: UUID,
    request: UpdateRuleRequest,
    _: str = Depends(require_admin_token),
):
    """Update alert rule."""
    repo = AlertsRepository(_db_pool)
    rule = await repo.update_rule(
        rule_id=rule_id,
        enabled=request.enabled,
        config=request.config,
        cooldown_minutes=request.cooldown_minutes,
    )
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Delete alert rule."""
    repo = AlertsRepository(_db_pool)
    deleted = await repo.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")


# =============================================================================
# Alert Events Endpoints
# =============================================================================


@router.get("")
async def list_events(
    workspace_id: UUID = Query(...),
    status: Optional[AlertStatus] = Query(None),
    severity: Optional[Severity] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    rule_type: Optional[RuleType] = Query(None),
    strategy_entity_id: Optional[UUID] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: str = Depends(require_admin_token),
):
    """List alert events with filters."""
    repo = AlertsRepository(_db_pool)
    events, total = await repo.list_events(
        workspace_id=workspace_id,
        status=status,
        severity=severity,
        acknowledged=acknowledged,
        rule_type=rule_type,
        strategy_entity_id=strategy_entity_id,
        limit=limit,
        offset=offset,
    )
    return {
        "items": events,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{event_id}")
async def get_event(
    event_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Get alert event by ID."""
    repo = AlertsRepository(_db_pool)
    event = await repo.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Alert not found")
    return event


@router.post("/{event_id}/acknowledge")
async def acknowledge_event(
    event_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Acknowledge alert event."""
    repo = AlertsRepository(_db_pool)
    success = await repo.acknowledge(event_id, acknowledged_by="admin")
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    return {"acknowledged": True, "event_id": str(event_id)}


@router.post("/{event_id}/unacknowledge")
async def unacknowledge_event(
    event_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Unacknowledge alert event."""
    repo = AlertsRepository(_db_pool)
    success = await repo.unacknowledge(event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or not acknowledged")
    return {"acknowledged": False, "event_id": str(event_id)}
```

**Step 4: Register router in main app**

Modify `app/admin/router.py` to include alerts router (add import and include_router).

**Step 5: Run test to verify it passes**

```bash
pytest tests/unit/admin/test_alerts_endpoints.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add app/admin/alerts.py tests/unit/admin/test_alerts_endpoints.py
git commit -m "feat(admin): add alert rules and events API endpoints"
```

---

## Task 8: Job Endpoint Integration

**Files:**
- Modify: `app/admin/router.py`
- Test: `tests/unit/admin/test_alerts_endpoints.py` (add job endpoint tests)

**Step 1: Add job endpoint to router**

Add to `app/admin/router.py`:

```python
from app.services.alerts.job import AlertEvaluatorJob

@router.post("/jobs/evaluate-alerts")
async def evaluate_alerts(
    workspace_id: UUID = Query(...),
    dry_run: bool = Query(False),
    _: str = Depends(require_admin_token),
):
    """Run alert evaluation job for workspace."""
    job = AlertEvaluatorJob(_db_pool)
    result = await job.run(workspace_id=workspace_id, dry_run=dry_run)

    if not result["lock_acquired"]:
        raise HTTPException(status_code=409, detail="Job already running")

    return result
```

**Step 2: Add test for job endpoint**

Add to `tests/unit/admin/test_alerts_endpoints.py`:

```python
class TestAlertJobEndpoint:
    """Tests for /admin/jobs/evaluate-alerts endpoint."""

    def test_evaluate_alerts_success(self, client, mock_db_pool):
        """Evaluate alerts job returns metrics."""
        workspace_id = uuid4()

        mock_job = MagicMock()
        mock_job.run = AsyncMock(
            return_value={
                "lock_acquired": True,
                "status": "completed",
                "metrics": {"rules_loaded": 0},
            }
        )

        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.services.alerts.job.AlertEvaluatorJob", return_value=mock_job
        ):
            response = client.post(
                f"/admin/jobs/evaluate-alerts?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"
```

**Step 3: Run tests**

```bash
pytest tests/unit/admin/test_alerts_endpoints.py -v
```

**Step 4: Commit**

```bash
git add app/admin/router.py tests/unit/admin/test_alerts_endpoints.py
git commit -m "feat(admin): add evaluate-alerts job endpoint"
```

---

## Task 9: UI - Alert List Page

**Files:**
- Create: `app/admin/templates/alerts.html`
- Modify: `app/admin/alerts.py` (add HTML route)

**Step 1: Create alerts list template**

Create `app/admin/templates/alerts.html`:

```html
{% extends "layout.html" %}

{% block title %}Alerts - Admin{% endblock %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h2 class="card-title">Alerts</h2>
        <span class="pagination-info">{{ total }} total</span>
    </div>

    <!-- Quick filter chips -->
    <div class="filter-chips" style="padding: 12px 16px; border-bottom: 1px solid var(--border);">
        <a href="?workspace_id={{ workspace_id }}&status=active&acknowledged=false"
           class="chip {% if status_filter == 'active' and acknowledged_filter == false %}active{% endif %}">
            Needs attention
        </a>
        <a href="?workspace_id={{ workspace_id }}&status=active"
           class="chip {% if status_filter == 'active' and acknowledged_filter is none %}active{% endif %}">
            Active
        </a>
        <a href="?workspace_id={{ workspace_id }}&status=resolved"
           class="chip {% if status_filter == 'resolved' %}active{% endif %}">
            Resolved
        </a>
        <a href="?workspace_id={{ workspace_id }}"
           class="chip {% if not status_filter %}active{% endif %}">
            All
        </a>
    </div>

    <!-- Filter bar -->
    <form class="filters" method="get">
        <input type="hidden" name="workspace_id" value="{{ workspace_id }}">
        <select name="severity">
            <option value="">All severities</option>
            <option value="high" {% if severity_filter == 'high' %}selected{% endif %}>High</option>
            <option value="medium" {% if severity_filter == 'medium' %}selected{% endif %}>Medium</option>
            <option value="low" {% if severity_filter == 'low' %}selected{% endif %}>Low</option>
        </select>
        <select name="rule_type">
            <option value="">All types</option>
            <option value="drift_spike" {% if rule_type_filter == 'drift_spike' %}selected{% endif %}>Drift Spike</option>
            <option value="confidence_drop" {% if rule_type_filter == 'confidence_drop' %}selected{% endif %}>Confidence Drop</option>
            <option value="combo" {% if rule_type_filter == 'combo' %}selected{% endif %}>Combo</option>
        </select>
        <button type="submit">Filter</button>
    </form>

    {% if alerts %}
    <table>
        <thead>
            <tr>
                <th>Last Seen</th>
                <th>Severity</th>
                <th>Type</th>
                <th>Strategy</th>
                <th>Regime</th>
                <th>Status</th>
                <th>Ack</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for alert in alerts %}
            <tr onclick="window.location='/admin/alerts/{{ alert.id }}'" style="cursor: pointer;">
                <td style="color: var(--text-muted); font-size: 13px;">
                    {{ alert.last_seen.strftime('%Y-%m-%d %H:%M') if alert.last_seen else '-' }}
                </td>
                <td>
                    {% if alert.severity == 'high' %}
                    <span class="badge badge-rejected">HIGH</span>
                    {% elif alert.severity == 'medium' %}
                    <span class="badge badge-pending">MEDIUM</span>
                    {% else %}
                    <span class="badge badge-weak">LOW</span>
                    {% endif %}
                </td>
                <td>{{ alert.rule_type | replace('_', ' ') | title }}</td>
                <td style="font-family: monospace; font-size: 12px;">
                    {{ (alert.strategy_entity_id | string)[:8] }}...
                </td>
                <td>{{ alert.regime_key }}</td>
                <td>
                    {% if alert.status == 'active' %}
                    <span class="badge badge-type">Active</span>
                    {% else %}
                    <span class="badge badge-verified">Resolved</span>
                    {% endif %}
                </td>
                <td>
                    {% if alert.acknowledged %}
                    <span style="color: var(--success);">✓</span>
                    {% else %}
                    <span style="color: var(--text-muted);">-</span>
                    {% endif %}
                </td>
                <td onclick="event.stopPropagation();">
                    {% if not alert.acknowledged %}
                    <form method="post" action="/admin/alerts/{{ alert.id }}/acknowledge" style="display: inline;">
                        <button type="submit" class="btn btn-secondary btn-sm">Ack</button>
                    </form>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <!-- Pagination -->
    <div class="pagination">
        <span class="pagination-info">
            Showing {{ offset + 1 }}-{{ [offset + limit, total]|min }} of {{ total }}
        </span>
        <div class="pagination-buttons">
            {% if offset > 0 %}
            <a href="?workspace_id={{ workspace_id }}&status={{ status_filter or '' }}&limit={{ limit }}&offset={{ [offset - limit, 0]|max }}"
               class="btn btn-secondary">Previous</a>
            {% endif %}
            {% if offset + limit < total %}
            <a href="?workspace_id={{ workspace_id }}&status={{ status_filter or '' }}&limit={{ limit }}&offset={{ offset + limit }}"
               class="btn btn-secondary">Next</a>
            {% endif %}
        </div>
    </div>

    {% else %}
    <div class="empty-state">
        <h3>No alerts found</h3>
        <p>{% if status_filter %}Try adjusting your filters{% else %}No alerts have been triggered yet{% endif %}</p>
    </div>
    {% endif %}
</div>

<style>
.filter-chips {
    display: flex;
    gap: 8px;
}
.chip {
    padding: 6px 12px;
    border-radius: 16px;
    background: var(--bg-secondary);
    color: var(--text-muted);
    text-decoration: none;
    font-size: 13px;
}
.chip:hover {
    background: var(--border);
}
.chip.active {
    background: var(--primary);
    color: white;
}
</style>
{% endblock %}
```

**Step 2: Add HTML route to alerts.py**

Add to `app/admin/alerts.py`:

```python
from pathlib import Path
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/page", response_class=HTMLResponse)
async def alerts_page(
    request: Request,
    workspace_id: UUID = Query(...),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    rule_type: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(20),
    offset: int = Query(0),
    _: str = Depends(require_admin_token),
):
    """Render alerts list page."""
    repo = AlertsRepository(_db_pool)

    status_enum = AlertStatus(status) if status else None
    severity_enum = Severity(severity) if severity else None
    rule_type_enum = RuleType(rule_type) if rule_type else None

    alerts, total = await repo.list_events(
        workspace_id=workspace_id,
        status=status_enum,
        severity=severity_enum,
        acknowledged=acknowledged,
        rule_type=rule_type_enum,
        limit=limit,
        offset=offset,
    )

    return templates.TemplateResponse(
        "alerts.html",
        {
            "request": request,
            "alerts": alerts,
            "total": total,
            "workspace_id": workspace_id,
            "status_filter": status,
            "severity_filter": severity,
            "rule_type_filter": rule_type,
            "acknowledged_filter": acknowledged,
            "limit": limit,
            "offset": offset,
        },
    )
```

**Step 3: Commit**

```bash
git add app/admin/templates/alerts.html app/admin/alerts.py
git commit -m "feat(admin): add alerts list page UI"
```

---

## Task 10: UI - Alert Detail Page

**Files:**
- Create: `app/admin/templates/alert_detail.html`
- Modify: `app/admin/alerts.py` (add detail HTML route)

**Step 1: Create alert detail template**

Create `app/admin/templates/alert_detail.html`:

```html
{% extends "layout.html" %}

{% block title %}Alert Detail - Admin{% endblock %}

{% block content %}
<div class="card">
    <div class="card-header">
        <div style="display: flex; align-items: center; gap: 12px;">
            <a href="/admin/alerts/page?workspace_id={{ alert.workspace_id }}" class="btn btn-secondary">← Back</a>
            <h2 class="card-title" style="margin: 0;">
                {% if alert.severity == 'high' %}
                <span class="badge badge-rejected">HIGH</span>
                {% elif alert.severity == 'medium' %}
                <span class="badge badge-pending">MEDIUM</span>
                {% else %}
                <span class="badge badge-weak">LOW</span>
                {% endif %}
                {{ alert.rule_type | replace('_', ' ') | title }}
            </h2>
            {% if alert.status == 'active' %}
            <span class="badge badge-type">Active</span>
            {% else %}
            <span class="badge badge-verified">Resolved</span>
            {% endif %}
        </div>
    </div>

    <div style="padding: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
        <!-- Why it triggered -->
        <div class="card" style="margin: 0;">
            <div class="card-header">
                <h3 class="card-title">Why it triggered</h3>
            </div>
            <div style="padding: 16px;">
                {% if alert.rule_type == 'drift_spike' %}
                <p><strong>Drift Score:</strong> {{ alert.context_json.current_drift | round(3) }}
                   (threshold: {{ alert.context_json.threshold }})</p>
                <p><strong>Consecutive buckets:</strong> {{ alert.context_json.consecutive_buckets }}</p>
                <p><strong>Recent values:</strong> {{ alert.context_json.recent_drifts }}</p>
                {% elif alert.rule_type == 'confidence_drop' %}
                <p><strong>Trend Delta:</strong> {{ alert.context_json.trend_delta | round(3) }}
                   (threshold: -{{ alert.context_json.trend_threshold }})</p>
                <p><strong>First half avg:</strong> {{ alert.context_json.first_half_avg | round(3) }}</p>
                <p><strong>Second half avg:</strong> {{ alert.context_json.second_half_avg | round(3) }}</p>
                {% elif alert.rule_type == 'combo' %}
                <div style="margin-bottom: 12px;">
                    <strong>Drift:</strong>
                    <p style="margin-left: 16px;">Score: {{ alert.context_json.drift.current_drift | round(3) }}</p>
                </div>
                <div>
                    <strong>Confidence:</strong>
                    <p style="margin-left: 16px;">Delta: {{ alert.context_json.confidence.trend_delta | round(3) }}</p>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- Timeline -->
        <div class="card" style="margin: 0;">
            <div class="card-header">
                <h3 class="card-title">Timeline</h3>
            </div>
            <div style="padding: 16px;">
                <p><strong>First seen:</strong> {{ alert.first_seen.strftime('%Y-%m-%d %H:%M:%S') if alert.first_seen else '-' }}</p>
                <p><strong>Activated:</strong> {{ alert.activated_at.strftime('%Y-%m-%d %H:%M:%S') if alert.activated_at else '-' }}</p>
                <p><strong>Last seen:</strong> {{ alert.last_seen.strftime('%Y-%m-%d %H:%M:%S') if alert.last_seen else '-' }}</p>
                {% if alert.resolved_at %}
                <p><strong>Resolved:</strong> {{ alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                {% endif %}
                {% if alert.acknowledged %}
                <p><strong>Acknowledged:</strong> {{ alert.acknowledged_at.strftime('%Y-%m-%d %H:%M:%S') if alert.acknowledged_at else 'Yes' }}
                   {% if alert.acknowledged_by %}by {{ alert.acknowledged_by }}{% endif %}</p>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Details -->
    <div style="padding: 16px; border-top: 1px solid var(--border);">
        <h3>Details</h3>
        <table style="width: auto;">
            <tr><td style="padding-right: 24px;"><strong>Strategy:</strong></td><td style="font-family: monospace;">{{ alert.strategy_entity_id }}</td></tr>
            <tr><td><strong>Regime:</strong></td><td>{{ alert.regime_key }}</td></tr>
            <tr><td><strong>Timeframe:</strong></td><td>{{ alert.timeframe }}</td></tr>
            <tr><td><strong>Fingerprint:</strong></td><td style="font-family: monospace; font-size: 12px;">{{ alert.fingerprint }}</td></tr>
        </table>
    </div>

    <!-- Actions -->
    <div style="padding: 16px; border-top: 1px solid var(--border); display: flex; gap: 12px;">
        {% if alert.context_json.deep_link %}
        <a href="/admin/analytics/regimes?workspace_id={{ alert.workspace_id }}&strategy={{ alert.context_json.deep_link.strategy_entity_id }}&timeframe={{ alert.context_json.deep_link.timeframe }}&regime={{ alert.context_json.deep_link.regime_key }}"
           class="btn btn-primary">View in Analytics</a>
        {% endif %}

        {% if alert.acknowledged %}
        <form method="post" action="/admin/alerts/{{ alert.id }}/unacknowledge">
            <button type="submit" class="btn btn-secondary">Unacknowledge</button>
        </form>
        {% else %}
        <form method="post" action="/admin/alerts/{{ alert.id }}/acknowledge">
            <button type="submit" class="btn btn-secondary">Acknowledge</button>
        </form>
        {% endif %}
    </div>
</div>
{% endblock %}
```

**Step 2: Add detail HTML route**

Add to `app/admin/alerts.py`:

```python
@router.get("/{event_id}/detail", response_class=HTMLResponse)
async def alert_detail_page(
    request: Request,
    event_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Render alert detail page."""
    repo = AlertsRepository(_db_pool)
    alert = await repo.get_event(event_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return templates.TemplateResponse(
        "alert_detail.html",
        {
            "request": request,
            "alert": alert,
        },
    )
```

**Step 3: Commit**

```bash
git add app/admin/templates/alert_detail.html app/admin/alerts.py
git commit -m "feat(admin): add alert detail page UI"
```

---

## Task 11: UI - Recent Alerts Panel

**Files:**
- Modify: `app/admin/templates/analytics_regimes.html`
- Modify: `app/admin/analytics.py` (fetch recent alerts)

**Step 1: Add alerts fetch to analytics endpoint**

Modify the analytics page to fetch recent alerts and pass to template.

**Step 2: Add Recent Alerts panel to template**

Add to the Overview section of `analytics_regimes.html`:

```html
<!-- Recent Alerts Panel -->
<div class="card recent-alerts-panel">
    <div class="card-header">
        <h3 class="card-title">Recent Alerts</h3>
        <div class="toggle-group">
            <button class="toggle-btn active" data-filter="active">Active</button>
            <button class="toggle-btn" data-filter="all">All</button>
        </div>
    </div>
    <div class="alerts-list">
        {% for alert in recent_alerts[:7] %}
        <div class="alert-row" onclick="window.location='/admin/alerts/{{ alert.id }}/detail'">
            <span class="badge {% if alert.severity == 'high' %}badge-rejected{% else %}badge-pending{% endif %}">
                {{ alert.severity | upper }}
            </span>
            <span class="alert-type">{{ alert.rule_type | replace('_', ' ') | title }}</span>
            <span class="alert-regime">{{ alert.regime_key }}</span>
            <span class="alert-status {% if alert.status == 'active' %}active{% endif %}">
                {{ alert.status }}
            </span>
        </div>
        {% else %}
        <div class="empty-state" style="padding: 24px;">
            <p>No recent alerts</p>
        </div>
        {% endfor %}
    </div>
    <div class="card-footer">
        <a href="/admin/alerts/page?workspace_id={{ workspace_id }}">View all alerts →</a>
    </div>
</div>
```

**Step 3: Commit**

```bash
git add app/admin/templates/analytics_regimes.html app/admin/analytics.py
git commit -m "feat(admin): add recent alerts panel to analytics overview"
```

---

## Task 12: Wire Everything Together

**Files:**
- Modify: `app/admin/router.py` (include alerts router)
- Modify: `app/main.py` (initialize alerts db pool)

**Step 1: Include alerts router**

Add to `app/admin/router.py`:

```python
from app.admin import alerts

# In the router setup
router.include_router(alerts.router)
```

**Step 2: Initialize alerts db pool in main.py**

Add to startup:

```python
from app.admin import alerts as admin_alerts

@app.on_event("startup")
async def startup():
    # ... existing startup code ...
    admin_alerts.set_db_pool(db_pool)
```

**Step 3: Run full test suite**

```bash
pytest tests/ -v
```

**Step 4: Commit**

```bash
git add app/admin/router.py app/main.py
git commit -m "feat(admin): wire alerts router and db pool"
```

---

## Task 13: Final Integration Tests

**Files:**
- Create: `tests/integration/test_alerts_integration.py`

**Step 1: Write integration tests**

```python
"""Integration tests for alerts system."""

import pytest
from uuid import uuid4

# Test full flow: create rule → run job → verify alert created
# Test alert lifecycle: active → ack → resolve
# Test UI navigation
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_alerts_integration.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_alerts_integration.py
git commit -m "test(alerts): add integration tests for alert system"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Database migrations | N/A |
| 2 | Pydantic models | test_models.py |
| 3 | Rule evaluators | test_evaluators.py |
| 4 | Alert repository | test_alerts_repo.py |
| 5 | Transition layer | test_transitions.py |
| 6 | Evaluator job | test_job.py |
| 7 | API endpoints (rules) | test_alerts_endpoints.py |
| 8 | Job endpoint | test_alerts_endpoints.py |
| 9 | Alert list page UI | Manual |
| 10 | Alert detail page UI | Manual |
| 11 | Recent alerts panel | Manual |
| 12 | Wire together | Full suite |
| 13 | Integration tests | test_alerts_integration.py |

**Total: 13 tasks**

After completing all tasks, run:
```bash
pytest tests/ -v
black app/ tests/
flake8 app/ tests/
```
