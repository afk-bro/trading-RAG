"""Alert models and schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


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
        """Return resolve consecutive buckets, defaulting to consecutive_buckets."""
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
