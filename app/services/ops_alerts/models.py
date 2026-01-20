"""Models and rule definitions for operational alerts."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID


class OpsRuleType(str, Enum):
    """Operational alert rule types."""

    HEALTH_DEGRADED = "health_degraded"
    WEAK_COVERAGE_P1 = "weak_coverage:P1"
    WEAK_COVERAGE_P2 = "weak_coverage:P2"
    DRIFT_SPIKE = "drift_spike"
    CONFIDENCE_DROP = "confidence_drop"
    # Strategy intelligence alerts (v1.5)
    STRATEGY_CONFIDENCE_LOW = "strategy_confidence_low"
    # Paper trading equity alerts
    WORKSPACE_DRAWDOWN_HIGH = "workspace_drawdown_high"


class Severity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DedupePeriod(str, Enum):
    """Deduplication window periods."""

    HOURLY = "hourly"
    DAILY = "daily"


@dataclass
class OpsAlertRule:
    """Definition of an operational alert rule."""

    rule_type: OpsRuleType
    description: str
    dedupe_period: DedupePeriod
    default_severity: Severity
    version: str = "v1"

    # For rules that need external data sources
    requires_health: bool = False
    requires_coverage: bool = False
    requires_match_runs: bool = False
    requires_strategy_intel: bool = False
    requires_equity_data: bool = False

    # Volume gating (for drift/confidence rules)
    min_sample_count: int = 0

    # Persistence gating (require N consecutive violations)
    persistence_count: int = 1

    def build_dedupe_key(self, date_str: str, extra: Optional[str] = None) -> str:
        """
        Build dedupe key for this rule.

        Format: {rule_type}:{date_str} or {rule_type}:{extra}:{date_str}
        """
        if extra:
            return f"{self.rule_type.value}:{extra}:{date_str}"
        return f"{self.rule_type.value}:{date_str}"

    def get_bucket_key(self, now: datetime) -> str:
        """Get the time bucket key based on dedupe period."""
        if self.dedupe_period == DedupePeriod.DAILY:
            return now.strftime("%Y-%m-%d")
        else:  # HOURLY
            return now.strftime("%Y-%m-%d-%H")


@dataclass
class AlertCondition:
    """Result of evaluating a rule condition."""

    triggered: bool
    severity: Severity
    payload: dict[str, Any] = field(default_factory=dict)
    dedupe_key: str = ""
    skip_reason: Optional[str] = None  # e.g., "insufficient_data", "gated"


@dataclass
class EvalContext:
    """Context passed to rule evaluation."""

    workspace_id: UUID
    now: datetime
    job_run_id: Optional[UUID] = None

    # Data sources (populated by evaluator)
    health_snapshot: Optional[Any] = None  # SystemHealthSnapshot
    coverage_stats: Optional[dict] = None  # Coverage counts by priority
    match_run_stats: Optional[dict] = None  # Recent match run aggregates
    strategy_intel: Optional[list[dict]] = None  # Active versions with intel snapshots
    equity_data: Optional[dict] = None  # Paper equity drawdown data per workspace


@dataclass
class EvalResult:
    """Result of evaluating all rules for a workspace."""

    workspace_id: UUID
    job_run_id: Optional[UUID]
    timestamp: datetime

    # Counts
    conditions_evaluated: int = 0
    alerts_triggered: int = 0
    alerts_new: int = 0
    alerts_updated: int = 0
    alerts_resolved: int = 0
    alerts_escalated: int = 0
    telegram_sent: int = 0

    # Auto-pause actions (guardrail)
    versions_auto_paused: int = 0
    auto_paused_version_ids: list[UUID] = field(default_factory=list)

    # Per-rule details
    by_rule_type: dict[str, dict] = field(default_factory=dict)

    # Any errors
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Rule Definitions (v1 - hardcoded)
# =============================================================================

RULES: dict[OpsRuleType, OpsAlertRule] = {
    OpsRuleType.HEALTH_DEGRADED: OpsAlertRule(
        rule_type=OpsRuleType.HEALTH_DEGRADED,
        description="System health is degraded or in error state",
        dedupe_period=DedupePeriod.DAILY,
        default_severity=Severity.HIGH,  # Escalates to CRITICAL for error/halted
        requires_health=True,
    ),
    OpsRuleType.WEAK_COVERAGE_P1: OpsAlertRule(
        rule_type=OpsRuleType.WEAK_COVERAGE_P1,
        description="P1 priority coverage gaps exist",
        dedupe_period=DedupePeriod.DAILY,
        default_severity=Severity.HIGH,
        requires_coverage=True,
    ),
    OpsRuleType.WEAK_COVERAGE_P2: OpsAlertRule(
        rule_type=OpsRuleType.WEAK_COVERAGE_P2,
        description="P2 priority coverage gaps exist",
        dedupe_period=DedupePeriod.DAILY,
        default_severity=Severity.MEDIUM,
        requires_coverage=True,
    ),
    OpsRuleType.DRIFT_SPIKE: OpsAlertRule(
        rule_type=OpsRuleType.DRIFT_SPIKE,
        description="Match quality has drifted significantly from baseline",
        dedupe_period=DedupePeriod.HOURLY,
        default_severity=Severity.MEDIUM,
        requires_match_runs=True,
        min_sample_count=50,  # Volume gate
    ),
    OpsRuleType.CONFIDENCE_DROP: OpsAlertRule(
        rule_type=OpsRuleType.CONFIDENCE_DROP,
        description="Match confidence has dropped below threshold",
        dedupe_period=DedupePeriod.HOURLY,
        default_severity=Severity.MEDIUM,
        requires_match_runs=True,
        min_sample_count=50,  # Volume gate
    ),
    OpsRuleType.STRATEGY_CONFIDENCE_LOW: OpsAlertRule(
        rule_type=OpsRuleType.STRATEGY_CONFIDENCE_LOW,
        description="Strategy version confidence score is below threshold",
        dedupe_period=DedupePeriod.DAILY,
        default_severity=Severity.MEDIUM,  # Escalates to HIGH for critical
        requires_strategy_intel=True,
        persistence_count=2,  # Require 2 consecutive low snapshots
    ),
    OpsRuleType.WORKSPACE_DRAWDOWN_HIGH: OpsAlertRule(
        rule_type=OpsRuleType.WORKSPACE_DRAWDOWN_HIGH,
        description="Paper trading drawdown exceeds threshold",
        dedupe_period=DedupePeriod.DAILY,
        default_severity=Severity.MEDIUM,  # Escalates to HIGH for critical
        requires_equity_data=True,
        persistence_count=2,  # Require 2 consecutive breaches
    ),
}


def get_all_rules() -> list[OpsAlertRule]:
    """Get all defined rules."""
    return list(RULES.values())


def get_rule(rule_type: OpsRuleType) -> OpsAlertRule:
    """Get a specific rule definition."""
    return RULES[rule_type]
