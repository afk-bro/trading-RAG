"""
Policy Engine: Evaluates trade intents against configurable rules.

The Policy Engine is the gatekeeper between what the trading brain wants
to do (TradeIntent) and what actually gets executed. It provides:

1. A pluggable rule interface for adding new safety checks
2. Ordered rule evaluation with short-circuit rejection
3. Full audit trail of which rules passed/failed
4. Support for rule-specific configuration

Rules are evaluated in priority order. The first rule to reject
terminates evaluation (short-circuit).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time

import structlog

from app.schemas import (
    TradeIntent,
    PolicyDecision,
    PolicyReason,
    CurrentState,
)


logger = structlog.get_logger(__name__)


# =============================================================================
# Rule Interface
# =============================================================================


@dataclass
class RuleResult:
    """Result from evaluating a single policy rule."""

    passed: bool
    reason: Optional[PolicyReason] = None
    reason_details: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    modified_quantity: Optional[float] = None


class PolicyRule(ABC):
    """
    Abstract base class for policy rules.

    Rules are evaluated in order by the PolicyEngine. Each rule
    can approve (pass), reject (fail with reason), or warn.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique rule identifier (e.g., 'kill_switch', 'drift_guard')."""
        pass

    @property
    def priority(self) -> int:
        """
        Evaluation priority (lower = earlier).

        Default is 100. Use lower numbers for critical safety rules
        that should short-circuit before others.
        """
        return 100

    @property
    def enabled(self) -> bool:
        """Whether rule is active. Override to make configurable."""
        return True

    @abstractmethod
    def evaluate(
        self,
        intent: TradeIntent,
        state: CurrentState,
    ) -> RuleResult:
        """
        Evaluate the rule against an intent.

        Args:
            intent: The trade intent to evaluate
            state: Current system/market state

        Returns:
            RuleResult with passed=True/False and optional details
        """
        pass


# =============================================================================
# Built-in Rules
# =============================================================================


class KillSwitchRule(PolicyRule):
    """
    Emergency kill switch - rejects all intents when active.

    This is the highest priority rule and should always be evaluated first.
    When the kill switch is active, no trading actions are allowed.
    """

    @property
    def name(self) -> str:
        return "kill_switch"

    @property
    def priority(self) -> int:
        return 0  # Highest priority - evaluate first

    def evaluate(
        self,
        intent: TradeIntent,
        state: CurrentState,
    ) -> RuleResult:
        if state.kill_switch_active:
            return RuleResult(
                passed=False,
                reason=PolicyReason.KILL_SWITCH_ACTIVE,
                reason_details="Global kill switch is active. All trading halted.",
            )

        if not state.trading_enabled:
            return RuleResult(
                passed=False,
                reason=PolicyReason.KILL_SWITCH_ACTIVE,
                reason_details="Trading is disabled for this workspace.",
            )

        return RuleResult(passed=True)


class DriftGuardRule(PolicyRule):
    """
    Regime drift protection - rejects intents when market regime
    has drifted significantly from training conditions.

    Uses the regime_distance_z from v1.5 Live Intelligence:
    - z < 2.0: Normal regime, trading allowed
    - z >= 2.0: Moderate drift, warning added
    - z >= 3.0: Severe drift, intent rejected

    The z-score measures how many standard deviations the current
    regime features are from the strategy's training distribution.
    """

    def __init__(
        self,
        warning_threshold: float = 2.0,
        reject_threshold: float = 3.0,
    ):
        """
        Initialize drift guard with configurable thresholds.

        Args:
            warning_threshold: Z-score above which to add warnings
            reject_threshold: Z-score above which to reject intent
        """
        self.warning_threshold = warning_threshold
        self.reject_threshold = reject_threshold

    @property
    def name(self) -> str:
        return "drift_guard"

    @property
    def priority(self) -> int:
        return 10  # High priority, after kill switch

    def evaluate(
        self,
        intent: TradeIntent,
        state: CurrentState,
    ) -> RuleResult:
        # If no regime data, pass with warning
        if state.regime_distance_z is None:
            return RuleResult(
                passed=True,
                warnings=["No regime distance data available - drift check skipped"],
            )

        z = state.regime_distance_z

        # Severe drift - reject
        if z >= self.reject_threshold:
            return RuleResult(
                passed=False,
                reason=PolicyReason.REGIME_DRIFT,
                reason_details=(
                    f"Regime drift too high (z={z:.2f} >= {self.reject_threshold}). "
                    f"Current market conditions differ significantly from training."
                ),
            )

        # Moderate drift - warn but allow
        warnings = []
        if z >= self.warning_threshold:
            warnings.append(
                f"Elevated regime drift (z={z:.2f}). "
                f"Consider reducing position size or monitoring closely."
            )

        return RuleResult(passed=True, warnings=warnings)


class MaxDrawdownRule(PolicyRule):
    """
    Rejects new positions when daily drawdown exceeds threshold.

    Only blocks opening new positions - allows closing positions
    to reduce exposure.
    """

    def __init__(self, max_drawdown_pct: float = 5.0):
        """
        Args:
            max_drawdown_pct: Maximum daily drawdown percentage (e.g., 5.0 = 5%)
        """
        self.max_drawdown_pct = max_drawdown_pct

    @property
    def name(self) -> str:
        return "max_drawdown"

    @property
    def priority(self) -> int:
        return 20

    def evaluate(
        self,
        intent: TradeIntent,
        state: CurrentState,
    ) -> RuleResult:
        # Only check for opening positions
        from app.schemas import IntentAction
        opening_actions = {
            IntentAction.OPEN_LONG,
            IntentAction.OPEN_SHORT,
            IntentAction.SCALE_IN,
        }

        if intent.action not in opening_actions:
            return RuleResult(passed=True)

        # No drawdown data - pass with warning
        if state.max_drawdown_today is None:
            return RuleResult(
                passed=True,
                warnings=["No drawdown data available - check skipped"],
            )

        if state.max_drawdown_today >= self.max_drawdown_pct:
            return RuleResult(
                passed=False,
                reason=PolicyReason.MAX_DRAWDOWN_EXCEEDED,
                reason_details=(
                    f"Daily drawdown ({state.max_drawdown_today:.1f}%) exceeds "
                    f"limit ({self.max_drawdown_pct:.1f}%). New positions blocked."
                ),
            )

        return RuleResult(passed=True)


# =============================================================================
# Policy Engine
# =============================================================================


class PolicyEngine:
    """
    Evaluates trade intents against a set of policy rules.

    The engine maintains an ordered list of rules and evaluates
    them in priority order. Evaluation stops at the first rejection.
    """

    def __init__(self, rules: Optional[list[PolicyRule]] = None):
        """
        Initialize with a list of rules.

        Args:
            rules: List of PolicyRule instances. If None, uses default rules.
        """
        if rules is None:
            rules = self._default_rules()

        # Sort by priority (lower = earlier)
        self._rules = sorted(rules, key=lambda r: r.priority)

        logger.info(
            "PolicyEngine initialized",
            rules=[r.name for r in self._rules],
            rule_count=len(self._rules),
        )

    @staticmethod
    def _default_rules() -> list[PolicyRule]:
        """Get default rule set."""
        return [
            KillSwitchRule(),
            DriftGuardRule(),
            MaxDrawdownRule(),
        ]

    @property
    def rules(self) -> list[PolicyRule]:
        """Get current rules (read-only)."""
        return list(self._rules)

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule and re-sort by priority."""
        self._rules.append(rule)
        self._rules = sorted(self._rules, key=lambda r: r.priority)
        logger.info("Rule added", rule=rule.name, priority=rule.priority)

    def evaluate(
        self,
        intent: TradeIntent,
        state: Optional[CurrentState] = None,
    ) -> PolicyDecision:
        """
        Evaluate an intent against all policy rules.

        Args:
            intent: The trade intent to evaluate
            state: Current system state (uses defaults if not provided)

        Returns:
            PolicyDecision with approval status and audit trail
        """
        start_time = time.perf_counter()

        # Use default state if not provided
        if state is None:
            state = CurrentState()

        rules_evaluated = []
        rules_passed = []
        rules_failed = []
        all_warnings = []

        log = logger.bind(
            intent_id=str(intent.id),
            correlation_id=intent.correlation_id,
            action=intent.action.value,
            symbol=intent.symbol,
        )

        # Evaluate rules in priority order
        for rule in self._rules:
            if not rule.enabled:
                log.debug("Rule disabled, skipping", rule=rule.name)
                continue

            rules_evaluated.append(rule.name)

            try:
                result = rule.evaluate(intent, state)
            except Exception as e:
                log.error("Rule evaluation failed", rule=rule.name, error=str(e))
                # Treat errors as failures (fail-safe)
                rules_failed.append(rule.name)
                return PolicyDecision(
                    approved=False,
                    reason=PolicyReason.KILL_SWITCH_ACTIVE,  # Use as generic safety rejection
                    reason_details=f"Rule '{rule.name}' raised exception: {str(e)}",
                    rules_evaluated=rules_evaluated,
                    rules_passed=rules_passed,
                    rules_failed=rules_failed,
                    evaluation_ms=int((time.perf_counter() - start_time) * 1000),
                )

            # Collect warnings
            all_warnings.extend(result.warnings)

            if result.passed:
                rules_passed.append(rule.name)
                log.debug("Rule passed", rule=rule.name)
            else:
                rules_failed.append(rule.name)
                log.info(
                    "Rule rejected intent",
                    rule=rule.name,
                    reason=result.reason.value if result.reason else None,
                    details=result.reason_details,
                )

                # Short-circuit on first rejection
                return PolicyDecision(
                    approved=False,
                    reason=result.reason or PolicyReason.KILL_SWITCH_ACTIVE,
                    reason_details=result.reason_details,
                    rules_evaluated=rules_evaluated,
                    rules_passed=rules_passed,
                    rules_failed=rules_failed,
                    warnings=all_warnings,
                    modified_quantity=result.modified_quantity,
                    evaluation_ms=int((time.perf_counter() - start_time) * 1000),
                )

        # All rules passed
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log.info(
            "Intent approved",
            rules_passed=rules_passed,
            warnings=all_warnings,
            elapsed_ms=elapsed_ms,
        )

        return PolicyDecision(
            approved=True,
            reason=PolicyReason.ALL_RULES_PASSED,
            reason_details=f"Passed all {len(rules_passed)} policy rules",
            rules_evaluated=rules_evaluated,
            rules_passed=rules_passed,
            rules_failed=[],
            warnings=all_warnings,
            evaluation_ms=elapsed_ms,
        )
