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
            trigger_value=drift_result.trigger_value,
            context={
                "drift": drift_result.context,
                "confidence": confidence_result.context,
            },
        )
