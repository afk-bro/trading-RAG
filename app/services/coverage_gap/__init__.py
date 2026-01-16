"""Coverage gap detection for Pine script matching."""

from app.services.coverage_gap.detector import (
    CoverageAssessment,
    CoverageReasonCode,
    assess_coverage,
    compute_intent_signature,
    generate_suggestions,
)
from app.services.coverage_gap.explanation import (
    ExplanationError,
    StrategyExplanation,
    generate_strategy_explanation,
)
from app.services.coverage_gap.repository import MatchRunRepository

__all__ = [
    "CoverageAssessment",
    "CoverageReasonCode",
    "ExplanationError",
    "MatchRunRepository",
    "StrategyExplanation",
    "assess_coverage",
    "compute_intent_signature",
    "generate_strategy_explanation",
    "generate_suggestions",
]
