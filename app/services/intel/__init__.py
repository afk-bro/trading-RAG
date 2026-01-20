"""Strategy Intelligence computation module.

v1.5 Live Intelligence - computes regime classification and confidence scores
for strategy versions based on market conditions and backtest performance.
"""

from app.services.intel.confidence import (
    compute_regime,
    compute_components,
    compute_confidence,
    ConfidenceContext,
    ConfidenceResult,
    DEFAULT_WEIGHTS,
)

__all__ = [
    "compute_regime",
    "compute_components",
    "compute_confidence",
    "ConfidenceContext",
    "ConfidenceResult",
    "DEFAULT_WEIGHTS",
]
