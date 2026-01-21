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

from app.services.intel.runner import (
    IntelRunner,
    compute_and_store_snapshot,
)

__all__ = [
    # Confidence computation
    "compute_regime",
    "compute_components",
    "compute_confidence",
    "ConfidenceContext",
    "ConfidenceResult",
    "DEFAULT_WEIGHTS",
    # Runner
    "IntelRunner",
    "compute_and_store_snapshot",
]
