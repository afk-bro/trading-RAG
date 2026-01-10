# Testing package - Test generation and orchestration for trading strategies
"""
Testing package for generating and orchestrating backtest runs.

This module provides:
- RunPlan: Plan for executing a batch of backtest variants
- RunVariant: Individual variant with parameter overrides
- RunResult: Results from executing a variant
- GeneratorConstraints: Constraints for test generation
- TestGenerator: Generates RunPlan variants from a base ExecutionSpec
"""

from app.services.testing.models import (
    canonical_json,
    hash_variant,
    apply_overrides,
    RunPlanStatus,
    GeneratorConstraints,
    RunVariant,
    RunPlan,
    VariantMetrics,
    RunResult,
)
from app.services.testing.test_generator import TestGenerator

__all__ = [
    "canonical_json",
    "hash_variant",
    "apply_overrides",
    "RunPlanStatus",
    "GeneratorConstraints",
    "RunVariant",
    "RunPlan",
    "VariantMetrics",
    "RunResult",
    "TestGenerator",
]
