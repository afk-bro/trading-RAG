"""
Parameter specifications and validation for strategy parameters.

Provides:
- ParamSpec: Type-safe parameter definition with bounds and defaults
- ParamType: Enum of supported parameter types
- validate_params: Validate params against specs
- repair_params: Fix out-of-bounds values with warnings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ParamType(str, Enum):
    """Supported parameter types."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"  # Categorical with choices


@dataclass
class ParamSpec:
    """
    Specification for a single strategy parameter.

    Defines the parameter's type, bounds, default, and metadata
    for validation, UI generation, and documentation.
    """

    name: str
    type: ParamType
    default: Any

    # Bounds (for numeric types)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None  # For grid search granularity

    # Choices (for enum type)
    choices: Optional[list[Any]] = None

    # Documentation
    description: str = ""
    unit: Optional[str] = None  # e.g., "bars", "percent", "dollars"

    # Constraints
    required: bool = True
    nullable: bool = False

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a value against this spec.

        Returns:
            (is_valid, error_message)
        """
        # Handle None
        if value is None:
            if self.nullable:
                return True, None
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None

        # Type validation
        if self.type == ParamType.INT:
            if not isinstance(value, (int, float)):
                return (
                    False,
                    f"Parameter '{self.name}' must be numeric, got {type(value).__name__}",
                )
            if isinstance(value, float) and not value.is_integer():
                return False, f"Parameter '{self.name}' must be an integer"

        elif self.type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return (
                    False,
                    f"Parameter '{self.name}' must be numeric, got {type(value).__name__}",
                )

        elif self.type == ParamType.BOOL:
            if not isinstance(value, bool):
                return (
                    False,
                    f"Parameter '{self.name}' must be boolean, got {type(value).__name__}",
                )

        elif self.type == ParamType.ENUM:
            if self.choices and value not in self.choices:
                return (
                    False,
                    f"Parameter '{self.name}' must be one of {self.choices}, got {value}",
                )

        # Bounds validation (for numeric types)
        if self.type in (ParamType.INT, ParamType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                return (
                    False,
                    f"Parameter '{self.name}' must be >= {self.min_value}, got {value}",
                )
            if self.max_value is not None and value > self.max_value:
                return (
                    False,
                    f"Parameter '{self.name}' must be <= {self.max_value}, got {value}",
                )

        return True, None

    def repair(self, value: Any) -> tuple[Any, Optional[str]]:
        """
        Attempt to repair an invalid value.

        Returns:
            (repaired_value, warning_message)
        """
        if value is None:
            if self.nullable:
                return None, None
            return (
                self.default,
                f"Parameter '{self.name}' was None, using default {self.default}",
            )

        # Type coercion
        if self.type == ParamType.INT:
            try:
                value = int(round(float(value)))
            except (ValueError, TypeError):
                return (
                    self.default,
                    f"Parameter '{self.name}' could not be converted to int",
                )

        elif self.type == ParamType.FLOAT:
            try:
                value = float(value)
            except (ValueError, TypeError):
                return (
                    self.default,
                    f"Parameter '{self.name}' could not be converted to float",
                )

        elif self.type == ParamType.BOOL:
            if not isinstance(value, bool):
                # Try common string conversions
                if isinstance(value, str):
                    if value.lower() in ("true", "1", "yes"):
                        value = True
                    elif value.lower() in ("false", "0", "no"):
                        value = False
                    else:
                        return (
                            self.default,
                            f"Parameter '{self.name}' could not be converted to bool",
                        )
                else:
                    value = bool(value)

        elif self.type == ParamType.ENUM:
            if self.choices and value not in self.choices:
                return (
                    self.default,
                    f"Parameter '{self.name}' value {value} not in choices, using default",
                )

        # Clamp to bounds
        warning = None
        if self.type in (ParamType.INT, ParamType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                warning = f"Parameter '{self.name}' clamped from {value} to min {self.min_value}"
                value = self.min_value
            if self.max_value is not None and value > self.max_value:
                warning = f"Parameter '{self.name}' clamped from {value} to max {self.max_value}"
                value = self.max_value

            # Convert to int if needed
            if self.type == ParamType.INT:
                value = int(value)

        return value, warning

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type.value,
            "default": self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "choices": self.choices,
            "description": self.description,
            "unit": self.unit,
            "required": self.required,
            "nullable": self.nullable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParamSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=ParamType(data["type"]),
            default=data["default"],
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            step=data.get("step"),
            choices=data.get("choices"),
            description=data.get("description", ""),
            unit=data.get("unit"),
            required=data.get("required", True),
            nullable=data.get("nullable", False),
        )


# =============================================================================
# Validation Functions
# =============================================================================


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repaired_params: Optional[dict] = None


def validate_params(
    params: dict[str, Any],
    specs: dict[str, ParamSpec],
    strict: bool = True,
) -> ValidationResult:
    """
    Validate parameters against specifications.

    Args:
        params: Parameter values to validate
        specs: Parameter specifications
        strict: If True, fail on unknown params; if False, ignore them

    Returns:
        ValidationResult with errors and warnings
    """
    errors = []
    warnings = []

    # Check for unknown parameters
    if strict:
        unknown = set(params.keys()) - set(specs.keys())
        if unknown:
            errors.append(f"Unknown parameters: {', '.join(sorted(unknown))}")

    # Check for missing required parameters
    for name, spec in specs.items():
        if spec.required and name not in params:
            errors.append(f"Missing required parameter: {name}")

    # Validate each parameter
    for name, value in params.items():
        if name not in specs:
            if strict:
                # Already reported as unknown
                pass
            else:
                warnings.append(f"Ignoring unknown parameter: {name}")
            continue

        spec = specs[name]
        is_valid, error = spec.validate(value)
        if not is_valid and error:
            errors.append(error)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def repair_params(
    params: dict[str, Any],
    specs: dict[str, ParamSpec],
) -> ValidationResult:
    """
    Repair parameters to conform to specifications.

    Attempts to fix:
    - Missing params (use defaults)
    - Out-of-bounds values (clamp to bounds)
    - Type mismatches (coerce if possible)

    Args:
        params: Parameter values to repair
        specs: Parameter specifications

    Returns:
        ValidationResult with repaired_params and warnings
    """
    repaired = {}
    warnings = []

    # Add defaults for missing params
    for name, spec in specs.items():
        if name not in params:
            if spec.required:
                repaired[name] = spec.default
                warnings.append(f"Using default for missing '{name}': {spec.default}")
        else:
            value = params[name]
            repaired_value, warning = spec.repair(value)
            repaired[name] = repaired_value
            if warning:
                warnings.append(warning)

    # Copy unknown params through unchanged
    for name, value in params.items():
        if name not in specs:
            repaired[name] = value
            warnings.append(f"Passing through unknown parameter: {name}")

    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=warnings,
        repaired_params=repaired,
    )


def validate_and_repair_params(
    params: dict[str, Any],
    specs: dict[str, ParamSpec],
) -> ValidationResult:
    """
    Validate and repair parameters in one pass.

    Returns valid params if possible, with warnings about repairs.

    Args:
        params: Parameter values
        specs: Parameter specifications

    Returns:
        ValidationResult with repaired_params if successful
    """
    # First try strict validation
    result = validate_params(params, specs, strict=False)

    if result.is_valid:
        # Still repair to ensure proper types and fill defaults
        repair_result = repair_params(params, specs)
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=result.warnings + repair_result.warnings,
            repaired_params=repair_result.repaired_params,
        )
    else:
        # Try to repair
        repair_result = repair_params(params, specs)
        # Re-validate repaired params (repaired_params is always set after repair)
        repaired = repair_result.repaired_params or {}
        revalidate = validate_params(repaired, specs, strict=False)

        return ValidationResult(
            is_valid=revalidate.is_valid,
            errors=revalidate.errors,
            warnings=result.warnings + repair_result.warnings,
            repaired_params=(
                repair_result.repaired_params if revalidate.is_valid else None
            ),
        )
