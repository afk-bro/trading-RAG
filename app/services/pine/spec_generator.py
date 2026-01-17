"""
Strategy spec generator from Pine Script inputs.

Converts parsed Pine Script input definitions into strategy specs
suitable for automated backtesting and parameter sweeps.

Pipeline:
    Pine Script → Parser → PineInput[] → SpecGenerator → StrategySpec
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from app.services.pine.models import InputType, PineInput, PineScriptEntry


@dataclass
class ParamSpec:
    """
    Strategy parameter specification for backtesting.

    Derived from Pine Script input.* declarations.
    """

    name: str  # Variable name (normalized to snake_case)
    display_name: str  # Human-readable title from input
    type: str  # "int", "float", "bool", "string", "source", etc.
    default: Any  # Default value (primitive or None)
    default_expr: Optional[str] = None  # Expression if default is non-primitive

    # Bounds for numeric types
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

    # Options for enum-like inputs
    options: Optional[list[Any]] = None

    # Metadata
    group: Optional[str] = None
    tooltip: Optional[str] = None

    # Auto-discovery flags
    sweepable: bool = False  # Has min/max or options for parameter sweep
    priority: int = 0  # Higher = more likely to affect strategy behavior

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.type,
            "default": self.default,
            "sweepable": self.sweepable,
            "priority": self.priority,
        }
        if self.default_expr:
            d["default_expr"] = self.default_expr
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.step is not None:
            d["step"] = self.step
        if self.options:
            d["options"] = self.options
        if self.group:
            d["group"] = self.group
        if self.tooltip:
            d["tooltip"] = self.tooltip
        return d


@dataclass
class StrategySpec:
    """
    Complete strategy specification for automated backtesting.

    Generated from Pine Script parsing + auto-discovery.
    """

    name: str  # Strategy name (from title or filename)
    source_path: str  # Relative path to Pine Script
    pine_version: str  # "4", "5", "6"
    params: list[ParamSpec] = field(default_factory=list)

    # Metadata
    description: Optional[str] = None
    sha256: Optional[str] = None  # Content hash for change detection

    # Auto-generated param space for sweeps
    sweep_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for spec_json column."""
        return {
            "name": self.name,
            "source_path": self.source_path,
            "pine_version": self.pine_version,
            "params": [p.to_dict() for p in self.params],
            "description": self.description,
            "sha256": self.sha256,
            "sweep_config": self.sweep_config,
        }

    @property
    def sweepable_params(self) -> list[ParamSpec]:
        """Return only parameters marked as sweepable."""
        return [p for p in self.params if p.sweepable]


# Type mapping from Pine InputType to spec type
INPUT_TYPE_TO_SPEC_TYPE: dict[InputType, str] = {
    InputType.INT: "int",
    InputType.FLOAT: "float",
    InputType.BOOL: "bool",
    InputType.STRING: "string",
    InputType.COLOR: "color",
    InputType.SOURCE: "source",
    InputType.TIMEFRAME: "timeframe",
    InputType.SESSION: "session",
    InputType.SYMBOL: "symbol",
    InputType.UNKNOWN: "unknown",
}

# Keywords that indicate high-priority parameters
HIGH_PRIORITY_KEYWORDS = {
    "length",
    "period",
    "lookback",
    "threshold",
    "multiplier",
    "factor",
    "level",
    "atr",
    "rsi",
    "ema",
    "sma",
    "bb",
    "stop",
    "take",
    "profit",
    "loss",
}

# Keywords that indicate low-priority parameters (visual/display)
LOW_PRIORITY_KEYWORDS = {
    "color",
    "style",
    "display",
    "show",
    "plot",
    "line",
    "label",
    "table",
    "bgcolor",
}


def _normalize_name(name: str) -> str:
    """
    Normalize parameter name to snake_case.

    Examples:
        "RSI Length" -> "rsi_length"
        "BB Multiplier" -> "bb_multiplier"
        "fastEMA" -> "fast_ema"
    """
    import re

    # Insert underscore before uppercase letters (for camelCase)
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    # Replace spaces and special chars with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Convert to lowercase
    name = name.lower()
    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)
    # Strip leading/trailing underscores
    return name.strip("_")


def _compute_priority(input: PineInput) -> int:
    """
    Compute priority score for parameter (higher = more important).

    Based on:
    - Type (int/float > bool > string > source/color)
    - Has bounds (min/max defined)
    - Name contains trading keywords
    """
    priority = 0
    name_lower = input.name.lower()

    # Type priority
    if input.type in (InputType.INT, InputType.FLOAT):
        priority += 10
    elif input.type == InputType.BOOL:
        priority += 5

    # Has bounds (likely meant to be tuned)
    if input.min_value is not None or input.max_value is not None:
        priority += 15

    # Has options (discrete choices)
    if input.options:
        priority += 5

    # High-priority keywords
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in name_lower:
            priority += 10
            break

    # Low-priority keywords (reduce)
    for kw in LOW_PRIORITY_KEYWORDS:
        if kw in name_lower:
            priority -= 20
            break

    return priority


def _is_sweepable(input: PineInput) -> bool:
    """
    Determine if parameter is suitable for parameter sweeps.

    Sweepable if:
    - Numeric type (int/float) with min/max bounds
    - Has discrete options
    - Bool type (always 2 options)
    """
    # Bool is always sweepable (true/false)
    if input.type == InputType.BOOL:
        return True

    # Numeric with bounds
    if input.type in (InputType.INT, InputType.FLOAT):
        if input.min_value is not None and input.max_value is not None:
            return True

    # Has options array
    if input.options and len(input.options) > 1:
        return True

    return False


def _generate_sweep_config(params: list[ParamSpec]) -> dict:
    """
    Generate default sweep configuration for sweepable parameters.

    Returns dict mapping param names to sweep values.
    """
    config = {}

    for p in params:
        if not p.sweepable:
            continue

        if p.type == "bool":
            config[p.name] = [True, False]
        elif p.options:
            config[p.name] = p.options
        elif p.type == "int" and p.min_value is not None and p.max_value is not None:
            # Generate 5 evenly-spaced integers
            min_v = int(p.min_value)
            max_v = int(p.max_value)
            step = max(1, (max_v - min_v) // 4)
            config[p.name] = list(range(min_v, max_v + 1, step))[:5]
        elif p.type == "float" and p.min_value is not None and p.max_value is not None:
            # Generate 5 evenly-spaced floats
            min_v = p.min_value
            max_v = p.max_value
            step = (max_v - min_v) / 4
            config[p.name] = [round(min_v + i * step, 4) for i in range(5)]

    return config


def pine_input_to_param_spec(input: PineInput) -> ParamSpec:
    """
    Convert a PineInput to a ParamSpec.

    Args:
        input: Parsed Pine Script input definition

    Returns:
        ParamSpec for strategy specification
    """
    return ParamSpec(
        name=_normalize_name(input.name),
        display_name=input.name,
        type=INPUT_TYPE_TO_SPEC_TYPE.get(input.type, "unknown"),
        default=input.default,
        default_expr=input.default_expr,
        min_value=input.min_value,
        max_value=input.max_value,
        step=input.step,
        options=input.options,
        group=input.group,
        tooltip=input.tooltip,
        sweepable=_is_sweepable(input),
        priority=_compute_priority(input),
    )


def generate_strategy_spec(entry: PineScriptEntry) -> StrategySpec:
    """
    Generate a complete strategy specification from a parsed Pine Script.

    Args:
        entry: Parsed Pine Script entry from registry

    Returns:
        StrategySpec ready for KB ingestion and backtesting
    """
    # Convert inputs to param specs
    params = [pine_input_to_param_spec(inp) for inp in entry.inputs]

    # Sort by priority (highest first)
    params.sort(key=lambda p: p.priority, reverse=True)

    # Generate sweep config for sweepable params
    sweep_config = _generate_sweep_config(params)

    # Use title or filename as strategy name
    name = entry.title or entry.rel_path.split("/")[-1].replace(".pine", "")

    return StrategySpec(
        name=name,
        source_path=entry.rel_path,
        pine_version=entry.pine_version.value if entry.pine_version else "unknown",
        params=params,
        description=entry.short_title,
        sha256=entry.sha256,
        sweep_config=sweep_config,
    )


def generate_specs_from_registry(
    entries: list[PineScriptEntry],
    strategies_only: bool = True,
) -> list[StrategySpec]:
    """
    Generate strategy specs from a Pine registry.

    Args:
        entries: List of parsed Pine Script entries
        strategies_only: If True, skip indicators/libraries

    Returns:
        List of StrategySpec objects
    """
    from app.services.pine.models import ScriptType

    specs = []
    for entry in entries:
        # Filter to strategies if requested
        if strategies_only and entry.script_type != ScriptType.STRATEGY:
            continue

        specs.append(generate_strategy_spec(entry))

    return specs
