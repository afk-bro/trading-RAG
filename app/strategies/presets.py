"""Strategy preset definitions.

Presets are static strategy templates that can be instantiated into any
workspace. Each preset defines the parameter schema, default params,
and strategy spec (rules, events) for a particular trading approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParamDef:
    """Single parameter definition within a preset's param schema."""

    type: str  # "int", "float", "str"
    default: Any
    choices: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""


@dataclass(frozen=True)
class StrategyPreset:
    """Immutable strategy template."""

    slug: str
    name: str
    description: str
    engine: str  # matches StrategyEngine values
    param_schema: dict[str, ParamDef]
    default_params: dict[str, Any]
    events: list[str] = field(default_factory=list)
    tags: dict[str, list[str]] = field(default_factory=dict)
    version: str = "1.0"
    schema_version: str = ""

    def to_param_space(self) -> dict[str, Any]:
        """Convert param_schema to a tune/WFO-compatible param_space dict.

        Returns format expected by the backtest tuner:
        ``{"param_name": [val1, val2, ...], ...}``
        """
        space: dict[str, Any] = {}
        for name, pdef in self.param_schema.items():
            if pdef.choices is not None:
                space[name] = pdef.choices
            elif (
                pdef.min is not None and pdef.max is not None and pdef.step is not None
            ):
                # Generate range
                vals: list[Any] = []
                v = pdef.min
                while v <= pdef.max + 1e-9:
                    vals.append(round(v, 6) if pdef.type == "float" else int(v))
                    v += pdef.step
                space[name] = vals
            else:
                space[name] = [pdef.default]
        return space

    def to_config_snapshot(self) -> dict[str, Any]:
        """Build config_snapshot dict for strategy_versions table."""
        return {
            "preset_slug": self.slug,
            "preset_version": self.version,
            "engine": self.engine,
            "schema_version": self.schema_version,
            "params": dict(self.default_params),
            "param_schema": {
                k: {
                    "type": v.type,
                    "default": v.default,
                    **({"choices": v.choices} if v.choices else {}),
                    **({"min": v.min} if v.min is not None else {}),
                    **({"max": v.max} if v.max is not None else {}),
                    **({"step": v.step} if v.step is not None else {}),
                    **({"description": v.description} if v.description else {}),
                }
                for k, v in self.param_schema.items()
            },
            "events": list(self.events),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "engine": self.engine,
            "version": self.version,
            "schema_version": self.schema_version,
            "param_schema": {
                k: {
                    "type": v.type,
                    "default": v.default,
                    **({"choices": v.choices} if v.choices else {}),
                    **({"min": v.min} if v.min is not None else {}),
                    **({"max": v.max} if v.max is not None else {}),
                    **({"step": v.step} if v.step is not None else {}),
                    **({"description": v.description} if v.description else {}),
                }
                for k, v in self.param_schema.items()
            },
            "default_params": dict(self.default_params),
            "events": list(self.events),
            "tags": dict(self.tags),
        }


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

# FROZEN: do not mutate. Clone to ny-am-orb-v1.1 for changes.
NY_AM_ORB_V1 = StrategyPreset(
    slug="ny-am-orb-v1",
    name="NY AM ORB v1",
    description=(
        "Opening Range Breakout during the New York AM session. "
        "Waits for the opening range to form, then trades confirmed "
        "breakouts with a stop at the opposite OR level."
    ),
    engine="orb",
    param_schema={
        "or_minutes": ParamDef(
            type="int",
            default=30,
            choices=[15, 30, 60],
            description="Opening range window in minutes from session open",
        ),
        "confirm_mode": ParamDef(
            type="str",
            default="close-beyond",
            choices=["close-beyond", "retest"],
            description="How breakout is confirmed",
        ),
        "stop_mode": ParamDef(
            type="str",
            default="or-opposite",
            choices=["or-opposite", "fixed-ticks"],
            description="Stop placement method",
        ),
        "target_r": ParamDef(
            type="float",
            default=1.5,
            min=1.0,
            max=2.0,
            step=0.1,
            description="Target as multiple of risk (R:R)",
        ),
        "max_trades": ParamDef(
            type="int",
            default=1,
            choices=[1, 2],
            description="Maximum trades per session",
        ),
        "session": ParamDef(
            type="str",
            default="NY AM",
            choices=["NY AM", "NY PM", "London"],
            description="Trading session window",
        ),
    },
    default_params={
        "or_minutes": 30,
        "confirm_mode": "close-beyond",
        "stop_mode": "or-opposite",
        "target_r": 1.5,
        "max_trades": 1,
        "session": "NY AM",
    },
    events=[
        "orb_range_update",
        "orb_range_locked",
        "setup_valid",
        "entry_signal",
    ],
    tags={
        "strategy_archetypes": ["breakout", "opening-range"],
        "indicators": ["price-action", "session-range"],
        "timeframe_buckets": ["intraday"],
        "topics": ["equities", "futures", "forex"],
    },
    version="1.0",
    schema_version="1.0.0",
)


# All available presets keyed by slug
PRESETS: dict[str, StrategyPreset] = {
    NY_AM_ORB_V1.slug: NY_AM_ORB_V1,
}
