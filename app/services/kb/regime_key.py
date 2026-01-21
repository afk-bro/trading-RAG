"""
Regime key utilities for v1.5 dimensional taxonomy.

Regime keys are canonical composites of fixed dimensions:
- trend: uptrend | downtrend | flat
- vol: low_vol | mid_vol | high_vol

Format: "trend=<value>|vol=<value>" (alphabetical dimension order)
"""

from dataclasses import dataclass
from typing import Literal, cast

# Valid dimension values (v1.5 invariant)
VALID_TREND_VALUES = frozenset(["uptrend", "downtrend", "flat"])
VALID_VOL_VALUES = frozenset(["low_vol", "mid_vol", "high_vol"])

TrendValue = Literal["uptrend", "downtrend", "flat"]
VolValue = Literal["low_vol", "mid_vol", "high_vol"]


@dataclass(frozen=True)
class RegimeDims:
    """
    Regime dimensions (v1.5).

    Immutable to ensure consistent hashing.
    """

    trend: TrendValue
    vol: VolValue

    def __post_init__(self):
        if self.trend not in VALID_TREND_VALUES:
            raise ValueError(
                f"Invalid trend value: {self.trend}. Must be one of {VALID_TREND_VALUES}"
            )
        if self.vol not in VALID_VOL_VALUES:
            raise ValueError(
                f"Invalid vol value: {self.vol}. Must be one of {VALID_VOL_VALUES}"
            )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {"trend": self.trend, "vol": self.vol}


def canonicalize_regime_key(dims: RegimeDims) -> str:
    """
    Generate canonical regime key from dimensions.

    Format: "trend=<value>|vol=<value>" (alphabetical order)

    Args:
        dims: Regime dimensions

    Returns:
        Canonical key string
    """
    # Alphabetical order: trend before vol
    return f"trend={dims.trend}|vol={dims.vol}"


def parse_regime_key(key: str) -> RegimeDims:
    """
    Parse canonical regime key into dimensions.

    Args:
        key: Canonical key string

    Returns:
        RegimeDims

    Raises:
        ValueError: If key format is invalid
    """
    parts = key.split("|")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid regime key format: {key}. Expected 2 parts, got {len(parts)}"
        )

    dims_dict = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(
                f"Invalid regime key format: {key}. Part missing '=': {part}"
            )
        dim_name, dim_value = part.split("=", 1)
        dims_dict[dim_name] = dim_value

    if "trend" not in dims_dict or "vol" not in dims_dict:
        raise ValueError(
            f"Invalid regime key format: {key}. Missing trend or vol dimension"
        )

    trend_val = dims_dict["trend"]
    vol_val = dims_dict["vol"]

    if trend_val not in VALID_TREND_VALUES:
        raise ValueError(f"Invalid trend value: {trend_val}")
    if vol_val not in VALID_VOL_VALUES:
        raise ValueError(f"Invalid vol value: {vol_val}")

    return RegimeDims(
        trend=cast(TrendValue, trend_val),
        vol=cast(VolValue, vol_val),
    )


def extract_marginal_keys(key: str) -> list[str]:
    """
    Extract single-dimension marginal keys for backoff queries.

    Args:
        key: Canonical composite key

    Returns:
        List of marginal keys (e.g., ["trend=uptrend", "vol=high_vol"])
    """
    return key.split("|")
