"""Enums, dataclasses, and configuration for the ICT Blueprint backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Bias(Enum):
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    BEARISH = "bearish"


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class SetupPhase(Enum):
    INACTIVE = "inactive"
    SCANNING = "scanning"
    SWEEP_PENDING = "sweep_pending"
    MSB_PENDING = "msb_pending"
    ENTRY_PENDING = "entry_pending"
    TIMED_OUT = "timed_out"


class EntryMode(Enum):
    BREAKER_RETEST = "breaker_retest"
    MSB_CLOSE = "msb_close"
    FVG_FILL = "fvg_fill"


class StopMode(Enum):
    BELOW_SWEEP = "below_sweep"
    BELOW_BREAKER = "below_breaker"


class TPMode(Enum):
    EXTERNAL_RANGE = "external_range"
    FIXED_RR = "fixed_rr"


class DeriskMode(Enum):
    MOVE_TO_BE = "move_to_be"
    HALF_OFF = "half_off"
    NONE = "none"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SwingPoint:
    index: int
    timestamp: int  # nanoseconds since epoch
    price: float
    is_high: bool


# Unique key for an order block: (msb_bar_index, ob_start_index, ob_end_index, side)
OBId = tuple[int, int, int, str]

# Unique key for a setup: (ob_id, side)
SetupKey = tuple[OBId, str]


@dataclass
class OrderBlock:
    top: float
    bottom: float
    bias: Bias
    ob_id: OBId
    anchor_swing: SwingPoint
    msb_bar_index: int
    attempts_used: int = 0
    invalidated: bool = False
    created_at_daily_index: int = 0
    last_setup_bar_index: int = -1  # tracks last timeout/exit bar for fresh L0 constraint


@dataclass
class TradingRange:
    high: float
    low: float
    midpoint: float
    bias: Bias


@dataclass
class BreakerZone:
    top: float
    bottom: float
    bar_index: int


@dataclass
class FVG:
    top: float
    bottom: float
    bar_index: int
    bullish: bool


@dataclass
class HTFState:
    bias: Bias = Bias.NEUTRAL
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    current_range: Optional[TradingRange] = None
    active_obs: list[OrderBlock] = field(default_factory=list)
    last_msb_bar_index: int = -1


@dataclass(frozen=True)
class HTFStateSnapshot:
    """Frozen view of HTF state at a point in time."""

    bias: Bias
    swing_highs: tuple[SwingPoint, ...]
    swing_lows: tuple[SwingPoint, ...]
    current_range: Optional[TradingRange]
    active_obs: tuple[OrderBlock, ...]
    last_msb_bar_index: int

    @staticmethod
    def from_state(state: HTFState) -> HTFStateSnapshot:
        return HTFStateSnapshot(
            bias=state.bias,
            swing_highs=tuple(state.swing_highs),
            swing_lows=tuple(state.swing_lows),
            current_range=state.current_range,
            active_obs=tuple(state.active_obs),
            last_msb_bar_index=state.last_msb_bar_index,
        )


@dataclass
class LTFSetup:
    ob: OrderBlock
    side: Side
    phase: SetupPhase = SetupPhase.SCANNING
    l0: Optional[SwingPoint] = None
    h0: Optional[SwingPoint] = None
    sweep_low: Optional[float] = None
    msb_bar_index: int = -1
    breaker: Optional[BreakerZone] = None
    fvg: Optional[FVG] = None
    bars_since_msb: int = 0
    last_exit_bar_index: int = -1

    @property
    def setup_key(self) -> SetupKey:
        return (self.ob.ob_id, self.side.value)


@dataclass
class Position:
    entry_time: int  # nanoseconds
    entry_price: float
    stop_price: float
    target_price: float
    side: Side
    size: float
    risk_points: float
    derisk_triggered: bool = False
    remaining_size: float = 0.0
    ob_id: Optional[OBId] = None

    def __post_init__(self) -> None:
        if self.remaining_size == 0.0:
            self.remaining_size = self.size


@dataclass
class ClosedTrade:
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    side: Side
    size: float
    pnl_points: float
    pnl_dollars: float
    exit_reason: str
    ob_id: Optional[OBId] = None


# ---------------------------------------------------------------------------
# Parameter config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICTBlueprintParams:
    """All tuneable parameters for the ICT Blueprint strategy."""

    # HTF
    swing_lookback: int = 1
    ob_candles: int = 1
    discount_threshold: float = 0.5
    range_anchor_mode: str = "immediate"
    max_ob_age_bars: int = 20

    # LTF
    ob_zone_entry_requirement: str = "close_inside"
    ob_zone_overlap_pct: float = 0.10
    ltf_swing_lookback: int = 1
    require_sweep: bool = True
    entry_mode: str = "breaker_retest"
    max_wait_bars_after_msb: int = 12
    breaker_candles: int = 1

    # Risk
    stop_mode: str = "below_sweep"
    tp_mode: str = "external_range"
    min_rr: float = 2.0
    fixed_rr: float = 3.0
    derisk_mode: str = "move_to_be"
    derisk_trigger_rr: float = 2.0
    max_attempts_per_ob: int = 2
    stop_buffer_ticks: float = 2.0
    point_value: float = 50.0
    risk_pct: float = 0.01

    @staticmethod
    def from_dict(d: dict) -> ICTBlueprintParams:
        valid_fields = {f.name for f in ICTBlueprintParams.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return ICTBlueprintParams(**filtered)
