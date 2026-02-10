"""Type definitions for the ORB (Opening Range Breakout) backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from enum import Enum
from typing import Any, Optional


class ORBPhase(Enum):
    """State machine phases for the ORB session."""

    PREMARKET = "premarket"
    OR_BUILD = "or_build"
    BREAKOUT_SCAN = "breakout_scan"
    ENTRY = "entry"
    TRADE_MGMT = "trade_mgmt"
    EXIT = "exit"
    LOCKOUT = "lockout"


@dataclass
class SessionWindow:
    """Trading session time boundaries (ET clock time)."""

    start: time
    end: time
    label: str


@dataclass
class ORBRange:
    """Opening range being built or locked."""

    high: float
    low: float
    start_bar_index: int
    lock_bar_index: int = -1
    locked: bool = False


@dataclass
class ORBParams:
    """Parsed engine parameters."""

    or_minutes: int = 30
    confirm_mode: str = "close-beyond"
    stop_mode: str = "or-opposite"
    target_r: float = 1.5
    max_trades: int = 1
    session: str = "NY AM"
    fixed_ticks: float = 50.0
    risk_pct: float = 0.01
    point_value: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ORBParams:
        return cls(
            or_minutes=int(d.get("or_minutes", 30)),
            confirm_mode=str(d.get("confirm_mode", "close-beyond")),
            stop_mode=str(d.get("stop_mode", "or-opposite")),
            target_r=float(d.get("target_r", 1.5)),
            max_trades=int(d.get("max_trades", 1)),
            session=str(d.get("session", "NY AM")),
            fixed_ticks=float(d.get("fixed_ticks", 50.0)),
            risk_pct=float(d.get("risk_pct", 0.01)),
            point_value=float(d.get("point_value", 1.0)),
        )


@dataclass
class ORBPosition:
    """Open position state."""

    entry_bar: int
    entry_time: str  # UTC ISO string
    entry_price: float
    stop: float
    target: float
    side: str  # "long" or "short"
    size: float
    risk_points: float


@dataclass
class ORBClosedTrade:
    """Completed trade record."""

    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    side: str
    size: float
    pnl: float
    exit_reason: str
    entry_bar: int = 0
    exit_bar: int = 0


@dataclass
class ORBSessionState:
    """Per-session mutable state."""

    phase: ORBPhase = ORBPhase.PREMARKET
    orb_range: Optional[ORBRange] = None
    position: Optional[ORBPosition] = None
    trade_count: int = 0
    session_date: Optional[date] = None
    or_bar_count: int = 0

    # Retest FSM fields
    pending_dir: Optional[str] = None
    pending_break_index: Optional[int] = None
    retest_confirmed: bool = False
    pending_expires_at_index: int = 0

    def reset(self) -> None:
        """Reset state for a new session."""
        self.phase = ORBPhase.PREMARKET
        self.orb_range = None
        self.position = None
        self.trade_count = 0
        self.session_date = None
        self.or_bar_count = 0
        self.pending_dir = None
        self.pending_break_index = None
        self.retest_confirmed = False
        self.pending_expires_at_index = 0


# Session window presets (ET times)
SESSION_WINDOWS: dict[str, SessionWindow] = {
    "NY AM": SessionWindow(start=time(9, 30), end=time(12, 0), label="NY AM"),
    "NY PM": SessionWindow(start=time(13, 0), end=time(16, 0), label="NY PM"),
    "London": SessionWindow(start=time(3, 0), end=time(6, 30), label="London"),
}
