"""ORB event payload contracts.

Single source of truth for event schema. Engine validates events against
these definitions in debug/test mode. Consumers (ReplayPanel, process_score,
loss_attribution) should reference these constants, not hardcode keys.

Versioned: bump ORB_EVENT_SCHEMA_VERSION when adding required keys or
changing semantics. Optional keys can be added without a version bump.
"""

from __future__ import annotations

ORB_EVENT_SCHEMA_VERSION = "1.0.0"

# Common keys present on every event
COMMON_REQUIRED_KEYS = frozenset(
    {
        "type",
        "bar_index",
        "ts",
        "session_date",
        "phase",
    }
)

# Per-type required payload keys (beyond common)
ORB_EVENT_TYPES: dict[str, frozenset[str]] = {
    "orb_range_update": frozenset(
        {
            "orb_high",
            "orb_low",
            "or_minutes",
            "or_start_index",
        }
    ),
    "orb_range_locked": frozenset(
        {
            "high",
            "low",
            "range",
            "or_minutes",
            "or_start_index",
            "or_lock_index",
            "or_bar_count_needed",
        }
    ),
    "setup_valid": frozenset(
        {
            "direction",
            "level",
            "confirm_mode",
            "trigger_price",
        }
    ),
    "entry_signal": frozenset(
        {
            "side",
            "price",
            "stop",
            "target",
            "size",
            "risk_points",
        }
    ),
}

# Union of all required keys (useful for quick membership tests)
ALL_REQUIRED_KEYS = COMMON_REQUIRED_KEYS | frozenset().union(*ORB_EVENT_TYPES.values())


def validate_events(events: list[dict]) -> list[str]:
    """Validate event list against contracts. Returns list of error strings.

    Cheap enough to run in tests and debug mode. Returns empty list if valid.
    """
    errors: list[str] = []
    for i, evt in enumerate(events):
        evt_type = evt.get("type", "<missing>")

        # Check common keys
        missing_common = COMMON_REQUIRED_KEYS - set(evt.keys())
        if missing_common:
            errors.append(
                f"Event {i} ({evt_type}): missing common keys {sorted(missing_common)}"
            )

        # Check type-specific keys
        type_keys = ORB_EVENT_TYPES.get(str(evt_type))
        if type_keys is None:
            errors.append(
                f"Event {i}: unknown type '{evt_type}'. "
                f"Valid: {sorted(ORB_EVENT_TYPES.keys())}"
            )
        else:
            missing_type = type_keys - set(evt.keys())
            if missing_type:
                errors.append(
                    f"Event {i} ({evt_type}): missing payload keys "
                    f"{sorted(missing_type)}"
                )

    return errors
