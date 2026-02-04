#!/usr/bin/env python3
"""
Register the ICT Blueprint strategy spec in the kb_strategy_specs table.

Creates the kb_entity (type=strategy) if it doesn't exist, then upserts the
full parameter specification into kb_strategy_specs.spec_json.

Usage:
    source .venv/bin/activate
    python scripts/register_ict_blueprint_spec.py
"""

import asyncio
import json
import os
import sys
from uuid import UUID

import asyncpg
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000001")

STRATEGY_NAME = "ICT Blueprint"
STRATEGY_DESCRIPTION = (
    "Trend-following pullback into institutional order flow zones. "
    "Daily HTF bias via swing/MSB structure, H1 LTF entry via OB zone "
    "sweep + breaker retest. Source: Trader Mine — 'The SIMPLE $10 Million ICT Blueprint'."
)

# ---------------------------------------------------------------------------
# Full spec_json following the ParamSpec / StrategySpec format from
# app/services/pine/spec_generator.py
# ---------------------------------------------------------------------------

SPEC_JSON = {
    "name": "ICT Blueprint",
    "source_path": "docs/plans/2026-02-03-ict-blueprint-strategy-spec.md",
    "pine_version": "n/a",
    "description": STRATEGY_DESCRIPTION,
    "sha256": None,
    "params": [
        # ── HTF Parameters ──────────────────────────────────────────────
        {
            "name": "swing_lookback",
            "display_name": "Swing Lookback",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 3,
            "step": 1,
            "group": "HTF",
            "tooltip": "Number of candles on each side for swing detection. 1 = 3-candle system.",
            "sweepable": True,
            "priority": 25,
        },
        {
            "name": "ob_candles",
            "display_name": "Order Block Candles",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 3,
            "step": 1,
            "group": "HTF",
            "tooltip": "Number of opposing candles to include in the OB zone.",
            "sweepable": True,
            "priority": 25,
        },
        {
            "name": "discount_threshold",
            "display_name": "Discount Threshold",
            "type": "float",
            "default": 0.5,
            "min": 0.382,
            "max": 0.618,
            "step": 0.001,
            "group": "HTF",
            "tooltip": "How deep into the range price must be for premium/discount eligibility.",
            "sweepable": True,
            "priority": 20,
        },
        {
            "name": "range_anchor_mode",
            "display_name": "Range Anchor Mode",
            "type": "string",
            "default": "immediate",
            "options": ["immediate"],
            "group": "HTF",
            "tooltip": "Which swing to anchor the trading range to. 'immediate' uses the most recent swing preceding the MSB.",
            "sweepable": False,
            "priority": 5,
        },
        # ── LTF Parameters ──────────────────────────────────────────────
        {
            "name": "ob_zone_entry_requirement",
            "display_name": "OB Zone Entry Requirement",
            "type": "string",
            "default": "close_inside",
            "options": ["touch", "close_inside", "percent_inside"],
            "group": "LTF",
            "tooltip": "How price must interact with the HTF OB zone to activate LTF scanning.",
            "sweepable": True,
            "priority": 20,
        },
        {
            "name": "ob_zone_overlap_pct",
            "display_name": "OB Zone Overlap %",
            "type": "float",
            "default": 0.10,
            "min": 0.05,
            "max": 0.20,
            "step": 0.01,
            "group": "LTF",
            "tooltip": "Minimum body overlap with OB zone (used only with percent_inside mode).",
            "sweepable": True,
            "priority": 10,
        },
        {
            "name": "ltf_swing_lookback",
            "display_name": "LTF Swing Lookback",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 2,
            "step": 1,
            "group": "LTF",
            "tooltip": "Number of candles on each side for H1 swing detection.",
            "sweepable": True,
            "priority": 25,
        },
        {
            "name": "require_sweep",
            "display_name": "Require Liquidity Sweep",
            "type": "bool",
            "default": True,
            "group": "LTF",
            "tooltip": "If true, require a sweep below L0 before MSB confirmation. False = more signals, lower win rate.",
            "sweepable": True,
            "priority": 15,
        },
        {
            "name": "entry_mode",
            "display_name": "Entry Mode",
            "type": "string",
            "default": "breaker_retest",
            "options": ["breaker_retest", "msb_close", "fvg_fill"],
            "group": "LTF",
            "tooltip": "How aggressively to enter after LTF MSB confirms. breaker_retest = conservative, msb_close = aggressive.",
            "sweepable": True,
            "priority": 25,
        },
        {
            "name": "max_wait_bars_after_msb",
            "display_name": "Max Wait Bars After MSB",
            "type": "int",
            "default": 12,
            "min": 5,
            "max": 30,
            "step": 1,
            "group": "LTF",
            "tooltip": "H1 candles to wait for a valid entry after LTF MSB before invalidating setup.",
            "sweepable": True,
            "priority": 15,
        },
        {
            "name": "breaker_candles",
            "display_name": "Breaker Candles",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 2,
            "step": 1,
            "group": "LTF",
            "tooltip": "Number of opposing candles forming the breaker zone after LTF MSB.",
            "sweepable": True,
            "priority": 20,
        },
        # ── Risk Parameters ─────────────────────────────────────────────
        {
            "name": "stop_mode",
            "display_name": "Stop Mode",
            "type": "string",
            "default": "below_sweep",
            "options": ["below_sweep", "below_breaker"],
            "group": "Risk",
            "tooltip": "Stop placement: below_sweep = wider/robust, below_breaker = tighter/better R:R.",
            "sweepable": True,
            "priority": 25,
        },
        {
            "name": "tp_mode",
            "display_name": "Take Profit Mode",
            "type": "string",
            "default": "external_range",
            "options": ["external_range", "fixed_rr"],
            "group": "Risk",
            "tooltip": "Target mode: external_range = HTF liquidity, fixed_rr = fixed risk multiple.",
            "sweepable": True,
            "priority": 25,
        },
        {
            "name": "min_rr",
            "display_name": "Minimum R:R",
            "type": "float",
            "default": 2.0,
            "min": 2.0,
            "max": 3.0,
            "step": 0.5,
            "group": "Risk",
            "tooltip": "Minimum reward:risk ratio to accept a trade. Skip if target/stop < this value.",
            "sweepable": True,
            "priority": 30,
        },
        {
            "name": "derisk_mode",
            "display_name": "De-Risk Mode",
            "type": "string",
            "default": "move_to_be",
            "options": ["move_to_be", "half_off", "none"],
            "group": "Risk",
            "tooltip": "Action at de-risk trigger: move stop to breakeven, close 50%, or none.",
            "sweepable": True,
            "priority": 15,
        },
        {
            "name": "derisk_trigger_rr",
            "display_name": "De-Risk Trigger R:R",
            "type": "float",
            "default": 2.0,
            "min": 1.5,
            "max": 2.5,
            "step": 0.5,
            "group": "Risk",
            "tooltip": "R-multiple at which the de-risk action triggers.",
            "sweepable": True,
            "priority": 15,
        },
        {
            "name": "max_attempts_per_ob",
            "display_name": "Max Attempts Per OB",
            "type": "int",
            "default": 2,
            "min": 1,
            "max": 3,
            "step": 1,
            "group": "Risk",
            "tooltip": "Maximum re-entry attempts within the same HTF OB zone. Each requires fresh setup.",
            "sweepable": True,
            "priority": 10,
        },
    ],
    "sweep_config": {
        "swing_lookback": [1, 2, 3],
        "ob_candles": [1, 2, 3],
        "discount_threshold": [0.382, 0.5, 0.618],
        "ob_zone_entry_requirement": ["touch", "close_inside", "percent_inside"],
        "ob_zone_overlap_pct": [0.05, 0.10, 0.20],
        "ltf_swing_lookback": [1, 2],
        "require_sweep": [True, False],
        "entry_mode": ["breaker_retest", "msb_close", "fvg_fill"],
        "max_wait_bars_after_msb": [5, 12, 20, 30],
        "breaker_candles": [1, 2],
        "stop_mode": ["below_sweep", "below_breaker"],
        "tp_mode": ["external_range", "fixed_rr"],
        "min_rr": [2.0, 2.5, 3.0],
        "derisk_mode": ["move_to_be", "half_off", "none"],
        "derisk_trigger_rr": [1.5, 2.0, 2.5],
        "max_attempts_per_ob": [1, 2, 3],
    },
}


async def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    print(f"Connecting to database...")
    conn = await asyncpg.connect(database_url, ssl="require", statement_cache_size=0)

    try:
        # 1. Upsert the strategy entity in kb_entities
        print(f"Upserting kb_entity: name={STRATEGY_NAME!r}, type=strategy")
        entity_row = await conn.fetchrow(
            """
            INSERT INTO kb_entities (workspace_id, type, name, aliases, description)
            VALUES ($1, 'strategy', $2, $3::jsonb, $4)
            ON CONFLICT (workspace_id, type, lower(name))
            DO UPDATE SET
                description = COALESCE(EXCLUDED.description, kb_entities.description),
                updated_at = NOW()
            RETURNING id, (xmax = 0) AS inserted
            """,
            WORKSPACE_ID,
            STRATEGY_NAME,
            json.dumps(["ICT Blueprint Strategy", "Trader Mine ICT", "10M ICT Blueprint"]),
            STRATEGY_DESCRIPTION,
        )

        entity_id = entity_row["id"]
        was_created = entity_row["inserted"]
        action = "Created" if was_created else "Updated"
        print(f"  {action} entity: {entity_id}")

        # 2. Upsert the strategy spec
        print("Upserting kb_strategy_specs row...")
        spec_row = await conn.fetchrow(
            """
            INSERT INTO kb_strategy_specs (
                strategy_entity_id, workspace_id, spec_json,
                derived_from_claim_ids, status, version
            )
            VALUES ($1, $2, $3::jsonb, '[]'::jsonb, 'draft', 1)
            ON CONFLICT (strategy_entity_id) DO UPDATE SET
                spec_json = EXCLUDED.spec_json,
                version = kb_strategy_specs.version + 1,
                updated_at = NOW()
            RETURNING id, version, status
            """,
            entity_id,
            WORKSPACE_ID,
            json.dumps(SPEC_JSON),
        )

        spec_id = spec_row["id"]
        version = spec_row["version"]
        status = spec_row["status"]

        print(f"  Spec ID:  {spec_id}")
        print(f"  Version:  {version}")
        print(f"  Status:   {status}")
        print(f"  Params:   {len(SPEC_JSON['params'])} parameters")
        print(f"  Sweepable: {sum(1 for p in SPEC_JSON['params'] if p['sweepable'])}")
        print()
        print("Registration complete.")
        print()
        print("Next steps:")
        print(f"  1. Approve:  PATCH /kb/strategies/{entity_id}/spec  "
              f'{{"status": "approved", "approved_by": "manual"}}')
        print(f"  2. Compile:  POST /kb/strategies/{entity_id}/compile?allow_draft=true")
        print(f"  3. View:     GET  /kb/strategies/{entity_id}/spec")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
