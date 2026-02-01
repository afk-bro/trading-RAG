#!/usr/bin/env python3
"""
Compare two Unicorn backtest JSON outputs side-by-side.

Standalone CLI — no framework deps (just json + argparse).

Usage:
    python scripts/compare_unicorn_runs.py run_a.json run_b.json
"""

import argparse
import json
import sys

# Inline set — keeps this script standalone (no imports from the model).
MANDATORY_CRITERIA = {"htf_bias", "stop_valid", "macro_window", "mss", "displacement"}


def compare_runs(a: dict, b: dict) -> str:
    """Compare two backtest JSON dicts and return a formatted report string."""
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Section 1: Identity
    # ------------------------------------------------------------------
    lines.append("=" * 70)
    lines.append("RUN COMPARISON")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Run A: {a.get('run_key', 'N/A')}")
    lines.append(f"Run B: {b.get('run_key', 'N/A')}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 2: Metrics delta table
    # ------------------------------------------------------------------
    lines.append("-" * 70)
    lines.append("METRICS DELTA")
    lines.append("-" * 70)

    metrics = [
        ("Trades", "trades_taken", "d"),
        ("Win Rate", "win_rate", "pct"),
        ("Profit Factor", "profit_factor", "f"),
        ("Expectancy (pts)", "expectancy_points", "f"),
        ("Total PnL (pts)", "total_pnl_points", "f"),
        ("Total PnL ($)", "total_pnl_dollars", "f"),
        ("Largest Loss (pts)", "largest_loss_points", "f"),
    ]

    header = f"{'Metric':<22} {'Run A':>12} {'Run B':>12} {'Delta':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for label, key, fmt in metrics:
        val_a = a.get(key, 0) or 0
        val_b = b.get(key, 0) or 0
        delta = val_b - val_a

        if fmt == "d":
            lines.append(f"{label:<22} {int(val_a):>12} {int(val_b):>12} {int(delta):>+12}")
        elif fmt == "pct":
            # win_rate may be 0-1 or 0-100; normalize to percentage display
            va = val_a * 100 if val_a <= 1.0 and val_a != 0 else val_a
            vb = val_b * 100 if val_b <= 1.0 and val_b != 0 else val_b
            d = vb - va
            lines.append(f"{label:<22} {va:>11.1f}% {vb:>11.1f}% {d:>+11.1f}%")
        else:
            lines.append(f"{label:<22} {val_a:>12.2f} {val_b:>12.2f} {delta:>+12.2f}")

    lines.append("")

    # ------------------------------------------------------------------
    # Section 3: Bottlenecks (top 3 [M] and top 3 [S] per run)
    # ------------------------------------------------------------------
    lines.append("-" * 70)
    lines.append("BOTTLENECKS")
    lines.append("-" * 70)

    for run_name, run_data in [("Run A", a), ("Run B", b)]:
        bottlenecks = run_data.get("criteria_bottlenecks", [])
        if not bottlenecks:
            lines.append(f"  {run_name}: (no bottleneck data)")
            continue

        # Sort by fail_rate descending
        sorted_bn = sorted(bottlenecks, key=lambda x: x.get("fail_rate", 0), reverse=True)

        mandatory = [b for b in sorted_bn if b.get("criterion", "") in MANDATORY_CRITERIA]
        scored = [b for b in sorted_bn if b.get("criterion", "") not in MANDATORY_CRITERIA]

        lines.append(f"  {run_name}:")
        lines.append(f"    Mandatory (top 3):")
        for bn in mandatory[:3]:
            crit = bn.get("criterion", "?")
            rate = bn.get("fail_rate", 0)
            lines.append(f"      [M] {crit:<24} {rate:.1%}")

        lines.append(f"    Scored (top 3):")
        for bn in scored[:3]:
            crit = bn.get("criterion", "?")
            rate = bn.get("fail_rate", 0)
            lines.append(f"      [S] {crit:<24} {rate:.1%}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Unicorn backtest JSON outputs",
    )
    parser.add_argument("run_a", help="Path to first run JSON file")
    parser.add_argument("run_b", help="Path to second run JSON file")
    args = parser.parse_args()

    with open(args.run_a) as f:
        a = json.load(f)
    with open(args.run_b) as f:
        b = json.load(f)

    print(compare_runs(a, b))


if __name__ == "__main__":
    main()
