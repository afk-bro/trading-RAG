#!/usr/bin/env bash
#
# A vs C scale-out comparison on real Databento data.
#
# Usage:
#   ./scripts/compare_scale_out.sh --databento-csv path/to/nq.csv \
#       --start-date 2024-06-01 --end-date 2024-09-30
#
# Both runs use identical settings:
#   - NY AM session only (--session-profile strict)
#   - eval-mode with governor
#   - worst intrabar policy, 2-tick slippage, $4.50 commission
#   - same seed (deterministic)
#
# Run A: baseline (no partial exit)
# Run C: prop_safe (33% off at +1R)
#
# Outputs JSON to tmp/ab_test/ and prints prop-survival comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$PROJECT_DIR/tmp/ab_test"
mkdir -p "$OUT_DIR"

# Forward all args (--databento-csv, --start-date, --end-date, etc.)
DATA_ARGS=("$@")

# Common flags for both runs
COMMON=(
    --symbol NQ
    --session-profile strict
    --eval-mode
    --slippage-ticks 2
    --commission 4.50
    --intrabar-policy worst
    --json
)

echo "=== Run A: Baseline (no scale-out) ==="
python "$SCRIPT_DIR/run_unicorn_backtest.py" \
    "${COMMON[@]}" \
    "${DATA_ARGS[@]}" \
    > "$OUT_DIR/run_a_baseline.json" 2>"$OUT_DIR/run_a.log"
echo "  -> $OUT_DIR/run_a_baseline.json"

echo "=== Run C: prop_safe (33% @ +1R) ==="
python "$SCRIPT_DIR/run_unicorn_backtest.py" \
    "${COMMON[@]}" \
    --scale-out prop_safe \
    "${DATA_ARGS[@]}" \
    > "$OUT_DIR/run_c_prop_safe.json" 2>"$OUT_DIR/run_c.log"
echo "  -> $OUT_DIR/run_c_prop_safe.json"

echo ""
echo "=== Comparison ==="
python "$SCRIPT_DIR/compare_unicorn_runs.py" \
    "$OUT_DIR/run_a_baseline.json" \
    "$OUT_DIR/run_c_prop_safe.json"

echo ""
echo "=== Prop Survival Metrics ==="
python3 -c "
import json, sys

with open('$OUT_DIR/run_a_baseline.json') as f:
    a = json.load(f)
with open('$OUT_DIR/run_c_prop_safe.json') as f:
    c = json.load(f)

pa, pc = a.get('prop_survival', {}), c.get('prop_survival', {})
ga, gc = a.get('governor_stats', {}), c.get('governor_stats', {})
sa, sc = None, c.get('scale_out_stats')

header = f\"{'Metric':<32} {'Baseline':>14} {'C (prop_safe)':>14} {'Delta':>12}\"
print(header)
print('-' * len(header))

rows = [
    ('Max Trailing DD (\$)',     'max_trailing_dd_dollars',    pa, pc, '\$'),
    ('Max Trailing DD (%)',      'max_trailing_dd_pct',        pa, pc, '%'),
    ('Worst Losing Streak',     'worst_losing_streak_setups', pa, pc, ''),
    ('Worst Intraday Loss (\$)', 'worst_intraday_loss_dollars',pa, pc, '\$'),
    ('Avg Win (R)',              'avg_win_r',                  pa, pc, 'R'),
    ('Avg Loss (R)',             'avg_loss_r',                 pa, pc, 'R'),
    ('Days Halted',              'days_halted',                ga, gc, ''),
    ('Loss Limit Halts',         'loss_limit_halts',           ga, gc, ''),
    ('Trade Limit Halts',        'trade_limit_halts',          ga, gc, ''),
    ('Signals Skipped',          'signals_skipped',            ga, gc, ''),
]

for label, key, da, dc, unit in rows:
    va = da.get(key, 0) or 0
    vc = dc.get(key, 0) or 0
    delta = vc - va
    if isinstance(va, float):
        print(f'{label:<32} {va:>14.2f} {vc:>14.2f} {delta:>+12.2f}')
    else:
        print(f'{label:<32} {va:>14} {vc:>14} {delta:>+12}')

# Dollar PnL (the real question)
va_d = a.get('total_pnl_dollars', 0)
vc_d = c.get('total_pnl_dollars', 0)
print(f\"{'Total PnL (\$)':<32} {va_d:>14.2f} {vc_d:>14.2f} {vc_d-va_d:>+12.2f}\")

# Expectancy
va_e = a.get('expectancy_points', 0)
vc_e = c.get('expectancy_points', 0)
print(f\"{'Expectancy/setup (pts)':<32} {va_e:>14.2f} {vc_e:>14.2f} {vc_e-va_e:>+12.2f}\")

# Profit factor
va_pf = a.get('profit_factor', 0)
vc_pf = c.get('profit_factor', 0)
print(f\"{'Profit Factor':<32} {va_pf:>14.2f} {vc_pf:>14.2f} {vc_pf-va_pf:>+12.2f}\")

# Scale-out diagnostics
if sc:
    print()
    print(f'Scale-out diagnostics (C):')
    print(f'  Setups entered:      {sc[\"setups_entered\"]}')
    print(f'  Total closed trades: {sc[\"total_closed_trades\"]}')
    print(f'  Partial legs:        {sc[\"partial_legs\"]}')

# Pass/fail verdict
acct = ga.get('eval_account_size', 0)
max_dd_limit = ga.get('eval_max_drawdown', 0)
if acct > 0 and max_dd_limit > 0:
    print()
    a_pass = pa.get('max_trailing_dd_dollars', 0) < max_dd_limit and not ga.get('drawdown_halt', False)
    c_pass = pc.get('max_trailing_dd_dollars', 0) < max_dd_limit and not gc.get('drawdown_halt', False)
    print(f'Eval pass (DD < \${max_dd_limit:,.0f}):  A={\"PASS\" if a_pass else \"FAIL\"}  C={\"PASS\" if c_pass else \"FAIL\"}')
"
