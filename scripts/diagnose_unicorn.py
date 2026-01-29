#!/usr/bin/env python3
"""
Diagnostic and parameter tuning script for Unicorn Model backtest.

Implements walk-forward validation and tests key parameter combinations.

Usage:
    python scripts/diagnose_unicorn.py --csv path/to/data.csv --symbol NQ
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest.data import DatabentoFetcher
from app.services.backtest.engines.unicorn_runner import (
    run_unicorn_backtest,
    IntrabarPolicy,
)
from app.services.strategy.strategies.unicorn_model import (
    UnicornConfig,
    SessionProfile,
    BiasDirection,
)

# Default direction filter and time-stop settings
DEFAULT_DIRECTION_FILTER = None  # None=both, BiasDirection.BULLISH=long-only
DEFAULT_TIME_STOP_MINUTES = None  # None=disabled, e.g., 30=exit if not +0.25R in 30 min
DEFAULT_TIME_STOP_R_THRESHOLD = 0.25


def run_single_backtest(htf_bars, ltf_bars, symbol, **kwargs):
    """Run a single backtest and return key metrics."""
    try:
        result = run_unicorn_backtest(
            symbol=symbol,
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            **kwargs
        )

        return {
            'trades': len(result.trades),
            'wins': result.wins,
            'losses': result.losses,
            'win_rate': result.win_rate,
            'pf': result.profit_factor,
            'pnl': result.total_pnl_handles,
            'avg_r': result.avg_r_multiple,
            'expectancy': result.expectancy_handles,
            'result': result,
        }
    except Exception as e:
        return {'error': str(e)}


def filter_bars_by_date(bars, start_date, end_date):
    """Filter bars to date range."""
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    return [b for b in bars if start_dt <= b.ts.replace(tzinfo=None) <= end_dt]


def expectancy_decomposition(result):
    """Print detailed expectancy breakdown."""
    print('=' * 60)
    print('EXPECTANCY DECOMPOSITION')
    print('=' * 60)

    trades = result.trades
    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars < 0]

    print(f'\nTotal Trades: {len(trades)}')
    print(f'  Wins: {len(wins)} ({100*len(wins)/len(trades):.1f}%)')
    print(f'  Losses: {len(losses)} ({100*len(losses)/len(trades):.1f}%)')

    win_rs = [t.r_multiple for t in wins if t.r_multiple]
    loss_rs = [t.r_multiple for t in losses if t.r_multiple]

    print(f'\nR-MULTIPLES:')
    if win_rs:
        print(f'  Avg Win: {statistics.mean(win_rs):+.2f}R')
        print(f'  Median Win: {statistics.median(win_rs):+.2f}R')
    if loss_rs:
        print(f'  Avg Loss: {statistics.mean(loss_rs):+.2f}R')
        print(f'  Median Loss: {statistics.median(loss_rs):+.2f}R')

    # Exit breakdown
    exit_reasons = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += t.pnl_handles
        if t.pnl_dollars > 0:
            exit_reasons[reason]['wins'] += 1

    print(f'\nEXIT BREAKDOWN:')
    for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]['count']):
        wr = 100 * stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        print(f'  {reason:12s} {stats["count"]:4d}  WR={wr:5.1f}%  PnL={stats["pnl"]:+.1f}h')

    # Direction breakdown
    longs = [t for t in trades if t.direction == BiasDirection.BULLISH]
    shorts = [t for t in trades if t.direction == BiasDirection.BEARISH]

    print(f'\nDIRECTION:')
    for name, subset in [('LONG', longs), ('SHORT', shorts)]:
        if subset:
            wins_sub = sum(1 for t in subset if t.pnl_dollars > 0)
            pnl = sum(t.pnl_handles for t in subset)
            wr = 100 * wins_sub / len(subset)
            avg_r = statistics.mean([t.r_multiple for t in subset if t.r_multiple])
            print(f'  {name:6s} {len(subset):4d}  WR={wr:5.1f}%  AvgR={avg_r:+.2f}R  PnL={pnl:+.1f}h')


def run_walk_forward(htf_bars, ltf_bars, symbol, splits, base_params):
    """Run walk-forward validation across time splits."""
    print('=' * 60)
    print('WALK-FORWARD VALIDATION')
    print('=' * 60)

    results = []
    for i, (train_range, test_range) in enumerate(splits):
        train_start, train_end = train_range
        test_start, test_end = test_range

        # Filter bars
        train_htf = filter_bars_by_date(htf_bars, train_start, train_end)
        train_ltf = filter_bars_by_date(ltf_bars, train_start, train_end)
        test_htf = filter_bars_by_date(htf_bars, test_start, test_end)
        test_ltf = filter_bars_by_date(ltf_bars, test_start, test_end)

        # Run on both periods
        train_result = run_single_backtest(train_htf, train_ltf, symbol, **base_params)
        test_result = run_single_backtest(test_htf, test_ltf, symbol, **base_params)

        print(f'\nSplit {i+1}: Train={train_start}→{train_end}, Test={test_start}→{test_end}')
        if 'error' not in train_result:
            print(f'  TRAIN: {train_result["trades"]} trades, WR={train_result["win_rate"]*100:.1f}%, '
                  f'PF={train_result["pf"]:.2f}, Exp={train_result["expectancy"]:+.2f}h')
        if 'error' not in test_result:
            print(f'  TEST:  {test_result["trades"]} trades, WR={test_result["win_rate"]*100:.1f}%, '
                  f'PF={test_result["pf"]:.2f}, Exp={test_result["expectancy"]:+.2f}h')

        results.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train': train_result,
            'test': test_result,
        })

    return results


def parameter_search(htf_bars, ltf_bars, symbol, base_params, search_space):
    """Run bounded parameter search."""
    print('=' * 60)
    print('PARAMETER SEARCH')
    print('=' * 60)

    results = []
    total_combinations = 1
    for param, values in search_space.items():
        total_combinations *= len(values)

    print(f'Testing {total_combinations} parameter combinations...\n')

    # Generate all combinations
    from itertools import product
    param_names = list(search_space.keys())
    param_values = [search_space[p] for p in param_names]

    for combo in product(*param_values):
        params = base_params.copy()
        combo_dict = dict(zip(param_names, combo))

        # Build config with the parameters
        config = UnicornConfig(
            min_criteria_score=combo_dict.get('min_criteria', params.get('min_criteria_score', 3)),
            stop_max_atr_mult=combo_dict.get('stop_atr_mult', 3.0),
            fvg_min_atr_mult=combo_dict.get('fvg_atr_mult', 0.3),
        )

        params['config'] = config
        params['min_criteria_score'] = combo_dict.get('min_criteria', 3)

        result = run_single_backtest(htf_bars, ltf_bars, symbol, **params)

        if 'error' not in result:
            results.append({
                'params': combo_dict,
                'trades': result['trades'],
                'win_rate': result['win_rate'],
                'pf': result['pf'],
                'pnl': result['pnl'],
                'expectancy': result['expectancy'],
            })

    # Sort by expectancy
    results.sort(key=lambda x: x['expectancy'], reverse=True)

    print(f'{"min_crit":>10s} {"stop_atr":>10s} {"fvg_atr":>10s} {"trades":>8s} '
          f'{"WR":>8s} {"PF":>8s} {"Exp":>10s}')
    print('-' * 70)

    for r in results[:15]:
        p = r['params']
        print(f'{p.get("min_criteria", 3):>10d} {p.get("stop_atr_mult", 3.0):>10.1f} '
              f'{p.get("fvg_atr_mult", 0.3):>10.2f} {r["trades"]:>8d} '
              f'{r["win_rate"]*100:>7.1f}% {r["pf"]:>8.2f} {r["expectancy"]:>+9.2f}h')

    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose and tune Unicorn Model")
    parser.add_argument('--csv', required=True, help='Path to Databento CSV')
    parser.add_argument('--symbol', default='NQ', help='Symbol (default: NQ)')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-06-30', help='End date')
    parser.add_argument('--mode', choices=['decomp', 'walk-forward', 'search', 'all'],
                        default='all', help='Analysis mode')

    args = parser.parse_args()

    # Load data
    print(f'Loading {args.symbol} data from {args.csv}...')
    fetcher = DatabentoFetcher()
    htf_bars, ltf_bars = fetcher.load_from_csv(
        csv_path=args.csv,
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        htf_interval='15m',
        ltf_interval='5m',
    )
    print(f'Loaded {len(htf_bars)} HTF bars, {len(ltf_bars)} LTF bars\n')

    # Base parameters
    base_params = {
        'dollars_per_trade': 1000.0,
        'min_criteria_score': 3,
        'slippage_ticks': 2.0,
        'commission_per_contract': 5.00,
        'intrabar_policy': IntrabarPolicy.WORST,
    }

    if args.mode in ['decomp', 'all']:
        # Run baseline and decompose
        result = run_single_backtest(htf_bars, ltf_bars, args.symbol, **base_params)
        if 'error' not in result:
            expectancy_decomposition(result['result'])

    if args.mode in ['walk-forward', 'all']:
        # Walk-forward splits
        splits = [
            # Train Jan-Apr, Test May-Jun
            (('2024-01-01', '2024-04-30'), ('2024-05-01', '2024-06-30')),
            # Train Mar-Jun, Test Jan-Feb
            (('2024-03-01', '2024-06-30'), ('2024-01-01', '2024-02-29')),
        ]
        run_walk_forward(htf_bars, ltf_bars, args.symbol, splits, base_params)

    if args.mode in ['search', 'all']:
        # Parameter search
        search_space = {
            'min_criteria': [3, 4, 5],
            'stop_atr_mult': [2.0, 2.5, 3.0, 3.5],
            'fvg_atr_mult': [0.2, 0.3, 0.4],
        }
        parameter_search(htf_bars, ltf_bars, args.symbol, base_params, search_space)


if __name__ == '__main__':
    main()
