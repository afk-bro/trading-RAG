"""Strategy Backtest admin endpoints.

Provides UI for running strategy backtests on historical data.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Path to historical data
DATA_PATH = Path(__file__).parent.parent.parent / "docs" / "historical_data" / "GLBX-20260129-JNB8PDSQ7C"
CSV_FILE = DATA_PATH / "glbx-mdp3-20210128-20260127.ohlcv-1m.csv"


@router.get("/backtests/strategy-test", response_class=HTMLResponse)
async def strategy_backtest_page(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """Render the strategy backtest configuration page."""
    return templates.TemplateResponse(
        "strategy_backtest.html",
        {"request": request},
    )


@router.post("/backtests/strategy-test/run", response_class=HTMLResponse)
async def run_strategy_backtest(
    request: Request,
    strategy: str = Form("unicorn_model"),
    symbol: str = Form("NQ"),
    start_date: str = Form(...),
    end_date: str = Form(...),
    direction: str = Form("long_only"),
    min_criteria: int = Form(3),
    slippage: float = Form(2.0),
    commission: float = Form(5.0),
    intrabar_policy: str = Form("worst"),
    _: bool = Depends(require_admin_token),
):
    """Run a strategy backtest and return results HTML."""
    try:
        # Import here to avoid circular imports
        from app.services.backtest.data import DatabentoFetcher
        from app.services.backtest.engines.unicorn_runner import (
            run_unicorn_backtest,
            IntrabarPolicy,
        )
        from app.services.strategy.strategies.unicorn_model import BiasDirection

        # Check if data file exists
        if not CSV_FILE.exists():
            return templates.TemplateResponse(
                "strategy_backtest_results.html",
                {
                    "request": request,
                    "error": f"Data file not found: {CSV_FILE}",
                },
            )

        # Load data
        logger.info(
            "strategy_backtest_loading_data",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        fetcher = DatabentoFetcher()
        htf_bars, ltf_bars = fetcher.load_from_csv(
            csv_path=str(CSV_FILE),
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            htf_interval="15m",
            ltf_interval="5m",
        )

        if len(htf_bars) < 50 or len(ltf_bars) < 30:
            return templates.TemplateResponse(
                "strategy_backtest_results.html",
                {
                    "request": request,
                    "error": f"Insufficient data: {len(htf_bars)} HTF bars, {len(ltf_bars)} LTF bars. Need at least 50 HTF and 30 LTF bars.",
                },
            )

        # Map direction filter
        direction_filter = None
        if direction == "long_only":
            direction_filter = BiasDirection.BULLISH
        elif direction == "short_only":
            direction_filter = BiasDirection.BEARISH

        # Map intrabar policy
        policy_map = {
            "worst": IntrabarPolicy.WORST,
            "best": IntrabarPolicy.BEST,
            "random": IntrabarPolicy.RANDOM,
        }
        policy = policy_map.get(intrabar_policy, IntrabarPolicy.WORST)

        # Run backtest
        logger.info(
            "strategy_backtest_running",
            symbol=symbol,
            htf_bars=len(htf_bars),
            ltf_bars=len(ltf_bars),
            direction=direction,
        )

        result = run_unicorn_backtest(
            symbol=symbol,
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            min_criteria_score=min_criteria,
            slippage_ticks=slippage,
            commission_per_contract=commission,
            intrabar_policy=policy,
            direction_filter=direction_filter,
        )

        # Build exit breakdown
        exit_breakdown = {}
        for trade in result.trades:
            reason = trade.exit_reason or "unknown"
            if reason not in exit_breakdown:
                exit_breakdown[reason] = {"count": 0, "pnl": 0.0, "wins": 0}
            exit_breakdown[reason]["count"] += 1
            exit_breakdown[reason]["pnl"] += trade.pnl_handles
            if trade.pnl_dollars > 0:
                exit_breakdown[reason]["wins"] += 1

        # Calculate win rate per exit reason
        for reason in exit_breakdown:
            count = exit_breakdown[reason]["count"]
            wins = exit_breakdown[reason]["wins"]
            exit_breakdown[reason]["win_rate"] = wins / count if count > 0 else 0

        # Build session stats for template
        session_stats = {}
        for session, stats in result.session_stats.items():
            session_stats[session.value] = {
                "trades": stats.trades_taken,
                "wins": stats.wins,
                "losses": stats.losses,
                "win_rate": stats.win_rate,
                "pnl": stats.total_pnl_handles,
            }

        # Build confidence buckets for template
        confidence_buckets = []
        for bucket in result.confidence_buckets:
            confidence_buckets.append({
                "min_conf": bucket.min_confidence,
                "max_conf": bucket.max_confidence,
                "trades": bucket.trade_count,
                "win_rate": bucket.win_rate,
                "avg_r": bucket.avg_r_multiple,
                "pnl": bucket.total_pnl,
            })

        # Build criteria bottlenecks for template
        criteria_bottlenecks = []
        for b in result.criteria_bottlenecks:
            criteria_bottlenecks.append({
                "criterion": b.criterion,
                "fail_rate": b.fail_rate,
                "fail_count": b.fail_count,
            })

        # Calculate R-metrics
        r_metrics = {
            "total_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "expectancy_r": 0.0,
            "r_distribution": [],
        }

        if result.trades:
            r_values = [t.r_multiple for t in result.trades]
            r_metrics["total_r"] = sum(r_values)

            win_r_values = [t.r_multiple for t in result.trades if t.pnl_dollars > 0]
            loss_r_values = [t.r_multiple for t in result.trades if t.pnl_dollars <= 0]

            if win_r_values:
                r_metrics["avg_win_r"] = sum(win_r_values) / len(win_r_values)
            if loss_r_values:
                r_metrics["avg_loss_r"] = sum(loss_r_values) / len(loss_r_values)

            # Expectancy in R: E = (Win% × Avg Win R) − (Loss% × |Avg Loss R|)
            win_pct = len(win_r_values) / len(result.trades)
            loss_pct = len(loss_r_values) / len(result.trades)
            r_metrics["expectancy_r"] = (win_pct * r_metrics["avg_win_r"]) - (loss_pct * abs(r_metrics["avg_loss_r"]))

            # R Distribution buckets
            buckets = [
                ("< -1.0R", lambda r: r < -1.0),
                ("-1.0 to 0R", lambda r: -1.0 <= r < 0),
                ("0 to +1R", lambda r: 0 <= r < 1.0),
                ("+1 to +2R", lambda r: 1.0 <= r < 2.0),
                ("+2R+", lambda r: r >= 2.0),
            ]
            for label, condition in buckets:
                count = len([r for r in r_values if condition(r)])
                r_metrics["r_distribution"].append({
                    "label": label,
                    "count": count,
                    "pct": count / len(result.trades) * 100 if result.trades else 0,
                })

        # Chart data for JavaScript
        chart_data = {
            "criteria_bottlenecks": criteria_bottlenecks,
            "r_distribution": r_metrics["r_distribution"],
            "session_stats": session_stats,
            "exit_breakdown": exit_breakdown,
        }

        # Template result object
        template_result = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "direction_filter": direction.replace("_", " ").title() if direction != "both" else None,
            "min_criteria": min_criteria,
            "slippage": slippage,
            "commission": commission,
            "intrabar_policy": intrabar_policy,
            "htf_bars": len(htf_bars),
            "ltf_bars": len(ltf_bars),
            "trades_taken": result.trades_taken,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor if result.profit_factor != float("inf") else 999.99,
            "total_pnl_handles": result.total_pnl_handles,
            "total_pnl_dollars": result.total_pnl_dollars,
            "expectancy_handles": result.expectancy_handles,
            "avg_r_multiple": result.avg_r_multiple,
            "avg_mfe": result.avg_mfe,
            "avg_mae": result.avg_mae,
            "mfe_capture_rate": result.mfe_capture_rate,
            "largest_win": result.largest_win_handles,
            "largest_loss": result.largest_loss_handles,
            "best_r": result.best_r_multiple,
            "worst_r": result.worst_r_multiple,
            "total_r": r_metrics["total_r"],
            "avg_win_r": r_metrics["avg_win_r"],
            "avg_loss_r": r_metrics["avg_loss_r"],
            "expectancy_r": r_metrics["expectancy_r"],
            "r_distribution": r_metrics["r_distribution"],
            "session_stats": session_stats,
            "exit_breakdown": exit_breakdown,
            "confidence_buckets": confidence_buckets,
            "confidence_correlation": result.confidence_win_correlation,
        }

        logger.info(
            "strategy_backtest_complete",
            symbol=symbol,
            trades=result.trades_taken,
            pf=result.profit_factor,
            pnl=result.total_pnl_handles,
        )

        return templates.TemplateResponse(
            "strategy_backtest_results.html",
            {
                "request": request,
                "result": template_result,
                "chart_data": chart_data,
            },
        )

    except Exception as e:
        logger.error("strategy_backtest_error", error=str(e))
        return templates.TemplateResponse(
            "strategy_backtest_results.html",
            {
                "request": request,
                "error": f"Backtest failed: {str(e)}",
            },
        )
