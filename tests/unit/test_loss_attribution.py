"""Tests for backtest coaching loss attribution."""

from app.services.backtest.loss_attribution import (
    COMPUTE_CEILING,
    LossAttributionResult,
    _compute_max_drawdown,
    _compute_size_clusters,
    _compute_time_clusters,
    _compute_total_return,
    compute_loss_attribution,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_trade(
    pnl,
    return_pct=None,
    side="long",
    hour=9,
    day="2024-01-15",
):
    if return_pct is None:
        return_pct = pnl / 10000.0
    return {
        "pnl": pnl,
        "return_pct": return_pct,
        "side": side,
        "t_entry": f"{day}T{hour:02d}:30:00",
        "t_exit": f"{day}T{hour + 1:02d}:00:00",
    }


# ── Time Clusters ────────────────────────────────────────────────────


class TestTimeClusters:
    def test_empty_losses(self):
        result = _compute_time_clusters([], 0.0)
        assert result == []

    def test_single_hour(self):
        trades = [_make_trade(-10, hour=9), _make_trade(-20, hour=9)]
        clusters = _compute_time_clusters(trades, -30.0)
        assert len(clusters) == 1
        assert clusters[0].label == "09:00"
        assert clusters[0].trade_count == 2
        assert clusters[0].total_loss == -30.0
        assert clusters[0].pct_of_total_losses == 100.0

    def test_multiple_hours(self):
        trades = [
            _make_trade(-10, hour=9),
            _make_trade(-20, hour=10),
            _make_trade(-30, hour=10),
        ]
        clusters = _compute_time_clusters(trades, -60.0)
        assert len(clusters) == 2
        assert clusters[0].label == "09:00"
        assert clusters[1].label == "10:00"
        assert clusters[1].trade_count == 2

    def test_no_timestamp(self):
        trades = [{"pnl": -10, "t_entry": None}]
        result = _compute_time_clusters(trades, -10.0)
        assert result == []


# ── Size Clusters ────────────────────────────────────────────────────


class TestSizeClusters:
    def test_empty(self):
        assert _compute_size_clusters([], 0.0) == []

    def test_three_buckets(self):
        trades = [
            _make_trade(-1),
            _make_trade(-2),
            _make_trade(-3),
            _make_trade(-50),
            _make_trade(-60),
            _make_trade(-70),
            _make_trade(-500),
            _make_trade(-600),
            _make_trade(-700),
        ]
        clusters = _compute_size_clusters(trades, sum(t["pnl"] for t in trades))
        labels = [c.label for c in clusters]
        assert "small" in labels
        assert "large" in labels

    def test_single_trade_no_crash(self):
        trades = [_make_trade(-100)]
        clusters = _compute_size_clusters(trades, -100.0)
        assert len(clusters) >= 1


# ── Return / Drawdown helpers ────────────────────────────────────────


class TestReturnHelpers:
    def test_total_return_empty(self):
        assert _compute_total_return([]) == 0.0

    def test_total_return_positive(self):
        trades = [{"return_pct": 0.10}, {"return_pct": 0.05}]
        ret = _compute_total_return(trades)
        expected = ((1.10 * 1.05) - 1) * 100
        assert abs(ret - expected) < 0.01

    def test_max_drawdown_no_trades(self):
        assert _compute_max_drawdown([]) == 0.0

    def test_max_drawdown_losing_sequence(self):
        trades = [
            {"return_pct": 0.10},
            {"return_pct": -0.20},
            {"return_pct": 0.05},
        ]
        dd = _compute_max_drawdown(trades)
        assert dd < 0  # Should be negative


# ── Full Attribution ─────────────────────────────────────────────────


class TestComputeLossAttribution:
    def test_no_losing_trades(self):
        trades = [_make_trade(10, return_pct=0.01) for _ in range(5)]
        result = compute_loss_attribution(trades)
        assert result.total_losses == 0
        assert result.total_loss_amount == 0.0
        assert result.time_clusters == []
        assert result.counterfactuals == []

    def test_all_losing_trades(self):
        trades = [
            _make_trade(-10, return_pct=-0.001, hour=9),
            _make_trade(-20, return_pct=-0.002, hour=10),
            _make_trade(-30, return_pct=-0.003, hour=11),
        ]
        result = compute_loss_attribution(trades)
        assert result.total_losses == 3
        assert result.total_loss_amount == -60.0
        assert len(result.time_clusters) == 3

    def test_regime_summary_present(self):
        trades = [_make_trade(-10, return_pct=-0.001)]
        regime = {"trend_tag": "downtrend", "vol_tag": "high"}
        result = compute_loss_attribution(trades, regime_is=regime)
        assert result.regime_summary is not None
        assert "downtrend" in result.regime_summary.regime_tags

    def test_counterfactual_skip_hour(self):
        # All losses in hour 14, gains in hour 9
        trades = [
            _make_trade(100, return_pct=0.10, hour=9, day="2024-01-15"),
            _make_trade(-50, return_pct=-0.05, hour=14, day="2024-01-15"),
            _make_trade(-40, return_pct=-0.04, hour=14, day="2024-01-16"),
        ]
        result = compute_loss_attribution(trades)
        # Should have a "skip hour 14" counterfactual
        skip_hour = [cf for cf in result.counterfactuals if "hour" in cf.description]
        assert len(skip_hour) >= 1

    def test_counterfactual_regime_filter(self):
        trades = [
            _make_trade(100, return_pct=0.10, side="long", hour=9),
            _make_trade(-50, return_pct=-0.05, side="short", hour=10),
        ]
        regime = {"trend_tag": "uptrend"}
        result = compute_loss_attribution(trades, regime_is=regime, regime_oos=None)
        regime_cf = [
            cf for cf in result.counterfactuals if "regime" in cf.description.lower()
        ]
        assert len(regime_cf) >= 1

    def test_compute_ceiling_no_counterfactuals(self):
        trades = [
            _make_trade(-1, return_pct=-0.0001) for _ in range(COMPUTE_CEILING + 1)
        ]
        result = compute_loss_attribution(trades)
        assert result.total_losses == COMPUTE_CEILING + 1
        assert result.counterfactuals == []
        # Clusters should still be computed
        assert len(result.time_clusters) >= 1

    def test_counterfactual_max_trades_per_day(self):
        # 10 trades on day 1, 1 trade on day 2
        trades = [
            _make_trade(-5, return_pct=-0.005, hour=9 + i, day="2024-01-15")
            for i in range(7)
        ] + [_make_trade(50, return_pct=0.05, hour=9, day="2024-01-16")]
        result = compute_loss_attribution(trades)
        # May or may not fire depending on whether cap < max
        # At minimum, should not crash
        assert isinstance(result, LossAttributionResult)

    def test_zero_loss_amount_no_division_error(self):
        # Edge case: pnl is exactly 0.0 for "losing" trades
        trades = [
            {
                "pnl": 0.0,
                "return_pct": 0.0,
                "side": "long",
                "t_entry": "2024-01-15T09:30:00",
            }
        ]
        # pnl=0 is not < 0, so no losses
        result = compute_loss_attribution(trades)
        assert result.total_losses == 0
