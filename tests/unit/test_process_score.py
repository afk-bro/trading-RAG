"""Tests for backtest coaching process score computation."""

from app.services.backtest.process_score import (
    COMPUTE_CEILING,
    ProcessScoreResult,
    _score_consistency,
    _score_exit_quality,
    _score_regime_alignment,
    _score_risk_discipline,
    _score_rule_adherence,
    compute_process_score,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_trades(n, pnl=10.0, return_pct=0.01, side="long"):
    return [
        {
            "pnl": pnl,
            "return_pct": return_pct,
            "side": side,
            "t_entry": f"2024-01-15T{(9 + i % 8):02d}:30:00",
            "t_exit": f"2024-01-15T{(10 + i % 8):02d}:00:00",
        }
        for i in range(n)
    ]


def _make_events(
    n_entry=5,
    n_setup=5,
    bar_gap=2,
    setup_offset=0,
):
    """Generate events with setup_valid preceding entry_signal."""
    events = []
    for i in range(n_setup):
        events.append(
            {
                "type": "setup_valid",
                "bar_index": i * bar_gap + setup_offset,
            }
        )
    for i in range(n_entry):
        events.append(
            {
                "type": "entry_signal",
                "bar_index": i * bar_gap + setup_offset + 1,
            }
        )
    return events


# ── Rule Adherence ────────────────────────────────────────────────────


class TestScoreRuleAdherence:
    def test_no_events_returns_none(self):
        assert _score_rule_adherence(None) is None
        assert _score_rule_adherence([]) is None

    def test_no_entry_signals_returns_none(self):
        events = [{"type": "setup_valid", "bar_index": 0}]
        assert _score_rule_adherence(events) is None

    def test_no_setups_returns_zero(self):
        events = [{"type": "entry_signal", "bar_index": 5}]
        assert _score_rule_adherence(events) == 0.0

    def test_all_matched(self):
        events = _make_events(n_entry=3, n_setup=3, bar_gap=3)
        score = _score_rule_adherence(events)
        assert score == 100.0

    def test_partial_match(self):
        events = [
            {"type": "setup_valid", "bar_index": 0},
            {"type": "entry_signal", "bar_index": 2},
            {"type": "entry_signal", "bar_index": 20},  # no setup nearby
        ]
        score = _score_rule_adherence(events)
        assert score == 50.0

    def test_setup_too_far_away(self):
        events = [
            {"type": "setup_valid", "bar_index": 0},
            {"type": "entry_signal", "bar_index": 100},
        ]
        score = _score_rule_adherence(events)
        assert score == 0.0


# ── Regime Alignment ──────────────────────────────────────────────────


class TestScoreRegimeAlignment:
    def test_no_trades_returns_none(self):
        assert _score_regime_alignment([], None, None) is None

    def test_no_regime_returns_none(self):
        trades = _make_trades(5)
        assert _score_regime_alignment(trades, None, None) is None

    def test_neutral_regime_returns_100(self):
        trades = _make_trades(5, side="long")
        regime = {"trend_tag": "flat", "vol_tag": "low", "tags": ["flat"]}
        score = _score_regime_alignment(trades, regime, None)
        assert score == 100.0

    def test_uptrend_all_long(self):
        trades = _make_trades(10, side="long")
        regime = {"trend_tag": "uptrend", "tags": ["uptrend"]}
        score = _score_regime_alignment(trades, regime, None)
        assert score == 100.0

    def test_uptrend_mixed(self):
        trades = _make_trades(8, side="long") + _make_trades(2, side="short")
        regime = {"trend_tag": "uptrend", "tags": ["uptrend"]}
        score = _score_regime_alignment(trades, regime, None)
        assert score == 80.0

    def test_downtrend_all_short(self):
        trades = _make_trades(5, side="short")
        regime = {"trend_tag": "downtrend", "tags": ["downtrend"]}
        score = _score_regime_alignment(trades, regime, None)
        assert score == 100.0


# ── Risk Discipline ──────────────────────────────────────────────────


class TestScoreRiskDiscipline:
    def test_no_trades_returns_none(self):
        assert _score_risk_discipline([]) is None

    def test_no_losing_trades(self):
        trades = _make_trades(10, pnl=100)
        assert _score_risk_discipline(trades) == 100.0

    def test_uniform_losses(self):
        trades = [{"pnl": -10.0} for _ in range(5)]
        score = _score_risk_discipline(trades)
        # All losses equal, ratio = 1.0, score = 100
        assert score == 100.0

    def test_one_outlier_loss(self):
        trades = [{"pnl": -10.0} for _ in range(4)] + [{"pnl": -100.0}]
        score = _score_risk_discipline(trades)
        # Worst/median ratio = 100/10 = 10, score = max(0, 100 - 9*25) = 0
        assert score <= 10.0

    def test_with_size_field(self):
        trades = [
            {"pnl": -10.0, "size": 1.0},
            {"pnl": -10.0, "size": 1.0},
            {"pnl": -10.0, "size": 1.0},
        ]
        score = _score_risk_discipline(trades)
        # Uniform sizes → good sizing CV, uniform losses → good loss ratio
        assert score >= 90.0


# ── Exit Quality ─────────────────────────────────────────────────────


class TestScoreExitQuality:
    def test_no_trades_returns_none(self):
        assert _score_exit_quality([]) is None

    def test_no_winners_returns_none(self):
        trades = [{"return_pct": -0.01}]
        assert _score_exit_quality(trades) is None

    def test_no_losers_returns_none(self):
        trades = [{"return_pct": 0.02}]
        assert _score_exit_quality(trades) is None

    def test_ratio_2_is_perfect(self):
        trades = [
            {"return_pct": 0.02},
            {"return_pct": -0.01},
        ]
        score = _score_exit_quality(trades)
        assert score == 100.0

    def test_ratio_1_is_partial(self):
        trades = [
            {"return_pct": 0.01},
            {"return_pct": -0.01},
        ]
        score = _score_exit_quality(trades)
        # ratio=1.0, score = (1.0 - 0.5) / 1.5 * 100 ≈ 33.3
        assert 30 <= score <= 40


# ── Consistency ──────────────────────────────────────────────────────


class TestScoreConsistency:
    def test_too_few_trades_returns_none(self):
        trades = _make_trades(4)
        assert _score_consistency(trades) is None

    def test_perfectly_consistent(self):
        trades = [{"return_pct": 0.01} for _ in range(10)]
        score = _score_consistency(trades)
        # CV = 0 → score 100
        assert score == 100.0

    def test_highly_variable(self):
        trades = [
            {"return_pct": 0.10},
            {"return_pct": -0.08},
            {"return_pct": 0.05},
            {"return_pct": -0.03},
            {"return_pct": 0.12},
        ]
        score = _score_consistency(trades)
        assert score is not None
        assert 0 <= score <= 100

    def test_zero_mean_returns_none(self):
        # Returns that sum to zero
        trades = [
            {"return_pct": 0.01},
            {"return_pct": -0.01},
            {"return_pct": 0.01},
            {"return_pct": -0.01},
            {"return_pct": 0.0},
        ]
        score = _score_consistency(trades)
        assert score is None  # zero mean → CV undefined

    def test_nan_return_ignored(self):
        trades = [
            {"return_pct": 0.01},
            {"return_pct": 0.02},
            {"return_pct": float("nan")},
            {"return_pct": 0.01},
            {"return_pct": 0.02},
            {"return_pct": 0.015},
        ]
        score = _score_consistency(trades)
        assert score is not None


# ── Composite Score ──────────────────────────────────────────────────


class TestComputeProcessScore:
    def test_empty_trades(self):
        result = compute_process_score([], None, None, None)
        assert result.total is None
        assert result.grade == "unavailable"

    def test_compute_ceiling_exceeded(self):
        trades = _make_trades(COMPUTE_CEILING + 1)
        result = compute_process_score(trades, None, None, None)
        assert result.total is None
        assert result.grade == "unavailable"
        assert result.components == []

    def test_event_ceiling_exceeded(self):
        trades = _make_trades(10)
        events = [
            {"type": "entry_signal", "bar_index": i} for i in range(COMPUTE_CEILING + 1)
        ]
        result = compute_process_score(trades, events, None, None)
        assert result.total is None
        assert result.grade == "unavailable"

    def test_full_score_with_all_data(self):
        trades = _make_trades(6, pnl=10, return_pct=0.01, side="long") + _make_trades(
            4, pnl=-5, return_pct=-0.005, side="long"
        )
        events = _make_events(n_entry=5, n_setup=5)
        regime = {"trend_tag": "uptrend", "tags": ["uptrend"]}

        result = compute_process_score(trades, events, regime, None)
        assert result.total is not None
        assert 0 <= result.total <= 100
        assert result.grade in ("A", "B", "C", "D", "F")
        assert len(result.components) == 5

    def test_weight_redistribution(self):
        # No events → rule_adherence unavailable
        # No regime → regime_alignment unavailable
        trades = _make_trades(10, pnl=10, return_pct=0.01)
        trades[7]["pnl"] = -5
        trades[7]["return_pct"] = -0.005
        trades[8]["pnl"] = -3
        trades[8]["return_pct"] = -0.003

        result = compute_process_score(trades, None, None, None)
        assert result.total is not None

        available = [c for c in result.components if c.available]
        unavailable = [c for c in result.components if not c.available]

        # rule_adherence and regime_alignment should be unavailable
        unavailable_names = {c.name for c in unavailable}
        assert "rule_adherence" in unavailable_names
        assert "regime_alignment" in unavailable_names

        # Available weights should sum to ~1.0
        total_weight = sum(c.weight for c in available)
        assert abs(total_weight - 1.0) < 0.01

    def test_grade_boundaries(self):
        # Test exact boundary values
        from app.services.backtest.process_score import _grade

        assert _grade(80.0) == "A"
        assert _grade(79.9) == "B"
        assert _grade(65.0) == "B"
        assert _grade(64.9) == "C"
        assert _grade(50.0) == "C"
        assert _grade(49.9) == "D"
        assert _grade(35.0) == "D"
        assert _grade(34.9) == "F"
        assert _grade(0.0) == "F"
        assert _grade(None) == "unavailable"

    def test_single_trade(self):
        trades = _make_trades(1, pnl=10, return_pct=0.01)
        result = compute_process_score(trades, None, None, None)
        # With only 1 trade, consistency should be unavailable (N < 5)
        consistency = next(
            (c for c in result.components if c.name == "consistency"), None
        )
        assert consistency is not None
        assert not consistency.available

    def test_process_score_not_outcome_dependent(self):
        """Process score should not increase when only profit factor improves."""
        base_trades = _make_trades(10, pnl=10, return_pct=0.01)
        base_trades[8] = {
            "pnl": -5,
            "return_pct": -0.005,
            "side": "long",
            "t_entry": "2024-01-15T09:30:00",
            "t_exit": "2024-01-15T10:00:00",
        }
        base_trades[9] = {
            "pnl": -5,
            "return_pct": -0.005,
            "side": "long",
            "t_entry": "2024-01-15T10:30:00",
            "t_exit": "2024-01-15T11:00:00",
        }

        # Higher profit trades (same structure, higher PnL)
        high_profit_trades = _make_trades(10, pnl=100, return_pct=0.10)
        high_profit_trades[8] = {
            "pnl": -5,
            "return_pct": -0.005,
            "side": "long",
            "t_entry": "2024-01-15T09:30:00",
            "t_exit": "2024-01-15T10:00:00",
        }
        high_profit_trades[9] = {
            "pnl": -5,
            "return_pct": -0.005,
            "side": "long",
            "t_entry": "2024-01-15T10:30:00",
            "t_exit": "2024-01-15T11:00:00",
        }

        base_result = compute_process_score(base_trades, None, None, None)
        high_result = compute_process_score(high_profit_trades, None, None, None)

        # Exit quality may differ (win/loss ratio), but risk discipline
        # and consistency should stay comparable
        base_risk = next(
            c for c in base_result.components if c.name == "risk_discipline"
        )
        high_risk = next(
            c for c in high_result.components if c.name == "risk_discipline"
        )
        assert abs(base_risk.score - high_risk.score) < 20

    def test_nan_inf_in_trades(self):
        trades = [
            {"pnl": float("nan"), "return_pct": 0.01, "side": "long"},
            {"pnl": 10.0, "return_pct": float("inf"), "side": "long"},
            {"pnl": 10.0, "return_pct": 0.01, "side": "long"},
            {"pnl": -5.0, "return_pct": -0.005, "side": "long"},
            {"pnl": 10.0, "return_pct": 0.01, "side": "long"},
            {"pnl": 10.0, "return_pct": 0.01, "side": "long"},
        ]
        # Should not raise
        result = compute_process_score(trades, None, None, None)
        assert isinstance(result, ProcessScoreResult)
