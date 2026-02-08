"""Unit tests for ICT Blueprint LTF entry logic."""

import pytest

from app.services.backtest.engines.ict_blueprint.ltf_entry import (
    advance_setup,
    check_ltf_msb,
    check_ob_zone_entry,
    check_sweep,
    create_setup_for_ob,
    define_breaker_zone,
    detect_fvg,
    find_l0_h0,
    get_active_setup_keys,
    select_candidate_setups,
    should_timeout,
)
from app.services.backtest.engines.ict_blueprint.types import (
    Bias,
    BreakerZone,
    FVG,
    HTFStateSnapshot,
    ICTBlueprintParams,
    LTFSetup,
    OrderBlock,
    SetupPhase,
    Side,
    SwingPoint,
    TradingRange,
)


def _make_ob(bias=Bias.BULLISH, top=50.0, bottom=45.0, msb_idx=5) -> OrderBlock:
    return OrderBlock(
        top=top, bottom=bottom, bias=bias,
        ob_id=(msb_idx, 3, 4, "long" if bias == Bias.BULLISH else "short"),
        anchor_swing=SwingPoint(2, 200, 45.0, True),
        msb_bar_index=msb_idx,
    )


def _make_htf_snapshot(bias=Bias.BULLISH) -> HTFStateSnapshot:
    return HTFStateSnapshot(
        bias=bias,
        swing_highs=(SwingPoint(2, 200, 55.0, True),),
        swing_lows=(SwingPoint(1, 100, 40.0, False),),
        current_range=TradingRange(high=55.0, low=40.0, midpoint=47.5, bias=bias),
        active_obs=(),
        last_msb_bar_index=5,
    )


# ---------------------------------------------------------------------------
# OB zone entry
# ---------------------------------------------------------------------------


class TestOBZoneEntry:
    def test_touch(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        assert check_ob_zone_entry(51.0, 52.0, 49.0, 51.0, ob, "touch") is True
        assert check_ob_zone_entry(51.0, 52.0, 51.0, 51.5, ob, "touch") is False

    def test_close_inside(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        assert check_ob_zone_entry(51.0, 52.0, 44.0, 47.0, ob, "close_inside") is True
        assert check_ob_zone_entry(51.0, 52.0, 44.0, 51.0, ob, "close_inside") is False

    def test_percent_inside(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        # Body from 46 to 48, fully inside OB → 100% overlap
        assert check_ob_zone_entry(46.0, 49.0, 44.0, 48.0, ob, "percent_inside", 0.10) is True
        # Body from 51 to 53, no overlap
        assert check_ob_zone_entry(51.0, 53.0, 50.5, 53.0, ob, "percent_inside", 0.10) is False


# ---------------------------------------------------------------------------
# Candidate setup selection
# ---------------------------------------------------------------------------


class TestCandidateSelection:
    def test_ordering_newest_first(self):
        ob1 = _make_ob(msb_idx=3)
        ob2 = _make_ob(msb_idx=7)
        s1 = LTFSetup(ob=ob1, side=Side.LONG, phase=SetupPhase.SCANNING)
        s2 = LTFSetup(ob=ob2, side=Side.LONG, phase=SetupPhase.SCANNING)
        result = select_candidate_setups([s1, s2])
        assert result[0].ob.msb_bar_index == 7

    def test_filters_timed_out(self):
        ob = _make_ob()
        s = LTFSetup(ob=ob, side=Side.LONG, phase=SetupPhase.TIMED_OUT)
        assert select_candidate_setups([s]) == []

    def test_filters_inactive(self):
        ob = _make_ob()
        s = LTFSetup(ob=ob, side=Side.LONG, phase=SetupPhase.INACTIVE)
        assert select_candidate_setups([s]) == []


# ---------------------------------------------------------------------------
# L0 / H0
# ---------------------------------------------------------------------------


class TestL0H0:
    def test_finds_l0_h0(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        lows = [SwingPoint(10, 1000, 46.0, False)]
        highs = [SwingPoint(8, 800, 52.0, True)]
        result = find_l0_h0(lows, highs, ob, current_index=12)
        assert result is not None
        l0, h0 = result
        assert l0.price == 46.0
        assert h0.price == 52.0

    def test_l0_above_ob_rejected(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        lows = [SwingPoint(10, 1000, 55.0, False)]  # Above OB top
        highs = [SwingPoint(8, 800, 60.0, True)]
        result = find_l0_h0(lows, highs, ob, current_index=12)
        assert result is None

    def test_freshness_constraint(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        lows = [SwingPoint(10, 1000, 46.0, False)]
        highs = [SwingPoint(8, 800, 52.0, True)]
        # last_exit_bar_index=10 means L0 at index 10 is stale
        result = find_l0_h0(lows, highs, ob, current_index=12, last_exit_bar_index=10)
        assert result is None

    def test_no_h0_before_l0(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        lows = [SwingPoint(2, 200, 46.0, False)]
        highs = [SwingPoint(5, 500, 52.0, True)]  # H0 is AFTER L0
        result = find_l0_h0(lows, highs, ob, current_index=7)
        assert result is None


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_sweep_detected(self):
        assert check_sweep(44.0, 46.0) == 44.0

    def test_no_sweep(self):
        assert check_sweep(47.0, 46.0) is None


# ---------------------------------------------------------------------------
# LTF MSB
# ---------------------------------------------------------------------------


class TestLTFMSB:
    def test_msb_confirmed(self):
        assert check_ltf_msb(53.0, 52.0) is True

    def test_no_msb(self):
        assert check_ltf_msb(51.0, 52.0) is False


# ---------------------------------------------------------------------------
# Breaker zone
# ---------------------------------------------------------------------------


class TestBreakerZone:
    def test_breaker_defined(self):
        # Bars: bearish at 8, confirmation (MSB) at 9
        bars = [
            (7, 700, 50.0, 52.0, 49.0, 51.0),  # bullish
            (8, 800, 51.0, 52.0, 48.0, 49.0),  # bearish
            (9, 900, 49.0, 55.0, 49.0, 54.0),  # MSB confirmation
        ]
        bz = define_breaker_zone(bars, 9, breaker_candles=1)
        assert bz is not None
        assert bz.top == 52.0
        assert bz.bottom == 48.0
        assert bz.bar_index == 8

    def test_no_bearish_before_msb(self):
        bars = [
            (7, 700, 49.0, 52.0, 49.0, 51.0),  # bullish
            (8, 800, 51.0, 55.0, 50.0, 54.0),  # bullish
            (9, 900, 54.0, 58.0, 53.0, 57.0),  # MSB
        ]
        bz = define_breaker_zone(bars, 9, breaker_candles=1)
        assert bz is None


# ---------------------------------------------------------------------------
# FVG
# ---------------------------------------------------------------------------


class TestFVG:
    def test_bullish_fvg(self):
        bars = [
            (0, 100, 10.0, 12.0, 9.0, 11.0),
            (1, 200, 11.0, 16.0, 11.0, 15.0),  # big up
            (2, 300, 15.0, 17.0, 14.0, 16.0),   # gap: bar[0].high=12 < bar[2].low=14
        ]
        fvg = detect_fvg(bars, 2)
        assert fvg is not None
        assert fvg.bullish is True
        assert fvg.bottom == 12.0
        assert fvg.top == 14.0

    def test_no_fvg(self):
        bars = [
            (0, 100, 10.0, 12.0, 9.0, 11.0),
            (1, 200, 11.0, 13.0, 10.0, 12.5),
            (2, 300, 12.0, 13.0, 11.0, 12.5),  # no gap
        ]
        fvg = detect_fvg(bars, 2)
        assert fvg is None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_exceeded(self):
        ob = _make_ob()
        s = LTFSetup(ob=ob, side=Side.LONG, phase=SetupPhase.ENTRY_PENDING, msb_bar_index=10)
        assert should_timeout(s, bar_index=25, max_wait_bars=12) is True

    def test_not_timed_out(self):
        ob = _make_ob()
        s = LTFSetup(ob=ob, side=Side.LONG, phase=SetupPhase.ENTRY_PENDING, msb_bar_index=10)
        assert should_timeout(s, bar_index=20, max_wait_bars=12) is False

    def test_wrong_phase(self):
        ob = _make_ob()
        s = LTFSetup(ob=ob, side=Side.LONG, phase=SetupPhase.SCANNING, msb_bar_index=10)
        assert should_timeout(s, bar_index=50, max_wait_bars=12) is False


# ---------------------------------------------------------------------------
# Full state machine sequence
# ---------------------------------------------------------------------------


class TestAdvanceSetup:
    def test_full_sequence_scanning_to_entry(self):
        ob = _make_ob(top=50.0, bottom=45.0)
        setup = create_setup_for_ob(ob, Bias.BULLISH)
        htf = _make_htf_snapshot()
        params = ICTBlueprintParams(
            entry_mode="msb_close",
            require_sweep=True,
            max_wait_bars_after_msb=12,
            ltf_swing_lookback=1,
            breaker_candles=1,
        )

        # Provide pre-existing swings for L0/H0
        ltf_lows = [SwingPoint(10, 1000, 46.0, False)]
        ltf_highs = [SwingPoint(8, 800, 52.0, True)]

        # Bar at index 12: scanning → finds L0/H0 → sweep_pending
        bar = (12, 1200, 47.0, 48.0, 47.0, 47.5)
        h1_bars = [bar]
        result = advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf)
        assert result is None
        assert setup.phase == SetupPhase.SWEEP_PENDING

        # Bar at index 13: sweep below L0 (46.0) → msb_pending
        bar = (13, 1300, 47.0, 48.0, 44.0, 45.0)
        h1_bars.append(bar)
        result = advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf)
        assert result is None
        assert setup.phase == SetupPhase.MSB_PENDING

        # Bar at index 14: close above H0 (52.0) → entry (msb_close mode)
        bar = (14, 1400, 50.0, 54.0, 49.0, 53.0)
        h1_bars.append(bar)
        result = advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf)
        assert result is not None
        assert result == 53.0  # msb_close = bar close

    def test_one_setup_per_ob_dedup(self):
        ob = _make_ob()
        s1 = LTFSetup(ob=ob, side=Side.LONG, phase=SetupPhase.SCANNING)
        existing = get_active_setup_keys([s1])
        assert ob.ob_id in existing

    def test_timeout_cleans_up_to_timed_out(self):
        """After MSB, if max_wait_bars exceeded, phase → TIMED_OUT."""
        ob = _make_ob(top=50.0, bottom=45.0)
        setup = create_setup_for_ob(ob, Bias.BULLISH)
        htf = _make_htf_snapshot()
        params = ICTBlueprintParams(
            entry_mode="breaker_retest", require_sweep=False,
            max_wait_bars_after_msb=5, ltf_swing_lookback=1, breaker_candles=1,
        )

        ltf_lows = [SwingPoint(10, 1000, 46.0, False)]
        ltf_highs = [SwingPoint(8, 800, 52.0, True)]

        # SCANNING → finds L0/H0 → MSB_PENDING (skip sweep)
        bar = (12, 1200, 47.0, 48.0, 47.0, 47.5)
        h1_bars = [bar]
        advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf)
        assert setup.phase == SetupPhase.MSB_PENDING

        # MSB confirmation
        bar = (13, 1300, 50.0, 54.0, 49.0, 53.0)
        h1_bars.append(bar)
        advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf)
        assert setup.phase == SetupPhase.ENTRY_PENDING

        # Advance past max_wait_bars without entry
        for i in range(14, 20):
            bar = (i, i * 100, 51.0, 52.0, 50.0, 51.5)  # no breaker retest
            h1_bars.append(bar)
            advance_setup(setup, bar, ltf_lows, ltf_highs, h1_bars, params, htf)

        assert setup.phase == SetupPhase.TIMED_OUT

    def test_timeout_propagates_bar_index_for_fresh_l0(self):
        """After timeout, new setup should not reuse stale L0 via last_setup_bar_index."""
        ob = _make_ob(top=50.0, bottom=45.0)
        # Simulate timeout at bar 20
        ob.last_setup_bar_index = 20
        new_setup = create_setup_for_ob(ob, Bias.BULLISH)
        assert new_setup.last_exit_bar_index == 20

        # L0 at index 10 should be rejected (stale)
        ltf_lows = [SwingPoint(10, 1000, 46.0, False)]
        ltf_highs = [SwingPoint(8, 800, 52.0, True)]
        result = find_l0_h0(ltf_lows, ltf_highs, ob, current_index=25,
                            last_exit_bar_index=new_setup.last_exit_bar_index)
        assert result is None  # L0 at 10 <= last_exit 20

        # L0 at index 25 should be accepted
        ltf_lows.append(SwingPoint(25, 2500, 47.0, False))
        ltf_highs.append(SwingPoint(23, 2300, 53.0, True))
        result = find_l0_h0(ltf_lows, ltf_highs, ob, current_index=27,
                            last_exit_bar_index=new_setup.last_exit_bar_index)
        assert result is not None

    def test_fvg_only_used_in_fvg_fill_mode(self):
        """FVG should not affect entry in breaker_retest mode."""
        ob = _make_ob(top=50.0, bottom=45.0)
        setup = create_setup_for_ob(ob, Bias.BULLISH)
        # Manually set to ENTRY_PENDING with breaker + FVG
        setup.phase = SetupPhase.ENTRY_PENDING
        setup.breaker = BreakerZone(top=48.0, bottom=46.0, bar_index=8)
        setup.fvg = FVG(top=49.0, bottom=47.0, bar_index=9, bullish=True)
        setup.msb_bar_index = 9

        from app.services.backtest.engines.ict_blueprint.ltf_entry import check_entry_trigger

        # Bar that touches breaker but NOT FVG — should still trigger in breaker mode
        entry = check_entry_trigger(
            47.5, 49.0, 46.5, 48.0, 10, setup, "breaker_retest", []
        )
        assert entry is not None  # Breaker retest triggers

        # In fvg_fill mode with no FVG overlap — should NOT trigger
        setup.fvg = FVG(top=52.0, bottom=51.0, bar_index=9, bullish=True)
        entry = check_entry_trigger(
            47.5, 49.0, 46.5, 48.0, 10, setup, "fvg_fill", []
        )
        assert entry is None  # FVG not filled (bar doesn't reach 52)
