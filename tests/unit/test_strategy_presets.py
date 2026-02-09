"""Tests for strategy presets module."""

import pytest

from app.strategies.presets import (
    NY_AM_ORB_V1,
    PRESETS,
)


class TestPresetRegistry:
    """Preset registry invariants."""

    def test_registry_has_orb_v1(self):
        assert "ny-am-orb-v1" in PRESETS

    def test_all_presets_have_unique_slugs(self):
        slugs = [p.slug for p in PRESETS.values()]
        assert len(slugs) == len(set(slugs))

    def test_all_presets_have_required_fields(self):
        for slug, preset in PRESETS.items():
            assert preset.slug == slug
            assert preset.name
            assert preset.description
            assert preset.engine
            assert preset.param_schema
            assert preset.default_params


class TestOrbV1Preset:
    """NY AM ORB v1 specific tests."""

    def test_slug(self):
        assert NY_AM_ORB_V1.slug == "ny-am-orb-v1"

    def test_engine(self):
        assert NY_AM_ORB_V1.engine == "python"

    def test_required_params_present(self):
        expected = {"or_minutes", "confirm_mode", "stop_mode", "target_r", "max_trades", "session"}
        assert set(NY_AM_ORB_V1.param_schema.keys()) == expected

    def test_defaults_match_schema(self):
        """Every default_params key must exist in param_schema."""
        for key, value in NY_AM_ORB_V1.default_params.items():
            assert key in NY_AM_ORB_V1.param_schema
            pdef = NY_AM_ORB_V1.param_schema[key]
            if pdef.choices:
                assert value in pdef.choices

    def test_events_defined(self):
        assert NY_AM_ORB_V1.events == [
            "orb_range_update",
            "orb_range_locked",
            "setup_valid",
            "entry_signal",
        ]

    def test_tags_populated(self):
        assert "breakout" in NY_AM_ORB_V1.tags["strategy_archetypes"]
        assert "intraday" in NY_AM_ORB_V1.tags["timeframe_buckets"]


class TestParamSpace:
    """to_param_space() produces tuner-compatible output."""

    def test_choices_params_are_lists(self):
        space = NY_AM_ORB_V1.to_param_space()
        assert space["or_minutes"] == [15, 30, 60]
        assert space["confirm_mode"] == ["close-beyond", "retest"]
        assert space["stop_mode"] == ["or-opposite", "fixed-ticks"]
        assert space["max_trades"] == [1, 2]
        assert space["session"] == ["NY AM", "NY PM", "London"]

    def test_range_param_generates_grid(self):
        space = NY_AM_ORB_V1.to_param_space()
        # target_r: min=1.0, max=2.0, step=0.1 => [1.0, 1.1, ..., 2.0]
        assert space["target_r"][0] == 1.0
        assert space["target_r"][-1] == 2.0
        assert len(space["target_r"]) == 11

    def test_range_values_are_floats(self):
        space = NY_AM_ORB_V1.to_param_space()
        for v in space["target_r"]:
            assert isinstance(v, float)


class TestConfigSnapshot:
    """to_config_snapshot() produces version-compatible output."""

    def test_has_preset_slug(self):
        snap = NY_AM_ORB_V1.to_config_snapshot()
        assert snap["preset_slug"] == "ny-am-orb-v1"

    def test_has_params(self):
        snap = NY_AM_ORB_V1.to_config_snapshot()
        assert snap["params"]["or_minutes"] == 30
        assert snap["params"]["target_r"] == 1.5

    def test_has_param_schema(self):
        snap = NY_AM_ORB_V1.to_config_snapshot()
        assert "or_minutes" in snap["param_schema"]
        assert snap["param_schema"]["or_minutes"]["type"] == "int"
        assert snap["param_schema"]["or_minutes"]["choices"] == [15, 30, 60]

    def test_has_events(self):
        snap = NY_AM_ORB_V1.to_config_snapshot()
        assert len(snap["events"]) == 4


class TestToDict:
    """to_dict() API serialization."""

    def test_roundtrip_fields(self):
        d = NY_AM_ORB_V1.to_dict()
        assert d["slug"] == "ny-am-orb-v1"
        assert d["name"] == "NY AM ORB v1"
        assert d["engine"] == "python"
        assert "or_minutes" in d["param_schema"]
        assert d["default_params"]["or_minutes"] == 30
        assert len(d["events"]) == 4

    def test_param_schema_includes_description(self):
        d = NY_AM_ORB_V1.to_dict()
        assert "description" in d["param_schema"]["or_minutes"]


class TestPresetImmutability:
    """Presets are frozen dataclasses."""

    def test_cannot_mutate_slug(self):
        with pytest.raises(AttributeError):
            NY_AM_ORB_V1.slug = "hacked"  # type: ignore[misc]

    def test_cannot_mutate_defaults(self):
        with pytest.raises(AttributeError):
            NY_AM_ORB_V1.default_params = {}  # type: ignore[misc]
