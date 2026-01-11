"""Unit tests for PlanBuilder."""


from app.services.testing.plan_builder import PlanBuilder


class TestPlanBuilder:
    """Tests for PlanBuilder."""

    def test_build_plan_has_three_sections(self):
        """build returns dict with inputs, resolved, provenance."""
        builder = PlanBuilder(
            base_spec={"strategy_name": "breakout"},
            objective="sharpe_dd_penalty",
            constraints={"max_variants": 25},
            dataset_ref="BTC_1h.csv",
            generator_name="grid_search_v1",
            generator_version="1.0.0",
        )

        builder.add_variant(0, {"lookback_days": 200}, "baseline")
        builder.add_variant(1, {"lookback_days": 252}, "grid")

        plan = builder.build()

        assert "inputs" in plan
        assert "resolved" in plan
        assert "provenance" in plan
        assert plan["resolved"]["n_variants"] == 2
        assert len(plan["resolved"]["variants"]) == 2

    def test_build_includes_fingerprints(self):
        """build includes fingerprints in provenance."""
        builder = PlanBuilder(
            base_spec={"strategy_name": "breakout"},
            objective="sharpe_dd_penalty",
            constraints={},
            dataset_ref="BTC_1h.csv",
            generator_name="grid_search_v1",
            generator_version="1.0.0",
        )

        plan = builder.build()

        assert "fingerprints" in plan["provenance"]
        assert "plan" in plan["provenance"]["fingerprints"]

    def test_build_fingerprint_is_stable(self):
        """Same inputs produce same fingerprint."""

        def make_builder():
            builder = PlanBuilder(
                base_spec={"strategy_name": "breakout", "param": 123},
                objective="sharpe",
                constraints={"max_variants": 10},
                dataset_ref="BTC_1h.csv",
                generator_name="grid_v1",
                generator_version="1.0.0",
            )
            builder.add_variant(0, {"x": 1}, "baseline")
            return builder

        plan1 = make_builder().build()
        plan2 = make_builder().build()

        # Fingerprints should match (ignoring timestamp differences)
        fp1 = plan1["provenance"]["fingerprints"]["plan"]
        fp2 = plan2["provenance"]["fingerprints"]["plan"]
        assert fp1 == fp2

    def test_add_variant_is_chainable(self):
        """add_variant returns self for chaining."""
        builder = PlanBuilder(
            base_spec={},
            objective="sharpe",
            constraints={},
            dataset_ref="test.csv",
            generator_name="test",
            generator_version="1.0",
        )

        result = builder.add_variant(0, {}, "baseline")

        assert result is builder

    def test_build_includes_seed_if_provided(self):
        """build includes seed in provenance if provided."""
        builder = PlanBuilder(
            base_spec={},
            objective="sharpe",
            constraints={},
            dataset_ref="test.csv",
            generator_name="random_search",
            generator_version="1.0",
            seed=42,
        )

        plan = builder.build()

        assert plan["provenance"]["seed"] == 42

    def test_build_seed_none_if_not_provided(self):
        """build has seed=None in provenance if not provided."""
        builder = PlanBuilder(
            base_spec={},
            objective="sharpe",
            constraints={},
            dataset_ref="test.csv",
            generator_name="grid",
            generator_version="1.0",
        )

        plan = builder.build()

        assert plan["provenance"]["seed"] is None

    def test_inputs_section_has_expected_fields(self):
        """inputs section contains expected fields."""
        builder = PlanBuilder(
            base_spec={"strategy": "bb_reversal"},
            objective="calmar",
            constraints={"max_dd": 0.2},
            dataset_ref="ETH_4h.csv",
            generator_name="sweep",
            generator_version="2.0",
            generator_config={"param_ranges": {"x": [1, 2, 3]}},
        )

        plan = builder.build()
        inputs = plan["inputs"]

        assert inputs["base_spec"] == {"strategy": "bb_reversal"}
        assert inputs["objective"]["name"] == "calmar"
        assert inputs["constraints"] == {"max_dd": 0.2}
        assert inputs["dataset_ref"] == "ETH_4h.csv"
        assert inputs["generator_config"] == {"param_ranges": {"x": [1, 2, 3]}}

    def test_provenance_has_generator_info(self):
        """provenance section contains generator info."""
        builder = PlanBuilder(
            base_spec={},
            objective="sharpe",
            constraints={},
            dataset_ref="test.csv",
            generator_name="grid_sweep_v2",
            generator_version="2.1.0",
        )

        plan = builder.build()
        generator = plan["provenance"]["generator"]

        assert generator["name"] == "grid_sweep_v2"
        assert generator["version"] == "2.1.0"

    def test_provenance_has_created_at(self):
        """provenance section contains created_at timestamp."""
        builder = PlanBuilder(
            base_spec={},
            objective="sharpe",
            constraints={},
            dataset_ref="test.csv",
            generator_name="test",
            generator_version="1.0",
        )

        plan = builder.build()

        assert "created_at" in plan["provenance"]
        # Should be ISO format
        assert "T" in plan["provenance"]["created_at"]
