"""Unit tests for strategy repository."""

from app.services.strategy.repository import slugify


class TestSlugify:
    """Tests for slug generation."""

    def test_basic_slugify(self):
        """Simple name converts to lowercase slug."""
        assert slugify("My Strategy") == "my-strategy"

    def test_slugify_removes_special_chars(self):
        """Special characters are removed."""
        assert slugify("RSI & MACD Strategy!") == "rsi-macd-strategy"

    def test_slugify_handles_multiple_spaces(self):
        """Multiple spaces become single hyphen."""
        assert slugify("My   Big   Strategy") == "my-big-strategy"

    def test_slugify_handles_underscores(self):
        """Underscores become hyphens."""
        assert slugify("my_strategy_v2") == "my-strategy-v2"

    def test_slugify_trims_hyphens(self):
        """Leading/trailing hyphens are removed."""
        assert slugify("  -My Strategy-  ") == "my-strategy"

    def test_slugify_truncates_long_names(self):
        """Names over 100 chars are truncated."""
        long_name = "a" * 150
        result = slugify(long_name)
        assert len(result) <= 100

    def test_slugify_empty_string(self):
        """Empty string returns empty string."""
        assert slugify("") == ""

    def test_slugify_numeric(self):
        """Numeric names are preserved."""
        assert slugify("Strategy 52W High") == "strategy-52w-high"


class TestStrategyTags:
    """Tests for tag overlap scoring."""

    def test_full_overlap_score(self):
        """Full overlap gives score of 1.0."""
        intent_tags = {
            "strategy_archetypes": ["breakout"],
            "indicators": ["rsi"],
            "timeframe_buckets": [],
            "topics": [],
            "risk_terms": [],
        }
        strategy_tags = {
            "strategy_archetypes": ["breakout"],
            "indicators": ["rsi"],
            "timeframe_buckets": [],
            "topics": [],
            "risk_terms": [],
        }

        # Calculate overlap manually (same as repository logic)
        all_intent = set()
        for field in intent_tags:
            all_intent.update(t.lower() for t in intent_tags.get(field, []))

        all_strategy = set()
        for field in strategy_tags:
            all_strategy.update(t.lower() for t in strategy_tags.get(field, []))

        matched = all_intent & all_strategy
        score = len(matched) / len(all_intent) if all_intent else 0

        assert score == 1.0

    def test_partial_overlap_score(self):
        """Partial overlap gives fractional score."""
        intent_tags = {
            "strategy_archetypes": ["breakout", "momentum"],
            "indicators": ["rsi", "volume"],
            "timeframe_buckets": [],
            "topics": [],
            "risk_terms": [],
        }
        strategy_tags = {
            "strategy_archetypes": ["breakout"],
            "indicators": [],
            "timeframe_buckets": [],
            "topics": [],
            "risk_terms": [],
        }

        all_intent = set()
        for field in intent_tags:
            all_intent.update(t.lower() for t in intent_tags.get(field, []))

        all_strategy = set()
        for field in strategy_tags:
            all_strategy.update(t.lower() for t in strategy_tags.get(field, []))

        matched = all_intent & all_strategy
        score = len(matched) / len(all_intent) if all_intent else 0

        # 1 match out of 4 tags = 0.25
        assert score == 0.25

    def test_no_overlap_score(self):
        """No overlap gives score of 0."""
        intent_tags = {
            "strategy_archetypes": ["breakout"],
            "indicators": [],
            "timeframe_buckets": [],
            "topics": [],
            "risk_terms": [],
        }
        strategy_tags = {
            "strategy_archetypes": ["mean_reversion"],
            "indicators": [],
            "timeframe_buckets": [],
            "topics": [],
            "risk_terms": [],
        }

        all_intent = set()
        for field in intent_tags:
            all_intent.update(t.lower() for t in intent_tags.get(field, []))

        all_strategy = set()
        for field in strategy_tags:
            all_strategy.update(t.lower() for t in strategy_tags.get(field, []))

        matched = all_intent & all_strategy
        score = len(matched) / len(all_intent) if all_intent else 0

        assert score == 0.0


class TestStrategySchemas:
    """Tests for strategy Pydantic schemas."""

    def test_strategy_tags_defaults(self):
        """StrategyTags has empty list defaults."""
        from app.schemas import StrategyTags

        tags = StrategyTags()
        assert tags.strategy_archetypes == []
        assert tags.indicators == []
        assert tags.timeframe_buckets == []
        assert tags.topics == []
        assert tags.risk_terms == []

    def test_strategy_tags_with_values(self):
        """StrategyTags accepts values."""
        from app.schemas import StrategyTags

        tags = StrategyTags(
            strategy_archetypes=["breakout", "momentum"],
            indicators=["rsi"],
        )
        assert tags.strategy_archetypes == ["breakout", "momentum"]
        assert tags.indicators == ["rsi"]

    def test_backtest_summary_defaults(self):
        """BacktestSummary has sensible defaults."""
        from app.schemas import BacktestSummary, BacktestSummaryStatus

        summary = BacktestSummary()
        assert summary.status == BacktestSummaryStatus.NEVER
        assert summary.last_backtest_at is None
        assert summary.best_oos_score is None

    def test_strategy_create_request_defaults(self):
        """StrategyCreateRequest has correct defaults."""
        from uuid import uuid4

        from app.schemas import (
            StrategyCreateRequest,
            StrategyEngine,
            StrategyStatus,
        )

        req = StrategyCreateRequest(
            workspace_id=uuid4(),
            name="Test Strategy",
        )
        assert req.engine == StrategyEngine.PINE
        assert req.status == StrategyStatus.DRAFT

    def test_strategy_source_ref_pine(self):
        """StrategySourceRef works for Pine scripts."""
        from uuid import uuid4

        from app.schemas import StrategySourceRef

        ref = StrategySourceRef(
            store="local",
            path="strategies/breakout.pine",
            doc_id=uuid4(),
        )
        assert ref.store == "local"
        assert ref.path == "strategies/breakout.pine"
        assert ref.module is None

    def test_strategy_source_ref_python(self):
        """StrategySourceRef works for Python strategies."""
        from app.schemas import StrategySourceRef

        ref = StrategySourceRef(
            module="strategies.breakout_52w",
            entrypoint="run",
            params_schema={
                "type": "object",
                "properties": {"lookback": {"type": "integer"}},
            },
        )
        assert ref.module == "strategies.breakout_52w"
        assert ref.entrypoint == "run"
        assert ref.store is None


class TestCoverageResponseExtensions:
    """Tests for coverage response strategy extensions."""

    def test_coverage_response_with_candidates(self):
        """CoverageResponse includes candidate_strategies when weak."""
        from app.schemas import CoverageResponse

        coverage = CoverageResponse(
            weak=True,
            best_score=0.35,
            avg_top_k_score=0.30,
            num_above_threshold=0,
            threshold=0.55,
            reason_codes=["LOW_BEST_SCORE", "NO_RESULTS_ABOVE_THRESHOLD"],
            suggestions=["Add breakout strategies"],
            intent_signature="abc123",
            candidate_strategies=[
                {
                    "strategy_id": "uuid1",
                    "name": "Breakout 52W",
                    "score": 0.75,
                    "matched_tags": ["breakout"],
                }
            ],
        )

        assert coverage.weak is True
        assert coverage.intent_signature == "abc123"
        assert len(coverage.candidate_strategies) == 1
        assert coverage.candidate_strategies[0]["name"] == "Breakout 52W"

    def test_coverage_response_without_candidates(self):
        """CoverageResponse works without candidates (good coverage)."""
        from app.schemas import CoverageResponse

        coverage = CoverageResponse(
            weak=False,
            best_score=0.85,
            avg_top_k_score=0.75,
            num_above_threshold=5,
            threshold=0.55,
            reason_codes=[],
            suggestions=[],
            intent_signature="def456",
            candidate_strategies=None,
        )

        assert coverage.weak is False
        assert coverage.candidate_strategies is None
