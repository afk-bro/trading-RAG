"""
Strategy runner for evaluating trading strategies.

The StrategyRunner is the main evaluation engine that:
1. Routes to strategy implementations by entry type
2. Handles max_positions gating (only blocks entries, never exits)
3. Returns consistent StrategyEvaluation for all cases
"""

from typing import Callable
from uuid import UUID, uuid4

from app.schemas import IntentAction, PaperState
from app.services.strategy.models import (
    ExecutionSpec,
    MarketSnapshot,
    StrategyEvaluation,
)
from app.services.strategy.strategies import (
    evaluate_breakout_52w_high,
    evaluate_unicorn_model,
)

# Type alias for strategy function signature
StrategyFn = Callable[
    [ExecutionSpec, MarketSnapshot, PaperState, UUID, bool],
    StrategyEvaluation,
]


class StrategyRunner:
    """
    Main evaluation engine for strategy execution.

    Responsibilities:
    - Register strategy implementations by entry type
    - Route evaluate() calls to the correct strategy
    - Handle max_positions gating (blocks entries, never exits)
    - Return consistent StrategyEvaluation for all code paths
    """

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyFn] = {}
        self._register_default_strategies()

    def register_strategy(self, entry_type: str, fn: StrategyFn) -> None:
        """
        Register a strategy implementation.

        Args:
            entry_type: The strategy type key (e.g., 'breakout_52w_high')
            fn: Strategy function with signature:
                (spec, snapshot, paper_state, evaluation_id, at_max_positions) -> StrategyEvaluation
        """
        self._strategies[entry_type] = fn

    def _register_default_strategies(self) -> None:
        """Register built-in strategies."""
        self.register_strategy("breakout_52w_high", evaluate_breakout_52w_high)
        self.register_strategy("unicorn_model", evaluate_unicorn_model)

    def evaluate(
        self,
        spec: ExecutionSpec,
        snapshot: MarketSnapshot,
        paper_state: PaperState,
    ) -> StrategyEvaluation:
        """
        Evaluate strategy and emit TradeIntents.

        This is the main entry point for strategy evaluation. It:
        1. Validates the strategy type is registered
        2. Checks if symbol is in spec's universe
        3. Determines max_positions context
        4. Delegates to the strategy implementation
        5. Post-filters to block entries (never exits) if at max positions

        Args:
            spec: The ExecutionSpec defining strategy configuration
            snapshot: Current market state for the symbol
            paper_state: Current paper trading state (positions, cash, etc.)

        Returns:
            StrategyEvaluation with intents, signals, and metadata

        Raises:
            ValueError: If spec.entry.type is not registered
        """
        # Generate evaluation_id (all intents share this as correlation_id)
        evaluation_id = uuid4()

        # Helper for consistent early returns
        def make_result(
            intents: list = None,
            signals: list = None,
            metadata: dict = None,
        ) -> StrategyEvaluation:
            return StrategyEvaluation(
                spec_id=str(spec.instance_id),
                symbol=snapshot.symbol,
                ts=snapshot.ts,
                intents=intents or [],
                signals=signals or [],
                metadata=metadata or {},
                evaluation_id=evaluation_id,
            )

        # 1. Validate spec.entry.type is registered
        strategy_fn = self._strategies.get(spec.entry.type)
        if not strategy_fn:
            raise ValueError(f"Unknown strategy type: {spec.entry.type}")

        # 2. Check if symbol in spec.symbols
        if snapshot.symbol not in spec.symbols:
            return make_result(signals=["symbol_not_in_spec"])

        # 3. Get max_positions context (passed to strategy, NOT a hard gate here)
        open_positions = [p for p in paper_state.positions.values() if p.quantity > 0]
        at_max_positions = len(open_positions) >= spec.risk.max_positions

        # 4. Delegate to strategy implementation
        result = strategy_fn(
            spec, snapshot, paper_state, evaluation_id, at_max_positions
        )

        # 5. Post-filter: ONLY block ENTRY intents if at max positions (never block exits)
        if at_max_positions:
            # Keep only exit actions (CLOSE_LONG, CLOSE_SHORT)
            exit_actions = {IntentAction.CLOSE_LONG, IntentAction.CLOSE_SHORT}
            filtered = [i for i in result.intents if i.action in exit_actions]
            if len(filtered) < len(result.intents):
                result.signals.append("entry_blocked_max_positions")
                result.intents = filtered

        return result
