"""Broker adapter factory."""

from typing import Optional

import structlog

from app.services.execution.paper_broker import PaperBroker
from app.repositories.trade_events import TradeEventsRepository
from app.repositories.paper_equity import PaperEquityRepository
from app.repositories.strategy_versions import StrategyVersionsRepository


logger = structlog.get_logger(__name__)

# Module-level singleton
_paper_broker: Optional[PaperBroker] = None


def get_paper_broker(
    events_repo: TradeEventsRepository,
    equity_repo: Optional[PaperEquityRepository] = None,
    version_repo: Optional[StrategyVersionsRepository] = None,
) -> PaperBroker:
    """
    Get or create paper broker instance.

    Uses singleton pattern to maintain state across requests.

    Args:
        events_repo: Trade events repository for journaling
        equity_repo: Optional equity snapshots repository for drawdown tracking
        version_repo: Optional strategy versions repository for state gating

    Returns:
        PaperBroker instance
    """
    global _paper_broker
    if _paper_broker is None:
        _paper_broker = PaperBroker(events_repo, equity_repo, version_repo)
        logger.info(
            "paper_broker_initialized",
            equity_tracking=equity_repo is not None,
            strategy_gating=version_repo is not None,
        )
    return _paper_broker


def reset_paper_broker() -> None:
    """
    Reset paper broker singleton.

    For testing purposes only.
    """
    global _paper_broker
    _paper_broker = None
    logger.warning("paper_broker_reset")
