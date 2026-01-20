"""Runner for strategy intelligence snapshot computation.

Orchestrates confidence computation and persistence to strategy_intel_snapshots.
Handles data fetching, deduplication, and batch processing.

v1.5 Step 2B - Runner + Persistence
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import pandas as pd
import structlog

from app.repositories.strategy_intel import StrategyIntelRepository, IntelSnapshot
from app.services.intel.confidence import (
    compute_confidence,
    ConfidenceContext,
    ConfidenceResult,
)

logger = structlog.get_logger(__name__)


class IntelRunner:
    """
    Runner for computing and persisting strategy intelligence snapshots.

    Coordinates:
    - Fetching version metadata
    - Fetching backtest/WFO metrics
    - Fetching OHLCV data
    - Computing confidence
    - Persisting to repository with deduplication
    """

    def __init__(
        self,
        pool,
        ohlcv_provider: Optional[Any] = None,
        engine_version: str = "intel_runner_v0.1",
    ):
        """
        Initialize runner.

        Args:
            pool: Database connection pool
            ohlcv_provider: Optional market data provider with get_ohlcv() method
            engine_version: Version string for provenance tracking
        """
        self._pool = pool
        self._intel_repo = StrategyIntelRepository(pool)
        self._ohlcv_provider = ohlcv_provider
        self._engine_version = engine_version

    async def run_for_version(
        self,
        version_id: UUID,
        as_of_ts: datetime,
        workspace_id: Optional[UUID] = None,
        force: bool = False,
    ) -> Optional[IntelSnapshot]:
        """
        Compute and persist intelligence snapshot for a specific version.

        Args:
            version_id: Strategy version UUID
            as_of_ts: Market time to compute intel for
            workspace_id: Optional workspace ID (fetched if not provided)
            force: If True, skip deduplication check

        Returns:
            Created IntelSnapshot, or None if deduplicated (same inputs_hash exists)
        """
        log = logger.bind(version_id=str(version_id), as_of_ts=as_of_ts.isoformat())

        # Step 1: Fetch version metadata
        version_data = await self._fetch_version_data(version_id)
        if not version_data:
            log.warning("intel_runner_version_not_found")
            return None

        workspace_id = workspace_id or version_data.get("workspace_id")
        if not workspace_id:
            log.error("intel_runner_no_workspace_id")
            return None

        strategy_entity_id = version_data.get("strategy_entity_id")

        # Step 2: Fetch backtest metrics
        backtest_metrics = await self._fetch_backtest_metrics(
            workspace_id, strategy_entity_id, version_id
        )

        # Step 3: Fetch OHLCV data (if provider available)
        ohlcv_df = await self._fetch_ohlcv(version_data, as_of_ts)
        latest_candle_ts = self._get_latest_candle_ts(ohlcv_df)

        # Step 4: Build context and compute confidence
        ctx = ConfidenceContext(
            version_id=version_id,
            as_of_ts=as_of_ts,
            ohlcv=ohlcv_df,
            backtest_metrics=backtest_metrics,
            wfo_metrics=None,  # TODO: Add WFO support
            latest_candle_ts=latest_candle_ts,
            strategy_regime_profile=version_data.get("regime_awareness"),
        )

        result = compute_confidence(ctx)

        # Step 5: Check for deduplication
        if not force:
            existing = await self._intel_repo.get_latest_snapshot(version_id)
            if existing and existing.inputs_hash == result.inputs_hash:
                log.debug(
                    "intel_runner_dedupe_skip",
                    inputs_hash=result.inputs_hash[:16],
                )
                return None

        # Step 6: Persist snapshot
        snapshot = await self._intel_repo.insert_snapshot(
            workspace_id=workspace_id,
            strategy_version_id=version_id,
            as_of_ts=as_of_ts,
            regime=result.regime,
            confidence_score=result.confidence_score,
            confidence_components=result.confidence_components,
            features=result.features,
            explain=result.explain,
            engine_version=self._engine_version,
            inputs_hash=result.inputs_hash,
        )

        log.info(
            "intel_snapshot_created",
            snapshot_id=str(snapshot.id),
            regime=result.regime,
            confidence=result.confidence_score,
        )

        return snapshot

    async def run_for_workspace_active(
        self,
        workspace_id: UUID,
        as_of_ts: datetime,
        force: bool = False,
    ) -> list[IntelSnapshot]:
        """
        Compute snapshots for all active versions in a workspace.

        Args:
            workspace_id: Workspace UUID
            as_of_ts: Market time to compute intel for
            force: If True, skip deduplication check

        Returns:
            List of created IntelSnapshots (excludes deduplicated)
        """
        log = logger.bind(workspace_id=str(workspace_id), as_of_ts=as_of_ts.isoformat())

        # Fetch all active versions for workspace
        active_versions = await self._fetch_active_versions(workspace_id)

        if not active_versions:
            log.info("intel_runner_no_active_versions")
            return []

        log.info(
            "intel_runner_batch_start",
            version_count=len(active_versions),
        )

        snapshots = []
        for version_data in active_versions:
            version_id = version_data["id"]
            try:
                snapshot = await self.run_for_version(
                    version_id=version_id,
                    as_of_ts=as_of_ts,
                    workspace_id=workspace_id,
                    force=force,
                )
                if snapshot:
                    snapshots.append(snapshot)
            except Exception as e:
                log.error(
                    "intel_runner_version_failed",
                    version_id=str(version_id),
                    error=str(e),
                )
                continue

        log.info(
            "intel_runner_batch_complete",
            total=len(active_versions),
            created=len(snapshots),
            dedupe_skipped=len(active_versions) - len(snapshots),
        )

        return snapshots

    # =========================================================================
    # Data Fetching Helpers
    # =========================================================================

    async def _fetch_version_data(self, version_id: UUID) -> Optional[dict]:
        """Fetch version with parent strategy data."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    v.id,
                    v.strategy_id,
                    v.strategy_entity_id,
                    v.config_snapshot,
                    v.regime_awareness,
                    v.state,
                    s.workspace_id
                FROM strategy_versions v
                JOIN strategies s ON v.strategy_id = s.id
                WHERE v.id = $1
                """,
                version_id,
            )

            if not row:
                return None

            data = dict(row)

            # Parse JSONB fields
            if isinstance(data.get("config_snapshot"), str):
                import json
                data["config_snapshot"] = json.loads(data["config_snapshot"])
            if isinstance(data.get("regime_awareness"), str):
                import json
                data["regime_awareness"] = json.loads(data["regime_awareness"])

            return data

    async def _fetch_active_versions(self, workspace_id: UUID) -> list[dict]:
        """Fetch all active versions for a workspace."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    v.id,
                    v.strategy_id,
                    v.strategy_entity_id,
                    v.config_snapshot,
                    v.regime_awareness,
                    s.workspace_id
                FROM strategy_versions v
                JOIN strategies s ON v.strategy_id = s.id
                WHERE s.workspace_id = $1
                  AND v.state = 'active'
                ORDER BY v.created_at DESC
                """,
                workspace_id,
            )

            result = []
            for row in rows:
                data = dict(row)
                if isinstance(data.get("config_snapshot"), str):
                    import json
                    data["config_snapshot"] = json.loads(data["config_snapshot"])
                if isinstance(data.get("regime_awareness"), str):
                    import json
                    data["regime_awareness"] = json.loads(data["regime_awareness"])
                result.append(data)

            return result

    async def _fetch_backtest_metrics(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        version_id: UUID,
    ) -> Optional[dict]:
        """
        Fetch latest backtest metrics for a strategy.

        Tries in order:
        1. Backtest linked to this specific version
        2. Latest completed backtest for strategy_entity_id
        """
        if not strategy_entity_id:
            return None

        async with self._pool.acquire() as conn:
            # First try version-linked backtest
            row = await conn.fetchrow(
                """
                SELECT summary
                FROM backtest_runs
                WHERE strategy_version_id = $1
                  AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                version_id,
            )

            if not row:
                # Fall back to entity-linked backtest
                row = await conn.fetchrow(
                    """
                    SELECT summary
                    FROM backtest_runs
                    WHERE workspace_id = $1
                      AND strategy_entity_id = $2
                      AND status = 'completed'
                    ORDER BY completed_at DESC
                    LIMIT 1
                    """,
                    workspace_id,
                    strategy_entity_id,
                )

            if not row or not row["summary"]:
                return None

            summary = row["summary"]
            if isinstance(summary, str):
                import json
                summary = json.loads(summary)

            # Map summary fields to expected metrics format
            return {
                "sharpe": summary.get("sharpe"),
                "return_pct": summary.get("return_pct"),
                "max_drawdown_pct": summary.get("max_drawdown_pct"),
                "trades": summary.get("trades"),
                "win_rate": summary.get("win_rate"),
            }

    async def _fetch_ohlcv(
        self,
        version_data: dict,
        as_of_ts: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for regime computation.

        Uses ohlcv_provider if available, otherwise returns None.
        """
        if not self._ohlcv_provider:
            return None

        try:
            # Extract symbol/timeframe from config or use defaults
            config = version_data.get("config_snapshot", {})
            symbol = config.get("symbol", "BTC/USDT")
            timeframe = config.get("timeframe", "1h")

            # Fetch last 100 bars
            ohlcv = await self._ohlcv_provider.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=100,
                end_time=as_of_ts,
            )

            if ohlcv is None or len(ohlcv) == 0:
                return None

            return ohlcv

        except Exception as e:
            logger.warning(
                "intel_runner_ohlcv_fetch_failed",
                error=str(e),
                version_id=str(version_data.get("id")),
            )
            return None

    def _get_latest_candle_ts(self, ohlcv: Optional[pd.DataFrame]) -> Optional[datetime]:
        """Get timestamp of latest candle from OHLCV data."""
        if ohlcv is None or len(ohlcv) == 0:
            return None

        try:
            # Try index first
            if hasattr(ohlcv.index, 'to_pydatetime'):
                ts = ohlcv.index[-1]
                if hasattr(ts, 'to_pydatetime'):
                    return ts.to_pydatetime()
                return ts

            # Try timestamp column
            for col in ['timestamp', 'ts', 'date', 'datetime']:
                if col in ohlcv.columns:
                    ts = ohlcv[col].iloc[-1]
                    if hasattr(ts, 'to_pydatetime'):
                        return ts.to_pydatetime()
                    return ts

            return None
        except Exception:
            return None


# =============================================================================
# Convenience Functions
# =============================================================================


async def compute_and_store_snapshot(
    pool,
    version_id: UUID,
    as_of_ts: datetime,
    workspace_id: Optional[UUID] = None,
    ohlcv_provider: Optional[Any] = None,
    force: bool = False,
) -> Optional[IntelSnapshot]:
    """
    Convenience function to compute and store a single snapshot.

    Args:
        pool: Database connection pool
        version_id: Strategy version UUID
        as_of_ts: Market time for computation
        workspace_id: Optional workspace ID
        ohlcv_provider: Optional market data provider
        force: Skip deduplication if True

    Returns:
        Created IntelSnapshot or None if deduplicated
    """
    runner = IntelRunner(pool, ohlcv_provider=ohlcv_provider)
    return await runner.run_for_version(
        version_id=version_id,
        as_of_ts=as_of_ts,
        workspace_id=workspace_id,
        force=force,
    )
