#!/usr/bin/env python
"""
CLI for KB ingestion and backfill operations.

Usage:
    python -m app.services.kb.cli ingest --workspace <id> [options]
    python -m app.services.kb.cli status --workspace <id>
    python -m app.services.kb.cli collection-info
    python -m app.services.kb.cli backfill-candidacy --workspace <id> [options]
    python -m app.services.kb.cli backfill-regime --workspace <id> --ohlcv-file <path> [options]

Examples:
    # Ingest all missing vectors
    python -m app.services.kb.cli ingest --workspace abc123

    # Dry run with limit
    python -m app.services.kb.cli ingest --workspace abc123 --dry-run --limit 100

    # Re-embed everything since a date
    python -m app.services.kb.cli ingest --workspace abc123 --reembed --since 2026-01-01

    # Check collection status
    python -m app.services.kb.cli collection-info

    # Backfill candidacy status for test_variant runs
    python -m app.services.kb.cli backfill-candidacy --workspace abc123 --since 2025-01-01 --dry-run

    # Backfill regime snapshots from OHLCV file
    python -m app.services.kb.cli backfill-regime --workspace abc123 --ohlcv-file data.csv --dry-run
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

# Configure logging for CLI
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


async def cmd_ingest(args: argparse.Namespace) -> int:
    """Run ingestion command."""
    from app.services.kb.ingestion import KBIngestionPipeline
    from app.services.kb.embed import get_kb_embedder
    from app.repositories.kb_trials import KBTrialRepository

    # Parse workspace ID
    try:
        workspace_id = UUID(args.workspace)
    except ValueError:
        logger.error("Invalid workspace ID", workspace=args.workspace)
        return 1

    # Parse since date if provided
    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error("Invalid date format", since=args.since)
            return 1

    # Initialize components
    embedder = get_kb_embedder()

    # Try to get Qdrant client
    repository = None
    if not args.dry_run:
        try:
            from qdrant_client import AsyncQdrantClient
            from app.config import get_settings

            settings = get_settings()
            client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            # Use versioned collection name
            _dim = await embedder.get_vector_dim()  # noqa: F841
            collection_name = embedder.get_collection_name()
            repository = KBTrialRepository(client, collection=collection_name)
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error=str(e))
            if not args.dry_run:
                return 1

    # Try to get tune repository
    tune_repo = None
    try:
        from app.repositories.backtest_tunes import BacktestTuneRepository
        from app.db import get_db_pool

        pool = await get_db_pool()
        tune_repo = BacktestTuneRepository(pool)
    except Exception as e:
        logger.warning("Failed to connect to database", error=str(e))

    # Create pipeline
    pipeline = KBIngestionPipeline(
        embedder=embedder,
        repository=repository,
        tune_repo=tune_repo,
    )

    # Run ingestion
    logger.info(
        "Starting ingestion",
        workspace=str(workspace_id),
        since=str(since) if since else None,
        limit=args.limit,
        dry_run=args.dry_run,
        reembed=args.reembed,
        only_missing=args.only_missing,
    )

    report = await pipeline.ingest_tune_runs(
        workspace_id=workspace_id,
        since=since,
        limit=args.limit,
        dry_run=args.dry_run,
        only_missing_vectors=args.only_missing,
        reembed=args.reembed,
    )

    # Print report
    print("\n" + "=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    print(f"Workspace:      {report.workspace_id}")
    print(f"Collection:     {report.collection_name}")
    print(f"Model:          {report.model_id}")
    print(f"Vector Dim:     {report.vector_dim}")
    print(f"Dry Run:        {report.dry_run}")
    print("-" * 60)
    print(f"Total Fetched:  {report.stats.total_fetched}")
    print(f"Total Skipped:  {report.stats.total_skipped}")
    print(f"Total Embedded: {report.stats.total_embedded}")
    print(f"Total Upserted: {report.stats.total_upserted}")
    print(f"Total Failed:   {report.stats.total_failed}")
    print(f"Duration:       {report.stats.duration_seconds:.2f}s")
    print("=" * 60)

    # Show dry-run preview if available
    if report.dry_run and report.preview:
        preview = report.preview
        print("\nDRY-RUN PREVIEW")
        print("-" * 60)
        print(f"Total Candidates:  {preview.total_candidates}")
        print(f"Would Upsert:      {preview.would_upsert}")
        print(f"Would Skip:        {preview.would_skip}")
        print(f"Would Reembed:     {preview.would_reembed}")

        if preview.samples:
            print("\nSample Actions:")
            for sample in preview.samples:
                print(f"  [{sample.action.upper():7}] {sample.tune_run_id[:8]}...")
                print(
                    f"           strategy={sample.strategy_name}, objective={sample.objective_type}"
                )
                print(f"           reason: {sample.reason}")
        print("-" * 60)

    if report.stats.failed_tune_run_ids:
        print(f"\nFailed IDs ({len(report.stats.failed_tune_run_ids)}):")
        for run_id in report.stats.failed_tune_run_ids[:10]:
            print(f"  - {run_id}")
        if len(report.stats.failed_tune_run_ids) > 10:
            print(f"  ... and {len(report.stats.failed_tune_run_ids) - 10} more")

    return 0 if report.stats.total_failed == 0 else 1


async def cmd_status(args: argparse.Namespace) -> int:
    """Show ingestion status for workspace."""
    from app.services.kb.embed import get_kb_embedder

    try:
        workspace_id = UUID(args.workspace)
    except ValueError:
        logger.error("Invalid workspace ID", workspace=args.workspace)
        return 1

    embedder = get_kb_embedder()

    # Get Qdrant info
    try:
        from qdrant_client import AsyncQdrantClient
        from app.config import get_settings
        from app.repositories.kb_trials import KBTrialRepository

        settings = get_settings()
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        dim = await embedder.get_vector_dim()
        collection_name = embedder.get_collection_name()
        repository = KBTrialRepository(client, collection=collection_name)

        # Get collection info
        info = await repository.get_collection_info()

        # Count by workspace
        count = await repository.count(workspace_id=workspace_id)

        print("\n" + "=" * 60)
        print("KB STATUS")
        print("=" * 60)
        print(f"Workspace:      {workspace_id}")
        print(f"Collection:     {collection_name}")
        print(f"Model:          {embedder.model_id}")
        print(f"Vector Dim:     {dim}")
        print("-" * 60)
        print(f"Collection Status: {info.get('status', 'unknown')}")
        print(f"Total Points:      {info.get('points_count', 0)}")
        print(f"Workspace Points:  {count}")
        print("=" * 60)

    except Exception as e:
        logger.error("Failed to get status", error=str(e))
        return 1

    return 0


async def cmd_collection_info(args: argparse.Namespace) -> int:
    """Show collection information."""
    from app.services.kb.embed import get_kb_embedder

    embedder = get_kb_embedder()

    try:
        from qdrant_client import AsyncQdrantClient
        from app.config import get_settings
        from app.repositories.kb_trials import KBTrialRepository

        settings = get_settings()
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        dim = await embedder.get_vector_dim()
        collection_name = embedder.get_collection_name()
        repository = KBTrialRepository(client, collection=collection_name)

        # Check if exists
        exists = await repository.collection_exists()

        print("\n" + "=" * 60)
        print("COLLECTION INFO")
        print("=" * 60)
        print(f"Collection Name:  {collection_name}")
        print(f"Model ID:         {embedder.model_id}")
        print(f"Vector Dim:       {dim}")
        print(f"Exists:           {exists}")

        if exists:
            info = await repository.get_collection_info()
            print("-" * 60)
            print(f"Status:           {info.get('status', 'unknown')}")
            print(f"Points Count:     {info.get('points_count', 0)}")
            print(f"Vectors Count:    {info.get('vectors_count', 0)}")

        print("=" * 60)

        # List all KB collections
        collections = await client.get_collections()
        kb_collections = [
            c.name
            for c in collections.collections
            if c.name.startswith("trading_kb_trials")
        ]

        if kb_collections:
            print("\nAll KB Collections:")
            for coll in sorted(kb_collections):
                marker = " (active)" if coll == collection_name else ""
                print(f"  - {coll}{marker}")

    except Exception as e:
        logger.error("Failed to get collection info", error=str(e))
        return 1

    return 0


async def cmd_backfill_candidacy(args: argparse.Namespace) -> int:
    """Backfill candidacy status for test_variant runs.

    Evaluates historical runs against the candidacy gate policy and marks
    eligible ones as 'candidate' for KB ingestion.
    """
    from app.db import get_db_pool
    from app.services.kb.candidacy import (
        is_candidate,
        VariantMetricsForCandidacy,
        CandidacyConfig,
    )
    from app.services.kb.types import RegimeSnapshot

    # Parse workspace ID
    try:
        workspace_id = UUID(args.workspace)
    except ValueError:
        logger.error("Invalid workspace ID", workspace=args.workspace)
        return 1

    # Parse since date if provided
    since: Optional[datetime] = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error("Invalid date format", since=args.since)
            return 1

    limit = args.limit or 1000
    dry_run = args.dry_run
    experiment_type = args.experiment_type or "sweep"
    require_regime = not args.no_regime_check

    logger.info(
        "Starting candidacy backfill",
        workspace=str(workspace_id),
        since=str(since) if since else None,
        limit=limit,
        dry_run=dry_run,
        experiment_type=experiment_type,
        require_regime=require_regime,
    )

    # Connect to database
    try:
        pool = await get_db_pool()
    except Exception as e:
        logger.error("Failed to connect to database", error=str(e))
        return 1

    # Query eligible runs
    query = """
        SELECT id, workspace_id, summary, regime_oos, trade_count, run_kind, status
        FROM backtest_runs
        WHERE run_kind = 'test_variant'
          AND status IN ('completed', 'success')
          AND kb_status = 'excluded'
          AND workspace_id = $1
          AND ($2::timestamp IS NULL OR created_at >= $2)
        ORDER BY created_at
        LIMIT $3
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, workspace_id, since, limit)

    logger.info("Fetched runs", count=len(rows))

    # Candidacy config
    config = CandidacyConfig(require_regime=require_regime)

    # Track results
    eligible_ids: list[UUID] = []
    rejection_reasons: Counter = Counter()
    total_scanned = len(rows)

    for row in rows:
        run_id = row["id"]
        summary = row["summary"]

        # Parse summary if it's a string
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {}

        if summary is None:
            summary = {}

        # Extract metrics from summary
        n_trades = row["trade_count"] or summary.get("trades", 0) or 0
        max_dd_pct = abs(summary.get("max_drawdown_pct", 0) or 0)
        sharpe = summary.get("sharpe")

        # Parse regime snapshot if present
        regime_oos = None
        if row["regime_oos"]:
            regime_data = row["regime_oos"]
            if isinstance(regime_data, str):
                try:
                    regime_data = json.loads(regime_data)
                except json.JSONDecodeError:
                    regime_data = None
            if regime_data:
                regime_oos = RegimeSnapshot.from_dict(regime_data)

        # Build metrics for candidacy check
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=n_trades,
            max_dd_frac_oos=max_dd_pct / 100.0,  # Convert percentage to fraction
            sharpe_oos=sharpe,
            overfit_gap=None,  # Not available for test_variants
        )

        # Evaluate candidacy
        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_oos,
            experiment_type=experiment_type,
            config=config,
        )

        if decision.eligible:
            eligible_ids.append(run_id)
        else:
            rejection_reasons[decision.reason] += 1

    # Update eligible runs (unless dry-run)
    updated_count = 0
    if eligible_ids and not dry_run:
        update_query = """
            UPDATE backtest_runs SET
                kb_status = 'candidate',
                kb_status_changed_at = NOW(),
                kb_status_changed_by = 'backfill',
                auto_candidate_gate = 'passed_all_gates'
            WHERE id = ANY($1)
        """
        async with pool.acquire() as conn:
            result = await conn.execute(update_query, eligible_ids)
            updated_count = int(result.split()[-1])

        logger.info("Updated runs", count=updated_count)

    # Print report
    print("\n" + "=" * 60)
    print("BACKFILL CANDIDACY REPORT")
    print("=" * 60)
    print(f"Workspace:       {workspace_id}")
    print(f"Since:           {since or 'all time'}")
    print(f"Limit:           {limit}")
    print(f"Experiment Type: {experiment_type}")
    print(f"Require Regime:  {require_regime}")
    print(f"Dry Run:         {dry_run}")
    print("-" * 60)
    print(f"Total Scanned:   {total_scanned}")
    print(f"Eligible:        {len(eligible_ids)}")
    print(f"Ineligible:      {sum(rejection_reasons.values())}")
    print(f"Updated:         {updated_count}" + (" (dry run)" if dry_run else ""))
    print("-" * 60)

    if rejection_reasons:
        print("Rejection Reasons:")
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason}: {count}")

    print("=" * 60)

    return 0


async def cmd_backfill_regime(args: argparse.Namespace) -> int:
    """Backfill regime snapshots from OHLCV file.

    Computes regime_oos for runs that are missing it, using provided OHLCV data.
    """
    import pandas as pd
    from pathlib import Path

    from app.db import get_db_pool
    from app.services.kb.regime import compute_regime_snapshot
    from app.services.kb.constants import REGIME_SCHEMA_VERSION

    # Parse workspace ID
    try:
        workspace_id = UUID(args.workspace)
    except ValueError:
        logger.error("Invalid workspace ID", workspace=args.workspace)
        return 1

    # Check OHLCV file
    ohlcv_path = Path(args.ohlcv_file)
    if not ohlcv_path.exists():
        logger.error("OHLCV file not found", path=str(ohlcv_path))
        return 1

    # Parse since date if provided
    since: Optional[datetime] = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error("Invalid date format", since=args.since)
            return 1

    limit = args.limit or 500
    dry_run = args.dry_run
    timeframe = args.timeframe or "1d"
    symbol_filter = args.symbol

    logger.info(
        "Starting regime backfill",
        workspace=str(workspace_id),
        ohlcv_file=str(ohlcv_path),
        symbol=symbol_filter,
        timeframe=timeframe,
        since=str(since) if since else None,
        limit=limit,
        dry_run=dry_run,
    )

    # Load OHLCV data
    try:
        if ohlcv_path.suffix.lower() == ".json":
            df = pd.read_json(ohlcv_path)
        else:
            df = pd.read_csv(ohlcv_path)
        logger.info("Loaded OHLCV data", rows=len(df))
    except Exception as e:
        logger.error("Failed to load OHLCV file", error=str(e))
        return 1

    # Connect to database
    try:
        pool = await get_db_pool()
    except Exception as e:
        logger.error("Failed to connect to database", error=str(e))
        return 1

    # Query runs needing regime backfill
    query = """
        SELECT id, workspace_id, dataset_meta, params, started_at, completed_at
        FROM backtest_runs
        WHERE run_kind = 'test_variant'
          AND status IN ('completed', 'success')
          AND regime_oos IS NULL
          AND workspace_id = $1
          AND ($2::text IS NULL OR dataset_meta->>'symbol' = $2)
          AND ($3::timestamp IS NULL OR created_at >= $3)
        ORDER BY created_at
        LIMIT $4
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, workspace_id, symbol_filter, since, limit)

    logger.info("Fetched runs needing regime", count=len(rows))

    # Compute regime for each run
    computed_count = 0
    skipped_reasons: Counter = Counter()

    for row in rows:
        run_id = row["id"]

        # For now, compute regime from the full OHLCV window
        # Future: extract OOS date range and slice the data
        try:
            regime = compute_regime_snapshot(
                df,
                source="backfill",
                timeframe=timeframe,
            )

            if not dry_run:
                update_query = """
                    UPDATE backtest_runs SET
                        regime_oos = $2,
                        regime_schema_version = $3
                    WHERE id = $1
                """
                async with pool.acquire() as conn:
                    await conn.execute(
                        update_query,
                        run_id,
                        json.dumps(regime.to_dict()),
                        REGIME_SCHEMA_VERSION,
                    )

            computed_count += 1

        except Exception as e:
            logger.warning("Failed to compute regime", run_id=str(run_id), error=str(e))
            skipped_reasons["computation_error"] += 1

    # Print report
    print("\n" + "=" * 60)
    print("BACKFILL REGIME REPORT")
    print("=" * 60)
    print(f"Workspace:       {workspace_id}")
    print(f"OHLCV File:      {ohlcv_path}")
    print(f"Symbol Filter:   {symbol_filter or 'all'}")
    print(f"Timeframe:       {timeframe}")
    print(f"OHLCV Rows:      {len(df)}")
    print(f"Dry Run:         {dry_run}")
    print("-" * 60)
    print(f"Total Scanned:   {len(rows)}")
    print(f"Computed:        {computed_count}")
    print(f"Skipped:         {sum(skipped_reasons.values())}")
    print(
        f"Updated:         {computed_count if not dry_run else 0}"
        + (" (dry run)" if dry_run else "")
    )

    if skipped_reasons:
        print("-" * 60)
        print("Skip Reasons:")
        for reason, count in skipped_reasons.most_common():
            print(f"  {reason}: {count}")

    print("=" * 60)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KB ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest tune_runs into KB")
    ingest_parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace ID",
    )
    ingest_parser.add_argument(
        "--since",
        "-s",
        help="Only process runs created after this date (ISO format)",
    )
    ingest_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Maximum number of runs to process",
    )
    ingest_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Don't actually upsert to Qdrant",
    )
    ingest_parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed all runs (ignore kb_ingested_at)",
    )
    ingest_parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Only process runs without vectors (default)",
    )
    ingest_parser.add_argument(
        "--all",
        action="store_true",
        dest="process_all",
        help="Process all runs, not just missing",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show ingestion status")
    status_parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace ID",
    )

    # Collection info command
    _collection_parser = subparsers.add_parser(  # noqa: F841
        "collection-info",
        help="Show collection information",
    )

    # Backfill candidacy command
    candidacy_parser = subparsers.add_parser(
        "backfill-candidacy",
        help="Backfill candidacy status for test_variant runs",
    )
    candidacy_parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace ID",
    )
    candidacy_parser.add_argument(
        "--since",
        "-s",
        help="Only process runs created after this date (ISO format)",
    )
    candidacy_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=1000,
        help="Maximum number of runs to process (default: 1000)",
    )
    candidacy_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview without writing changes",
    )
    candidacy_parser.add_argument(
        "--experiment-type",
        "-e",
        default="sweep",
        help="Experiment type for gate check (default: sweep)",
    )
    candidacy_parser.add_argument(
        "--no-regime-check",
        action="store_true",
        help="Skip regime requirement in candidacy check",
    )

    # Backfill regime command
    regime_parser = subparsers.add_parser(
        "backfill-regime",
        help="Backfill regime snapshots from OHLCV file",
    )
    regime_parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        help="Workspace ID",
    )
    regime_parser.add_argument(
        "--ohlcv-file",
        "-f",
        required=True,
        help="Path to OHLCV data file (CSV or JSON)",
    )
    regime_parser.add_argument(
        "--symbol",
        help="Filter runs by symbol (from dataset_meta)",
    )
    regime_parser.add_argument(
        "--timeframe",
        "-t",
        default="1d",
        help="Timeframe for regime computation (default: 1d)",
    )
    regime_parser.add_argument(
        "--since",
        "-s",
        help="Only process runs created after this date (ISO format)",
    )
    regime_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=500,
        help="Maximum number of runs to process (default: 500)",
    )
    regime_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview without writing changes",
    )

    args = parser.parse_args()

    # Handle --all flag
    if hasattr(args, "process_all") and args.process_all:
        args.only_missing = False

    # Run appropriate command
    if args.command == "ingest":
        exit_code = asyncio.run(cmd_ingest(args))
    elif args.command == "status":
        exit_code = asyncio.run(cmd_status(args))
    elif args.command == "collection-info":
        exit_code = asyncio.run(cmd_collection_info(args))
    elif args.command == "backfill-candidacy":
        exit_code = asyncio.run(cmd_backfill_candidacy(args))
    elif args.command == "backfill-regime":
        exit_code = asyncio.run(cmd_backfill_regime(args))
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
