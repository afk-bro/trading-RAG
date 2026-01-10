#!/usr/bin/env python
"""
CLI for KB ingestion operations.

Usage:
    python -m app.services.kb.cli ingest --workspace <id> [options]
    python -m app.services.kb.cli status --workspace <id>
    python -m app.services.kb.cli collection-info

Examples:
    # Ingest all missing vectors
    python -m app.services.kb.cli ingest --workspace abc123

    # Dry run with limit
    python -m app.services.kb.cli ingest --workspace abc123 --dry-run --limit 100

    # Re-embed everything since a date
    python -m app.services.kb.cli ingest --workspace abc123 --reembed --since 2026-01-01

    # Check collection status
    python -m app.services.kb.cli collection-info
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
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
    from app.services.kb.ingestion import KBIngestionPipeline, get_ingestion_pipeline
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
            dim = await embedder.get_vector_dim()
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
    collection_parser = subparsers.add_parser(
        "collection-info",
        help="Show collection information",
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
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
