#!/usr/bin/env python3
"""
Validation harness for /query/compare endpoint.

Runs representative questions through the compare endpoint and generates
metrics to decide whether reranking is worth the operational complexity.

Usage:
    python scripts/validate_rerank_compare.py \
        --base-url http://localhost:8000 \
        --workspace-id YOUR-WORKSPACE-ID \
        --questions-file questions.txt \
        --out-jsonl compare_logs.jsonl

    # With custom retrieval params
    python scripts/validate_rerank_compare.py \
        --base-url http://localhost:8000 \
        --workspace-id YOUR-WORKSPACE-ID \
        --retrieve-k 30 \
        --top-k 5 \
        --questions-file questions.txt

Decision thresholds (configure via args):
    --impact-threshold 0.15   # Impact rate >= 15% to justify PR3
    --p95-threshold 1000      # p95 rerank_ms <= 1000ms
    --fallback-threshold 0.05 # Fallback rate <= 5%
"""

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class ValidationResult:
    """Summary metrics from validation run."""

    total_queries: int
    impacted_count: int
    impact_rate: float
    median_rank_delta: Optional[float]
    max_rank_delta: Optional[int]
    p50_rerank_ms: Optional[float]
    p95_rerank_ms: Optional[float]
    fallback_count: int
    fallback_rate: float
    timeout_count: int
    timeout_rate: float
    errors: int

    def passes_thresholds(
        self,
        impact_threshold: float = 0.15,
        p95_threshold: float = 1000.0,
        fallback_threshold: float = 0.05,
    ) -> tuple[bool, list[str]]:
        """Check if results pass decision thresholds."""
        failures = []

        if self.impact_rate < impact_threshold:
            failures.append(
                f"Impact rate {self.impact_rate:.1%} < {impact_threshold:.0%} threshold"
            )

        if self.p95_rerank_ms and self.p95_rerank_ms > p95_threshold:
            failures.append(
                f"p95 rerank_ms {self.p95_rerank_ms:.0f}ms > {p95_threshold:.0f}ms threshold"
            )

        if self.fallback_rate > fallback_threshold:
            failures.append(
                f"Fallback rate {self.fallback_rate:.1%} > {fallback_threshold:.0%} threshold"
            )

        return len(failures) == 0, failures


def percentile(data: list[float], p: float) -> Optional[float]:
    """Compute percentile (0-1) of sorted data."""
    if not data:
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = int(p * (n - 1))
    return sorted_data[idx]


def median(data: list[float]) -> Optional[float]:
    """Compute median of data."""
    if not data:
        return None
    return statistics.median(data)


async def run_compare(
    client: httpx.AsyncClient,
    base_url: str,
    workspace_id: str,
    question: str,
    retrieve_k: int,
    top_k: int,
) -> dict:
    """Run a single compare query."""
    response = await client.post(
        f"{base_url}/query/compare",
        json={
            "workspace_id": workspace_id,
            "question": question,
            "retrieve_k": retrieve_k,
            "top_k": top_k,
            "skip_neighbors": True,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


async def validate(
    base_url: str,
    workspace_id: str,
    questions: list[str],
    retrieve_k: int,
    top_k: int,
    out_jsonl: Path,
    concurrency: int = 5,
) -> ValidationResult:
    """Run validation and compute metrics."""
    results = []
    errors = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(q: str, idx: int) -> Optional[dict]:
        nonlocal errors
        async with semaphore:
            try:
                async with httpx.AsyncClient() as client:
                    result = await run_compare(
                        client, base_url, workspace_id, q, retrieve_k, top_k
                    )
                    print(f"[{idx+1}/{len(questions)}] OK: {q[:50]}...")
                    return {
                        "question": q,
                        "jaccard": result["metrics"]["jaccard"],
                        "spearman": result["metrics"]["spearman"],
                        "rank_delta_mean": result["metrics"]["rank_delta_mean"],
                        "rank_delta_max": result["metrics"]["rank_delta_max"],
                        "overlap_count": result["metrics"]["overlap_count"],
                        "union_count": result["metrics"]["union_count"],
                        "rerank_ms": result["reranked"]["meta"].get("rerank_ms"),
                        "rerank_total_ms": result["reranked"]["meta"].get("total_ms"),
                        "rerank_state": result["reranked"]["meta"].get("rerank_state"),
                        "rerank_timeout": result["reranked"]["meta"].get(
                            "rerank_timeout", False
                        ),
                        "rerank_fallback": result["reranked"]["meta"].get(
                            "rerank_fallback", False
                        ),
                        "vector_top5": result["metrics"]["vector_only_ids"][:5],
                        "reranked_top5": result["metrics"]["reranked_ids"][:5],
                    }
            except Exception as e:
                print(f"[{idx+1}/{len(questions)}] ERROR: {q[:50]}... - {e}")
                errors += 1
                return None

    # Run all queries with concurrency limit
    tasks = [run_one(q, i) for i, q in enumerate(questions)]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    # Write JSONL
    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} results to {out_jsonl}")

    # Compute metrics
    impacted = [r for r in results if r["jaccard"] < 0.8]
    rank_deltas = [r["rank_delta_mean"] for r in impacted if r["rank_delta_mean"] is not None]
    max_deltas = [r["rank_delta_max"] for r in impacted if r["rank_delta_max"] is not None]
    rerank_times = [r["rerank_total_ms"] for r in results if r["rerank_total_ms"] is not None]
    fallbacks = [r for r in results if r.get("rerank_fallback")]
    timeouts = [r for r in results if r.get("rerank_timeout")]

    return ValidationResult(
        total_queries=len(results),
        impacted_count=len(impacted),
        impact_rate=len(impacted) / len(results) if results else 0,
        median_rank_delta=median(rank_deltas),
        max_rank_delta=max(max_deltas) if max_deltas else None,
        p50_rerank_ms=percentile(rerank_times, 0.5),
        p95_rerank_ms=percentile(rerank_times, 0.95),
        fallback_count=len(fallbacks),
        fallback_rate=len(fallbacks) / len(results) if results else 0,
        timeout_count=len(timeouts),
        timeout_rate=len(timeouts) / len(results) if results else 0,
        errors=errors,
    )


def print_report(result: ValidationResult, thresholds: dict) -> None:
    """Print validation report."""
    print("\n" + "=" * 60)
    print("RERANK VALIDATION REPORT")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Value':>15} {'Threshold':>15}")
    print("-" * 60)

    # Impact
    impact_thresh = f">= {thresholds['impact']:.0%}"
    impact_status = "PASS" if result.impact_rate >= thresholds["impact"] else "FAIL"
    print(f"{'Impact rate':<30} {result.impact_rate:>14.1%} {impact_thresh:>15} [{impact_status}]")
    print(f"{'  Impacted queries':<30} {result.impacted_count:>15}")

    # Magnitude
    if result.median_rank_delta is not None:
        print(f"{'Median rank_delta (impacted)':<30} {result.median_rank_delta:>15.2f}")
    if result.max_rank_delta is not None:
        print(f"{'Max rank_delta (impacted)':<30} {result.max_rank_delta:>15}")

    # Latency
    if result.p50_rerank_ms is not None:
        print(f"{'p50 rerank_total_ms':<30} {result.p50_rerank_ms:>14.0f}ms")
    if result.p95_rerank_ms is not None:
        p95_thresh = f"<= {thresholds['p95']:.0f}ms"
        p95_status = "PASS" if result.p95_rerank_ms <= thresholds["p95"] else "FAIL"
        print(f"{'p95 rerank_total_ms':<30} {result.p95_rerank_ms:>14.0f}ms {p95_thresh:>15} [{p95_status}]")

    # Reliability
    fallback_thresh = f"<= {thresholds['fallback']:.0%}"
    fallback_status = "PASS" if result.fallback_rate <= thresholds["fallback"] else "FAIL"
    print(f"{'Fallback rate':<30} {result.fallback_rate:>14.1%} {fallback_thresh:>15} [{fallback_status}]")
    print(f"{'  Fallbacks':<30} {result.fallback_count:>15}")
    print(f"{'  Timeouts':<30} {result.timeout_count:>15}")

    # Errors
    if result.errors > 0:
        print(f"{'Errors':<30} {result.errors:>15}")

    # Decision
    print("\n" + "-" * 60)
    passes, failures = result.passes_thresholds(
        thresholds["impact"], thresholds["p95"], thresholds["fallback"]
    )

    if passes:
        print("DECISION: BUILD PR3 - All thresholds passed")
    else:
        print("DECISION: SKIP PR3 - Thresholds not met:")
        for f in failures:
            print(f"  - {f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate rerank compare endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", required=True, help="API base URL")
    parser.add_argument("--workspace-id", required=True, help="Workspace UUID")
    parser.add_argument(
        "--questions-file", required=True, type=Path, help="File with questions (one per line)"
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("compare_logs.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument("--retrieve-k", type=int, default=50, help="Candidates to retrieve")
    parser.add_argument("--top-k", type=int, default=10, help="Results to return")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent requests")
    parser.add_argument(
        "--impact-threshold",
        type=float,
        default=0.15,
        help="Min impact rate to pass (0-1)",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=1000.0,
        help="Max p95 rerank_ms to pass",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.05,
        help="Max fallback rate to pass (0-1)",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Output summary JSON file (optional)",
    )

    args = parser.parse_args()

    # Load questions
    questions = [
        line.strip()
        for line in args.questions_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if not questions:
        print(f"No questions found in {args.questions_file}")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions from {args.questions_file}")
    print(f"Config: retrieve_k={args.retrieve_k}, top_k={args.top_k}")
    print(f"Target: {args.base_url}/query/compare")
    print()

    # Run validation
    result = asyncio.run(
        validate(
            base_url=args.base_url,
            workspace_id=args.workspace_id,
            questions=questions,
            retrieve_k=args.retrieve_k,
            top_k=args.top_k,
            out_jsonl=args.out_jsonl,
            concurrency=args.concurrency,
        )
    )

    # Print report
    thresholds = {
        "impact": args.impact_threshold,
        "p95": args.p95_threshold,
        "fallback": args.fallback_threshold,
    }
    print_report(result, thresholds)

    # Optionally write summary JSON
    if args.out_summary:
        summary = {
            "total_queries": result.total_queries,
            "impacted_count": result.impacted_count,
            "impact_rate": result.impact_rate,
            "median_rank_delta": result.median_rank_delta,
            "max_rank_delta": result.max_rank_delta,
            "p50_rerank_ms": result.p50_rerank_ms,
            "p95_rerank_ms": result.p95_rerank_ms,
            "fallback_count": result.fallback_count,
            "fallback_rate": result.fallback_rate,
            "timeout_count": result.timeout_count,
            "timeout_rate": result.timeout_rate,
            "errors": result.errors,
            "thresholds": thresholds,
            "passes": result.passes_thresholds(
                thresholds["impact"], thresholds["p95"], thresholds["fallback"]
            )[0],
        }
        args.out_summary.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {args.out_summary}")

    # Exit code based on decision
    passes, _ = result.passes_thresholds(
        thresholds["impact"], thresholds["p95"], thresholds["fallback"]
    )
    sys.exit(0 if passes else 1)


if __name__ == "__main__":
    main()
