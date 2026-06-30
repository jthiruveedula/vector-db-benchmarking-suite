"""Combine JSON result files (one BenchmarkResult per file, or a list) into a summary."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from vectordb_bench.benchmark import BenchmarkResult


def load_results_from_dir(results_dir: str | Path) -> list[BenchmarkResult]:
    """Load every *.json file in a directory as one or more BenchmarkResults."""
    results_dir = Path(results_dir)
    results: list[BenchmarkResult] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            results.append(BenchmarkResult(**record))
    return results


def aggregate_results(results: list[BenchmarkResult]) -> dict:
    """Group results by (db_name, operation) and summarize key metrics.

    Returns a dict keyed by "db_name:operation" with mean/best QPS, latency,
    recall, and cost — useful when multiple benchmark runs exist per DB.
    """
    groups: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for r in results:
        groups.setdefault((r.db_name, r.operation), []).append(r)

    summary = {}
    for (db_name, operation), group in groups.items():
        key = f"{db_name}:{operation}"
        summary[key] = {
            "db_name": db_name,
            "operation": operation,
            "n_runs": len(group),
            "mean_qps": mean(r.qps for r in group),
            "best_qps": max(r.qps for r in group),
            "mean_p99_ms": mean(r.p99_ms for r in group),
            "mean_recall_at_k": mean(r.recall_at_k for r in group),
            "mean_cost_per_1k_queries_usd": mean(r.cost_per_1k_queries_usd for r in group),
        }
    return summary


def save_results(results: list[BenchmarkResult], path: str | Path) -> None:
    """Write a list of BenchmarkResults to a single JSON file."""
    Path(path).write_text(json.dumps([asdict(r) for r in results], indent=2))


def write_summary(results: list[BenchmarkResult], path: str | Path) -> None:
    """Aggregate results and write the summary dict to a JSON file."""
    Path(path).write_text(json.dumps(aggregate_results(results), indent=2))
