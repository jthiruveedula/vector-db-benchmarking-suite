"""Matplotlib charts from BenchmarkResult lists.

matplotlib is an optional dependency (install with the `viz` extra); the
import is guarded so the rest of the package works without it.
"""

from __future__ import annotations

from pathlib import Path

from vectordb_bench.benchmark import BenchmarkResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization: pip install 'vectordb-bench[viz]'"
        ) from exc
    return plt


def plot_qps_comparison(results: list[BenchmarkResult], output_path: str | Path | None = None):
    """Bar chart of QPS per database, sorted descending."""
    plt = _require_matplotlib()
    ordered = sorted(results, key=lambda r: r.qps, reverse=True)
    names = [r.db_name for r in ordered]
    qps = [r.qps for r in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, qps, color="#4C72B0")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title("Throughput Comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_latency_percentiles(results: list[BenchmarkResult], output_path: str | Path | None = None):
    """Grouped bar chart of P50/P95/P99 latency per database."""
    plt = _require_matplotlib()
    names = [r.db_name for r in results]
    p50 = [r.p50_ms for r in results]
    p95 = [r.p95_ms for r in results]
    p99 = [r.p99_ms for r in results]

    x = range(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], p50, width, label="P50")
    ax.bar(list(x), p95, width, label="P95")
    ax.bar([i + width for i in x], p99, width, label="P99")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Percentile Comparison")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig
