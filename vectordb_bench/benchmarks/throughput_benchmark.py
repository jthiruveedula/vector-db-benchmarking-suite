"""QPS under concurrent load, sweeping a list of thread-pool sizes.

For each concurrency level, runs VectorDBBenchmarker.benchmark_query and
records QPS so callers can plot a concurrency-vs-throughput curve.
"""

from __future__ import annotations

from dataclasses import dataclass

from vectordb_bench.benchmark import BenchmarkConfig, VectorDBAdapter, VectorDBBenchmarker


@dataclass
class ThroughputPoint:
    concurrency: int
    qps: float
    p99_ms: float


@dataclass
class ThroughputResult:
    db_name: str
    points: list[ThroughputPoint]

    def best(self) -> ThroughputPoint:
        return max(self.points, key=lambda p: p.qps)


def run_throughput_benchmark(
    db_name: str,
    adapter: VectorDBAdapter,
    n_vectors: int = 10_000,
    vector_dim: int = 128,
    n_queries: int = 200,
    top_k: int = 10,
    concurrency_levels: list[int] | None = None,
    seed: int = 42,
) -> ThroughputResult:
    """Sweep concurrency levels and measure QPS/P99 at each."""
    concurrency_levels = concurrency_levels or [1, 2, 4, 8]

    config = BenchmarkConfig(
        n_vectors=n_vectors, vector_dim=vector_dim, n_queries=n_queries,
        top_k=top_k, n_threads=concurrency_levels[0], warm_up_queries=min(20, n_queries), seed=seed,
    )
    bench = VectorDBBenchmarker(config)
    bench.benchmark_insert(db_name, adapter)

    points = []
    for concurrency in concurrency_levels:
        bench.config.n_threads = concurrency
        result = bench.benchmark_query(db_name, adapter)
        points.append(ThroughputPoint(concurrency=concurrency, qps=result.qps, p99_ms=result.p99_ms))

    return ThroughputResult(db_name=db_name, points=points)
