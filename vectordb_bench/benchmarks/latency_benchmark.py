"""P50/P95/P99 single-threaded query latency measurement.

Runs queries sequentially so latency numbers reflect per-query cost without
concurrency effects, then reports percentiles plus min/max.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from vectordb_bench.benchmark import VectorDBAdapter


@dataclass
class LatencyResult:
    db_name: str
    n_queries: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    avg_ms: float


def run_latency_benchmark(
    db_name: str,
    adapter: VectorDBAdapter,
    n_vectors: int = 10_000,
    vector_dim: int = 128,
    n_queries: int = 200,
    top_k: int = 10,
    warm_up_queries: int = 20,
    seed: int = 42,
) -> LatencyResult:
    """Insert a synthetic corpus, then measure sequential query latency percentiles."""
    rng = np.random.default_rng(seed)
    corpus = rng.random((n_vectors, vector_dim), dtype=np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    queries = rng.random((n_queries, vector_dim), dtype=np.float32)

    batch_size = 1000
    for i in range(0, n_vectors, batch_size):
        adapter.insert_batch(corpus[i:i + batch_size], ids[i:i + batch_size])

    for i in range(min(warm_up_queries, n_queries)):
        adapter.query(queries[i], top_k)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        adapter.query(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

    lats = sorted(latencies)
    return LatencyResult(
        db_name=db_name,
        n_queries=len(latencies),
        min_ms=lats[0],
        p50_ms=lats[int(len(lats) * 0.50)],
        p95_ms=lats[int(len(lats) * 0.95)],
        p99_ms=lats[min(int(len(lats) * 0.99), len(lats) - 1)],
        max_ms=lats[-1],
        avg_ms=statistics.mean(latencies),
    )
