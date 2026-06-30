"""Vector DB Benchmarking Suite — orchestrator.

Runs insert/query benchmarks across pluggable vector DB adapters and reports
QPS, latency percentiles, recall@k, and cost per query. Concrete adapters
live in vectordb_bench.connectors; this module only defines the protocol
and the benchmark runner.
"""

from __future__ import annotations
import time
import statistics
import logging
from dataclasses import dataclass, field, asdict
from typing import Protocol
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    n_vectors: int = 100_000
    vector_dim: int = 768
    n_queries: int = 1_000
    top_k: int = 10
    n_threads: int = 4
    warm_up_queries: int = 100
    seed: int = 42


@dataclass
class BenchmarkResult:
    db_name: str
    operation: str  # insert, query
    n_queries: int
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    recall_at_k: float
    cost_per_1k_queries_usd: float
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.db_name:20s} | QPS={self.qps:8.1f} | "
            f"P99={self.p99_ms:7.1f}ms | Recall@{self.config.get('top_k', 10)}={self.recall_at_k:.3f} | "
            f"$/1k={self.cost_per_1k_queries_usd:.4f}"
        )


class VectorDBAdapter(Protocol):
    """Protocol all connectors implement."""

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None: ...
    def query(self, vector: np.ndarray, top_k: int) -> list[str]: ...
    def count(self) -> int: ...
    def cost_per_1k_queries(self) -> float: ...


class VectorDBBenchmarker:
    """Runs benchmarks across multiple vector DB adapters."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        np.random.seed(config.seed)
        self._corpus_vectors = np.random.rand(config.n_vectors, config.vector_dim).astype(np.float32)
        self._corpus_ids = [f"vec_{i}" for i in range(config.n_vectors)]
        self._query_vectors = np.random.rand(config.n_queries, config.vector_dim).astype(np.float32)

    def benchmark_insert(self, db_name: str, adapter: VectorDBAdapter, batch_size: int = 1000) -> BenchmarkResult:
        """Benchmark batch insert throughput."""
        latencies = []
        total_start = time.perf_counter()
        for i in range(0, self.config.n_vectors, batch_size):
            batch_vecs = self._corpus_vectors[i:i+batch_size]
            batch_ids = self._corpus_ids[i:i+batch_size]
            t0 = time.perf_counter()
            adapter.insert_batch(batch_vecs, batch_ids)
            latencies.append((time.perf_counter() - t0) * 1000)

        total_s = time.perf_counter() - total_start
        qps = self.config.n_vectors / total_s
        lats = sorted(latencies)

        return BenchmarkResult(
            db_name=db_name, operation="insert",
            n_queries=len(latencies), qps=qps,
            p50_ms=lats[int(len(lats)*0.5)],
            p95_ms=lats[int(len(lats)*0.95)],
            p99_ms=lats[int(len(lats)*0.99)],
            avg_ms=statistics.mean(latencies),
            recall_at_k=1.0,  # N/A for insert
            cost_per_1k_queries_usd=adapter.cost_per_1k_queries(),
            config=asdict(self.config),
        )

    def benchmark_query(
        self,
        db_name: str,
        adapter: VectorDBAdapter,
        ground_truth: dict[int, list[str]] | None = None,
    ) -> BenchmarkResult:
        """Benchmark query latency, QPS, and recall."""
        # Warm up
        for i in range(min(self.config.warm_up_queries, len(self._query_vectors))):
            adapter.query(self._query_vectors[i], self.config.top_k)

        latencies = []
        recalls = []

        def run_query(i: int):
            t0 = time.perf_counter()
            results = adapter.query(self._query_vectors[i], self.config.top_k)
            lat = (time.perf_counter() - t0) * 1000
            recall = 1.0
            if ground_truth and i in ground_truth:
                hits = len(set(results) & set(ground_truth[i]))
                recall = hits / len(ground_truth[i])
            return lat, recall

        with ThreadPoolExecutor(max_workers=self.config.n_threads) as ex:
            futures = [ex.submit(run_query, i) for i in range(self.config.n_queries)]
            for f in futures:
                lat, rec = f.result()
                latencies.append(lat)
                recalls.append(rec)

        lats = sorted(latencies)
        total_s = sum(latencies) / 1000
        qps = self.config.n_queries / total_s if total_s > 0 else 0

        return BenchmarkResult(
            db_name=db_name, operation="query",
            n_queries=self.config.n_queries, qps=qps,
            p50_ms=lats[int(len(lats)*0.50)],
            p95_ms=lats[int(len(lats)*0.95)],
            p99_ms=lats[int(len(lats)*0.99)],
            avg_ms=statistics.mean(latencies),
            recall_at_k=statistics.mean(recalls),
            cost_per_1k_queries_usd=adapter.cost_per_1k_queries(),
            config=asdict(self.config),
        )

    def print_leaderboard(self, results: list[BenchmarkResult]) -> None:
        print(f"\n{'DB':20s} | {'QPS':>8s} | {'P99 (ms)':>8s} | {'Recall':>8s} | {'$/1k':>8s}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x.qps, reverse=True):
            print(r.summary())
