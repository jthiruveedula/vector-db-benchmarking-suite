"""Vector DB Benchmarking Suite
Comprehensive benchmarking framework for comparing vector databases:
BigQuery Vector Search, Pinecone, Weaviate, Chroma, and pgvector.
Measures QPS, P99 latency, recall@k, and cost per query.
"""

from __future__ import annotations
import time
import random
import statistics
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol
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
    """Protocol for vector DB adapters."""

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None: ...
    def query(self, vector: np.ndarray, top_k: int) -> list[str]: ...
    def count(self) -> int: ...
    def cost_per_1k_queries(self) -> float: ...


class BigQueryVectorAdapter:
    """BigQuery VECTOR_SEARCH adapter."""

    COST_PER_1K = 0.005  # approximate

    def __init__(self, project_id: str, dataset: str = "benchmark", table: str = "vectors") -> None:
        from google.cloud import bigquery
        self.bq = bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset}.{table}"
        self._ensure_table(project_id, dataset, table)

    def _ensure_table(self, project_id: str, dataset: str, table: str) -> None:
        from google.cloud import bigquery
        schema = [
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]
        ds = bigquery.Dataset(f"{project_id}.{dataset}")
        self.bq.create_dataset(ds, exists_ok=True)
        t = bigquery.Table(self.table_ref, schema=schema)
        self.bq.create_table(t, exists_ok=True)

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        rows = [{"id": id_, "embedding": vec.tolist()} for id_, vec in zip(ids, vectors)]
        errors = self.bq.insert_rows_json(self.table_ref, rows)
        if errors:
            raise RuntimeError(f"BQ insert errors: {errors}")

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        emb_str = ", ".join(str(v) for v in vector.tolist())
        sql = f"""
            SELECT id, COSINE_DISTANCE(embedding, [{emb_str}]) AS score
            FROM `{self.table_ref}`
            ORDER BY score ASC
            LIMIT {top_k}
        """
        return [r.id for r in self.bq.query(sql).result()]

    def count(self) -> int:
        return list(self.bq.query(f"SELECT COUNT(*) AS n FROM `{self.table_ref}`").result())[0].n

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K


class InMemoryFAISSAdapter:
    """FAISS in-memory adapter for baseline comparison."""

    COST_PER_1K = 0.0  # self-hosted

    def __init__(self, dim: int, index_type: str = "Flat") -> None:
        import faiss
        self.dim = dim
        if index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)
        else:
            self.index = faiss.IndexFlatL2(dim)
        self._ids: list[str] = []

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(vectors.astype(np.float32))
        self.index.add(vectors.astype(np.float32))
        self._ids.extend(ids)

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        _, indices = self.index.search(vector.reshape(1, -1).astype(np.float32), top_k)
        return [self._ids[i] for i in indices[0] if i >= 0 and i < len(self._ids)]

    def count(self) -> int:
        return self.index.ntotal

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K


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

