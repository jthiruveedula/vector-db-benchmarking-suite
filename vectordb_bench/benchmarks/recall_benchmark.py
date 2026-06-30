"""Recall@k evaluation against a synthetic dataset with exact ground truth.

Unlike VectorDBBenchmarker.benchmark_query (which times queries against the
benchmarker's own random corpus), this wrapper loads/generates a dataset with
brute-force ground truth, inserts it into the adapter, and reports recall@k
in isolation from latency/QPS concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

from vectordb_bench.benchmark import VectorDBAdapter
from vectordb_bench.datasets.dataset_loader import load_dataset


@dataclass
class RecallResult:
    db_name: str
    top_k: int
    n_queries: int
    recall_at_k: float


def run_recall_benchmark(
    db_name: str,
    adapter: VectorDBAdapter,
    n_vectors: int = 10_000,
    dim: int = 128,
    n_queries: int = 100,
    top_k: int = 10,
    metric: str = "cosine",
    seed: int = 42,
) -> RecallResult:
    """Insert a synthetic corpus into adapter and measure recall@top_k."""
    dataset = load_dataset(
        n_vectors=n_vectors, dim=dim, n_queries=n_queries, top_k=top_k, metric=metric, seed=seed
    )

    batch_size = 1000
    for i in range(0, dataset.n_vectors, batch_size):
        adapter.insert_batch(
            dataset.corpus_vectors[i:i + batch_size],
            dataset.corpus_ids[i:i + batch_size],
        )

    recalls = []
    for i, query_vec in enumerate(dataset.query_vectors):
        results = adapter.query(query_vec, top_k)
        truth = dataset.ground_truth[i]
        hits = len(set(results) & set(truth))
        recalls.append(hits / len(truth) if truth else 1.0)

    return RecallResult(
        db_name=db_name,
        top_k=top_k,
        n_queries=n_queries,
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
    )
