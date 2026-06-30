"""Synthetic vector dataset generator with brute-force kNN ground truth.

Generates a random corpus + query set and computes exact nearest neighbors
for each query so recall@k can be measured against an ANN index without
needing a real embedding dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticDataset:
    corpus_vectors: np.ndarray  # (n_vectors, dim) float32
    corpus_ids: list[str]
    query_vectors: np.ndarray  # (n_queries, dim) float32
    ground_truth: dict[int, list[str]]  # query index -> top_k true neighbor ids

    @property
    def n_vectors(self) -> int:
        return self.corpus_vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.corpus_vectors.shape[1]


def _brute_force_knn(
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
    metric: str = "cosine",
    block_size: int = 256,
) -> np.ndarray:
    """Exact top_k neighbor indices for each query, computed in blocks to bound memory."""
    n_queries = query_vectors.shape[0]
    neighbor_idx = np.empty((n_queries, top_k), dtype=np.int64)

    if metric == "cosine":
        corpus_norm = corpus_vectors / (np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-12)
        query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12)
    else:
        corpus_norm, query_norm = corpus_vectors, query_vectors

    for start in range(0, n_queries, block_size):
        end = min(start + block_size, n_queries)
        block = query_norm[start:end]
        if metric == "cosine":
            # BLAS backends (e.g. Apple Accelerate) can raise spurious FP
            # warnings on matmul that don't reflect actual NaN/inf results.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                scores = block @ corpus_norm.T  # higher is closer
            top = np.argpartition(-scores, kth=min(top_k, scores.shape[1] - 1), axis=1)[:, :top_k]
            # Sort each row's top-k candidates by actual score, descending.
            row_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-row_scores, axis=1)
            sorted_top = np.take_along_axis(top, order, axis=1)
        else:  # euclidean
            dists = np.linalg.norm(block[:, None, :] - corpus_norm[None, :, :], axis=2)
            sorted_top = np.argsort(dists, axis=1)[:, :top_k]
        neighbor_idx[start:end] = sorted_top

    return neighbor_idx


def generate_synthetic_dataset(
    n_vectors: int = 10_000,
    dim: int = 128,
    n_queries: int = 100,
    top_k: int = 10,
    metric: str = "cosine",
    seed: int = 42,
) -> SyntheticDataset:
    """Generate random corpus + queries and compute exact ground-truth neighbors."""
    rng = np.random.default_rng(seed)
    corpus_vectors = rng.random((n_vectors, dim), dtype=np.float32)
    query_vectors = rng.random((n_queries, dim), dtype=np.float32)
    corpus_ids = [f"vec_{i}" for i in range(n_vectors)]

    neighbor_idx = _brute_force_knn(corpus_vectors, query_vectors, top_k, metric=metric)
    ground_truth = {
        i: [corpus_ids[j] for j in neighbor_idx[i]] for i in range(n_queries)
    }

    return SyntheticDataset(
        corpus_vectors=corpus_vectors,
        corpus_ids=corpus_ids,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
    )
