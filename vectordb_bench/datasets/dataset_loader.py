"""Load a cached dataset from disk, generating + caching it if missing.

Caches as .npz (vectors) + .json (ids/ground truth) so repeated benchmark
runs against the same config don't re-pay generation cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vectordb_bench.datasets.synthetic_generator import SyntheticDataset, generate_synthetic_dataset

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vectordb_bench" / "datasets"


def _cache_key(n_vectors: int, dim: int, n_queries: int, top_k: int, metric: str, seed: int) -> str:
    return f"synthetic_n{n_vectors}_d{dim}_q{n_queries}_k{top_k}_{metric}_s{seed}"


def load_dataset(
    n_vectors: int = 10_000,
    dim: int = 128,
    n_queries: int = 100,
    top_k: int = 10,
    metric: str = "cosine",
    seed: int = 42,
    cache_dir: str | Path | None = None,
    force_regenerate: bool = False,
) -> SyntheticDataset:
    """Load a synthetic dataset from cache, generating it if absent or forced."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(n_vectors, dim, n_queries, top_k, metric, seed)
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    if not force_regenerate and npz_path.exists() and meta_path.exists():
        return _read_cache(npz_path, meta_path)

    dataset = generate_synthetic_dataset(
        n_vectors=n_vectors, dim=dim, n_queries=n_queries, top_k=top_k, metric=metric, seed=seed
    )
    _write_cache(dataset, npz_path, meta_path)
    return dataset


def _write_cache(dataset: SyntheticDataset, npz_path: Path, meta_path: Path) -> None:
    np.savez_compressed(
        npz_path,
        corpus_vectors=dataset.corpus_vectors,
        query_vectors=dataset.query_vectors,
    )
    meta_path.write_text(
        json.dumps(
            {
                "corpus_ids": dataset.corpus_ids,
                "ground_truth": {str(k): v for k, v in dataset.ground_truth.items()},
            }
        )
    )


def _read_cache(npz_path: Path, meta_path: Path) -> SyntheticDataset:
    arrays = np.load(npz_path)
    meta = json.loads(meta_path.read_text())
    ground_truth = {int(k): v for k, v in meta["ground_truth"].items()}
    return SyntheticDataset(
        corpus_vectors=arrays["corpus_vectors"],
        corpus_ids=meta["corpus_ids"],
        query_vectors=arrays["query_vectors"],
        ground_truth=ground_truth,
    )
