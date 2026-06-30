from __future__ import annotations

import numpy as np

from vectordb_bench.datasets.dataset_loader import load_dataset


def test_load_dataset_generates_and_caches(tmp_path):
    ds1 = load_dataset(n_vectors=30, dim=4, n_queries=5, top_k=2, seed=9, cache_dir=tmp_path)
    cached_files = list(tmp_path.glob("*"))
    assert len(cached_files) == 2  # .npz + .json

    ds2 = load_dataset(n_vectors=30, dim=4, n_queries=5, top_k=2, seed=9, cache_dir=tmp_path)
    assert np.array_equal(ds1.corpus_vectors, ds2.corpus_vectors)
    assert ds1.corpus_ids == ds2.corpus_ids
    assert ds1.ground_truth == ds2.ground_truth


def test_force_regenerate_overwrites_cache(tmp_path):
    ds1 = load_dataset(n_vectors=20, dim=4, n_queries=3, top_k=2, seed=1, cache_dir=tmp_path)
    ds2 = load_dataset(
        n_vectors=20, dim=4, n_queries=3, top_k=2, seed=1, cache_dir=tmp_path, force_regenerate=True
    )
    # Same seed => same data even after forced regeneration.
    assert np.array_equal(ds1.corpus_vectors, ds2.corpus_vectors)
