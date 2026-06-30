from __future__ import annotations

import numpy as np

from vectordb_bench.datasets.synthetic_generator import generate_synthetic_dataset


def test_shapes_and_ids():
    ds = generate_synthetic_dataset(n_vectors=200, dim=16, n_queries=10, top_k=5, seed=1)
    assert ds.corpus_vectors.shape == (200, 16)
    assert ds.query_vectors.shape == (10, 16)
    assert len(ds.corpus_ids) == 200
    assert ds.n_vectors == 200
    assert ds.dim == 16


def test_ground_truth_keys_and_lengths():
    ds = generate_synthetic_dataset(n_vectors=200, dim=16, n_queries=10, top_k=5, seed=1)
    assert set(ds.ground_truth.keys()) == set(range(10))
    for neighbors in ds.ground_truth.values():
        assert len(neighbors) == 5
        assert all(n in ds.corpus_ids for n in neighbors)


def test_ground_truth_matches_brute_force_cosine():
    ds = generate_synthetic_dataset(n_vectors=100, dim=8, n_queries=3, top_k=4, metric="cosine", seed=7)

    corpus_norm = ds.corpus_vectors / np.linalg.norm(ds.corpus_vectors, axis=1, keepdims=True)
    for i, query in enumerate(ds.query_vectors):
        q_norm = query / np.linalg.norm(query)
        scores = corpus_norm @ q_norm
        expected_idx = np.argsort(-scores)[:4]
        expected_ids = [ds.corpus_ids[j] for j in expected_idx]
        assert ds.ground_truth[i] == expected_ids


def test_deterministic_with_seed():
    ds1 = generate_synthetic_dataset(n_vectors=50, dim=8, n_queries=5, top_k=3, seed=42)
    ds2 = generate_synthetic_dataset(n_vectors=50, dim=8, n_queries=5, top_k=3, seed=42)
    assert np.array_equal(ds1.corpus_vectors, ds2.corpus_vectors)
    assert ds1.ground_truth == ds2.ground_truth


def test_euclidean_metric_runs():
    ds = generate_synthetic_dataset(n_vectors=50, dim=8, n_queries=5, top_k=3, metric="euclidean", seed=3)
    assert len(ds.ground_truth) == 5
