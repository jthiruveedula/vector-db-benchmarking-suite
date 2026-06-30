from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from vectordb_bench.connectors.faiss_connector import InMemoryFAISSAdapter


def test_insert_and_count():
    adapter = InMemoryFAISSAdapter(dim=8)
    vectors = np.random.rand(10, 8).astype(np.float32)
    ids = [f"v{i}" for i in range(10)]
    adapter.insert_batch(vectors, ids)
    assert adapter.count() == 10


def test_query_returns_exact_match_for_flat_index():
    adapter = InMemoryFAISSAdapter(dim=4)
    vectors = np.eye(4, dtype=np.float32)
    ids = ["a", "b", "c", "d"]
    adapter.insert_batch(vectors, ids)

    results = adapter.query(vectors[2], top_k=1)
    assert results == ["c"]


def test_query_top_k_returns_correct_count():
    adapter = InMemoryFAISSAdapter(dim=16)
    vectors = np.random.rand(50, 16).astype(np.float32)
    ids = [f"v{i}" for i in range(50)]
    adapter.insert_batch(vectors, ids)

    results = adapter.query(vectors[0], top_k=5)
    assert len(results) == 5
    assert "v0" in results


def test_cost_is_zero():
    adapter = InMemoryFAISSAdapter(dim=4)
    assert adapter.cost_per_1k_queries() == 0.0
