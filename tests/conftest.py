"""Shared fixtures: a trivial in-memory fake adapter with no external services."""

from __future__ import annotations

import numpy as np
import pytest


class FakeInMemoryAdapter:
    """Brute-force in-memory adapter used to test the orchestrator without FAISS/network deps."""

    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._ids: list[str] = []

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        self._vectors.extend(vectors)
        self._ids.extend(ids)

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        if not self._vectors:
            return []
        dists = [np.linalg.norm(vector - v) for v in self._vectors]
        order = np.argsort(dists)[:top_k]
        return [self._ids[i] for i in order]

    def count(self) -> int:
        return len(self._ids)

    def cost_per_1k_queries(self) -> float:
        return 0.0


@pytest.fixture
def fake_adapter() -> FakeInMemoryAdapter:
    return FakeInMemoryAdapter()
