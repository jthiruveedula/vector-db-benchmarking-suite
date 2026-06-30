"""FAISS in-memory adapter — Meta's library, no persistence, no network cost."""

from __future__ import annotations

import numpy as np


class InMemoryFAISSAdapter:
    """FAISS in-memory adapter for baseline comparison."""

    COST_PER_1K = 0.0  # self-hosted, in-memory

    def __init__(self, dim: int, index_type: str = "Flat") -> None:
        import faiss
        self.dim = dim
        if index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.index = faiss.IndexFlatL2(dim)
        self._ids: list[str] = []

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(vectors.astype(np.float32))
        self.index.add(vectors.astype(np.float32))
        self._ids.extend(ids)

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        _, indices = self.index.search(vector.reshape(1, -1).astype(np.float32), top_k)
        return [self._ids[i] for i in indices[0] if 0 <= i < len(self._ids)]

    def count(self) -> int:
        return self.index.ntotal

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K
