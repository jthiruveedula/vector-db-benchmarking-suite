"""Pinecone managed vector DB adapter (pod-based or serverless)."""

from __future__ import annotations

import numpy as np


class PineconeAdapter:
    """Adapter for Pinecone's managed vector search service."""

    # Approximate list price for a serverless starter index (us-east-1),
    # illustrative only — see Pinecone pricing for current figures.
    COST_PER_1K = 0.40

    def __init__(
        self,
        api_key: str,
        index_name: str = "benchmark",
        dim: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self._pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        if index_name not in self._pc.list_indexes().names():
            self._pc.create_index(
                name=index_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self._pc.Index(index_name)

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        vectors_list = [(id_, vec.tolist()) for id_, vec in zip(ids, vectors)]
        self.index.upsert(vectors=vectors_list)

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        result = self.index.query(vector=vector.tolist(), top_k=top_k, include_values=False)
        return [match["id"] for match in result["matches"]]

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        return int(stats["total_vector_count"])

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K
