"""Qdrant adapter — Rust-based, HNSW indexing, self-hosted or Cloud."""

from __future__ import annotations

import numpy as np


class QdrantAdapter:
    """Adapter for a self-hosted or Qdrant Cloud instance."""

    # Self-hosted compute cost estimate, illustrative only.
    COST_PER_1K = 0.04

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "benchmark",
        dim: int = 768,
        distance: str = "Cosine",
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance[distance.upper()]),
            )

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=idx, vector=vec.tolist(), payload={"ext_id": id_})
            for idx, (id_, vec) in enumerate(zip(ids, vectors))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        results = self._client.query_points(
            collection_name=self.collection_name, query=vector.tolist(), limit=top_k
        )
        return [point.payload["ext_id"] for point in results.points]

    def count(self) -> int:
        return self._client.count(self.collection_name).count

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K
