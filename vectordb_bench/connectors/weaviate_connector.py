"""Weaviate adapter — open source, HNSW + hybrid (BM25 + dense) search."""

from __future__ import annotations

import numpy as np


class WeaviateAdapter:
    """Adapter for a self-hosted or WCS Weaviate instance."""

    # Self-hosted compute cost estimate, illustrative only.
    COST_PER_1K = 0.05

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: str | None = None,
        collection_name: str = "Benchmark",
        dim: int = 768,
    ) -> None:
        import weaviate
        from weaviate.classes.init import Auth
        from weaviate.classes.config import Configure

        if api_key:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url, auth_credentials=Auth.api_key(api_key)
            )
        else:
            self._client = weaviate.connect_to_local()

        self.collection_name = collection_name
        if not self._client.collections.exists(collection_name):
            self._client.collections.create(
                collection_name,
                vector_config=Configure.Vectors.self_provided(),
            )
        self._collection = self._client.collections.get(collection_name)

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        with self._collection.batch.dynamic() as batch:
            for id_, vec in zip(ids, vectors):
                batch.add_object(properties={"ext_id": id_}, vector=vec.tolist())

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        result = self._collection.query.near_vector(near_vector=vector.tolist(), limit=top_k)
        return [obj.properties["ext_id"] for obj in result.objects]

    def count(self) -> int:
        return self._collection.aggregate.over_all(total_count=True).total_count

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K

    def close(self) -> None:
        self._client.close()
