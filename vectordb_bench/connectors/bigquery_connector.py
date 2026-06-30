"""BigQuery VECTOR_SEARCH adapter.

Not part of the README's primary 5-way comparison table (Pinecone, Weaviate,
Qdrant, pgvector, FAISS), but kept available as a connector for teams already
on BigQuery who want to benchmark it against the others.
"""

from __future__ import annotations

import numpy as np


class BigQueryVectorAdapter:
    """BigQuery VECTOR_SEARCH adapter."""

    COST_PER_1K = 0.005  # approximate, on-demand query pricing

    def __init__(self, project_id: str, dataset: str = "benchmark", table: str = "vectors") -> None:
        from google.cloud import bigquery
        self.bq = bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset}.{table}"
        self._ensure_table(project_id, dataset, table)

    def _ensure_table(self, project_id: str, dataset: str, table: str) -> None:
        from google.cloud import bigquery
        schema = [
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]
        ds = bigquery.Dataset(f"{project_id}.{dataset}")
        self.bq.create_dataset(ds, exists_ok=True)
        t = bigquery.Table(self.table_ref, schema=schema)
        self.bq.create_table(t, exists_ok=True)

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        rows = [{"id": id_, "embedding": vec.tolist()} for id_, vec in zip(ids, vectors)]
        errors = self.bq.insert_rows_json(self.table_ref, rows)
        if errors:
            raise RuntimeError(f"BQ insert errors: {errors}")

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        emb_str = ", ".join(str(v) for v in vector.tolist())
        sql = f"""
            SELECT id, COSINE_DISTANCE(embedding, [{emb_str}]) AS score
            FROM `{self.table_ref}`
            ORDER BY score ASC
            LIMIT {top_k}
        """
        return [r.id for r in self.bq.query(sql).result()]

    def count(self) -> int:
        return list(self.bq.query(f"SELECT COUNT(*) AS n FROM `{self.table_ref}`").result())[0].n

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K
