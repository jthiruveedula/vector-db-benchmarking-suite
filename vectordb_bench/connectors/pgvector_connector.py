"""pgvector adapter — PostgreSQL extension, familiar SQL interface."""

from __future__ import annotations

import numpy as np


class PgVectorAdapter:
    """Adapter for a PostgreSQL instance with the pgvector extension."""

    # Self-hosted compute cost estimate, illustrative only.
    COST_PER_1K = 0.03

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/postgres",
        table: str = "benchmark_vectors",
        dim: int = 768,
        index_type: str = "hnsw",
    ) -> None:
        import psycopg

        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                f"(ext_id TEXT PRIMARY KEY, embedding vector({dim}))"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} "
                f"USING {index_type} (embedding vector_cosine_ops)"
            )

    def insert_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        with self._conn.cursor() as cur:
            rows = [(id_, vec.tolist()) for id_, vec in zip(ids, vectors)]
            cur.executemany(
                f"INSERT INTO {self.table} (ext_id, embedding) VALUES (%s, %s) "
                f"ON CONFLICT (ext_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                rows,
            )

    def query(self, vector: np.ndarray, top_k: int) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT ext_id FROM {self.table} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector.tolist(), top_k),
            )
            return [row[0] for row in cur.fetchall()]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]

    def cost_per_1k_queries(self) -> float:
        return self.COST_PER_1K

    def close(self) -> None:
        self._conn.close()
