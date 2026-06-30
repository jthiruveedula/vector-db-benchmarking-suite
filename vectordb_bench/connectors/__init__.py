"""Vector DB connector adapters.

Each connector implements the VectorDBAdapter protocol (insert_batch, query,
count, cost_per_1k_queries) defined in vectordb_bench.benchmark. Heavy client
libraries are imported lazily inside __init__ so this package can be imported
without any of them installed.
"""

from vectordb_bench.connectors.faiss_connector import InMemoryFAISSAdapter
from vectordb_bench.connectors.pinecone_connector import PineconeAdapter
from vectordb_bench.connectors.weaviate_connector import WeaviateAdapter
from vectordb_bench.connectors.qdrant_connector import QdrantAdapter
from vectordb_bench.connectors.pgvector_connector import PgVectorAdapter
from vectordb_bench.connectors.bigquery_connector import BigQueryVectorAdapter

__all__ = [
    "InMemoryFAISSAdapter",
    "PineconeAdapter",
    "WeaviateAdapter",
    "QdrantAdapter",
    "PgVectorAdapter",
    "BigQueryVectorAdapter",
]
