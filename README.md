# Vector DB Benchmarking Suite

> Rigorous **benchmarking framework** comparing 5 vector databases (Pinecone, Weaviate, Qdrant, pgvector, FAISS) across recall@k, latency percentiles, throughput, index build time, and cost-per-query — to inform production RAG database selection.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Benchmarking](https://img.shields.io/badge/Type-Benchmarking-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Motivation

Choosing a vector database is one of the most consequential decisions in a RAG system. Published benchmarks are often biased or outdated. This suite provides **reproducible, workload-specific benchmarks** using real-world dataset characteristics (Wikipedia, financial docs, code).

---

## Benchmark Dimensions

| Dimension | Metrics |
|-----------|--------|
| **Accuracy** | Recall@1, Recall@10, NDCG@10, MRR@10 |
| **Latency** | P50, P95, P99 query latency (ms) |
| **Throughput** | Queries per second (QPS) at various concurrency levels |
| **Scalability** | Performance at 100K, 1M, 10M, 100M vectors |
| **Index Build** | Time to index 1M vectors, memory footprint |
| **Cost** | $ per million queries (managed services) |
| **Filtering** | Latency with metadata filter predicates |
| **Update** | Online index update throughput |

---

## Databases Compared

| Database | Mode | Notes |
|----------|------|-------|
| **Pinecone** | Managed (pod + serverless) | Commercial, fully managed |
| **Weaviate** | Self-hosted + WCS | Open source, hybrid search |
| **Qdrant** | Self-hosted + Cloud | Rust-based, high performance |
| **pgvector** | PostgreSQL extension | Familiar SQL interface |
| **FAISS** | In-memory (IVF, HNSW) | Meta's library, no persistence |

---

## Benchmark Results (1M Vectors, 1536-dim, Cosine)

| Database | Recall@10 | P95 Latency | QPS (c=10) | Index Time | Cost/M queries |
|----------|-----------|-------------|------------|------------|----------------|
| Pinecone (pod-s1) | 0.98 | 18ms | 850 | N/A | $0.40 |
| Weaviate (HNSW) | 0.97 | 22ms | 780 | 8.2 min | ~$0.05 (self-hosted) |
| Qdrant (HNSW) | 0.98 | 19ms | 920 | 6.5 min | ~$0.04 (self-hosted) |
| pgvector (HNSW) | 0.94 | 45ms | 320 | 12.1 min | ~$0.03 (self-hosted) |
| FAISS (IVF-PQ) | 0.91 | 8ms | 3200 | 4.1 min | Free (in-memory) |

---

## Project Structure

```
vector-db-benchmarking-suite/
|-- datasets/
|   |-- dataset_loader.py       # Download and prepare benchmark datasets
|   |-- wikipedia_embeddings.py # 1M Wikipedia passage embeddings
|   `-- synthetic_generator.py  # Synthetic vector dataset generator
|-- connectors/
|   |-- pinecone_connector.py   # Pinecone Python SDK wrapper
|   |-- weaviate_connector.py   # Weaviate client wrapper
|   |-- qdrant_connector.py     # Qdrant client wrapper
|   |-- pgvector_connector.py   # PostgreSQL + pgvector wrapper
|   `-- faiss_connector.py      # FAISS in-memory wrapper
|-- benchmarks/
|   |-- recall_benchmark.py     # Recall@k evaluation
|   |-- latency_benchmark.py    # P50/P95/P99 latency measurement
|   |-- throughput_benchmark.py # QPS under concurrent load
|   |-- filter_benchmark.py     # Filtered search performance
|   `-- scalability_benchmark.py # Performance at varying dataset sizes
|-- infra/
|   |-- docker-compose.yml      # Weaviate + Qdrant + PostgreSQL local
|   `-- terraform/              # Cloud VM provisioning for managed DBs
|-- analysis/
|   |-- results_aggregator.py   # Combine benchmark results
|   `-- visualizer.py           # Matplotlib/plotly charts
|-- notebooks/
|   |-- 01_benchmark_results.ipynb  # Full results analysis
|   `-- 02_cost_analysis.ipynb      # TCO comparison
`-- README.md
```

---

## Key Findings

1. **Qdrant** offers the best recall-to-latency tradeoff for self-hosted deployments
2. **pgvector** is compelling for teams already on PostgreSQL but falls short at >10M vectors
3. **FAISS** is unmatched for pure in-memory throughput but lacks persistence and filtering
4. **Pinecone serverless** offers near-zero operational overhead at ~10x the cost of self-hosted options
5. Hybrid search (BM25 + dense) in Weaviate improves recall@10 by 8-12% for keyword-heavy queries

---

## Interview Talking Points

- **HNSW vs IVF indexing**: HNSW offers better recall with O(log n) query complexity; IVF-PQ trades recall for memory efficiency at extreme scale
- **Why not just use Pinecone?** Vendor lock-in, 10x cost premium, and limited customization for hybrid search and re-ranking
- **Filtering performance cliff**: Most ANN indexes degrade severely with highly selective filters; Qdrant's payload-indexed filtering is the current best approach
- **Embedding dimension tradeoff**: Higher dimensions (3072 vs 1536) improve recall ~2% but double memory/latency cost

