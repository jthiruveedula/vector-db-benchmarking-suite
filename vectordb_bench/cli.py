"""CLI entrypoint: a zero-external-dependency FAISS demo benchmark.

Generates a synthetic dataset, runs insert + query benchmarks against the
in-memory FAISS adapter, and prints a leaderboard. Requires only numpy and
faiss-cpu — no managed services, network, or API keys.
"""

from __future__ import annotations

import argparse
import sys

from vectordb_bench.benchmark import BenchmarkConfig, VectorDBBenchmarker
from vectordb_bench.datasets.dataset_loader import load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vectordb_bench",
        description="Run a quick FAISS-only vector DB benchmark using a synthetic dataset.",
    )
    parser.add_argument("--n-vectors", type=int, default=20_000, help="corpus size")
    parser.add_argument("--dim", type=int, default=128, help="vector dimensionality")
    parser.add_argument("--n-queries", type=int, default=200, help="number of queries")
    parser.add_argument("--top-k", type=int, default=10, help="neighbors per query")
    parser.add_argument("--index-type", default="Flat", choices=["Flat", "IVF", "HNSW"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import faiss  # noqa: F401
    except ImportError:
        print(
            "faiss is not installed. Install it with: pip install -e '.[faiss]'",
            file=sys.stderr,
        )
        return 1

    from vectordb_bench.connectors.faiss_connector import InMemoryFAISSAdapter

    print(f"Generating synthetic dataset: n_vectors={args.n_vectors}, dim={args.dim} ...")
    # Ground truth uses euclidean distance to match FAISS's IndexFlatL2/IVF/HNSW metric,
    # so the "Flat" index type benchmarks at ~1.0 recall as expected for exact search.
    dataset = load_dataset(
        n_vectors=args.n_vectors, dim=args.dim, n_queries=args.n_queries,
        top_k=args.top_k, metric="euclidean", seed=args.seed,
    )

    config = BenchmarkConfig(
        n_vectors=dataset.n_vectors, vector_dim=dataset.dim,
        n_queries=args.n_queries, top_k=args.top_k, seed=args.seed,
    )
    bench = VectorDBBenchmarker(config)
    # Reuse the dataset's corpus/ground truth instead of the benchmarker's own random data.
    bench._corpus_vectors = dataset.corpus_vectors
    bench._corpus_ids = dataset.corpus_ids
    bench._query_vectors = dataset.query_vectors

    adapter = InMemoryFAISSAdapter(dim=args.dim, index_type=args.index_type)

    print("Running insert benchmark ...")
    insert_result = bench.benchmark_insert("FAISS (in-memory)", adapter)
    print(insert_result.summary())

    print("Running query benchmark ...")
    query_result = bench.benchmark_query("FAISS (in-memory)", adapter, ground_truth=dataset.ground_truth)
    print(query_result.summary())

    bench.print_leaderboard([insert_result, query_result])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
