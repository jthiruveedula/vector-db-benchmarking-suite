from __future__ import annotations

from vectordb_bench.benchmark import BenchmarkConfig, VectorDBBenchmarker


def make_benchmarker(n_vectors=50, vector_dim=8, n_queries=20, top_k=5, n_threads=2) -> VectorDBBenchmarker:
    config = BenchmarkConfig(
        n_vectors=n_vectors, vector_dim=vector_dim, n_queries=n_queries,
        top_k=top_k, n_threads=n_threads, warm_up_queries=2, seed=1,
    )
    return VectorDBBenchmarker(config)


def test_benchmark_insert_returns_expected_fields(fake_adapter):
    bench = make_benchmarker()
    result = bench.benchmark_insert("fake-db", fake_adapter, batch_size=10)

    assert result.db_name == "fake-db"
    assert result.operation == "insert"
    assert result.qps > 0
    assert result.p50_ms >= 0
    assert result.cost_per_1k_queries_usd == 0.0
    assert fake_adapter.count() == 50


def test_benchmark_query_without_ground_truth_has_recall_one(fake_adapter):
    bench = make_benchmarker()
    bench.benchmark_insert("fake-db", fake_adapter, batch_size=10)
    result = bench.benchmark_query("fake-db", fake_adapter)

    assert result.operation == "query"
    assert result.n_queries == 20
    assert result.recall_at_k == 1.0
    assert result.qps > 0
    assert result.p50_ms <= result.p95_ms <= result.p99_ms


def test_benchmark_query_with_perfect_ground_truth_has_recall_one(fake_adapter):
    bench = make_benchmarker(n_queries=5, top_k=3)
    bench.benchmark_insert("fake-db", fake_adapter, batch_size=10)

    # Build ground truth from the adapter's own (exact) results since FakeInMemoryAdapter is brute-force.
    ground_truth = {
        i: fake_adapter.query(bench._query_vectors[i], 3) for i in range(5)
    }
    result = bench.benchmark_query("fake-db", fake_adapter, ground_truth=ground_truth)
    assert result.recall_at_k == 1.0


def test_benchmark_query_with_wrong_ground_truth_has_low_recall(fake_adapter):
    bench = make_benchmarker(n_queries=5, top_k=3)
    bench.benchmark_insert("fake-db", fake_adapter, batch_size=10)

    ground_truth = {i: ["nonexistent_1", "nonexistent_2", "nonexistent_3"] for i in range(5)}
    result = bench.benchmark_query("fake-db", fake_adapter, ground_truth=ground_truth)
    assert result.recall_at_k == 0.0


def test_print_leaderboard_runs_without_error(fake_adapter, capsys):
    bench = make_benchmarker()
    insert_result = bench.benchmark_insert("fake-db", fake_adapter, batch_size=10)
    query_result = bench.benchmark_query("fake-db", fake_adapter)

    bench.print_leaderboard([insert_result, query_result])
    captured = capsys.readouterr()
    assert "fake-db" in captured.out
