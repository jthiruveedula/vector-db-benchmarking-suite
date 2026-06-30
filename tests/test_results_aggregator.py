from __future__ import annotations

import json

from vectordb_bench.analysis.results_aggregator import (
    aggregate_results,
    load_results_from_dir,
    save_results,
)
from vectordb_bench.benchmark import BenchmarkResult


def make_result(db_name="faiss", operation="query", qps=100.0, p99=5.0, recall=0.9, cost=0.0):
    return BenchmarkResult(
        db_name=db_name, operation=operation, n_queries=10, qps=qps,
        p50_ms=1.0, p95_ms=3.0, p99_ms=p99, avg_ms=2.0,
        recall_at_k=recall, cost_per_1k_queries_usd=cost, config={"top_k": 10},
    )


def test_aggregate_groups_by_db_and_operation():
    results = [make_result(qps=100), make_result(qps=200), make_result(db_name="other", qps=50)]
    summary = aggregate_results(results)

    assert "faiss:query" in summary
    assert "other:query" in summary
    assert summary["faiss:query"]["n_runs"] == 2
    assert summary["faiss:query"]["mean_qps"] == 150
    assert summary["faiss:query"]["best_qps"] == 200


def test_save_and_load_results_round_trip(tmp_path):
    results = [make_result(qps=123.0), make_result(db_name="other", qps=456.0)]
    save_results(results, tmp_path / "run1.json")

    loaded = load_results_from_dir(tmp_path)
    assert len(loaded) == 2
    assert {r.db_name for r in loaded} == {"faiss", "other"}


def test_load_results_from_dir_handles_single_object_json(tmp_path):
    result = make_result()
    (tmp_path / "single.json").write_text(json.dumps(result.__dict__))

    loaded = load_results_from_dir(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].db_name == "faiss"
