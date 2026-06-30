"""Thin, task-focused wrappers around VectorDBBenchmarker."""

from vectordb_bench.benchmarks.recall_benchmark import run_recall_benchmark
from vectordb_bench.benchmarks.latency_benchmark import run_latency_benchmark
from vectordb_bench.benchmarks.throughput_benchmark import run_throughput_benchmark

__all__ = ["run_recall_benchmark", "run_latency_benchmark", "run_throughput_benchmark"]
