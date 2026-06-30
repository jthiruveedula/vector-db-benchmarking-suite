"""Dataset generation and loading utilities for benchmark corpora."""

from vectordb_bench.datasets.synthetic_generator import SyntheticDataset, generate_synthetic_dataset
from vectordb_bench.datasets.dataset_loader import load_dataset

__all__ = ["SyntheticDataset", "generate_synthetic_dataset", "load_dataset"]
