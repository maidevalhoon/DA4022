"""Evaluation package for benchmarks and metrics."""
from .metrics import (
    compute_auroc,
    compute_prr,
    compute_ece,
    compute_memory_savings,
)
from .benchmarks import load_benchmark

__all__ = [
    "compute_auroc",
    "compute_prr",
    "compute_ece",
    "compute_memory_savings",
    "load_benchmark",
]
