"""Tests for Evaluation Metrics."""
import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_auroc,
    compute_prr,
    compute_ece,
    compute_memory_savings,
)


def test_compute_auroc():
    # Perfect predictor: high uncertainty for incorrect, low for correct
    correctness = [1, 1, 1, 0, 0, 0]
    uncertainty = [0.1, 0.2, 0.15, 0.8, 0.9, 0.85]
    auroc = compute_auroc(correctness, uncertainty)
    assert auroc == 1.0

    # Inverted confidence
    confidence = [0.9, 0.8, 0.85, 0.2, 0.1, 0.15]
    auroc_inv = compute_auroc(correctness, confidence, invert=True)
    assert auroc_inv == 1.0


def test_compute_prr():
    correctness = [1, 1, 1, 1, 0, 0, 0, 0]
    uncertainty = [0.1, 0.2, 0.25, 0.3, 0.7, 0.8, 0.85, 0.9]
    res = compute_prr(correctness, uncertainty)

    assert "prr" in res
    assert res["prr"] > 0.0
    assert len(res["accuracies"]) > 0
    # Accuracies should increase as uncertain samples are rejected
    assert res["accuracies"][-1] >= res["accuracies"][0]


def test_compute_ece():
    correctness = [1, 1, 0, 0]
    confidence = [0.9, 0.8, 0.2, 0.1]
    ece = compute_ece(correctness, confidence)
    assert 0.0 <= ece <= 1.0


def test_compute_memory_savings():
    budgets = [0.25, 0.25, 0.50, 1.00]
    res = compute_memory_savings(budgets, baseline_budget=1.0)
    assert np.isclose(res["mean_allocated_budget"], 0.50)
    assert np.isclose(res["savings_percentage"], 50.0)
