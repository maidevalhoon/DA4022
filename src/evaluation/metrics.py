"""
Evaluation Metrics for Uncertainty Quantification and Adaptive Allocation.

Implements standard benchmark metrics from Paper A (Chen et al.) and Paper B (Hamer et al.):
1. AUROC (Area Under ROC Curve)
2. PRR (Prediction Rejection Ratio) — Paper A Table 1 metric
3. ECE (Expected Calibration Error)
4. Memory Savings & KV Token Ratios
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(
    correctness_labels: Union[List[int], np.ndarray],
    uncertainty_scores: Union[List[float], np.ndarray],
    invert: bool = False,
) -> float:
    """
    Compute AUROC for uncertainty scores predicting incorrectness.

    Args:
        correctness_labels: 1 if model answer is correct, 0 if incorrect.
        uncertainty_scores: Higher score = more uncertain / likely wrong.
        invert: If True, treats uncertainty_scores as confidence (high = likely correct).

    Returns:
        auroc: float in [0.0, 1.0] (multiplied by 100 for percentage reporting).
    """
    y_true = np.asarray(correctness_labels, dtype=int)
    scores = np.asarray(uncertainty_scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5  # Degenerate case when all items have identical label

    # Invert if score is confidence (we want positive class = incorrect query)
    # So high uncertainty -> predicts incorrect label (1 - y_true)
    y_incorrect = 1 - y_true
    if invert:
        scores = -scores

    try:
        score_val = float(roc_auc_score(y_incorrect, scores))
    except ValueError:
        score_val = 0.5

    return score_val


def compute_prr(
    correctness_labels: Union[List[int], np.ndarray],
    uncertainty_scores: Union[List[float], np.ndarray],
    rejection_rates: Optional[List[float]] = None,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute Prediction Rejection Ratio (PRR) as defined in Paper A.
    Measures how quickly error drops when high-uncertainty queries are rejected.

    PRR = (Area under Error-Rejection Curve - Oracle Area) normalized,
    or relative accuracy improvement across rejection fractions [0.1, 0.2, ..., 0.9].

    Returns:
        dict with:
            prr: float (0 to 100%)
            rejection_rates: list of rejection fractions
            accuracies: list of accuracies at each rejection rate
            error_rates: list of error rates at each rejection rate
    """
    y_correct = np.asarray(correctness_labels, dtype=int)
    scores = np.asarray(uncertainty_scores, dtype=float)
    n = len(y_correct)

    if n == 0:
        return {"prr": 0.0, "rejection_rates": [], "accuracies": [], "error_rates": []}

    base_acc = float(np.mean(y_correct))
    base_err = 1.0 - base_acc

    if rejection_rates is None:
        rejection_rates = [round(r, 2) for r in np.linspace(0.05, 0.80, 16)]

    # Sort queries from lowest uncertainty to highest uncertainty
    # Queries to keep first are the ones with lowest uncertainty
    sort_order = np.argsort(scores)  # ascending: low uncertainty -> high uncertainty

    accuracies = []
    error_rates = []

    for r in rejection_rates:
        keep_n = max(1, int(round((1.0 - r) * n)))
        kept_indices = sort_order[:keep_n]
        acc_at_r = float(np.mean(y_correct[kept_indices]))
        accuracies.append(acc_at_r)
        error_rates.append(1.0 - acc_at_r)

    # Compute PRR as area of accuracy gain over base accuracy
    acc_gains = [max(0.0, a - base_acc) for a in accuracies]
    max_possible_gain = max(1e-6, 1.0 - base_acc)
    norm_gain = float(np.mean(acc_gains) / max_possible_gain * 100.0)

    return {
        "prr": norm_gain,
        "base_accuracy": base_acc,
        "rejection_rates": rejection_rates,
        "accuracies": accuracies,
        "error_rates": error_rates,
    }


def compute_ece(
    correctness_labels: Union[List[int], np.ndarray],
    confidence_scores: Union[List[float], np.ndarray],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures calibration between confidence scores and empirical correctness.
    """
    y_true = np.asarray(correctness_labels, dtype=int)
    conf = np.asarray(confidence_scores, dtype=float)
    n = len(y_true)

    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(conf[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_memory_savings(
    allocated_budgets: Union[List[float], np.ndarray],
    baseline_budget: float = 1.0,
) -> Dict[str, float]:
    """
    Computes percentage memory savings relative to static baseline budget.
    """
    budgets = np.asarray(allocated_budgets, dtype=float)
    mean_budget = float(np.mean(budgets))
    savings_pct = float(max(0.0, (1.0 - mean_budget / baseline_budget) * 100.0))
    return {
        "mean_allocated_budget": mean_budget,
        "baseline_budget": baseline_budget,
        "savings_percentage": savings_pct,
    }
