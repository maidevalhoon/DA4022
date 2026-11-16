"""
Metrics used by both reference papers to judge an uncertainty score.

Convention everywhere: `correct` is 1 if the model's answer was right,
`uncertainty` is higher when the model is LESS sure.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """
    Probability that a wrong answer gets a higher uncertainty than a right one.
    0.5 = useless score, 1.0 = perfect.
    """
    wrong = 1 - np.asarray(correct, dtype=int)
    if wrong.min() == wrong.max():
        return float("nan")                 # all right or all wrong: AUROC undefined
    return float(roc_auc_score(wrong, uncertainty))


def _rejection_area(correct: np.ndarray, order: np.ndarray) -> float:
    """Reject questions in `order`; average the accuracy of what is kept."""
    kept = correct[order[::-1]]             # rejected-last first
    accuracy_of_kept = np.cumsum(kept) / np.arange(1, len(kept) + 1)
    return float(accuracy_of_kept.mean())


def prr(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """
    Prediction Rejection Ratio (Malinin et al., used in Paper A).
    We reject the most uncertain questions first and watch accuracy rise.
        PRR = (area with our score - area random) / (area oracle - area random)
    0 = no better than rejecting at random, 1 = as good as knowing the labels.
    """
    correct = np.asarray(correct, dtype=float)
    ours = _rejection_area(correct, np.argsort(-np.asarray(uncertainty), kind="stable"))
    oracle = _rejection_area(correct, np.argsort(correct, kind="stable"))
    random = float(correct.mean())
    if oracle == random:
        return float("nan")
    return (ours - random) / (oracle - random)


def ece(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error: does "70% confident" mean right 70% of the time?
    `confidence` must be in [0, 1].  0 = perfectly calibrated.
    """
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)
