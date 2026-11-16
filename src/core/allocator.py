"""
Uncertainty-Calibrated Allocation (UCA) - our contribution.

The setting.  A question can be answered at several memory levels: 0, 1, 2, 4 or
8 retrieved passages in the prompt.  More passages cost more KV-cache memory
and usually help.  Memory is limited, so not every question can have the
maximum.  Who should get how much?

Two ways to use uncertainty for this are implemented here.

A) ESCALATION (the main method).
   Start every question with no passages.  After each level, read the model's
   confidence.  If the model is confident enough, answer; otherwise give it the
   next level of memory and check again, up to a maximum level.
   Two knobs: the confidence threshold and the maximum level.

B) ONE-SHOT ALLOCATION (the simpler variant we tried first).
   Read the confidence once, before retrieving anything, and decide the level
   for every question from that single number.

Both need the raw confidence score to be turned into "probability that the
answer is correct".  That is what calibration does, and it is learned from a
held-out set of questions.  Nothing in this file touches the model.
"""

from typing import List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


# --------------------------------------------------------------------------
# Calibration: raw confidence score -> probability of a correct answer
# --------------------------------------------------------------------------
class Calibrator:
    """
    Fits a rising staircase through (score, was the answer correct?) pairs.
    This is isotonic regression: the only assumption is "a higher score never
    means a lower chance of being right".
    """

    def __init__(self):
        self.curve = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, confidence: np.ndarray, correct: np.ndarray) -> "Calibrator":
        self.curve.fit(np.asarray(confidence, float), np.asarray(correct, float))
        return self

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        return self.curve.predict(np.asarray(confidence, float))


# --------------------------------------------------------------------------
# A) Escalation
# --------------------------------------------------------------------------
def fit_calibrators(confidence: np.ndarray, correct: np.ndarray) -> List[Calibrator]:
    """
    One calibration curve per check level c:
        confidence read at level c  ->  P(the answer at level c is correct).
    confidence: [n, n_checks]   correct: [n, n_levels]   (calibration questions)
    """
    return [Calibrator().fit(confidence[:, c], correct[:, c]) for c in range(confidence.shape[1])]


def probability_correct(calibrators: List[Calibrator], confidence: np.ndarray) -> np.ndarray:
    """Apply the curves: confidence [m, n_checks] -> P(correct) [m, n_checks]."""
    return np.column_stack([calibrators[c].predict(confidence[:, c])
                            for c in range(len(calibrators))])


def escalate(p_correct: np.ndarray, threshold: float, max_level: int) -> np.ndarray:
    """
    p_correct[i, c] = calibrated probability that question i is answered
    correctly at level c.

    Each question stops at the first level where p_correct >= threshold.
    A question that is never confident stops at max_level anyway.
    Returns the final level index for every question.
    """
    confident = p_correct[:, :max_level] >= threshold
    return np.where(confident.any(axis=1), confident.argmax(axis=1), max_level)


# --------------------------------------------------------------------------
# B) One-shot allocation
# --------------------------------------------------------------------------
def fit_gain_table(uncertainty: np.ndarray, correct: np.ndarray,
                   n_bins: int = 5, smoothing: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    uncertainty: [n]            score per calibration question (higher = less sure)
    correct:     [n, n_levels]  1 if the answer at that memory level was right

    Sort questions into bins by uncertainty and measure the accuracy of every
    memory level inside every bin.  Returns (bin_edges, table [bins, levels]).
    Each cell is a smoothed mean so a small bin cannot give a wild estimate.
    """
    edges = np.unique(np.quantile(uncertainty, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.digitize(uncertainty, edges, right=True)
    overall = correct.mean(axis=0)
    table = np.zeros((len(edges) + 1, correct.shape[1]))
    for b in range(len(edges) + 1):
        members = correct[bins == b]
        table[b] = (members.sum(axis=0) + smoothing * overall) / (len(members) + smoothing)
    return edges, table


def allocate(uncertainty: np.ndarray, edges: np.ndarray, table: np.ndarray,
             level_cost: np.ndarray, budget_per_question: float) -> np.ndarray:
    """
    Start everyone at the cheapest level, then keep buying the upgrade with the
    best (expected accuracy gain) / (extra tokens) until the budget is spent.
    This greedy rule is the standard way to solve this kind of knapsack problem.
    Returns a level index for every question.
    """
    n = len(uncertainty)
    bins = np.digitize(uncertainty, edges, right=True)
    level = np.zeros(n, dtype=int)
    remaining = budget_per_question * n - level_cost[0] * n
    exhausted = set()

    while True:
        best, best_ratio = None, 0.0
        for b in range(table.shape[0]):
            for lo in np.unique(level[bins == b]):
                for hi in range(lo + 1, table.shape[1]):
                    if (b, lo, hi) in exhausted:
                        continue
                    ratio = (table[b, hi] - table[b, lo]) / (level_cost[hi] - level_cost[lo])
                    if ratio > best_ratio:
                        best, best_ratio = (b, lo, hi), ratio
        if best is None:
            break                                   # nothing left that is expected to help

        b, lo, hi = best
        step_cost = level_cost[hi] - level_cost[lo]
        candidates = np.where((bins == b) & (level == lo))[0]
        candidates = candidates[np.argsort(-uncertainty[candidates])]   # least sure first
        chosen = candidates[: int(remaining // step_cost)]
        level[chosen] = hi
        remaining -= step_cost * len(chosen)
        if len(chosen) < len(candidates):
            exhausted.add(best)                     # could not afford everyone; try cheaper upgrades
    return level


# --------------------------------------------------------------------------
# Reference points
# --------------------------------------------------------------------------
def best_fixed_mix(accuracy: np.ndarray, cost: np.ndarray, budget: float) -> List[Tuple[int, float]]:
    """
    The best policy that ignores uncertainty: give a random share of the
    questions one fixed level and the rest another, so that the average cost
    equals the budget.  Returns [(level, share of questions), ...].
    """
    accuracy, cost = np.asarray(accuracy, float), np.asarray(cost, float)
    affordable = np.flatnonzero(cost <= budget + 1e-9)
    if len(affordable) == 0:
        return [(int(np.argmin(cost)), 1.0)]
    single = affordable[np.argmax(accuracy[affordable])]
    best_value, best = accuracy[single], [(int(single), 1.0)]
    for b in np.flatnonzero(cost > budget + 1e-9):
        share_b = (budget - cost[affordable]) / (cost[b] - cost[affordable])
        value = (1 - share_b) * accuracy[affordable] + share_b * accuracy[b]
        k = int(np.argmax(value))
        if value[k] > best_value + 1e-12:
            best_value = value[k]
            best = [(int(affordable[k]), float(1 - share_b[k])), (int(b), float(share_b[k]))]
    return best


def fixed_mix_accuracy(accuracy: np.ndarray, cost: np.ndarray, budget: float) -> float:
    """Accuracy of the best uncertainty-blind policy at a given average memory."""
    return float(sum(share * accuracy[level] for level, share in best_fixed_mix(accuracy, cost, budget)))


def allocate_oracle(correct: np.ndarray, level_cost: np.ndarray,
                    budget_per_question: float) -> np.ndarray:
    """
    Upper bound that cheats by looking at the answers: give each question the
    cheapest level at which it is correct, cheapest fixes first.
    """
    n = len(correct)
    level = np.zeros(n, dtype=int)
    remaining = budget_per_question * n - level_cost[0] * n
    fixable = [(int(np.argmax(row)), i) for i, row in enumerate(correct)
               if not row[0] and row.any()]
    for cheapest_level, i in sorted(fixable):
        step_cost = level_cost[cheapest_level] - level_cost[0]
        if step_cost <= remaining:
            level[i] = cheapest_level
            remaining -= step_cost
    return level
