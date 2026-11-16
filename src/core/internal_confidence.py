"""
Internal Confidence (IC) - Paper A, "Query-Level Uncertainty in Large Language Models"
(Chen, de Melo, Suchanek, Varoquaux, ICLR 2026).

Idea: before generating anything, ask the model "can you answer this? Yes or No"
and read how strongly its hidden states point to "Yes".  This is done at every
layer and at the last few tokens, then averaged with weights that are largest
at the last layer / last token and decay quickly away from it.

The forward pass itself lives in llm.py (LLM.yes_no_confidence), which returns
the matrix  p_yes[layer, token].  This file only does the aggregation, so it is
plain numpy and easy to test.
"""

import numpy as np


def attenuated_weights(n: int, alpha: float = 1.0) -> np.ndarray:
    """
    Attenuated Encoding weights (Paper A, Eq. 4) centred on the LAST position:

        w_j = exp(-alpha * |center - j|^2) / sum_j exp(-alpha * |center - j|^2)

    With alpha = 1 (the paper's setting) the last position gets ~72% of the
    weight, its neighbour ~27%, the next ~1%: a soft "look mostly at the end".
    """
    distance = (n - 1) - np.arange(n)
    weights = np.exp(-alpha * distance.astype(float) ** 2)
    return weights / weights.sum()


def internal_confidence(p_yes: np.ndarray, alpha: float = 1.0) -> float:
    """
    IC = sum over layers l and tokens n of  w_layer[l] * w_token[n] * P(Yes | h_n^l)
    (Paper A, Eq. 3).  p_yes has shape [n_layers, n_tokens].
    """
    layer_weights = attenuated_weights(p_yes.shape[0], alpha)
    token_weights = attenuated_weights(p_yes.shape[1], alpha)
    return float(layer_weights @ p_yes @ token_weights)


def p_yes_top_right(p_yes: np.ndarray) -> float:
    """Baseline from Paper A: P(Yes) at the last layer and last token only (Eq. 1)."""
    return float(p_yes[-1, -1])


def p_yes_naive_average(p_yes: np.ndarray) -> float:
    """Baseline from Paper A: plain average over all layers and tokens."""
    return float(p_yes.mean())
