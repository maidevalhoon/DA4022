"""Tests for Internal Confidence Gate (Paper A)."""
import numpy as np
import pytest
from src.core.internal_confidence import ICGate


def test_attenuated_weights_properties():
    n_layers = 16
    n_tokens = 20
    weights = ICGate.compute_attenuated_weights(n_layers, n_tokens, alpha=1.0)

    # Must sum to 1.0
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    # Peak must be at last layer, last token (bottom-right cell)
    assert np.argmax(weights) == weights.size - 1
    assert weights[-1, -1] == np.max(weights)
    # Weight at (0, 0) should be strictly less than at (-1, -1)
    assert weights[0, 0] < weights[-1, -1]


def test_locality_score():
    # Delta function at center should have Loc = 1.0
    w_delta = np.zeros(10)
    w_delta[9] = 1.0
    loc_delta = ICGate.locality_score(w_delta, 9)
    assert np.isclose(loc_delta, 1.0)

    # Uniform weights should have lower locality
    w_uniform = np.ones(10) / 10.0
    loc_uniform = ICGate.locality_score(w_uniform, 9)
    assert 0.0 < loc_uniform < loc_delta


def test_ic_gate_score_mock():
    gate = ICGate(mock_mode=True, alpha=1.0, tau_high=0.65, tau_low=0.30)
    res = gate.score("What is the capital of France?")

    assert 0.0 <= res["ic_score"] <= 1.0
    assert 0.0 <= res["top_right_score"] <= 1.0
    assert 0.0 <= res["locality"] <= 1.0
    assert res["route"] in ["fast_path", "uncertain", "abstain"]
    assert res["p_yes_matrix"].ndim == 2


def test_ic_gate_routing():
    gate = ICGate(mock_mode=True, tau_high=0.70, tau_low=0.30)

    # Test route helper
    route_name, score_val = gate.route("Test query")
    assert route_name in ["fast_path", "uncertain", "abstain"]
    assert 0.0 <= score_val <= 1.0
