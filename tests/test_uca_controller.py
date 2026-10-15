"""Tests for UCA Controller (Novel Contribution)."""
import numpy as np
import pytest
from src.core.internal_confidence import ICGate
from src.core.semantic_dpp import SemanticDPP
from src.core.uca_controller import UCAController


@pytest.fixture
def uca():
    ic = ICGate(mock_mode=True, tau_high=0.65, tau_low=0.30)
    dpp = SemanticDPP(mock_mode=True, m=3)
    return UCAController(
        ic_gate=ic,
        semantic_dpp=dpp,
        tau_high=0.65,
        tau_low=0.30,
        r_min=0.10,
        r_max=0.95,
        mock_mode=True,
    )


def test_fast_path_allocation(uca):
    # Easy query with high confidence
    res = uca.compute_compression_config("What is the capital of France?", difficulty="easy")
    assert res["route"] == "fast_path"
    assert res["kv_budget"] == 0.25
    assert res["beam_width"] == 1
    assert res["trigger_rag"] is False
    assert res["tier_used"] == "tier_0"
    assert len(res["compression_ratios"]) == 3
    assert np.allclose(res["compression_ratios"], 0.10)


def test_abstain_path_allocation(uca):
    # Hard query with very low confidence
    res = uca.compute_compression_config("Prove Fermat's Last Theorem in one line", difficulty="hard")
    assert res["route"] == "abstain"
    assert res["kv_budget"] == 1.00
    assert res["beam_width"] == 4
    assert res["trigger_rag"] is True
    assert res["tier_used"] == "tier_0"
    assert len(res["compression_ratios"]) == 3
    assert np.allclose(res["compression_ratios"], 0.95)


def test_uncertain_cascade_allocation(uca):
    # Medium query falling into uncertain zone (0.30 <= IC < 0.65)
    res = uca.compute_compression_config("Explain how transformers scale with sequence length", difficulty="medium")
    assert res["route"] in ["standard", "augmented"]
    assert 0.25 <= res["kv_budget"] <= 1.00
    assert 1 <= res["beam_width"] <= 4
    assert res["tier_used"] == "tier_0_and_tier_1"
    assert res["dpp_eigenvalues"] is not None
    assert len(res["compression_ratios"]) == 3
    assert (res["compression_ratios"] >= 0.10).all()
    assert (res["compression_ratios"] <= 0.95).all()


def test_continuous_budget_monotonicity(uca):
    # Testing mathematical formula KV_budget = 0.25 + 0.75 * UCA
    b_low = uca._compute_kv_budget(0.0)
    b_mid = uca._compute_kv_budget(0.5)
    b_high = uca._compute_kv_budget(1.0)
    assert b_low == 0.25
    assert np.isclose(b_mid, 0.625)
    assert b_high == 1.00
    assert b_low < b_mid < b_high
