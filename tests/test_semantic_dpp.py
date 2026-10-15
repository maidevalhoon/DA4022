"""Tests for SemanticDPP (Paper B)."""
import numpy as np
import pytest
import torch
from src.core.semantic_dpp import SemanticDPP, SemanticSubspaceProjector


def test_projector_orthogonality_and_normalization():
    d, p = 512, 128
    projector = SemanticSubspaceProjector(d=d, p=p)

    # Input batch of random vectors
    x = torch.randn(4, d)
    out = projector(x)

    assert out.shape == (4, p)
    # Norms should be 1.0 (unit hypersphere)
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_kernel_matrix_properties():
    dpp = SemanticDPP(hidden_dim=256, subspace_dim=64, alpha=1.0, mock_mode=True)
    embeddings = torch.randn(5, 256)
    K = dpp.compute_kernel_matrix(embeddings)

    assert K.shape == (5, 5)
    # Diagonal must be 1.0 (self-distance is 0 -> exp(0) = 1)
    diag = torch.diag(K)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)
    # Must be symmetric
    assert torch.allclose(K, K.T, atol=1e-5)
    # Off-diagonals must be in (0, 1]
    assert (K > 0.0).all() and (K <= 1.0).all()


def test_uncertainty_score():
    dpp = SemanticDPP(hidden_dim=256, subspace_dim=64, m=4, mock_mode=True)
    res = dpp.uncertainty_score("What is quantum entanglement?")

    assert "log_det" in res
    assert "eigenvalues" in res
    assert "uncertainty" in res
    assert "dpp_c_clusters" in res

    assert len(res["eigenvalues"]) == 4
    # All eigenvalues should be non-negative
    assert (res["eigenvalues"] >= 0.0).all()
    # dpp_c_clusters should be an integer in [1, m]
    assert 1 <= res["dpp_c_clusters"] <= 4
    # Normalized uncertainty in [0, 1]
    assert 0.0 <= res["uncertainty"] <= 1.0


def test_dpp_marginal_loss():
    L = torch.eye(3) + 0.2 * torch.ones(3, 3)
    loss = SemanticDPP.dpp_marginal_loss(L, [0, 1])
    assert not torch.isnan(loss)
    assert isinstance(loss.item(), float)
