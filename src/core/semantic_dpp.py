"""
SemanticDPP and SemanticDPP-C.

Direct implementation of Paper B:
"SemanticDPP: Efficient Uncertainty Quantification in LLMs"
(Anonymous, Google DeepMind — ICLR 2026)

Key Paper Details Implemented:
- SemanticSubspaceProjector: φ: R^d -> R^p (single-layer NN, orthogonal init)
- DPP L-ensemble kernel (Eq. 1 in Paper B):
    K(x_i, x_j) = exp(-α ||φ(x_i) - φ(x_j)||²)
- Likelihood of subset Y (L-ensemble DPP):
    P_L(Y) = det(L_Y) / det(L + I)
- Training objective (Marginal Likelihood maximization):
    log P_L(Y) = log det(L_Y) - log det(L + I)
- Uncertainty score (Eq. 4 in Paper B):
    U(S) = log det(L_S + I) - log det(2I)
- SemanticDPP-C: soft clustering via greedy MAP selection and trace(K_marginal)

UCA Novel Extensions Beyond Paper B:
1. Eigenspectrum Extraction:
   Exposes all eigenvalues {σ_1, ..., σ_m} of L via torch.linalg.eigvalsh.
   In UCA, contested dimensions (high σ_i) require high fidelity, while
   stable dimensions (low σ_i) allow aggressive compression.
2. Cluster Count as Beam Width Floor:
   SemanticDPP-C cluster count k = max(1, round(trace(K_marginal))) directly
   determines the generation beam width floor in UCA.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSubspaceProjector(nn.Module):
    """
    Learned projection φ: R^d -> R^p mapping internal LLM activations
    to a low-dimensional semantic manifold.
    Initialized with orthogonal weights per Paper B Section 3.1.
    Outputs are L2-normalized onto the unit hypersphere.
    """

    def __init__(self, d: int = 4096, p: int = 256):
        super().__init__()
        self.d = d
        self.p = p
        self.proj = nn.Linear(d, p, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects and normalizes input embeddings onto unit sphere."""
        projected = self.proj(x)
        return F.normalize(projected, p=2, dim=-1)


class SemanticDPP:
    """
    SemanticDPP uncertainty estimator + SemanticDPP-C clustering.
    Computes kernel matrix, log-determinant uncertainty, eigenspectrum,
    and semantic cluster count.
    """

    def __init__(
        self,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        projector: Optional[SemanticSubspaceProjector] = None,
        hidden_dim: int = 4096,
        subspace_dim: int = 256,
        alpha: float = 1.0,
        m: int = 3,
        temperature: float = 0.8,
        eps: float = 1e-6,
        device: str = "auto",
        mock_mode: bool = False,
        seed: int = 42,
    ):
        self.alpha = float(alpha)
        self.m = int(m)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.mock_mode = mock_mode
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim

        if projector is not None:
            self.projector = projector.to(self.device)
        else:
            self.projector = SemanticSubspaceProjector(d=hidden_dim, p=subspace_dim).to(self.device)

        self.model = model
        self.tokenizer = tokenizer
        if model is None or mock_mode:
            self.mock_mode = True

    def compute_kernel_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        DPP L-ensemble kernel matrix (Paper B Eq. 1):
            K(x_i, x_j) = exp(-α ||φ(x_i) - φ(x_j)||²)

        Args:
            embeddings: Tensor of shape (m, d)
        Returns:
            K: Tensor of shape (m, m) with entries in (0, 1]
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor (m, d), got shape {embeddings.shape}")

        # Project to semantic manifold and normalize: (m, p)
        projected = self.projector(embeddings.to(self.device))

        # Pairwise squared Euclidean distance: ||u - v||²
        diff = projected.unsqueeze(0) - projected.unsqueeze(1)  # (m, m, p)
        sq_dist = torch.sum(diff ** 2, dim=-1)  # (m, m)

        # RBF similarity kernel
        K = torch.exp(-self.alpha * sq_dist)
        return K

    def compute_marginal_kernel(self, L: torch.Tensor) -> torch.Tensor:
        """
        Computes DPP marginal kernel:
            K_marginal = L (L + I)⁻¹
        """
        m = L.shape[0]
        I = torch.eye(m, device=L.device, dtype=L.dtype)
        K_marg = L @ torch.linalg.inv(L + I)
        return K_marg

    def estimate_clusters_c(self, K: torch.Tensor) -> int:
        """
        SemanticDPP-C cluster count estimation (Paper B Algorithm 1):
            k = max(1, round(trace(K_marginal)))
        """
        m = K.shape[0]
        I = torch.eye(m, device=K.device, dtype=K.dtype)
        L = K + self.eps * I
        K_marg = self.compute_marginal_kernel(L)
        trace_val = torch.trace(K_marg).item()
        k = max(1, int(round(trace_val)))
        return min(k, m)

    def greedy_map_clustering(self, K: torch.Tensor) -> List[int]:
        """
        Greedy MAP selection (Paper B Algorithm 1):
        Iteratively adds items that maximize log det(L_{Y U {j}}).
        Returns list of selected cluster center indices.
        """
        m = K.shape[0]
        I = torch.eye(m, device=K.device, dtype=K.dtype)
        L = K + self.eps * I

        selected = []
        remaining = list(range(m))

        # First item: highest self-similarity / quality
        best_first = int(torch.argmax(torch.diag(L)).item())
        selected.append(best_first)
        remaining.remove(best_first)

        while remaining:
            best_cand = None
            best_det = -float("inf")
            for cand in remaining:
                sub_indices = selected + [cand]
                sub_L = L[np.ix_(sub_indices, sub_indices)]
                det_val = torch.linalg.det(sub_L).item()
                if det_val > best_det:
                    best_det = det_val
                    best_cand = cand

            # Stop if adding items reduces determinant (diversity trade-off reached)
            prev_L = L[np.ix_(selected, selected)]
            prev_det = torch.linalg.det(prev_L).item()
            if best_cand is not None and best_det > prev_det:
                selected.append(best_cand)
                remaining.remove(best_cand)
            else:
                break

        return selected

    def uncertainty_score_from_embeddings(
        self, embeddings: torch.Tensor
    ) -> Dict[str, Union[float, np.ndarray, int, torch.Tensor]]:
        """
        Compute full SemanticDPP uncertainty metrics from pre-extracted embeddings.
        """
        m = embeddings.shape[0]
        K = self.compute_kernel_matrix(embeddings)

        I = torch.eye(m, device=K.device, dtype=K.dtype)
        L = K + self.eps * I

        # Paper B Eq. 4: U(S) = log det(L_S + I) - log det(2I)
        log_det_L_plus_I = torch.logdet(L + I).item()
        log_det_2I = m * math.log(2.0)
        log_det = float(log_det_L_plus_I - log_det_2I)

        # UCA Key Extension 1: Full Eigenspectrum of L
        eigenvalues = torch.linalg.eigvalsh(L)
        eigenvalues_np = eigenvalues.detach().cpu().numpy()
        eigenvalues_np = np.clip(eigenvalues_np, 0.0, None)

        # UCA Key Extension 2: SemanticDPP-C cluster count
        dpp_c_clusters = self.estimate_clusters_c(K)

        # Normalized uncertainty in [0, 1] via sigmoid
        uncertainty_norm = float(torch.sigmoid(torch.tensor(log_det / max(1, m), dtype=torch.float32)).item())

        return {
            "log_det": log_det,
            "eigenvalues": eigenvalues_np,
            "eigenvalues_desc": np.sort(eigenvalues_np)[::-1],
            "kernel_matrix": K.detach().cpu(),
            "uncertainty": uncertainty_norm,
            "dpp_c_clusters": dpp_c_clusters,
            "trace_k": float(torch.trace(K).item()),
        }

    def _sample_embeddings(self, query: str, difficulty: Optional[str] = None) -> torch.Tensor:
        """Sample m responses using the LLM and extract mean-pooled last-layer hidden states."""
        if self.mock_mode:
            return self._sample_mock_embeddings(query, difficulty=difficulty)

        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        sampled_vecs = []

        with torch.no_grad():
            for _ in range(self.m):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=self.temperature,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                step_states = [step[-1][:, -1, :] for step in out.hidden_states]
                stacked = torch.stack(step_states, dim=1)  # (1, n_tokens, hidden_dim)
                mean_embedding = stacked.mean(dim=1).squeeze(0)  # (hidden_dim,)
                sampled_vecs.append(mean_embedding)

        return torch.stack(sampled_vecs, dim=0)  # (m, hidden_dim)

    def _sample_mock_embeddings(self, query: str, difficulty: Optional[str] = None) -> torch.Tensor:
        """
        Generates realistic synthetic embeddings for testing.
        Simulates:
        - Confident queries: samples clustered close together (high kernel, low log_det).
        - Uncertain queries: samples spread across diverse directions (low kernel, high log_det).
        """
        q_hash = abs(hash(query)) % 1000000
        rng = np.random.default_rng(q_hash)

        q_lower = query.lower()
        if difficulty == "easy":
            cluster_dispersion = 0.08  # Very tight cluster (confident)
        elif difficulty == "hard":
            cluster_dispersion = 0.95  # Diverse cluster (uncertain)
        elif difficulty == "medium":
            cluster_dispersion = 0.42  # Moderate dispersion
        elif any(w in q_lower for w in ["what is", "capital", "who", "when"]):
            cluster_dispersion = 0.15  # Low dispersion (confident)
        elif any(w in q_lower for w in ["prove", "solve", "why", "difference"]):
            cluster_dispersion = 0.85  # High dispersion (uncertain)
        else:
            cluster_dispersion = 0.50

        # Base semantic direction
        base_dir = rng.normal(0, 1, size=(self.hidden_dim,))
        base_dir = base_dir / np.linalg.norm(base_dir)

        embeddings = []
        scale = cluster_dispersion / np.sqrt(self.hidden_dim)
        for _ in range(self.m):
            noise = rng.normal(0, scale, size=(self.hidden_dim,))
            vec = base_dir + noise
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        tensor_emb = torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)
        return tensor_emb

    def uncertainty_score(
        self, query: str, difficulty: Optional[str] = None
    ) -> Dict[str, Union[float, np.ndarray, int, torch.Tensor]]:
        """
        Main end-to-end SemanticDPP uncertainty computation.
        """
        embeddings = self._sample_embeddings(query, difficulty=difficulty)
        return self.uncertainty_score_from_embeddings(embeddings)

    @staticmethod
    def dpp_marginal_loss(
        L: torch.Tensor,
        cluster_indices: List[int],
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        DPP Marginal Likelihood Loss for training projector φ (Paper B Section 3.1):
            L_DPP = - (log det(L_Y) - log det(L + I))
        Minimizing this loss maximizes the likelihood of observing cluster subset Y.
        """
        m = L.shape[0]
        I = torch.eye(m, device=L.device, dtype=L.dtype)
        reg_L = L + eps * I

        if not cluster_indices:
            sub_logdet = torch.tensor(0.0, device=L.device)
        else:
            idx = torch.tensor(cluster_indices, device=L.device, dtype=torch.long)
            sub_L = reg_L[idx][:, idx]
            sub_logdet = torch.logdet(sub_L)

        total_logdet = torch.logdet(reg_L + I)
        return -(sub_logdet - total_logdet)
