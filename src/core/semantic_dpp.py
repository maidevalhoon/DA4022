"""
SemanticDPP - Paper B, "SemanticDPP: Efficient Uncertainty Quantification in LLMs"
(anonymous submission, under review at ICLR 2026).

Idea: sample a few answers to the same question.  If they all mean the same
thing the model is sure; if they scatter, it is guessing.  "Scatter" is measured
with a determinantal point process (DPP) on the answers' embeddings:

    kernel      L[i, j] = exp(-alpha * || phi(x_i) - phi(x_j) ||^2)          (Eq. 1)
    training    maximise  log det(L_Y) - log det(L + I)                      (Eq. 2-3)
                where Y holds one answer from each distinct-meaning cluster
    uncertainty log det(K_S),   K = L (L + I)^-1                             (Eq. 4)

phi is a small one-hidden-layer network that learns which directions of the
embedding space carry meaning.  Higher log det = more diverse answers = more
uncertain.

Two honest simplifications compared with the paper (both stated in the report):
  * Pseudo-labels: the paper clusters sampled answers with an entailment model.
    TriviaQA answers are a few words long, so we cluster by normalised string
    match instead (cluster_answers below).  No second model needed.
  * Whitening: we whiten with PCA to a few dozen dimensions because we only
    have a few hundred calibration questions.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.data import normalize


# --------------------------------------------------------------------------
# Pseudo-labels: which sampled answers say the same thing?
# --------------------------------------------------------------------------
def cluster_answers(answers: List[str]) -> List[int]:
    """
    "Leader" clustering (the algorithm Paper B uses, Section 2.1), with a string
    test instead of an entailment model: two answers match if one contains the
    other after normalisation ("Paris" ~ "paris, france").
    Returns one cluster id per answer.
    """
    leaders, labels = [], []
    for answer in answers:
        text = normalize(answer)
        for cluster_id, leader in enumerate(leaders):
            if text == leader or (text and leader and (text in leader or leader in text)):
                labels.append(cluster_id)
                break
        else:
            leaders.append(text)
            labels.append(len(leaders) - 1)
    return labels


def lexical_diversity(answers: List[str]) -> float:
    """Simple baseline: entropy of the string-match clusters (a cheap 'semantic entropy')."""
    counts = np.bincount(cluster_answers(answers)).astype(float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------
# The DPP
# --------------------------------------------------------------------------
class SemanticDPP:
    def __init__(self, pca_dim: int = 32, hidden: int = 64, out_dim: int = 8,
                 alpha: float = 1.0, jitter: float = 1e-3,
                 steps: int = 200, lr: float = 1e-2, weight_decay: float = 1e-3,
                 trainable: bool = True, seed: int = 0):
        self.pca_dim, self.alpha, self.jitter = pca_dim, alpha, jitter
        self.steps, self.lr, self.weight_decay = steps, lr, weight_decay
        self.trainable = trainable          # False -> phi is the identity (untrained baseline)
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(pca_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.mean = self.components = self.scale = None

    # ------------------------------------------------------------- whitening
    def _fit_whitening(self, flat: np.ndarray):
        self.mean = flat.mean(axis=0)
        _, singular, vt = np.linalg.svd(flat - self.mean, full_matrices=False)
        self.components = vt[: self.pca_dim].T                          # [d, pca_dim]
        self.scale = singular[: self.pca_dim] / np.sqrt(len(flat) - 1)   # std of each component

    def _whiten(self, embeddings: np.ndarray) -> torch.Tensor:
        z = (embeddings.astype(np.float64) - self.mean) @ self.components / self.scale
        return torch.tensor(z, dtype=torch.float32)

    # ---------------------------------------------------------------- kernel
    def _kernel(self, z: torch.Tensor) -> torch.Tensor:
        """L-ensemble kernel for a batch of answer sets.  z: [..., m, pca_dim] -> [..., m, m]."""
        if self.trainable:
            z = self.phi(z)
        else:
            z = z / np.sqrt(z.shape[-1])    # identity phi; keep distances on a sane scale
        L = torch.exp(-self.alpha * torch.cdist(z, z) ** 2)
        return L + self.jitter * torch.eye(z.shape[-2])

    # ------------------------------------------------------------------- fit
    def fit(self, embeddings: np.ndarray, clusters: np.ndarray) -> List[float]:
        """
        embeddings: [n_questions, m, d]   sampled-answer embeddings
        clusters:   [n_questions, m]      pseudo-label cluster id of each answer
        Returns the training loss curve.
        """
        n, m, d = embeddings.shape
        self._fit_whitening(embeddings.reshape(n * m, d))
        if not self.trainable:
            return []

        z = self._whiten(embeddings)
        clusters = torch.tensor(clusters)
        eye = torch.eye(m)
        optimiser = torch.optim.Adam(self.phi.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        losses = []
        for _ in range(self.steps):
            # Y = one random representative per cluster, encoded as a 0/1 mask.
            random_rank = torch.rand(n, m)
            best_in_cluster = torch.zeros(n, m).scatter_reduce(
                1, clusters, random_rank, reduce="amax", include_self=False)
            mask = (random_rank == best_in_cluster.gather(1, clusters)).float()

            L = self._kernel(z)
            # log det of the sub-matrix L_Y, written so it can be batched:
            # keep rows/cols in Y, put 1 on the diagonal for the rest.
            L_Y = mask.unsqueeze(2) * L * mask.unsqueeze(1) + torch.diag_embed(1 - mask)
            log_likelihood = torch.logdet(L_Y) - torch.logdet(L + eye)      # Eq. 2
            loss = -log_likelihood.mean()                                   # Eq. 3

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        return losses

    # ----------------------------------------------------------------- score
    @torch.no_grad()
    def _eigenvalues(self, embeddings: np.ndarray) -> torch.Tensor:
        L = self._kernel(self._whiten(embeddings))
        return torch.linalg.eigvalsh(L).clamp(min=self.jitter)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Uncertainty = log det(K_S) with K = L (L + I)^-1                    (Eq. 4)
                    = sum_i log( lambda_i / (1 + lambda_i) )   over eigenvalues of L.
        Accepts [m, d] or a batch [n, m, d].  Higher = more uncertain.
        """
        lam = self._eigenvalues(embeddings)
        return (torch.log(lam) - torch.log1p(lam)).sum(dim=-1).numpy()

    def expected_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """trace(K): the DPP's estimate of how many distinct answers there are (Paper B, Alg. 1)."""
        lam = self._eigenvalues(embeddings)
        return (lam / (1 + lam)).sum(dim=-1).numpy()
