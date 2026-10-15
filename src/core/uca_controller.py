"""
Uncertainty-Calibrated Adaptive Compression (UCA) Controller.

This is the central novel contribution of the project, uniting:
1. Paper A (ICLR 2026): Query-Level Uncertainty (Internal Confidence) as a zero-generation pre-filter.
2. Paper B (ICLR 2026): SemanticDPP L-ensemble diversity and eigenspectrum decomposition.

Workflow:
- Tier 0: Single forward pass Internal Confidence (IC) evaluates query solvability.
  - If IC >= tau_high (0.65) -> Fast Path: 25% KV cache budget, greedy decoding (beam=1), max compression.
  - If IC < tau_low (0.30) -> Abstain / RAG: 100% KV cache budget, beam=4, uncompressed.
  - If tau_low <= IC < tau_high -> Cascade to Tier 1.
- Tier 1: SemanticDPP computes response diversity:
  - Eigenspectrum {σ_1, ..., σ_m} maps to per-dimension compression ratios:
      r_i = clip(1 - α · σ_i / σ_max, r_min, r_max)
  - SemanticDPP-C cluster count k determines the generation beam width floor.
  - Composite UCA score = 0.4 · (1 - IC) + 0.6 · DPP_uncertainty.
  - Continuous KV cache budget = 0.25 + 0.75 · UCA.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .internal_confidence import ICGate
from .semantic_dpp import SemanticDPP


class UCAController:
    """
    Orchestrates multi-tier uncertainty estimation and dynamic allocation.
    """

    def __init__(
        self,
        ic_gate: Optional[ICGate] = None,
        semantic_dpp: Optional[SemanticDPP] = None,
        tau_high: float = 0.65,
        tau_low: float = 0.30,
        r_min: float = 0.10,
        r_max: float = 0.95,
        alpha_scale: float = 2.0,
        mock_mode: bool = False,
    ):
        self.mock_mode = mock_mode
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.alpha_scale = float(alpha_scale)

        if ic_gate is not None:
            self.ic = ic_gate
        else:
            self.ic = ICGate(mock_mode=mock_mode or True, tau_high=self.tau_high, tau_low=self.tau_low)

        if semantic_dpp is not None:
            self.dpp = semantic_dpp
        else:
            self.dpp = SemanticDPP(mock_mode=mock_mode or True)

    def compute_compression_config(
        self,
        query: str,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Union[str, float, int, bool, np.ndarray, None]]:
        """
        Main UCA routing, allocation, and compression configuration.

        Args:
            query: Input prompt / question
            difficulty: Optional difficulty hint ('easy', 'medium', 'hard') for simulation/testing

        Returns:
            Dictionary containing:
                route: str ('fast_path' | 'standard' | 'augmented' | 'abstain')
                ic_score: float
                uca_score: float in [0.0, 1.0]
                compression_ratios: np.ndarray of per-dimension ratios
                kv_budget: float in [0.25, 1.0]
                beam_width: int in [1, 4]
                trigger_rag: bool
                dpp_eigenvalues: Optional[np.ndarray]
                dpp_c_clusters: int
                tier_used: str ('tier_0' | 'tier_0_and_tier_1')
        """
        # --- TIER 0: Pre-Generation Internal Confidence Gate ---
        ic_res = self.ic.score(query, difficulty=difficulty)
        ic_score = float(ic_res["ic_score"])

        # High confidence (Fast Path)
        if ic_score >= self.tau_high:
            return self._fast_path_config(ic_score)

        # Very low confidence (Abstain / RAG fallback)
        if ic_score < self.tau_low:
            return self._abstain_config(ic_score)

        # --- TIER 1: DPP Eigenspectrum & Semantic Diversity (Uncertain queries only) ---
        dpp_res = self.dpp.uncertainty_score(query, difficulty=difficulty)
        eigenvalues = np.asarray(dpp_res["eigenvalues"], dtype=np.float32)
        dpp_uncertainty = float(dpp_res["uncertainty"])
        dpp_c_clusters = int(dpp_res.get("dpp_c_clusters", 1))

        # Per-Dimension Compression Ratio mapping:
        # r_i = clip(1 - alpha · σ_i / σ_max, r_min, r_max)
        sigma_max = float(eigenvalues.max()) if len(eigenvalues) > 0 else 1.0
        sigma_max = max(1e-6, sigma_max)
        ratios = 1.0 - self.alpha_scale * (eigenvalues / sigma_max)
        ratios = np.clip(ratios, self.r_min, self.r_max)

        # Continuous composite UCA score:
        # Paper A IC uncertainty: (1 - ic_score)
        # Paper B DPP uncertainty: dpp_uncertainty
        # Weights (0.4 vs 0.6) reflect SemanticDPP's higher discriminative AUROC
        ic_uncertainty = 1.0 - ic_score
        combined_uca = 0.4 * ic_uncertainty + 0.6 * dpp_uncertainty
        combined_uca = float(np.clip(combined_uca, 0.0, 1.0))

        route = "augmented" if combined_uca > 0.55 else "standard"
        kv_budget = self._compute_kv_budget(combined_uca)

        # Beam width: UCA score drives continuous width, with DPP-C cluster count as floor
        base_beam = self._compute_beam_width(combined_uca)
        beam_width = min(4, max(base_beam, dpp_c_clusters))

        return {
            "route": route,
            "ic_score": ic_score,
            "uca_score": combined_uca,
            "compression_ratios": ratios,
            "kv_budget": kv_budget,
            "beam_width": beam_width,
            "trigger_rag": combined_uca > 0.60,
            "dpp_eigenvalues": eigenvalues,
            "dpp_c_clusters": dpp_c_clusters,
            "tier_used": "tier_0_and_tier_1",
        }

    def _fast_path_config(self, ic_score: float) -> Dict:
        """High confidence: maximum compression, greedy decoding, zero DPP overhead."""
        m = getattr(self.dpp, "m", 3)
        return {
            "route": "fast_path",
            "ic_score": ic_score,
            "uca_score": max(0.0, float(1.0 - ic_score)),
            "compression_ratios": np.full(m, self.r_min, dtype=np.float32),
            "kv_budget": 0.25,
            "beam_width": 1,
            "trigger_rag": False,
            "dpp_eigenvalues": None,
            "dpp_c_clusters": 1,
            "tier_used": "tier_0",
        }

    def _abstain_config(self, ic_score: float) -> Dict:
        """Very low confidence: full uncompressed memory, wide beam, external RAG triggered."""
        m = getattr(self.dpp, "m", 3)
        return {
            "route": "abstain",
            "ic_score": ic_score,
            "uca_score": min(1.0, float(1.0 - ic_score)),
            "compression_ratios": np.full(m, self.r_max, dtype=np.float32),
            "kv_budget": 1.0,
            "beam_width": 4,
            "trigger_rag": True,
            "dpp_eigenvalues": None,
            "dpp_c_clusters": 1,
            "tier_used": "tier_0",
        }

    @staticmethod
    def _compute_kv_budget(uca_score: float) -> float:
        """
        Continuous KV Cache Budget mapping:
            KV_budget = 0.25 + 0.75 · UCA
        """
        return float(np.clip(0.25 + 0.75 * uca_score, 0.25, 1.0))

    @staticmethod
    def _compute_beam_width(uca_score: float) -> int:
        """
        Continuous beam width mapping:
            Beam_width = round(1 + 3 · UCA) in [1, 4]
        """
        return int(np.clip(round(1.0 + 3.0 * uca_score), 1, 4))
