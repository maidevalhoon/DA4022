"""
Internal Confidence (IC) Gate.

Direct implementation of Paper A:
"Query-Level Uncertainty in Large Language Models"
(Chen, de Melo, Suchanek, Varoquaux — ICLR 2026)

Key Paper Details Implemented:
- Self-evaluation prompt:
    "Respond only with 'Yes' or 'No' to indicate whether you are capable of
     answering the following query accurately. Answer Yes or No:\n{query}"
- Single forward pass with zero token generation cost.
- Logits computed for P(YES) vs P(NO) across layer and token positions.
- Attenuated Encoding (Eq. 4 in Paper A):
    δ_j^(i) = exp(-α |i - j|²) / Σ exp(-α |i - j|²), with α = 1.0
- Decision center fixed at (last layer, last token) — strictly training-free.
- Formal locality metric (Eq. 5 in Paper A):
    Loc(δ^(i)) = Σ_j δ_j^(i) / 2^|i - j|
- Dual-threshold 3-way routing (UCA extension):
    IC >= tau_high (0.65) -> fast_path (max compression, greedy)
    tau_low <= IC < tau_high -> uncertain (route to SemanticDPP)
    IC < tau_low (0.30) -> abstain / full memory / RAG
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class ICGate:
    """
    Query-Level Internal Confidence (IC) Gate.
    Estimates model uncertainty before generating any tokens using a single forward pass.
    """

    PROMPT_TEMPLATE = (
        "Respond only with 'Yes' or 'No' to indicate whether you are capable of "
        "answering the following query accurately. Answer Yes or No:\n{query}"
    )

    def __init__(
        self,
        model_name_or_instance: Optional[Union[str, torch.nn.Module]] = None,
        tokenizer: Optional[object] = None,
        alpha: float = 1.0,
        tau_high: float = 0.65,
        tau_low: float = 0.30,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        target_tokens: Optional[List[int]] = None,
        mock_mode: bool = False,
        seed: int = 42,
    ):
        self.alpha = float(alpha)
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.mock_mode = mock_mode
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if mock_mode or model_name_or_instance is None:
            self.mock_mode = True
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
            self.yes_token_id = 7566  # Default Llama Yes ID
            self.no_token_id = 2360   # Default Llama No ID
            return

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

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if isinstance(model_name_or_instance, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_instance, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_instance,
                torch_dtype=torch_dtype if self.device != "cpu" else torch.float32,
                device_map=self.device if self.device != "mps" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            if self.device == "mps":
                self.model = self.model.to("mps")
        else:
            self.model = model_name_or_instance
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        # Resolve target token IDs for ' Yes' and ' No'
        if target_tokens is not None and len(target_tokens) >= 2:
            self.yes_token_id = target_tokens[0]
            self.no_token_id = target_tokens[1]
        else:
            self.yes_token_id, self.no_token_id = self._resolve_target_tokens()

    def _resolve_target_tokens(self) -> Tuple[int, int]:
        """Resolve token IDs for ' Yes' and ' No' from the active tokenizer."""
        cand_yes = self.tokenizer.encode(" Yes", add_special_tokens=False)
        cand_no = self.tokenizer.encode(" No", add_special_tokens=False)
        if not cand_yes:
            cand_yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        if not cand_no:
            cand_no = self.tokenizer.encode("No", add_special_tokens=False)
        return cand_yes[-1], cand_no[-1]

    @staticmethod
    def compute_attenuated_weights(
        n_layers: int,
        n_tokens: int,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """
        Compute Attenuated Encoding 2D weight matrix centered at (last layer, last token).
        Implements Paper A Eq. 4:
            δ_j^(i) = exp(-α |i - j|²) / Σ exp(-α |i - j|²)
        where distance |i - j| is the normalized distance in the (layer, token) grid.
        """
        center_l = n_layers - 1
        center_t = n_tokens - 1

        # Normalized coordinates so both axes contribute proportionally
        norm_l = max(1, center_l)
        norm_t = max(1, center_t)

        l_indices = np.arange(n_layers)[:, None]
        t_indices = np.arange(n_tokens)[None, :]

        # Normalized Euclidean distance squared: |i - j|²
        dist_sq = ((l_indices - center_l) / norm_l) ** 2 + ((t_indices - center_t) / norm_t) ** 2
        raw_weights = np.exp(-alpha * dist_sq)
        weights = raw_weights / np.sum(raw_weights)
        return weights

    @staticmethod
    def locality_score(weight_vector: np.ndarray, center_idx: int) -> float:
        """
        Compute Attenuated Encoding locality metric (Eq. 5 in Paper A):
            Loc(δ^(i)) = Σ_j δ_j / 2^|i - j|  ∈ [0, 1]
        """
        weights_flat = np.asarray(weight_vector).flatten()
        J = len(weights_flat)
        c = int(center_idx)
        denom = np.array([2.0 ** abs(c - j) for j in range(J)], dtype=np.float64)
        return float(np.sum(weights_flat / denom))

    def score(
        self,
        query: str,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Compute Internal Confidence score for a query.

        Returns:
            dict containing:
                ic_score: float in [0, 1]
                top_right_score: float (P(YES) at last layer, last token alone)
                locality: float
                route: str ('fast_path' | 'uncertain' | 'abstain')
                p_yes_matrix: np.ndarray of shape (n_layers, n_tokens)
        """
        if self.mock_mode:
            return self._score_mock(query, difficulty=difficulty)

        prompt = self.PROMPT_TEMPLATE.format(query=query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: tuple of (batch, seq_len, hidden_dim)
        # index 0 is embeddings, index 1..L are transformer layer outputs
        hidden_states = outputs.hidden_states[1:]
        n_layers = len(hidden_states)
        seq_len = inputs["input_ids"].shape[1]

        # Extract unembedding weights for Yes / No
        # If model has lm_head, project hidden states directly
        lm_head = self.model.get_output_embeddings()
        w_unemb = lm_head.weight  # shape (vocab_size, hidden_dim)
        w_yes = w_unemb[self.yes_token_id].unsqueeze(1)  # (hidden_dim, 1)
        w_no = w_unemb[self.no_token_id].unsqueeze(1)    # (hidden_dim, 1)

        p_yes_matrix = np.zeros((n_layers, seq_len), dtype=np.float32)

        for l_idx, h_layer in enumerate(hidden_states):
            h_last = h_layer[0]  # shape (seq_len, hidden_dim)
            logit_yes = torch.matmul(h_last, w_yes).squeeze(-1)  # (seq_len,)
            logit_no = torch.matmul(h_last, w_no).squeeze(-1)    # (seq_len,)
            # Binary softmax between YES and NO logits
            logits_binary = torch.stack([logit_yes, logit_no], dim=-1)
            probs = F.softmax(logits_binary, dim=-1)[:, 0]  # P(YES)
            p_yes_matrix[l_idx] = probs.detach().cpu().numpy()

        weights = self.compute_attenuated_weights(n_layers, seq_len, alpha=self.alpha)
        ic_score = float(np.sum(weights * p_yes_matrix))
        top_right_score = float(p_yes_matrix[-1, -1])

        # Compute locality centered at (last_layer, last_token)
        weights_flat = weights.flatten()
        center_flat_idx = len(weights_flat) - 1
        loc = self.locality_score(weights_flat, center_flat_idx)

        route = self._determine_route(ic_score)

        return {
            "ic_score": ic_score,
            "top_right_score": top_right_score,
            "locality": loc,
            "route": route,
            "p_yes_matrix": p_yes_matrix,
        }

    def _determine_route(self, ic_score: float) -> str:
        """Dual-threshold routing decision."""
        if ic_score >= self.tau_high:
            return "fast_path"
        elif ic_score >= self.tau_low:
            return "uncertain"
        else:
            return "abstain"

    def route(
        self,
        query: str,
        tau_high: Optional[float] = None,
        tau_low: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Convenience method returning (route_str, ic_score)."""
        th_high = tau_high if tau_high is not None else self.tau_high
        th_low = tau_low if tau_low is not None else self.tau_low
        res = self.score(query)
        score_val = res["ic_score"]
        if score_val >= th_high:
            return "fast_path", score_val
        elif score_val >= th_low:
            return "uncertain", score_val
        else:
            return "abstain", score_val

    def _score_mock(
        self,
        query: str,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Deterministic, realistic mock scorer for testing and CPU-only environments.
        Computes synthetic layer-token matrix where confidence peaks near (last layer, last token).
        """
        # Seed hash from query string for deterministic responses
        q_hash = abs(hash(query)) % 1000000
        rng = np.random.default_rng(q_hash)

        # Classify query pseudo-difficulty based on difficulty hint or keywords
        q_lower = query.lower()
        if difficulty == "easy":
            base_p = 0.90 + rng.uniform(-0.04, 0.04)
        elif difficulty == "hard":
            base_p = 0.18 + rng.uniform(-0.04, 0.04)
        elif difficulty == "medium":
            base_p = 0.52 + rng.uniform(-0.06, 0.06)
        elif any(w in q_lower for w in ["what is", "capital", "who", "when"]):
            base_p = 0.88 + rng.uniform(-0.04, 0.04)  # Factual / confident
        elif any(w in q_lower for w in ["prove", "solve", "calculate", "integrate", "derivative"]):
            base_p = 0.48 + rng.uniform(-0.12, 0.12)  # Complex reasoning / uncertain
        elif any(w in q_lower for w in ["unanswerable", "fictional", "impossible", "nonsense"]):
            base_p = 0.22 + rng.uniform(-0.08, 0.08)  # Abstain zone
        else:
            base_p = 0.55 + rng.uniform(-0.15, 0.15)

        base_p = float(np.clip(base_p, 0.05, 0.95))

        n_layers = 16
        n_tokens = 12
        p_matrix = np.zeros((n_layers, n_tokens), dtype=np.float32)

        # Simulate Paper A finding: P(YES) climbs towards last layer & last token
        for l in range(n_layers):
            layer_progress = (l + 1) / n_layers
            for t in range(n_tokens):
                token_progress = (t + 1) / n_tokens
                cell_signal = base_p * (0.6 + 0.4 * layer_progress * token_progress)
                noise = rng.normal(0, 0.04)
                p_matrix[l, t] = np.clip(cell_signal + noise, 0.01, 0.99)

        weights = self.compute_attenuated_weights(n_layers, n_tokens, alpha=self.alpha)
        ic_score = float(np.sum(weights * p_matrix))
        top_right_score = float(p_matrix[-1, -1])

        weights_flat = weights.flatten()
        loc = self.locality_score(weights_flat, len(weights_flat) - 1)
        route = self._determine_route(ic_score)

        return {
            "ic_score": ic_score,
            "top_right_score": top_right_score,
            "locality": loc,
            "route": route,
            "p_yes_matrix": p_matrix,
        }
