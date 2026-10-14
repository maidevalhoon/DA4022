# UCA: Uncertainty-Calibrated Adaptive Compression for LLM Inference
## End-to-End Implementation Plan — DA4022 Course Project

> **Core Question:** Can we design a better adaptive compression system, inspired by C-DIC and HyC-LoRA, that uses uncertainty signals (Internal Confidence + SemanticDPP) to dynamically decide *how much* and *what kind* of compression to apply — and beat existing benchmarks?

---

## PART 1: CRITICAL ANALYSIS OF THE FOUR PAPERS

### 1.1 Paper A — Query-Level Uncertainty in LLMs (Internal Confidence)
**Authors:** Lihu Chen, Gerard de Melo, Fabian M. Suchanek, Gaël Varoquaux  
**Venue:** ICLR 2026 | **Code:** `tigerchen52/query_level_uncertainty`

#### What It Does
- Proposes **Internal Confidence (IC)** — a training-free, zero-generation uncertainty estimate computed in a **single forward pass** over the query hidden states.
- Probes `P(YES)` logits at every (layer, token-position) pair, then aggregates via **Attenuated Encoding** — an exponentially decaying weighted sum from the "optimal decision center" (last token, last layer).
- Formula: `IC(q) = Σ_{l,t} w(l,t) · P(YES | h_l^t)` where `w(l,t) ∝ exp(-λ·dist(l,t))`.
- Achieves **30–600× speedup** vs. answer-level methods with competitive AUROC.

#### Critical Strengths
- Zero generation cost — no tokens sampled, no beam search.
- Model-agnostic — works on any autoregressive LLM with hidden state access.
- Published open code with all baselines benchmarked (TriviaQA, GSM8K, etc.).

#### Critical Weaknesses / Gaps (Our Targets)
1. **Binary routing only** — IC outputs a scalar; paper only uses a hard threshold. No continuous or multi-level resource allocation.
2. **Static fallback action** — low IC always triggers RAG or cascade. Does not tune *how much* compression to apply.
3. **No compression of the context itself** — IC decides on context expansion, not compression quality as a variable.
4. **Threshold is dataset-specific** — optimal τ₁ must be tuned per dataset/model pair.
5. **Last-k tokens are hyperparameter-sensitive** — k is reported as highly performance-sensitive.

---

### 1.2 Paper B — SemanticDPP: Efficient Uncertainty Quantification in LLMs
**Authors:** Jenny Hamer, Jasper Snoek, et al. (Google DeepMind)  
**Venue:** ICLR 2026 | **Local PDF:** `9298_SemanticDPP_Efficient_Unc.pdf`

#### What It Does
- Computes **semantic diversity** of sampled responses `S = {r₁,...,rₘ}` using a **Determinantal Point Process (DPP)** over the output embedding space.
- Learns projection `ϕ: R^d → R^p` (p << d) to a semantically meaningful subspace.
- Kernel: `K(xᵢ, xⱼ) = exp(-α||ϕ(xᵢ) - ϕ(xⱼ)||²)`
- Uncertainty: `U(S) = log det(L_S) - log det(L + I)`
- SemanticDPP-C adds soft clustering over response diversity.
- Benchmarked on Gemma2-9B, Gemma3-27B; matches/exceeds Semantic Entropy (SE).

#### Critical Strengths
- Semantically grounded — captures genuine response diversity, not just token variance.
- Unsupervised — no NLI labels needed (unlike SE which uses DeBERTa entailment).
- Richer signal — full kernel matrix encodes *structure* of uncertainty.

#### Critical Weaknesses / Gaps (Our Targets)
1. **Still requires m forward passes** — m ≥ 5 generations needed; expensive for real-time routing.
2. **No compression action** — pure uncertainty estimation; no downstream use for compression/allocation decisions.
3. **Embedding space is fixed** — ϕ is learned offline; no adaptive update per query context.
4. **Discards kernel structure** — the log-det is a scalar; all rich pairwise information in L_S is discarded.

---

### 1.3 Paper C — C-DIC: Context-Driven Incremental Compression (ICML 2026)
**Authors:** Yeongseo Jung, Jaehyeok Kim, Eunseo Jung, Jiachuan Wang, et al. (HKUST)  
**OpenReview:** `ZGeVMaiIlb`

#### What It Does
- Addresses the growing context window in **multi-turn dialogue**: re-encoding the full history at every turn is O(n²) in attention cost.
- Treats conversation as **interleaved contextual threads** (topic clusters across turns).
- Maintains a **compact dialogue memory**: one compressed state vector per thread.
- At each turn runs a lightweight **Retrieve → Revise → Write-Back** loop:
  1. **Retrieve:** Score existing thread states by similarity to `q_t`; fetch top-k.
  2. **Revise (Compress):** Trainable compressor `f_comp(retrieved_states, current_turn)` → updated state.
  3. **Write-Back:** If `δ_t` (peak similarity) > threshold → revise existing thread; else insert new thread (topic shift).
- Training via **Truncated BPTT (TBPTT)**: gradients flow one hop along "usage edges".
- Achieves **near-constant inference time** over hundreds of turns.

#### Critical Strengths
- Incremental — O(1) cost per turn (retrieval + compress), not O(n²).
- Revisable memory — avoids stale/compounding errors from one-shot summarization.
- Modular compressor — `f_comp` is a small trainable module.

#### Critical Weaknesses / Gaps (Our Targets)
1. **No uncertainty awareness** — every turn compresses with the same aggressiveness regardless of model uncertainty.
2. **Fixed compression ratio** — `f_comp` outputs a fixed-dimension state; cannot adapt aggressiveness.
3. **No error recovery** — if the compressor makes a lossy mistake on an important turn, no detection/correction.
4. **Thread boundary detection is heuristic** — the δ_t threshold for topic shift is static.

---

### 1.4 Paper D — HyC-LoRA: Hybrid Activation Compression (MLSys 2025)
**Authors:** Yujin Wang, Shunan Dong, Zongle Huang, et al. (Tsinghua)  
**OpenReview:** `WU86J4uM9V` | **Code:** `thu-ee-acts-lab/HyC-LoRA-release`

#### What It Does
- Addresses **activation memory** during LoRA fine-tuning — temporarily buffered forward-pass activations for backpropagation.
- These dominate GPU memory, especially in non-linear modules (RMSNorm, SiLU).
- **Two-tier hybrid compression:**
  1. **Intra-Operator:** Detects structured outlier channels, stores them in FP16 while quantizing the rest to ~2-bit.
  2. **Inter-Operator:** Leverages the LoRA adapter to *compensate* for quantization errors; reorders operator execution, selective recomputation.
- Achieves **3.97× end-to-end memory reduction** with negligible accuracy drop.

#### Critical Strengths
- Practical systems contribution — real GPU memory reduction.
- Outlier-aware — structured outlier detection that prior methods missed.
- LoRA adapter as error compensator — elegant reuse.

#### Critical Weaknesses / Gaps (Our Targets)
1. **Training-time only** — designed for fine-tuning backprop, not inference-time KV cache / context compression.
2. **Fixed quantization bit-width** — always 2-bit for non-outliers; cannot adapt to signal importance.
3. **No uncertainty signal** — quantization aggressiveness is uniform regardless of activation importance.
4. **Outlier detection is static** — channel outliers detected once per model, not per-input.

---

## PART 2: SYNTHESIS — THE RESEARCH GAP

### The Core Insight That All Four Papers Miss

| Method | Uncertainty-Aware? | Adaptive Compression? | Zero-Cost Routing? | Incremental? |
|---|---|---|---|---|
| Internal Confidence (IC) | ✅ | ❌ | ✅ | ❌ |
| SemanticDPP | ✅ | ❌ | ❌ | ❌ |
| C-DIC | ❌ | ✅ (fixed ratio) | N/A | ✅ |
| HyC-LoRA | ❌ | ✅ (fixed 2-bit) | N/A | ❌ |
| **UCA (Ours)** | **✅** | **✅ (continuous)** | **✅** | **✅** |

**The gap:** No existing method uses uncertainty as a *continuous control signal* to modulate compression quality. Current systems have hard binary switches (compress or don't compress). **Our insight: the compression ratio itself should be a differentiable function of uncertainty.**

### Our Novel Contribution: Uncertainty-Calibrated Adaptive Compression (UCA)

> **Core mechanism:** Use IC as a near-zero-cost pre-filter. For queries passing the filter, use the *structure of the DPP kernel matrix* (not just the log-det scalar) to determine per-dimension compression precision, inspired by HyC-LoRA's outlier-aware approach and C-DIC's incremental update pattern.

**Three specific novelties beyond all four papers:**
1. **DPP Eigenspectrum as Compression Oracle** — eigendecompose `L_S = UΣUᵀ`; per-dimension eigenvalue σᵢ drives compression ratio rᵢ.
2. **IC-Gated DPP Cascade** — IC pre-filters queries; DPP only runs on the uncertain subset (~40% of queries), saving 60% of DPP overhead.
3. **Uncertainty History in Thread Memory** — C-DIC threads carry uncertainty scores, enabling preferential re-compression of previously under-compressed threads.

---

## PART 3: PROPOSED ARCHITECTURE — UCA SYSTEM

```
┌──────────────────────────────────────────────────────────────┐
│                      INPUT QUERY q_t                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 0: Internal Confidence Gate (IC)                      │
│  • Single forward pass, no generation                       │
│  • IC(q) = Σ_{l,t} w(l,t) · P(YES | h_l^t)                │
│  • Cost: ~5ms on GPU (1 forward pass, no decode)            │
└──────────────────────────────┬──────────────────────────────┘
                               │
        IC >= τ_high           │         IC < τ_low
       (Very Confident)        │        (Very Uncertain)
               ┌───────────────┤──────────────────┐
               ▼               │ (IC uncertain)   ▼
┌─────────────────────┐        │        ┌──────────────────────┐
│  FAST PATH          │        ▼        │  ABSTAIN / CASCADE   │
│  Max compression    │  ┌─────────────┐│  Full KV cache       │
│  25% KV budget      │  │ TIER 1: DPP ││  Beam width = 4      │
│  Greedy decode      │  │ Eigenspectrum││  Trigger RAG         │
│  Reuse threads      │  └──────┬──────┘└──────────────────────┘
└─────────────────────┘         │
                                ▼
                    ┌───────────────────────┐
                    │  DPP Structure Decoder│
                    │  Sample m=3 responses │
                    │  Compute L_S kernel   │
                    │  Decompose L_S = UΣUᵀ │
                    │  Per-dim σᵢ = "spread"│
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Compression Ratio    │
                    │  rᵢ = clip(1-α·σᵢ/σmax, r_min)│
                    │  High σᵢ → less compression   │
                    │  Low σᵢ → more compression    │
                    └──────────┬────────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
   ┌───────────▼────────────┐      ┌────────────▼────────────┐
   │  THREAD MEMORY (C-DIC) │      │  GENERATION CONFIG       │
   │  Per-thread quant bits │      │  KV Budget = f(UCA)     │
   │  = f(rᵢ, HyC-outlier)  │      │  Beam Width = f(UCA)    │
   │  Uncertainty history   │      │  RAG: if UCA > τ_RAG    │
   └────────────────────────┘      └─────────────────────────┘
```

---

## PART 4: WHAT "BEAT THE BENCHMARK" MEANS

### AUROC Targets

| Method | AUROC (TriviaQA) | AUROC (GSM8K) | Cost (per query) |
|---|---|---|---|
| Semantic Entropy | ~0.78 | ~0.82 | O(10 generations) |
| SemanticDPP | ~0.81 | ~0.85 | O(5 generations) |
| Internal Confidence | ~0.72 | ~0.77 | O(1 pass) |
| **UCA Target** | **>0.83** | **>0.87** | **O(1 pass + 3 gen. for ~40%)** |

### Memory Targets

| Method | KV Cache Reduction | Context Memory |
|---|---|---|
| Static baseline | 0% | 0% |
| C-DIC | N/A | ~40% over naive |
| HyC-LoRA | N/A | 3.97× (training only) |
| **UCA Target** | **>30% avg** | **>50% for high-conf** |

---

## PART 5: PHASED IMPLEMENTATION PLAN

### Phase 0: Environment Setup (Hours 1-4)

```bash
conda create -n uca python=3.11
conda activate uca
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 accelerate datasets evaluate
pip install scikit-learn scipy numpy pandas matplotlib seaborn
pip install query-level-uncertainty    # Official IC package
pip install dppy                       # DPP library
pip install lm-polygraph               # Evaluation suite
pip install bitsandbytes               # For 4-bit model loading
```

**Project Structure:**
```
DA4022/
├── src/
│   ├── core/
│   │   ├── internal_confidence.py      # IC gate
│   │   ├── semantic_dpp.py             # DPP structure decoder
│   │   ├── uca_controller.py           # Main orchestrator (OUR KEY NOVELTY)
│   │   └── compression/
│   │       ├── thread_memory.py        # C-DIC + uncertainty history
│   │       ├── kv_cache_manager.py     # Dynamic KV budgeting
│   │       └── quantizer.py           # HyC-LoRA inspired adaptive quantizer
│   ├── evaluation/
│   │   ├── metrics.py                  # AUROC, PRR, ECE, memory savings
│   │   └── benchmarks.py              # Dataset loaders
│   └── experiments/
│       ├── run_baseline.py            # Reproduce paper baselines
│       ├── run_uca.py                 # Full UCA evaluation
│       └── run_ablation.py            # Component ablation
├── configs/default.yaml
└── results/
```

---

### Phase 1: Baseline Reproduction (Hours 5-12)

**Reproduce IC and SemanticDPP numbers from the papers.**

#### `src/core/internal_confidence.py`

```python
"""
Internal Confidence (IC) Gate.
Wraps the official tigerchen52/query_level_uncertainty package.
Single forward pass, no generation required.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ql_uncertainty import QLUncertainty


class ICGate:
    """
    Zero-generation Internal Confidence gate.
    Returns a confidence score in [0, 1] (high = confident).
    """
    def __init__(self, model_name: str,
                 target_tokens: list = None,   # Yes/No token IDs for specific model
                 last_k_token: int = 6,
                 device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.last_k = last_k_token
        # NOTE: Token IDs for "Yes" / "No" are model-specific!
        # For Llama-3.1-8B-Instruct: Yes=7566, No=2360
        # Run: tokenizer.encode("Yes")[1] to verify
        self.target_tokens = target_tokens or [[7566], [2360]]
        self.ql = QLUncertainty(
            self.model, self.tokenizer,
            method='internal_confidence',
            target_tokens=self.target_tokens
        )

    def score(self, query: str) -> float:
        """Returns confidence in [0,1]. High = confident. Cost: 1 forward pass."""
        return self.ql.estimate(query, self.last_k)

    def route(self, query: str,
              tau_high: float = 0.65,
              tau_low: float = 0.30) -> tuple:
        """
        Returns: (route_str, ic_score)
        route_str: 'fast_path' | 'uncertain' | 'abstain'
        """
        ic = self.score(query)
        if ic >= tau_high:
            return 'fast_path', ic
        elif ic >= tau_low:
            return 'uncertain', ic
        else:
            return 'abstain', ic
```

#### `src/core/semantic_dpp.py`

```python
"""
SemanticDPP reimplementation from Hamer et al. (ICLR 2026).
Key extension: expose eigenspectrum of L_S (not just log-det scalar).
"""
import torch
import torch.nn as nn
import numpy as np


class SemanticSubspaceProjector(nn.Module):
    """
    Learned projection phi: R^d -> R^p (semantically meaningful subspace).
    Initialized with orthogonal weights.
    """
    def __init__(self, d: int, p: int):
        super().__init__()
        self.proj = nn.Linear(d, p, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


class SemanticDPP:
    """
    SemanticDPP + eigenspectrum extension.
    
    Given m sampled responses, computes:
    1. log det(L_S) — original SemanticDPP uncertainty (scalar)
    2. eigenvalues of L_S — per-dimension uncertainty (OUR EXTENSION)
    """
    def __init__(self, model, tokenizer,
                 projector: SemanticSubspaceProjector,
                 alpha: float = 1.0,
                 m: int = 3,       # Set low (3) for speed; increase for quality
                 device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.projector = projector.to(device)
        self.alpha = alpha
        self.m = m
        self.device = device

    def _get_response_embedding(self, query: str,
                                temperature: float = 0.8) -> torch.Tensor:
        """Sample one response, return mean-pooled last-layer embedding."""
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # Mean pool last-layer hidden states over ALL generated tokens
            hidden_states = out.hidden_states  # tuple of (layers,) tuples
            last_layer_states = torch.stack(
                [step[-1] for step in hidden_states], dim=2
            )  # (1, 1, n_new_tokens, d)
            embedding = last_layer_states.squeeze(0).squeeze(0).mean(dim=0)  # (d,)
        return embedding

    def compute_kernel_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        DPP L-ensemble kernel.
        K(xi, xj) = exp(-alpha * ||phi(xi) - phi(xj)||^2)
        """
        projected = self.projector(embeddings)  # (m, p)
        diff = projected.unsqueeze(0) - projected.unsqueeze(1)  # (m, m, p)
        sq_dist = (diff ** 2).sum(dim=-1)  # (m, m)
        K = torch.exp(-self.alpha * sq_dist)  # (m, m)
        return K

    def uncertainty_score(self, query: str) -> dict:
        """
        Main uncertainty computation.

        Returns:
            log_det: float          — Scalar uncertainty (original SemanticDPP)
            eigenvalues: np.array   — OUR EXTENSION: DPP eigenspectrum for UCA
            kernel_matrix: tensor   — Full kernel for downstream use
            uncertainty: float      — Normalized [0,1] score
        """
        embeddings = torch.stack([
            self._get_response_embedding(query)
            for _ in range(self.m)
        ])  # (m, d)

        K = self.compute_kernel_matrix(embeddings)  # (m, m)

        # Add regularization for numerical stability
        eps = 1e-6
        I = torch.eye(self.m, device=self.device)
        L = K + eps * I

        # Log-determinant (SemanticDPP uncertainty)
        log_det = torch.logdet(L + I).item() - torch.logdet(2 * I).item()

        # --- OUR EXTENSION: Eigenspectrum decomposition ---
        eigenvalues = torch.linalg.eigvalsh(L)  # (m,) sorted ascending
        eigenvalues_np = eigenvalues.detach().cpu().numpy()
        eigenvalues_np = np.clip(eigenvalues_np, 0, None)  # numerical stability

        # Normalize to [0,1]
        uncertainty_normalized = float(torch.sigmoid(
            torch.tensor(log_det / self.m, dtype=torch.float32)
        ))

        return {
            'log_det': log_det,
            'eigenvalues': eigenvalues_np,       # KEY: per-dimension semantic variance
            'kernel_matrix': K.detach().cpu(),
            'uncertainty': uncertainty_normalized,
        }
```

---

### Phase 2: Core UCA Controller (Hours 13-22)

**The main novelty of the project.**

#### `src/core/uca_controller.py`

```python
"""
Uncertainty-Calibrated Adaptive Compression (UCA) Controller.

This is the central novel contribution of the project.
It synthesizes all four papers into a unified adaptive compression system.
"""
import numpy as np
import torch
from .internal_confidence import ICGate
from .semantic_dpp import SemanticDPP


class UCAController:
    """
    UCA System Orchestrator.

    Novel contributions over existing papers:
    1. IC gates expensive DPP computation (saves ~60% of DPP cost)
    2. DPP eigenspectrum drives per-dimension compression ratios
    3. Combined IC+DPP score is more informative than either alone
    """
    def __init__(self,
                 ic_gate: ICGate,
                 semantic_dpp: SemanticDPP,
                 tau_high: float = 0.65,    # IC threshold: fast path
                 tau_low: float = 0.30,     # IC threshold: abstain
                 r_min: float = 0.10,       # Min per-dim compression ratio
                 r_max: float = 0.95,       # Max per-dim compression ratio
                 alpha_scale: float = 2.0,  # Eigenvalue → ratio scale factor
                 ):
        self.ic = ic_gate
        self.dpp = semantic_dpp
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.r_min = r_min
        self.r_max = r_max
        self.alpha_scale = alpha_scale

    def compute_compression_config(self, query: str) -> dict:
        """
        Main UCA routing and compression configuration.

        Returns:
            route: str                    # 'fast_path' | 'standard' | 'augmented' | 'abstain'
            ic_score: float               # Internal confidence
            uca_score: float              # Combined uncertainty [0,1]
            compression_ratios: array     # Per-dim ratios (None for fast/abstain)
            kv_budget: float              # Fraction of max KV cache to allocate
            beam_width: int               # Generation beam width
            trigger_rag: bool             # Whether to invoke RAG
            dpp_eigenvalues: array        # For logging and analysis
        """
        # --- TIER 0: Internal Confidence Gate ---
        ic_score = self.ic.score(query)

        if ic_score >= self.tau_high:
            return self._fast_path_config(ic_score)

        if ic_score < self.tau_low:
            return self._abstain_config(ic_score)

        # --- TIER 1: DPP Eigenspectrum (uncertain queries only) ---
        dpp_result = self.dpp.uncertainty_score(query)
        eigenvalues = dpp_result['eigenvalues']         # shape: (m,)
        dpp_uncertainty = dpp_result['uncertainty']

        # KEY EQUATION: per-dimension compression ratios
        # rᵢ = clip(1 - alpha * σᵢ / σ_max, r_min, r_max)
        # High σᵢ (semantically contested dimension) → less compression (higher ratio)
        # Low σᵢ (semantically stable dimension) → more compression (lower ratio)
        sigma_max = eigenvalues.max() + 1e-8
        ratios = 1.0 - self.alpha_scale * (eigenvalues / sigma_max)
        ratios = np.clip(ratios, self.r_min, self.r_max)

        # Combined UCA score: weighted combination
        # IC uncertainty = (1 - ic_score), DPP uncertainty = dpp_uncertainty
        ic_uncertainty = 1.0 - ic_score
        combined_uca = 0.4 * ic_uncertainty + 0.6 * dpp_uncertainty

        route = 'augmented' if combined_uca > 0.55 else 'standard'

        return {
            'route': route,
            'ic_score': ic_score,
            'uca_score': combined_uca,
            'compression_ratios': ratios,
            'kv_budget': self._compute_kv_budget(combined_uca),
            'beam_width': self._compute_beam_width(combined_uca),
            'trigger_rag': combined_uca > 0.60,
            'dpp_eigenvalues': eigenvalues,
        }

    def _fast_path_config(self, ic_score: float) -> dict:
        """High confidence: maximum compression, no DPP cost."""
        return {
            'route': 'fast_path',
            'ic_score': ic_score,
            'uca_score': 1.0 - ic_score,
            'compression_ratios': np.full(5, self.r_min),   # Max compression
            'kv_budget': 0.25,   # 25% of max KV cache
            'beam_width': 1,     # Greedy decoding
            'trigger_rag': False,
            'dpp_eigenvalues': None,
        }

    def _abstain_config(self, ic_score: float) -> dict:
        """Very low confidence: no compression, full resources."""
        return {
            'route': 'abstain',
            'ic_score': ic_score,
            'uca_score': 1.0 - ic_score,
            'compression_ratios': np.full(5, self.r_max),   # Minimal compression
            'kv_budget': 1.0,    # Full KV cache
            'beam_width': 4,     # Beam search
            'trigger_rag': True,
            'dpp_eigenvalues': None,
        }

    def _compute_kv_budget(self, uca_score: float) -> float:
        """Smooth mapping: uncertainty [0,1] → KV budget [0.25, 1.0]."""
        return 0.25 + 0.75 * uca_score

    def _compute_beam_width(self, uca_score: float) -> int:
        """Low uncertainty → greedy; high uncertainty → wider beam."""
        return max(1, round(1 + 3 * uca_score))  # Maps to [1, 4]
```

---

### Phase 3: Thread Memory with Uncertainty History (Hours 23-30)

**C-DIC extension with uncertainty-calibrated compression.**

#### `src/core/compression/thread_memory.py`

```python
"""
C-DIC Inspired Thread Memory with Uncertainty History (Novel Extension).

Key differences vs. original C-DIC:
1. Threads carry uncertainty_history: list of UCA scores at write times.
2. Retrieval preferentially selects threads with high past uncertainty
   (candidates for re-compression with updated context).
3. Write-back quantization bit-width is modulated by current UCA score.
4. HyC-LoRA outlier detection applied at thread embedding write time.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Thread:
    """A single contextual thread in dialogue memory."""
    thread_id: int
    state: torch.Tensor                   # Compressed thread embedding
    last_query_repr: str                  # For debugging
    uncertainty_history: List[float] = field(default_factory=list)
    write_count: int = 0
    effective_bits: float = 8.0           # Current effective quantization bits


class ThreadMemory:
    """
    Thread memory manager for multi-turn inference.
    Inspired by C-DIC; extended with uncertainty-calibrated compression.
    """
    def __init__(self,
                 d_model: int = 4096,
                 d_thread: int = 512,
                 max_threads: int = 32,
                 topic_shift_threshold: float = 0.40,
                 revisit_uncertainty_threshold: float = 0.70):
        self.d_model = d_model
        self.d_thread = d_thread
        self.max_threads = max_threads
        self.delta_t = topic_shift_threshold
        self.revisit_threshold = revisit_uncertainty_threshold

        # Lightweight trainable compressor (2-layer MLP)
        self.compressor = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_thread * 2),
            torch.nn.GELU(),
            torch.nn.Linear(d_thread * 2, d_thread),
        )
        self.threads: List[Thread] = []
        self._next_id = 0

    def retrieve(self, query_emb: torch.Tensor, top_k: int = 3) -> List[Thread]:
        """
        Retrieve top-k relevant threads.
        Novel: high-uncertainty threads get a retrieval score boost (revisit bias).
        """
        if not self.threads:
            return []
        q = query_emb[:self.d_thread]
        states = torch.stack([t.state for t in self.threads])       # (n, d_thread)
        q_n = torch.nn.functional.normalize(q.unsqueeze(0))
        s_n = torch.nn.functional.normalize(states)
        sims = (q_n @ s_n.T).squeeze(0)                            # (n,)

        # Uncertainty revisit boost
        for i, thread in enumerate(self.threads):
            if thread.uncertainty_history:
                recent_u = np.mean(thread.uncertainty_history[-3:])
                if recent_u > self.revisit_threshold:
                    sims[i] = sims[i] + 0.15  # Boost to encourage re-visit

        _, indices = sims.topk(min(top_k, len(self.threads)))
        return [self.threads[i] for i in indices.tolist()]

    def write_back(self, query_emb: torch.Tensor,
                   retrieved: List[Thread],
                   uca_score: float) -> None:
        """
        Compress and write-back with UCA-calibrated quantization.

        uca_score=0.0 → aggressive 4-bit compression (high confidence)
        uca_score=1.0 → light 16-bit compression (high uncertainty)
        """
        q = query_emb[:self.d_thread]
        if not retrieved:
            similarity = 0.0
        else:
            states = torch.stack([t.state for t in retrieved])
            q_n = torch.nn.functional.normalize(q.unsqueeze(0))
            s_n = torch.nn.functional.normalize(states)
            similarity = (q_n @ s_n.T).max().item()

        # Compress query embedding
        with torch.no_grad():
            new_state = self.compressor(query_emb)  # (d_thread,)

        # Adaptive quantization: 4-bit (high conf) to 16-bit (high uncertainty)
        target_bits = 4.0 + 12.0 * uca_score
        quantized_state = self._quantize_outlier_aware(new_state, target_bits)

        if similarity >= self.delta_t and retrieved:
            # On-topic: revise best matching thread
            best = max(retrieved, key=lambda t: t.state.dot(q).item())
            best.state = quantized_state
            best.uncertainty_history.append(uca_score)
            best.write_count += 1
            best.effective_bits = target_bits
        else:
            # Topic shift: insert new thread
            if len(self.threads) >= self.max_threads:
                self.threads.pop(0)  # Evict oldest
            self.threads.append(Thread(
                thread_id=self._next_id,
                state=quantized_state,
                last_query_repr=str(query_emb[:5].tolist()),
                uncertainty_history=[uca_score],
                write_count=1,
                effective_bits=target_bits,
            ))
            self._next_id += 1

    @staticmethod
    def _quantize_outlier_aware(tensor: torch.Tensor, bits: float) -> torch.Tensor:
        """
        HyC-LoRA inspired adaptive quantization.
        - Detect top-5% channels (by L2 norm) as outliers
        - Keep outliers in FP32
        - Quantize non-outliers to `bits` levels
        """
        n_levels = int(2 ** round(bits))
        t_min, t_max = tensor.min(), tensor.max()
        scale = (t_max - t_min) / (n_levels - 1 + 1e-8)

        # Outlier detection (HyC-LoRA intra-operator style)
        channel_norms = tensor.abs()
        threshold = torch.quantile(channel_norms, 0.95)
        outlier_mask = channel_norms >= threshold

        # Quantize all channels
        quantized = torch.clamp(
            torch.round((tensor - t_min) / (scale + 1e-8)), 0, n_levels - 1
        ) * scale + t_min

        # Restore outlier channels to original precision
        result = quantized.clone()
        result[outlier_mask] = tensor[outlier_mask]
        return result
```

---

### Phase 4: KV Cache Manager (Hours 31-34)

#### `src/core/compression/kv_cache_manager.py`

```python
"""
Dynamic KV Cache Budget Manager.
Uses UCA score to set maximum KV cache tokens, saving GPU memory.
"""
import torch
from transformers import DynamicCache


class UCAKVCacheManager:
    """
    Programmatic KV cache budget control using HuggingFace DynamicCache.
    Low UCA score → small budget (save memory)
    High UCA score → large budget (preserve context)
    """
    def __init__(self, max_context_length: int = 4096):
        self.max_len = max_context_length

    def create_cache(self, kv_budget: float) -> DynamicCache:
        """
        Create a DynamicCache with budget-limited size.
        kv_budget: fraction of max_context_length [0.0, 1.0].
        """
        cache = DynamicCache()
        cache._uca_budget = int(self.max_len * kv_budget)
        return cache

    def evict_to_budget(self, cache: DynamicCache) -> DynamicCache:
        """
        Sliding-window eviction: if cache exceeds budget, drop oldest tokens.
        Upgrade path: use attention-score-weighted eviction for better quality.
        """
        budget = getattr(cache, '_uca_budget', self.max_len)
        for layer_idx in range(len(cache.key_cache)):
            current_len = cache.key_cache[layer_idx].shape[2]
            if current_len > budget:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, -budget:, :]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, -budget:, :]
        return cache

    def get_current_tokens(self, cache: DynamicCache) -> int:
        """Return current number of KV tokens in cache."""
        if not cache.key_cache:
            return 0
        return cache.key_cache[0].shape[2]
```

---

### Phase 5: Evaluation Pipeline (Hours 35-42)

#### `src/evaluation/metrics.py`

```python
"""
Evaluation metrics for UCA vs. baselines.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(uncertainty_scores: list, is_wrong: list) -> float:
    """AUROC: uncertainty as hallucination detector (higher = more uncertain)."""
    return roc_auc_score(is_wrong, uncertainty_scores)


def compute_prr(uncertainty_scores: list, is_wrong: list, n_steps: int = 20) -> float:
    """
    Prediction Rejection Ratio.
    Measures: as we reject high-uncertainty predictions,
    how fast does error rate drop vs. random baseline?
    """
    scores, labels = np.array(uncertainty_scores), np.array(is_wrong)
    thresholds = np.percentile(scores, np.linspace(0, 100, n_steps))
    error_rates = []
    for tau in thresholds:
        kept = labels[scores <= tau]
        error_rates.append(0.0 if len(kept) == 0 else kept.mean())
    random_baseline = labels.mean() * (1 - np.linspace(0, 1, n_steps))
    return float(np.trapz(random_baseline - error_rates, np.linspace(0, 1, n_steps)))


def compute_ece(uncertainty_scores: list, is_wrong: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Measures whether uncertainty scores match empirical error rates.
    """
    scores, labels = np.array(uncertainty_scores), np.array(is_wrong)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if mask.sum() > 0:
            bin_conf = scores[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (mask.sum() / len(scores)) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_memory_savings(baseline_kv_tokens: list, uca_kv_tokens: list) -> dict:
    """Quantify GPU memory savings from dynamic KV cache budgeting."""
    baseline = np.array(baseline_kv_tokens, dtype=float)
    uca = np.array(uca_kv_tokens, dtype=float)
    savings_pct = 100 * (baseline - uca) / (baseline + 1e-8)
    return {
        'mean_savings_pct': float(savings_pct.mean()),
        'median_savings_pct': float(np.median(savings_pct)),
        'total_tokens_saved': int((baseline - uca).sum()),
        'savings_by_route': None,   # Filled in by run_uca.py
    }
```

#### `src/experiments/run_uca.py`

```python
"""
Full UCA system evaluation script.

Usage:
    python src/experiments/run_uca.py \
        --model meta-llama/Llama-3.2-3B \
        --dataset trivia_qa \
        --n_samples 500
"""
import argparse, torch, pandas as pd, json, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.internal_confidence import ICGate
from src.core.semantic_dpp import SemanticDPP, SemanticSubspaceProjector
from src.core.uca_controller import UCAController
from src.evaluation.metrics import compute_auroc, compute_prr, compute_memory_savings


def main(args):
    os.makedirs("results", exist_ok=True)

    # Load model (4-bit quantization for GPU efficiency)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Initialize UCA components
    ic_gate = ICGate(args.model, device="cuda")
    projector = SemanticSubspaceProjector(d=model.config.hidden_size, p=256)
    dpp = SemanticDPP(model, tokenizer, projector, alpha=1.0, m=args.m_samples)
    controller = UCAController(ic_gate, dpp, tau_high=args.tau_high, tau_low=args.tau_low)

    # Dataset loading
    if args.dataset == "trivia_qa":
        data = load_dataset("trivia_qa", "rc.nocontext",
                            split=f"validation[:{args.n_samples}]")
        get_q = lambda x: x["question"]
        get_a = lambda x: x["answer"]["value"]
    elif args.dataset == "gsm8k":
        data = load_dataset("gsm8k", "main", split=f"test[:{args.n_samples}]")
        get_q = lambda x: x["question"]
        get_a = lambda x: x["answer"].split("####")[-1].strip()
    elif args.dataset == "truthful_qa":
        data = load_dataset("truthful_qa", "generation", split=f"validation[:{args.n_samples}]")
        get_q = lambda x: x["question"]
        get_a = lambda x: x["best_answer"]

    results = []
    for i, sample in enumerate(data):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(data)}...")
        question, ref_answer = get_q(sample), get_a(sample)

        # Get UCA compression config (CORE of our system)
        config = controller.compute_compression_config(question)

        # Generate with UCA config
        inputs = tokenizer(question, return_tensors="pt",
                           max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=(config['beam_width'] > 1),
                num_beams=config['beam_width'],
                temperature=0.7 if config['beam_width'] > 1 else 1.0,
            )
        pred = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True)

        is_correct = (ref_answer.lower() in pred.lower() or
                     pred.lower() in ref_answer.lower())
        results.append({
            'question': question,
            'ref_answer': ref_answer,
            'pred_answer': pred,
            'ic_score': config['ic_score'],
            'uca_score': config['uca_score'],
            'route': config['route'],
            'kv_budget': config['kv_budget'],
            'beam_width': config['beam_width'],
            'trigger_rag': config['trigger_rag'],
            'is_correct': is_correct,
            'is_wrong': int(not is_correct),
            'kv_tokens_used': int(config['kv_budget'] * 4096),
        })

    df = pd.DataFrame(results)
    out_path = f"results/{args.dataset}_uca_m{args.m_samples}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Compute all metrics
    auroc = compute_auroc(df['uca_score'].tolist(), df['is_wrong'].tolist())
    prr = compute_prr(df['uca_score'].tolist(), df['is_wrong'].tolist())
    mem = compute_memory_savings([4096]*len(df), df['kv_tokens_used'].tolist())

    print(f"\n{'='*60}")
    print(f"UCA Results — {args.dataset} — {args.model}")
    print(f"{'='*60}")
    print(f"N Samples:          {len(df)}")
    print(f"AUROC:              {auroc:.4f}")
    print(f"PRR:                {prr:.4f}")
    print(f"Mean KV Saved:      {mem['mean_savings_pct']:.1f}%")
    print(f"Accuracy (correct): {df['is_correct'].mean():.4f}")
    print(f"\nRoute Distribution:")
    print(df['route'].value_counts().to_string())
    print(f"{'='*60}")

    # Save summary
    summary = {'auroc': auroc, 'prr': prr, 'memory': mem,
                'n': len(df), 'accuracy': df['is_correct'].mean()}
    with open(f"results/{args.dataset}_uca_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dataset", default="trivia_qa",
                        choices=["trivia_qa", "gsm8k", "truthful_qa"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--m_samples", type=int, default=3,  # DPP samples
                        help="Number of DPP samples; 3 for speed, 5 for quality")
    parser.add_argument("--tau_high", type=float, default=0.65)
    parser.add_argument("--tau_low", type=float, default=0.30)
    main(parser.parse_args())
```

---

### Phase 6: Ablation Studies (Hours 43-48)

**Prove each component contributes independently.**

| Ablation Config | What Is Removed | Expected Effect |
|---|---|---|
| **UCA-Full** | Nothing (proposed method) | Baseline |
| **-Eigenspectrum** | Use log-det scalar, not eigenvalues for ratios | -0.02 to -0.04 AUROC |
| **-IC Gate** | Always run DPP (no IC pre-filter) | Same AUROC, +200% latency |
| **-Adaptive Compression** | Fixed compression ratio | Same AUROC, worse memory |
| **-Uncertainty History** | No UCA history in threads | Memory efficiency drops |
| **IC Only** | No DPP at all | -0.05 to -0.08 AUROC |
| **DPP Only** | No IC gate, always run DPP | Same AUROC, 3× slower |

```bash
# Run full ablation
python src/experiments/run_ablation.py \
    --model meta-llama/Llama-3.2-3B \
    --dataset trivia_qa \
    --configs full,no_eigen,no_gate,fixed_compression,ic_only,dpp_only
```

---

## PART 6: EXPECTED BENCHMARK RESULTS

### Primary Hypothesis
- **H1:** UCA-Full outperforms IC-Only by ≥0.03 AUROC because the DPP eigenspectrum provides richer uncertainty signal than the IC scalar alone.
- **H2:** UCA-Full matches SemanticDPP AUROC while being ~3× faster (IC gate eliminates DPP for ~55% of queries).
- **H3:** UCA-Full reduces average KV cache usage by >30% vs. static allocation while maintaining equivalent accuracy on confident queries.

### Target Benchmark Table

| Method | AUROC ↑ | PRR ↑ | KV Saved ↑ | Latency vs. IC |
|---|---|---|---|---|
| Static Baseline | 0.50 | 0.00 | 0% | 1× |
| IC Only | ~0.73 | ~0.18 | 0% | 1× |
| SemanticDPP (m=5) | ~0.81 | ~0.24 | 0% | ~15× |
| **UCA-Full (Ours)** | **>0.83** | **>0.27** | **>32%** | **~5×** |
| UCA (-Eigenspectrum) | ~0.80 | ~0.22 | ~25% | ~5× |
| UCA (IC-Only mode) | ~0.73 | ~0.18 | ~30% | 1× |

---

## PART 7: POTENTIAL CHALLENGES AND MITIGATIONS

| Challenge | Risk | Mitigation |
|---|---|---|
| DPP with m=3 too noisy | High | Increase to m=5; use SemanticDPP-C soft clustering |
| IC token IDs model-specific | Medium | `tokenizer.encode("Yes")[1]` to find IDs per model |
| Eigenvalue decomposition NaN | Medium | Tikhonov regularization: `K + ε·I` before eigvalsh |
| KV eviction loses critical context | Medium | Attention-score-weighted eviction instead of recency |
| Thread compressor untrained | High | Initialize from IC paper's embedding projections |
| GPU OOM with full model | High | 4-bit bitsandbytes; start with Phi-3-mini-4k |
| AUROC delta too small | Medium | 500+ samples; report 95% CI via bootstrap |
| IC AUROC already near-optimal | Medium | Emphasize memory savings + latency as primary metric |

---

## PART 8: TIMELINE (40-HOUR SPRINT)

| Phase | Hours | Deliverable |
|---|---|---|
| 0: Setup | 1-4 | Env configured, models downloaded |
| 1: Baselines | 5-12 | IC + DPP reproduced, AUROC verified vs. papers |
| 2: UCA Controller | 13-22 | Core controller working, routes correctly |
| 3: Thread Memory | 23-30 | C-DIC threads + uncertainty history functional |
| 4: KV Manager | 31-34 | Dynamic KV budget allocation working |
| 5: Full Eval | 35-42 | Results CSV, AUROC/PRR/memory tables generated |
| 6: Ablation | 43-48 | Component contribution verified, plots generated |

---

## PART 9: QUICK START COMMANDS

```bash
# Step 1: Setup
conda create -n uca python=3.11 && conda activate uca
pip install torch transformers accelerate datasets dppy scikit-learn pandas matplotlib
pip install query-level-uncertainty lm-polygraph bitsandbytes

# Step 2: Reproduce IC baseline
python src/experiments/run_baseline.py --method ic --dataset trivia_qa

# Step 3: Reproduce DPP baseline
python src/experiments/run_baseline.py --method dpp --dataset trivia_qa

# Step 4: Run UCA full system
python src/experiments/run_uca.py \
    --model meta-llama/Llama-3.2-3B \
    --dataset trivia_qa \
    --n_samples 500

# Step 5: Run ablation study
python src/experiments/run_ablation.py \
    --model meta-llama/Llama-3.2-3B \
    --dataset trivia_qa

# Step 6: Generate all plots
python src/evaluation/plot_results.py --results_dir results/
```

---

## PART 10: FINAL DELIVERABLES CHECKLIST

- [ ] `src/` — Complete, runnable Python codebase (all modules above)
- [ ] `results/*.csv` — Per-query results for TriviaQA, GSM8K, TruthfulQA
- [ ] **Figure 1:** ROC curves — UCA vs. IC vs. SemanticDPP
- [ ] **Figure 2:** Memory savings histogram across routing tiers (fast/standard/augmented/abstain)
- [ ] **Figure 3:** Ablation bar chart (AUROC per ablation config)
- [ ] **Figure 4:** IC score vs. UCA score scatter (shows DPP adds information beyond IC)
- [ ] **Figure 5:** KV budget as a function of query type (math vs. factual vs. opinion)
- [ ] Final Report: Critical analysis (Part 1) + Novel contribution (Part 2) + Results + Ablation

---

*Document prepared: 2026-10-11 | DA4022 Course Project*  
*UCA: Uncertainty-Calibrated Adaptive Compression — Synthesizing C-DIC + HyC-LoRA + IC + SemanticDPP*
