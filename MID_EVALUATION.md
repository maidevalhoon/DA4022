# Project 7: Uncertainty for Optimal Memory Allocation
**DA4022 Mid-Evaluation Report**

**Team:**
Deval Singhal (230101035) · Jayant Kumar (230101050) · Sonal Mangesh Wagh (230108054)
Sristi Vats (230122057) · Shourya Veer Singh (230106069)

---

## 1. Background

Large Language Models (LLMs) are increasingly deployed in settings where computational resources are limited. A key bottleneck is how memory and compute are allocated during inference — particularly KV cache size and generation strategy (greedy vs. beam search). Currently, these are fixed regardless of how confident the model is in its output.

Two recent ICLR 2026 papers address uncertainty estimation in LLMs from complementary angles:

**Paper 1 — Query-Level Uncertainty (Internal Confidence):** Proposes estimating model uncertainty *before* generating any tokens, via a single forward pass. The method, called Internal Confidence (IC), probes `P(YES)` logits at every (layer, token) position and aggregates them using Attenuated Encoding centered at the last layer and last token. This produces a cheap, training-free uncertainty scalar that predicts whether the model can correctly answer a query.

**Paper 2 — SemanticDPP:** Proposes estimating uncertainty *after* sampling a small set of responses, by measuring their semantic diversity using a Determinantal Point Process (DPP) over internal model embeddings. A learned projection maps embeddings to a semantic subspace, and the log-determinant of the resulting kernel matrix captures how diverse (uncertain) the responses are. An extension, SemanticDPP-C, uses greedy MAP clustering to identify the number of semantically distinct clusters.

Neither paper connects uncertainty to downstream memory or compute allocation.

---

## 2. Research Gap and Novel Contribution

### 2.1 What the Papers Leave Open

| Aspect | Paper 1 (IC) | Paper 2 (SemanticDPP) |
|---|---|---|
| Uncertainty type | Pre-generation scalar | Post-generation diversity |
| Computation cost | ~1 forward pass (zero decode) | m forward passes (m ≥ 3) |
| Output | Single IC score ∈ [0,1] | log-det scalar + kernel matrix |
| Downstream use | Binary: trigger RAG / abstain | Binary: abstain threshold |
| Memory allocation | ✗ Not addressed | ✗ Not addressed |
| Adaptive compression | ✗ Not addressed | ✗ Not addressed |

**The core gap:** Both papers produce uncertainty signals but use them only as binary on/off switches. Neither uses uncertainty as a *continuous control variable* to modulate how much memory or compute is allocated per query.

### 2.2 Our Novel Contribution

We propose **Uncertainty-Calibrated Adaptive Compression (UCA)**: a system that uses IC as a zero-cost pre-filter and the DPP eigenspectrum as a richer, per-dimension uncertainty signal to continuously control KV cache budget and generation strategy.

Three specific contributions beyond both papers:

1. **IC-Gated DPP Cascade:** IC pre-filters queries. Only uncertain queries (~40–50%) proceed to the more expensive DPP computation, saving roughly half the DPP overhead.

2. **DPP Eigenspectrum as Memory Oracle:** Paper 2 computes `L_S` but discards all per-dimension structure, retaining only `log det(L_S)`. We expose the eigenspectrum `{σ₁, …, σₘ}` of `L_S` and map each eigenvalue to a per-dimension compression ratio:
   ```
   rᵢ = clip(1 − α · σᵢ / σ_max,  r_min,  r_max)
   ```
   High σᵢ (contested semantic dimension) → less compression. Low σᵢ → aggressive compression.

3. **Continuous KV Budget Control:** The combined UCA score drives KV cache size and beam width continuously, rather than a binary choice:
   ```
   UCA = 0.4 · (1 − IC) + 0.6 · DPP_uncertainty
   KV_budget = 0.25 + 0.75 · UCA
   ```

---

## 3. Proposed Architecture

```
INPUT QUERY q
      │
      ▼
┌─────────────────────────────────┐
│  TIER 0: Internal Confidence    │  ← Paper 1 (IC)
│  Single forward pass            │
│  IC(q) = Σ w(l,n)·P(YES|h_n^l) │
└──────────────┬──────────────────┘
               │
   IC ≥ τ_high │                   IC < τ_low
  (Confident)  │               (Very Uncertain)
       ┌───────┤──────────────────────┐
       ▼       │ (Uncertain zone)     ▼
  FAST PATH    ▼               ABSTAIN / RAG
  Max compress ┌─────────────────┐   Full KV cache
  25% KV cache │ TIER 1: DPP     │   Beam width = 4
  Greedy decode│ Sample m=3      │
               │ Build L_S kernel│  ← Paper 2 (SemanticDPP)
               │ Eigendecompose  │
               │ L_S = UΣUᵀ      │
               └────────┬────────┘
                        │
               ┌────────▼────────┐
               │ Compression     │
               │ rᵢ = f(σᵢ/σmax)│
               └────────┬────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  KV Budget = f(UCA)         Beam Width = f(UCA)
  [0.25 → 1.0]               [1 → 4]
```

**Key design decisions grounded in the papers:**
- `τ_high = 0.65`, `τ_low = 0.30` — informed by IC's reported adaptive RAG threshold experiments (Paper 1, Section 5)
- `α = 1.0` in Attenuated Encoding — fixed per Paper 1, Section 4.1 (training-free)
- `m = 3` DPP samples — Paper 2 uses m=5 for peak AUROC; we use 3 since IC gates most queries
- DPP-C cluster count `k = max(1, round(trace(K)))` — Paper 2, Algorithm 1; used as beam width floor

---

## 4. Implementation Progress

### 4.1 Completed

- **Environment setup:** Python 3.11, PyTorch 2.3, HuggingFace Transformers, `query-level-uncertainty` package installed and verified.

- **IC Gate (`src/core/internal_confidence.py`):** Implemented `ICGate` class wrapping the official `tigerchen52/query_level_uncertainty` package. The self-evaluation prompt template and Attenuated Encoding with `α=1.0` are in place. Tested on a small batch of TriviaQA samples with Llama-3.2-3B; IC scores are produced correctly and routing into fast/uncertain/abstain buckets is working.

- **SemanticDPP (`src/core/semantic_dpp.py`):** `SemanticSubspaceProjector` (orthogonal-initialized single-layer NN, `d→256`) and kernel matrix computation `K(xᵢ,xⱼ) = exp(−α||φ(xᵢ)−φ(xⱼ)||²)` implemented. The `uncertainty_score()` function returns `log_det`, normalized uncertainty, eigenvalues, and DPP-C cluster count. Eigenvalue extraction via `torch.linalg.eigvalsh` verified numerically.

- **UCA Controller (`src/core/uca_controller.py`):** Wires IC and DPP together. `compute_compression_config()` produces route, KV budget, beam width, and per-dimension compression ratios for a given query. Tested manually on a handful of queries; routing logic is correct.

### 4.2 In Progress

- **Projector training:** The `SemanticSubspaceProjector` needs to be trained by maximizing the DPP marginal likelihood `log det(L_Y) − log det(L + I)` on pseudolabeled TriviaQA clusters. Training loop is written; optimization not yet converged.

- **Baseline IC reproduction:** Running IC on TriviaQA validation split (Llama-3.2-3B) to verify AUROC against paper numbers. Partially processed.

### 4.3 Not Yet Started

- Full UCA end-to-end evaluation (`run_uca.py`)
- SemanticDPP standalone AUROC measurement
- Ablation experiments

---

## 5. Evaluation Plan

We will evaluate UCA against two standalone baselines drawn from the two provided papers.

**Baselines:**
- **IC Only** (Paper 1 standalone): IC score alone for routing; no DPP.
- **SemanticDPP Only** (Paper 2 standalone): DPP for every query; log-det for abstain threshold.

**Metrics:**
- **AUROC:** uncertainty score vs. correctness label (primary metric, used in both papers).
- **PRR (Prediction Rejection Ratio):** how fast the error rate drops as high-uncertainty queries are rejected.
- **Average KV tokens used:** measures memory savings from adaptive budgeting.
- **Latency per query:** IC adds ~1 forward pass; DPP adds ~3 generation passes.

**Datasets:** TriviaQA, GSM8K (same as Paper 1 evaluation). Model: Llama-3.2-3B.

---

## References

1. Chen, L., de Melo, G., Suchanek, F. M., & Varoquaux, G. (2026). *Query-Level Uncertainty in Large Language Models*. ICLR 2026. https://openreview.net/pdf?id=11QZITAMUO

2. Hamer, J., Snoek, J., et al. (2026). *SemanticDPP: Efficient Uncertainty Quantification in LLMs*. ICLR 2026. https://openreview.net/pdf?id=C8yjlrx4Wo
