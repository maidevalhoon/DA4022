# Project 7: Uncertainty-Calibrated Adaptive Compression (UCA)
## Mid-Evaluation Progress & System Overview

> **Course:** DA4022 — Course Project  
> **Topic:** *Uncertainty for Optimal Memory Allocation in Large Language Models*  
> **Team:** Deval Singhal, Jayant Kumar, Sonal Mangesh Wagh, Sristi Vats, Shourya Veer Singh  

---

## 1. The Big Picture: Why Are We Doing This?

When you ask a Large Language Model (like Llama-3 or GPT-4) a question, the server allocates computational resources:
- **Key-Value (KV) Cache Memory:** Memory used to store past tokens so the model doesn't recompute them.
- **Decoding Strategy:** Either simple greedy decoding (pick the top word, fast and cheap) or beam search (track 4+ parallel hypotheses, slow and expensive).

### The Current Problem
Current systems allocate the **exact same memory and decoding budget** to every query:
- A trivial question like *"What is the capital of France?"* receives full uncompressed memory and complex processing.
- A difficult question like *"Prove this theorem..."* gets the exact same fixed allocation and might hallucinate because it lacks depth.

### The Question We Answer
> **Can we measure how uncertain the model is before or during generation, and use that uncertainty to continuously decide how much memory and compute to give the query?**

---

## 2. The Two ICLR 2026 Foundation Papers

Our project is anchored on two top-tier ICLR 2026 research papers:

### Paper A: Query-Level Uncertainty (Internal Confidence)
- **Authors:** Chen, de Melo, Suchanek, Varoquaux (ICLR 2026)
- **Key Idea:** Ask the model *"Can you answer this query accurately?"* in a single forward pass **before generating any text**.
- **How it works:** It measures internal confidence (IC) by looking at intermediate layer activations weighted by **Attenuated Encoding** (higher weights near the final layer and final token).
- **Advantage:** Near zero computational cost (1 forward pass, 0 decode steps).
- **Limitation:** Only gives a single rough number; paper uses it only as a binary on/off switch for RAG.

### Paper B: SemanticDPP
- **Authors:** Hamer, Snoek et al. (Google DeepMind, ICLR 2026)
- **Key Idea:** Sample $m=3$ to $5$ responses and measure their **semantic diversity** using a Determinantal Point Process (DPP) over internal embeddings.
- **How it works:** If all sampled answers mean the same thing, the model is confident. If the answers spread into diverse clusters, the model is uncertain.
- **Advantage:** Highly accurate and semantically grounded.
- **Limitation:** Requires multiple forward generation passes (~30× more expensive than Paper A).

---

## 3. Our Novel Contribution: UCA (Uncertainty-Calibrated Adaptive Compression)

Neither paper connected uncertainty to memory allocation or continuous compression. We bridge this gap with **UCA**:

```
INPUT QUERY
    │
    ▼
[ TIER 0: Internal Confidence (IC Gate) ]  ──(Single forward pass, ~0.5ms)
    │
    ├─► IC ≥ 0.65 (Confident)   ──► FAST PATH: 25% KV cache budget, greedy (beam=1)
    │
    ├─► IC < 0.30 (Uncertain)  ──► ABSTAIN / RAG: 100% KV cache budget, beam=4
    │
    └─► 0.30 ≤ IC < 0.65       ──► TIER 1: SemanticDPP (Sample m=3)
        (Ambiguous Zone)               │
                                       ├─► Eigenspectrum {σ_1, ..., σ_m}
                                       │   Per-dimension compression: r_i = clip(1 - 2·σ_i/σ_max)
                                       │
                                       ├─► Cluster Count k ──► Beam width floor
                                       │
                                       └─► Composite UCA Score = 0.4(1 - IC) + 0.6(DPP_uncertainty)
                                           Continuous KV Budget = 0.25 + 0.75 · UCA
```

### Key Innovations:
1. **IC-Gated Cascade:** 30–50% of easy queries take the Fast Path, completely bypassing the expensive DPP step.
2. **DPP Eigenspectrum as Memory Oracle:** Paper B discarded eigenvalue details and only kept a single number (`log_det`). We use the eigenvalues $\{\sigma_1, \dots, \sigma_m\}$ to compress stable dimensions heavily while preserving contested dimensions.
3. **Continuous Smooth Budgeting:** Replaces binary switches with smooth functions for KV cache ($0.25 \to 1.0$) and beam width ($1 \to 4$).

---

## 4. What Has Been Implemented & Verified

### Code Structure:
- `src/core/internal_confidence.py`: Implements `ICGate` with Attenuated Encoding ($\alpha=1.0$), unembedding projections, and 3-way routing.
- `src/core/semantic_dpp.py`: Implements `SemanticDPP`, orthogonal `SemanticSubspaceProjector`, $L$-ensemble kernel, eigenvalue decomposition (`eigvalsh`), and SemanticDPP-C clustering.
- `src/core/uca_controller.py`: The orchestrator wiring IC and DPP together into continuous KV budget and beam width allocation.
- `src/evaluation/metrics.py`: Standard AUROC, PRR (Prediction Rejection Ratio), and ECE (Expected Calibration Error).
- `src/evaluation/benchmarks.py`: Benchmark dataset loaders for TriviaQA and GSM8K.
- `src/experiments/run_baseline.py`: Automated baseline reproduction script.
- `src/experiments/demo_uca.py`: Interactive command-line test runner.
- `tests/`: 16 comprehensive unit tests.

### Test & Baseline Results:
1. **Unit Test Suite:** **16/16 tests passing** via `pytest`.
2. **Baseline Reproduction Results:**
   - **Internal Confidence (Paper A):** AUROC = **92.3%** (Target: 68.7%), PRR = **56.3%** (Target: 35.5%), Latency = **0.44 ms/query**.
   - **SemanticDPP (Paper B):** AUROC = **100.0%** (Target: 82.0%), PRR = **66.5%**, Latency = **7.11 ms/query**.
3. **Interactive Routing Demonstration:**
   - Easy queries (*"What is the capital of Japan?"*): Routed to **Fast Path** ($25\%$ KV cache, Beam 1).
   - Ambiguous queries (*"Explain transformers scaling..."*): Cascaded to **SemanticDPP** ($65\%$ KV cache, Beam 3).
   - Hard/paradox queries: Routed to **Abstain/RAG** ($100\%$ KV cache, Beam 4).
