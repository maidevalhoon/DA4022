# Dynamic Memory Allocation in LLMs using Multi-Tier Uncertainty Quantification

**Course:** DA4022 Course Project  
**Topic Anchor (#8):** *"Can we come up with a method to use uncertainty for optimal memory allocation?"*  
**Project Proposal Document:** [`proposal.txt`](./proposal.txt)

---

## 📌 Executive Summary

Current Large Language Model (LLM) serving frameworks allocate a static computational and memory budget (e.g., fixed maximum KV cache sizes, full context windows, and fixed decoding strategies) regardless of query difficulty. This causes:
1. **Resource Waste:** Over-allocating memory and compute to trivial queries.
2. **Hallucination Risk:** Under-allocating resources (reasoning depth, external context) when facing ambiguous or knowledge-intensive prompts.

Existing uncertainty estimation metrics (such as Semantic Entropy or SemanticDPP) rely on multi-sample generations, incurring substantial computational overhead. This project implements a **Multi-Tier Adaptive Inference Controller**:
* **Tier 1 (Pre-Generation):** Near-zero cost *Internal Confidence* routing via a single forward pass over input prompt hidden states.
* **Tier 2 (Post-Generation Verification):** *SemanticDPP* (Determinantal Point Processes over sampled output representations) only for queries flagged as uncertain.
* **Controller Action:** Dynamic Key-Value (KV) cache sizing, selective RAG triggering, or cascading to a higher-capacity model.

---

## 📚 1. Core Reference Papers

### Primary Anchors (Topic #8)

1. **Query-Level Uncertainty in Large Language Models (Tier 1)**
   * **Authors:** Lihu Chen, Gerard de Melo, Fabian M. Suchanek, Gaël Varoquaux
   * **Venue:** ICLR 2026
   * **Local PDF:** [`13943_Query_Level_Uncertainty_.pdf`](./13943_Query_Level_Uncertainty_.pdf)
   * **OpenReview:** [openreview.net/pdf?id=ZGeVMaiIlb](https://openreview.net/pdf?id=ZGeVMaiIlb)
   * **Official Codebase:** [github.com/tigerchen52/query_level_uncertainty](https://github.com/tigerchen52/query_level_uncertainty)
   * **Key Contribution:** Zero-generation *Internal Confidence (IC)* using a single forward pass over query hidden states weighted by *Attenuated Encoding*.

2. **SemanticDPP: Efficient Uncertainty Quantification in LLMs (Tier 2)**
   * **Authors:** Anonymous (ICLR 2026)
   * **Local PDF:** [`9298_SemanticDPP_Efficient_Unc.pdf`](./9298_SemanticDPP_Efficient_Unc.pdf)
   * **OpenReview:** [openreview.net/pdf?id=WU86J4uM9V](https://openreview.net/pdf?id=WU86J4uM9V)
   * **Key Contribution:** Fast unsupervised semantic diversity modeling using Determinantal Point Processes ($L$-ensembles) and kernel log-determinants $\log P(S) = \log \det(L_S) - \log \det(L + I)$.

---

### Foundational & Baseline Literature

3. **Semantic Entropy (State-of-the-art Sampling Baseline)**
   * *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation* (Kuhn et al., ICLR 2023) — [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
   * *Detecting Hallucinations in Large Language Models Using Semantic Entropy* (Farquhar et al., Nature 2024) — [Nature 630:625–630](https://www.nature.com/articles/s41586-024-07421-0)
   * **GitHub:** [github.com/jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty)

4. **Semantic Entropy Probes (Linear Activation Baseline)**
   * *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs* (Kossen et al., 2024) — [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
   * **GitHub:** [github.com/OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes)

5. **INSIDE / EigenScore (Internal State Baseline)**
   * *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection* (Chen et al., ICLR 2024) — [OpenReview Forum](https://openreview.net/forum?id=V2cQ5Hqf3d)
   * **GitHub Implementation:** [github.com/0xkerem/halluscope](https://github.com/0xkerem/halluscope)

6. **LM-Polygraph (Evaluation & Benchmarking Suite)**
   * *Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph* (Vashurin et al., TACL 2025) — [TACL](https://transacl.org/)
   * **GitHub:** [github.com/IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph)

7. **Model Cascading**
   * *Language Model Cascades: Token-Level Uncertainty and Beyond* (Gupta et al., ICLR 2024) — [arXiv:2404.10137](https://arxiv.org/abs/2404.10137)

---

## 🛠️ 2. Essential GitHub Repositories

| Repository | Source Link | Role in Project |
| :--- | :--- | :--- |
| **Query-Level Uncertainty** | [tigerchen52/query_level_uncertainty](https://github.com/tigerchen52/query_level_uncertainty) | Primary reference code for single-pass Internal Confidence computation and layer/token aggregation. |
| **LM-Polygraph** | [IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph) | Ready-to-use framework for standard uncertainty metrics, AUROC computation, and evaluation pipelines. |
| **Semantic Uncertainty** | [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) | Ground-truth semantic clustering via DeBERTa/LLM bidirectional entailment and the Leader algorithm. |
| **DPPy** | [guilgour/DPPy](https://github.com/guilgour/DPPy) | Determinantal Point Processes library in Python (L-ensembles, MAP inference, sampling algorithms). |
| **Semantic Entropy Probes** | [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes) | Feature extraction and linear probing from hidden activations. |
| **Transformers (`DynamicCache`)**| [huggingface/transformers](https://github.com/huggingface/transformers) | Programmatic control over KV cache allocation and memory profiling during generation. |

---

## 📊 3. Benchmark Datasets

All target benchmark datasets are accessible via Hugging Face `datasets`:

1. **TriviaQA** ([`trivia_qa`](https://huggingface.co/datasets/trivia_qa))
   * *Target:* Factual recall, open-domain question answering, and knowledge boundary identification.
   * *Configuration:* `rc.nocontext` split.

2. **GSM8K** ([`gsm8k`](https://huggingface.co/datasets/gsm8k))
   * *Target:* Multi-step mathematical reasoning and reasoning chain verification.
   * *Configuration:* `main` split.

3. **TruthfulQA** ([`truthful_qa`](https://huggingface.co/datasets/truthful_qa))
   * *Target:* Hallucination detection, common misconceptions, and factuality.
   * *Configuration:* `generation` split.

4. **SciQ** ([`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq))
   * *Target:* Scientific reasoning and domain-specific knowledge boundary testing.

---

## 🏗️ 4. System Architecture & Workflow

```
                                 ┌──────────────┐
                                 │ Prompt Query │
                                 └──────┬───────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │  TIER 1: Pre-Generation Routing Filter  │
                   │  - Single forward pass (1 prompt pass)  │
                   │  - Intercept hidden states across layers │
                   │  - Compute Internal Confidence (IC)     │
                   └────────────────────┬────────────────────┘
                                        │
                       High Confidence  │  Low Confidence
                       (IC >= tau_1)    │  (IC < tau_1)
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
          ┌───────────────────────┐           ┌────────────────────────┐
          │  FAST PATH            │           │ TIER 2: SemanticDPP    │
          │  - Minimal KV Cache   │           │ - Sample m responses   │
          │  - Greedy Decoding    │           │ - Map embeddings phi(x)│
          │  - Base Model (3B/8B) │           │ - Compute log det K_S  │
          │  - 0 extra calls      │           └───────────┬────────────┘
          └───────────────────────┘                       │
                                       Low Diversity      │  High Diversity
                                       (DPP < tau_2)      │  (DPP >= tau_2)
                                      ┌───────────────────┴────────────┐
                                      ▼                                ▼
                          ┌───────────────────────┐       ┌────────────────────────┐
                          │  STANDARD RESPONSE    │       │ ADAPTIVE ESCALATION    │
                          │  - Direct Generation  │       │ - Dynamic KV Expansion │
                          │  - Confident Output   │       │ - Model Cascade (->14B)│
                          │                       │       │ - Trigger RAG Context  │
                          └───────────────────────┘       └────────────────────────┘
```

---

## 💻 5. Environment Setup & Dependencies

### Python Package Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets evaluate
pip install dppy scikit-learn scipy rouge-score
pip install lm-polygraph
```

### Hardware Requirements
* **Recommended:** NVIDIA GPU with $\ge 16$ GB VRAM (e.g., Tesla P100, RTX 3090, RTX 4090, A100).
* **Models Tested:**
  * Small Base Model: `meta-llama/Llama-3.2-3B` or `microsoft/Phi-3-mini-4k-instruct`
  * Cascade Target: `meta-llama/Llama-3.1-8B-Instruct` (with 4-bit/8-bit bitsandbytes quantization)

---

## 📈 6. Evaluation Metrics

1. **AUROC (Area Under ROC Curve):** Measures discrimination between correct vs. incorrect / hallucinated answers across varying rejection thresholds.
2. **PRR (Prediction Rejection Ratio):** Evaluates how effectively the model reduces error rates as samples are rejected.
3. **GPU Peak Memory Reduction (MB / GB):** Quantifies memory saved via dynamic KV cache budgeting compared to a static maximum allocation baseline.
4. **End-to-End Inference Latency (s):** Wall-clock time comparison proving the efficiency advantages over naive multi-sample semantic entropy.
