# UCA: Uncertainty-Calibrated Adaptive Compression

Implementation of **Uncertainty-Calibrated Adaptive Compression (UCA)** for optimal memory and compute allocation in LLMs, built on:
- **Paper A:** *Query-Level Uncertainty in LLMs* (Chen et al., ICLR 2026)
- **Paper B:** *SemanticDPP: Efficient Uncertainty Quantification in LLMs* (Hamer et al., Google DeepMind, ICLR 2026)

---

## 🚀 Quick Start Commands

### 1. Environment Setup

Activate the project virtual environment (pre-configured with Python 3.11, PyTorch with MPS, Transformers, and Datasets):

```bash
# Activate existing environment
source .venv/bin/activate
```

*(Optional fresh setup from scratch):*
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch transformers scikit-learn scipy numpy datasets accelerate pytest pandas weasyprint
```

---

### 2. Run All Unit Tests

Verify all 16 mathematical components, matrix properties, and routing logic:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

Expected output:
```text
16 passed in ~3.2s
```

---

### 3. Run the Interactive UCA Controller Demo

Test how UCA dynamically routes queries across Fast Path (25% memory), Cascaded DPP (adaptive memory), and Abstain (100% memory):

```bash
# Run on built-in multi-difficulty test suite
PYTHONPATH=. .venv/bin/python src/experiments/demo_uca.py

# Test any custom query
PYTHONPATH=. .venv/bin/python src/experiments/demo_uca.py --query "What is the capital of Japan?"
PYTHONPATH=. .venv/bin/python src/experiments/demo_uca.py --query "Prove that every planar graph is 4-colorable."
```

Sample output:
```text
==========================================================================================
                     UCA ADAPTIVE COMPRESSION CONTROLLER DEMO
==========================================================================================
Query                                            | Route      | IC    | UCA   | KV%   | Beam | Tiers Used
---------------------------------------------------------------------------------------------------------
What is the capital of Japan?                    | fast_path  | 0.66  | 0.34  | 25%   | 1    | tier_0
In what year did the Apollo 11 mission land o... | standard   | 0.39  | 0.53  | 64%   | 3    | tier_0_and_tier_1
Prove that every planar graph is 4-colorable ... | abstain    | 0.15  | 0.85  | 100%  | 4    | tier_0
==========================================================================================
```

---

### 4. Reproduce Paper Baselines

Reproduce Internal Confidence (Paper A) and SemanticDPP (Paper B) metrics on TriviaQA and GSM8K:

```bash
# Run both baselines on TriviaQA (50 samples)
PYTHONPATH=. .venv/bin/python src/experiments/run_baseline.py --method both --dataset trivia_qa --n_samples 50

# Run both baselines on GSM8K
PYTHONPATH=. .venv/bin/python src/experiments/run_baseline.py --method both --dataset gsm8k --n_samples 50

# Run only Internal Confidence (Paper A)
PYTHONPATH=. .venv/bin/python src/experiments/run_baseline.py --method ic --dataset trivia_qa

# Run only SemanticDPP (Paper B)
PYTHONPATH=. .venv/bin/python src/experiments/run_baseline.py --method dpp --dataset trivia_qa
```

Results are saved to:
- `results/baseline_ic_trivia_qa.csv` & `.json`
- `results/baseline_dpp_trivia_qa.csv` & `.json`
- `results/baseline_summary_trivia_qa.json`

---

### 5. Compile Mid-Evaluation Report PDF

To recompile the official 3-page Mid-Evaluation Report PDF:

```bash
python3 generate_mid_evaluation_pdf.py
```

Outputs:
- `MID_EVALUATION.pdf` (Verified exactly 3 pages, vector SVG flowchart, zero black-box glyphs).

---

## 📁 Repository Structure

```text
DA4022/
├── configs/
│   └── default.yaml                # System hyperparameters (tau_high, tau_low, alpha, etc.)
├── src/
│   ├── core/
│   │   ├── internal_confidence.py  # Paper A: Internal Confidence Gate (Tier 0)
│   │   ├── semantic_dpp.py         # Paper B: SemanticDPP & Eigenspectrum (Tier 1)
│   │   └── uca_controller.py       # Our Novel Contribution: Continuous UCA Controller
│   ├── evaluation/
│   │   ├── metrics.py              # AUROC, PRR, ECE, Memory Savings
│   │   └── benchmarks.py           # TriviaQA & GSM8K data loaders
│   └── experiments/
│       ├── run_baseline.py         # Paper A & B baseline reproduction runner
│       └── demo_uca.py             # Interactive routing demonstration
├── tests/
│   ├── test_internal_confidence.py # IC gate & Attenuated Encoding unit tests
│   ├── test_semantic_dpp.py        # Projector & DPP kernel unit tests
│   ├── test_metrics.py             # AUROC / PRR / ECE mathematical tests
│   └── test_uca_controller.py      # Multi-tier routing & allocation unit tests
├── results/                        # Generated benchmark CSVs & JSON summaries
├── MID_EVALUATION.md               # Mid-evaluation report markdown source
├── MID_EVALUATION.pdf              # Compiled 3-page submission PDF
├── PROJECT_OVERVIEW.md             # Simple conceptual explanation of the system
└── README.md                       # Command & execution reference manual
```
