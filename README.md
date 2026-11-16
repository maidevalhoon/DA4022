# UCA — Uncertainty-Calibrated Allocation of KV-cache memory

DA4022 course project 7: *Can we come up with a method to use uncertainty for
optimal memory allocation?*

We measure how sure a language model is about its answer and use that to decide
how much KV-cache memory (how many retrieved passages) each question gets.
Every question starts with no passages; it only gets more memory while the
model is still unsure. Confident questions stay cheap, and the saved memory is
exactly what a fixed "same for everyone" setting wastes.

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — the idea, the method, and the results, explained simply.
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — review of the earlier work, the plan, and the progress log.

Built on two papers:
- **Paper A** — *Query-Level Uncertainty in Large Language Models*, Chen, de Melo, Suchanek, Varoquaux, ICLR 2026 (Internal Confidence).
- **Paper B** — *SemanticDPP: Efficient Uncertainty Quantification in LLMs*, anonymous, under review at ICLR 2026.

## Setup

Python 3.11. Tested on an Apple M1 laptop with 8 GB RAM (the model runs on the
Apple GPU); a CUDA GPU or plain CPU also works.

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install torch transformers datasets scikit-learn numpy matplotlib pytest
```

The model, `Qwen/Qwen2.5-1.5B-Instruct` (3 GB, no licence gate), downloads
automatically the first time it is used.

## Run

All commands are run from the repository root.

```bash
python -m pytest tests -q
```
Unit tests for the maths (weights, DPP, allocator, metrics). Takes ~10 seconds, needs no model.

```bash
python -m src.experiments.prepare_data --n 600
```
Step 1. Streams 600 TriviaQA questions, retrieves the top-8 passages for each with BM25, and writes `data/triviaqa_rag.json`. Already included in the repo, so you can skip this.

```bash
python -m src.experiments.collect --limit 400
```
Step 2. Runs the real model on 400 questions at every memory level (0, 1, 2, 4, 8 passages): the answer, its probability, Internal Confidence, and 5 sampled answers for SemanticDPP. Writes `results/runs.jsonl` and `results/runs.npz`. Slow (about 20 s per question on the M1, so over two hours) but resumable — run it again and it continues. Also already included. On an 8 GB machine, close other apps and do not start a second model process while it runs.

```bash
python -m src.experiments.analyze
```
Step 3. Computes every table and figure from the saved runs. About a minute, no model needed. Writes `results/summary.json` and `results/figures/`.

```bash
python -m src.experiments.demo --n 10
```
Live demo on 10 questions the experiments never saw: answers each one with as few passages as its confidence allows (at most 2 by default) and reports the KV memory used against giving everyone 2 passages.

```bash
python -m src.experiments.demo --n 10 --max_passages 8 --threshold 0.8
```
Same, with a higher memory ceiling and a stricter confidence threshold.

```bash
python -m src.experiments.demo --question "Who wrote the novel Dracula?"
```
Confidence readings for any single question.

## Layout

```text
src/core/llm.py                  model wrapper: answer + its probability, sampled answers, P(Yes) per layer, KV bytes per token
src/core/internal_confidence.py  Paper A: Internal Confidence
src/core/semantic_dpp.py         Paper B: SemanticDPP with a learned kernel
src/core/allocator.py            ours: calibration, escalation policy, one-shot allocation, baselines
src/evaluation/data.py           TriviaQA loading, BM25 retrieval, answer checking
src/evaluation/metrics.py        AUROC, PRR, ECE
src/experiments/                 prepare_data, collect, analyze, demo
tests/test_project.py            unit tests
data/triviaqa_rag.json           600 questions with retrieved passages (400 used for experiments, the rest for the demo)
results/                         saved model runs, summary.json, figures/
temperory/                       the two papers and the course brief
mid_report/, MID_EVALUATION.*    mid-term submission (kept as submitted)
```
