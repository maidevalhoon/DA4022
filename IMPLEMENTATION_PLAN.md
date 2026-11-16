# Implementation plan and progress log

**Project 7 - Uncertainty for optimal memory allocation (DA4022)**
Final report, video and working code are due **26 November 2026**.

This file is the working document: what we found when we reviewed the earlier
work, what we decided to change, the step-by-step plan, and a log that we update
as each step is finished. For the explanation of the idea and the results, read
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

## 1. Review of the work done before 16 November

We went through every file in the repo. The short version: the earlier work had
a reasonable idea and tidy code structure, but **nothing had been run on a real
language model**, and several parts of the design could not be defended in a
viva. Details:

| What we found | Why it is a problem | What we did |
|---|---|---|
| Every result came from `mock_mode`. The "Internal Confidence" was a random number chosen from a difficulty label (`easy` → 0.9, `hard` → 0.18), and the "correct / incorrect" label was drawn from the **same** difficulty label. | The reported AUROC of 92.3% (IC) and 100% (SemanticDPP) measured only that the simulator agrees with itself. They are not results. The "0.44 ms per query" latency was the time to draw a random number. | Deleted the mock mode and all result files produced by it. Every number in the repo now comes from a real model run. |
| The "dataset" was 12 hand-written TriviaQA-style questions and 5 GSM8K questions, repeated to fill 50 rows. | Not the benchmark the papers use. | Real TriviaQA (600 questions with their Wikipedia evidence pages), streamed from the HuggingFace hub. |
| The real-model code path of `ICGate` had never been executed. It did not use the chat template, and it used a 2-D distance for the weights instead of the paper's layer-weights × token-weights. | Would not have reproduced Paper A even if run. | Rewritten to follow Paper A's Eq. 1, 3 and 4, and tested on a real model. |
| `SemanticDPP` used a random, untrained projection, and the uncertainty formula was labelled "Eq. 4" but was a different expression (`log det(L+I) − log det(2I)`). | Paper B's whole point is that the kernel is *learned*; and the formula was not the paper's. | Implemented the learned kernel (Eq. 1), the training loss (Eq. 2–3) and the real score `log det(L(L+I)⁻¹)` (Eq. 4). |
| The "novel contribution" mapped the eigenvalues of the 3×3 answer-similarity matrix to "per-dimension compression ratios". | Those 3 numbers describe how the 3 sampled answers differ from each other. They do not correspond to any dimension of the KV cache, so there is nothing for the ratios to compress. The ratios were computed and then never used. | Dropped. |
| "KV budget = 0.25 + 0.75·UCA" and "beam width 1–4" were printed but never applied to a model, and the weights 0.4 / 0.6 and thresholds 0.65 / 0.30 were picked by hand. | No memory was ever saved or measured, so the central question of the project was not answered. | Replaced with a memory knob that is actually applied and measured (next section), and an allocation rule whose numbers are learned from held-out data. |
| Paper B was attributed to named Google DeepMind authors. | The PDF in `temperory/` is an anonymous submission, "under review at ICLR 2026". We cannot verify any author names. | Cited as anonymous / under review. |
| The 1,400-line plan also pulled in two more papers (C-DIC, HyC-LoRA), "thread memory", and a KV-cache manager. | Far too much for the time left and not needed to answer the project question. | Cut. |
| Model in the config was Llama-3.2-3B. | Gated download, ~6.5 GB; the development laptop has 8 GB RAM and under 8 GB of free disk. | Qwen2.5-1.5B-Instruct (3 GB, not gated, runs on the M1 GPU). |

One thing to be aware of: **the mid-evaluation report quotes the simulated
numbers and says IC was "tested on TriviaQA with Llama-3.2-3B"**. We have left
that submitted report untouched, but the final report should state clearly that
the mid-term numbers were from a simulation and are superseded.

What we kept from the earlier work: the choice of the two papers' signals, the
idea of using the cheap one (IC) first and the expensive one (SemanticDPP) only
when IC is unsure, the name UCA, and the evaluation metrics (AUROC, PRR, ECE).

---

## 2. The simplified project

**Question from the course:** can we use uncertainty for optimal memory allocation?

**Our concrete version.** In retrieval-augmented question answering, the big
memory cost is the KV cache, and the KV cache grows in direct proportion to the
number of tokens in the prompt. Retrieved passages are most of the prompt. So:

> *How many retrieved passages should each question get, when KV-cache memory is limited?*

- **Memory knob:** 0, 1, 2, 4 or 8 retrieved passages (≈ 50 to ≈ 1,200 prompt tokens,
  i.e. 1.4 MB to 35 MB of KV cache per question on our model).
- **Confidence signals:** Internal Confidence from Paper A (one forward pass, no
  generation), SemanticDPP from Paper B (5 sampled answers), and the probability
  the model gives its own answer (free, a standard baseline).
- **Our method, UCA (Uncertainty-Calibrated Allocation):** start every question
  with no passages; read the confidence; if the calibrated probability of being
  right is above a threshold, answer; otherwise add passages and check again, up
  to a maximum. Two knobs: the threshold and the maximum.
- **What we compare against:** the same memory for everyone, including the best
  random mix of two fixed levels (the strongest policy that ignores
  uncertainty); deciding once before retrieval, as in Paper A; and an oracle.
- **What we measure:** accuracy against exact KV-cache memory, and the time cost
  of the confidence checks.

---

## 3. Plan

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Review earlier work, read both papers again | Section 1 above | ✅ 16 Nov |
| 2 | Real model wrapper: answer + probability, sampling from the shared prompt cache, P(Yes) per layer, KV bytes per token | `src/core/llm.py` | ✅ 16 Nov |
| 3 | Real dataset: TriviaQA + BM25 passage retrieval + answer checking | `src/evaluation/data.py`, `data/triviaqa_rag.json` | ✅ 16 Nov |
| 4 | Internal Confidence (Paper A) | `src/core/internal_confidence.py` | ✅ 16 Nov |
| 5 | SemanticDPP with a learned kernel (Paper B) | `src/core/semantic_dpp.py` | ✅ 16 Nov |
| 6 | Metrics (AUROC, PRR, ECE) and unit tests | `src/evaluation/metrics.py`, `tests/` | ✅ 16 Nov |
| 7 | Run the model on 400 questions × 5 memory levels | `results/runs.jsonl`, `results/runs.npz` | ✅ 17 Nov |
| 8 | Allocation policies: escalation, one-shot, baselines | `src/core/allocator.py` | ✅ 16 Nov |
| 9 | Analysis: signal quality, accuracy vs memory, time cost, figures | `results/summary.json`, `results/figures/` | ✅ 17 Nov |
| 10 | Live demo on unseen questions | `src/experiments/demo.py` | ✅ 17 Nov |
| 11 | Write-up of idea, method and results | `PROJECT_OVERVIEW.md` | ✅ 17 Nov |
| 12 | Final report (LaTeX and PDF) | `final_report/` | ✅ 17 Nov |
| 13 | Presentation video | - | ⬜ team |

---

## 4. Progress log

**16 Nov, afternoon - review and reset.**
Read all the existing code and both papers. Found that every reported number
came from the mock mode (Section 1). Decided to restart the experiments on a
real model and to cut the design down to something we can run and explain.

**16 Nov - model and data.**
Qwen2.5-1.5B-Instruct runs on the M1 GPU in half precision. Measured the KV
cache directly from the model's cache tensors: 28,672 bytes per token, which
matches the formula 2 × 28 layers × 2 KV heads × 128 dims × 2 bytes. Built the
TriviaQA subset: 600 questions, Wikipedia evidence cut into 100-word passages,
top 8 by BM25.

**16 Nov - first attempt: decide once, before retrieval.**
Ran 120 questions with closed-book Internal Confidence and SemanticDPP, and an
allocator that gives more passages to less confident questions. It did not beat
random allocation. The reason turned out to be basic: about three quarters of
the questions need passages no matter what, so knowing who is confident
*without* passages can only save the memory of the remaining quarter. Even a
perfect closed-book signal would have gained only about 2 points on average.
Lesson: the useful question is not "is this question hard?" but "is the memory
I have given it so far enough?".

**16 Nov - a mistake worth recording.**
We started a small test while the main run was going. Two copies of the model
do not fit in 8 GB; the laptop froze and the run was lost. Since then: one
model process at a time, checkpoints every 10 questions, and the GPU cache is
released after every question (without that, time per question doubled).

**16 Nov, evening - second design: escalation.**
Re-collected with confidence read at every memory level, not only before
retrieval. Rewrote sampling so that the 5 sampled answers are decoded from the
prompt's cached keys and values instead of re-reading the prompt 5 times: the
sampling cost at 4 passages dropped from 7.1 s to 1.4 s, and it no longer needs
5 copies of the prompt in memory.

**16 Nov, night - what did not work, part two.**
With the first 100–200 questions we tried two cleverer policies: a rule based
on "expected accuracy gain per extra token", and choosing a separate cutoff for
every memory budget on the calibration half. Both looked good on calibration
questions and worse on test questions - with only 100–200 calibration questions
there are too many settings to choose between. We went back to the simplest
rule (one threshold, one maximum level) and report whole curves on held-out
questions instead of hand-picked operating points.

**17 Nov, 00:08 - full run finished.**
400 questions × 5 memory levels in 2 hours on the M1 (about 18 s per question).
Analysis on 20 random 200/200 halvings:

- Same memory for everyone: 32.1% / 52.0% / 57.5% / 62.6% / 65.9% accuracy at
  0 / 1 / 2 / 4 / 8 passages (1.5 to 34.4 MB of KV cache per question).
- Signal quality (AUROC, no passages → with passages): Internal Confidence
  0.64 → 0.75, SemanticDPP 0.83 → 0.71–0.74, answer probability 0.87 → 0.77–0.79.
- Escalation with answer probability, at most 2 passages, threshold 0.70:
  57.6% accuracy with 7.8 MB, against 57.5% with 9.8 MB for "everyone gets 2" -
  21% less KV memory, and +2.6 ± 1.0 points over the best uncertainty-blind mix
  at the same memory. Up to +7.2 ± 1.8 points at very small budgets.
- With at most 8 passages the saving is 36–58% but accuracy drops by 1–5 points;
  against the blind mix at equal memory it is about level.
- Deciding once before retrieval: no measurable gain (+0.3 ± 1.0).
- Cost: the checks add time (1.75 s vs 1.05 s per question in the best case).

**17 Nov - demo and write-up.**
The demo ran on 10 questions that were not part of the experiments: 5 of 10
correct using 81% of the memory of "everyone gets 2". `PROJECT_OVERVIEW.md`
written with the method, all results, the limits, and viva notes. 13 unit tests
pass.

**17 Nov - final report.**
Wrote the report in LaTeX from the overview: same results, plus the equations
for both papers, the algorithm, and a section on what changed since the
mid-term. Compiles with `tectonic final_report.tex` from the `final_report` folder.

---

## 5. What is left for the team

- **Final report (40%)**: written in `final_report/final_report.tex`, compiled to
  `final_report/final_report.pdf` (12 pages). Everyone should read it once and
  check the team details before submission.
- **Video (25%)**: suggested flow - the problem in one slide; the two papers in
  one slide each; our method with the `where_questions_stop` figure; the
  `accuracy_vs_memory` figure; a screen recording of
  `python -m src.experiments.demo --n 10`; limits and what we would do next.
- **Working code (10%)**: done. `README.md` lists the commands. Everybody on the
  team should run the tests and the demo once before the viva.
- **If there is time**: the two extensions in the "what we would do next"
  section of the overview (reusing the cache between escalation steps, and a
  larger model).
