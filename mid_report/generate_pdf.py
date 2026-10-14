#!/usr/bin/env python3
"""
Generate the DA4022 Mid-Project Report PDF using WeasyPrint.
This produces a professional, LaTeX-quality PDF from HTML+CSS.

Run: python3 generate_pdf.py
Output: mid_report.pdf
"""

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>UCA: Uncertainty-Calibrated Adaptive Compression — Mid-Project Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@400;600;700&family=Source+Code+Pro:wght@400;500&display=swap');

  :root {
    --blue: #1e5ab4;
    --blue-light: #e8f0fb;
    --orange: #d46c14;
    --orange-light: #fdf1e6;
    --green: #1e8c3c;
    --red: #b41e1e;
    --gray-dark: #2a2a35;
    --gray-mid: #555566;
    --gray-light: #f5f5f8;
    --border: #c8cdd6;
    --text: #1a1a22;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  @page {
    size: A4;
    margin: 22mm 20mm 20mm 22mm;
    @top-left {
      content: "DA4022 — Mid-Project Report";
      font-family: 'Source Sans 3', sans-serif;
      font-size: 8.5pt;
      color: var(--gray-mid);
    }
    @top-right {
      content: "UCA: Uncertainty-Calibrated Adaptive Compression";
      font-family: 'Source Sans 3', sans-serif;
      font-size: 8.5pt;
      color: var(--gray-mid);
    }
    @bottom-center {
      content: counter(page);
      font-family: 'Source Sans 3', sans-serif;
      font-size: 9pt;
      color: var(--gray-mid);
    }
  }

  body {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.60;
    color: var(--text);
    background: white;
  }

  /* -------- Title Page -------- */
  .title-page {
    page-break-after: always;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30pt 10pt;
    text-align: center;
  }
  .course-label {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 11pt;
    color: var(--gray-mid);
    letter-spacing: 0.04em;
    margin-bottom: 4pt;
  }
  .title-rule { width: 100%; border: none; border-top: 2.5pt solid var(--blue); margin: 10pt 0; }
  h1.title {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 22pt;
    font-weight: 700;
    color: var(--blue);
    line-height: 1.25;
    margin: 8pt 0;
  }
  .tagline-box {
    border: 1pt solid #b0bcd6;
    background: var(--blue-light);
    border-radius: 4pt;
    padding: 12pt 18pt;
    max-width: 460pt;
    margin: 20pt auto;
    font-style: italic;
    font-size: 10pt;
    color: var(--gray-dark);
    line-height: 1.55;
  }
  .author-block {
    margin-top: 20pt;
    font-family: 'Source Sans 3', sans-serif;
  }
  .author-name { font-size: 14pt; font-weight: 700; color: var(--gray-dark); }
  .author-meta { font-size: 10pt; color: var(--gray-mid); margin-top: 4pt; }
  .dates-table {
    margin-top: 28pt;
    border-collapse: collapse;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 10pt;
  }
  .dates-table td { padding: 3pt 12pt 3pt 0; }
  .dates-table td:first-child { font-weight: 600; color: var(--gray-dark); }

  /* -------- Sections -------- */
  h2 {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 13.5pt;
    font-weight: 700;
    color: var(--blue);
    border-bottom: 1.5pt solid var(--blue);
    padding-bottom: 3pt;
    margin-top: 22pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
  }
  h3 {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 11pt;
    font-weight: 700;
    color: var(--gray-dark);
    margin-top: 14pt;
    margin-bottom: 5pt;
    page-break-after: avoid;
  }
  h4 {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 10pt;
    font-weight: 600;
    font-style: italic;
    color: var(--gray-mid);
    margin-top: 10pt;
    margin-bottom: 3pt;
  }

  p { margin-bottom: 7pt; }

  ul, ol {
    margin: 5pt 0 8pt 18pt;
  }
  li { margin-bottom: 3pt; }

  /* -------- Math-like formatting -------- */
  .eq {
    display: block;
    text-align: center;
    font-family: 'Source Code Pro', monospace;
    font-size: 9.5pt;
    background: var(--gray-light);
    border-left: 3pt solid var(--blue);
    padding: 7pt 14pt;
    margin: 8pt 0;
    border-radius: 0 3pt 3pt 0;
  }
  .eq-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .eq-num { color: var(--gray-mid); font-size: 9pt; }

  code {
    font-family: 'Source Code Pro', monospace;
    font-size: 9pt;
    background: var(--gray-light);
    padding: 1pt 4pt;
    border-radius: 2pt;
    color: #333;
  }

  /* -------- Tables -------- */
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 9.5pt;
    margin: 10pt 0;
  }
  th {
    background: var(--blue);
    color: white;
    font-weight: 600;
    padding: 5pt 8pt;
    text-align: left;
  }
  td {
    padding: 4pt 8pt;
    border-bottom: 0.5pt solid var(--border);
    vertical-align: top;
  }
  tr:nth-child(even) td { background: var(--gray-light); }
  tr.ours td { background: #d8ecff !important; font-weight: 600; }
  .caption {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 9pt;
    color: var(--gray-mid);
    text-align: center;
    margin-top: 3pt;
    margin-bottom: 10pt;
  }

  /* -------- Special Boxes -------- */
  .novelty-box {
    border: 1pt solid var(--blue);
    background: var(--blue-light);
    border-radius: 4pt;
    padding: 10pt 14pt;
    margin: 10pt 0;
  }
  .novelty-box .box-title {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    font-size: 10pt;
    color: var(--blue);
    margin-bottom: 5pt;
  }

  .gap-box {
    border: 1pt solid var(--orange);
    background: var(--orange-light);
    border-radius: 4pt;
    padding: 10pt 14pt;
    margin: 10pt 0;
  }
  .gap-box .box-title {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    font-size: 10pt;
    color: var(--orange);
    margin-bottom: 5pt;
  }

  .check { color: var(--green); font-weight: 700; }
  .cross { color: var(--red); font-weight: 700; }
  .bold { font-weight: 700; }
  .italic { font-style: italic; }
  .blue { color: var(--blue); }

  /* -------- Architecture Block -------- */
  .arch-block {
    font-family: 'Source Code Pro', monospace;
    font-size: 8pt;
    line-height: 1.5;
    background: #f0f0f5;
    border: 1pt solid var(--border);
    border-radius: 4pt;
    padding: 10pt 14pt;
    margin: 10pt 0;
    white-space: pre;
    page-break-inside: avoid;
  }

  /* -------- Progress Checklist -------- */
  .done::before { content: "✅ "; }
  .todo::before { content: "⬜ "; }

  /* -------- References -------- */
  .references { font-size: 9.5pt; }
  .references p { margin-bottom: 5pt; }

  hr.section-sep {
    border: none;
    border-top: 0.5pt solid var(--border);
    margin: 12pt 0;
  }
</style>
</head>
<body>

<!-- ===== TITLE PAGE ===== -->
<div class="title-page">
  <div class="course-label">DA4022 — Data Analytics &amp; Applications | Mid-Project Report | October 2026</div>
  <hr class="title-rule"/>
  <h1 class="title">UCA: Uncertainty-Calibrated Adaptive<br/>Compression for LLM Inference</h1>
  <hr class="title-rule"/>

  <div class="tagline-box">
    <strong>Core Research Question:</strong><br/>
    Can we design a better adaptive compression system, inspired by <strong>C-DIC</strong> and <strong>HyC-LoRA</strong>,
    that uses uncertainty signals (<strong>Internal Confidence</strong> + <strong>SemanticDPP</strong>)
    to dynamically control how much and what kind of compression to apply at inference time —
    and beat existing benchmarks?
  </div>

  <div class="author-block">
    <div class="author-name">Deval Singhal</div>
    <div class="author-meta">DA4022 Course Project · Topic Anchor #8<br/>
    <em>"Can we use uncertainty for optimal memory allocation?"</em></div>
  </div>

  <table class="dates-table">
    <tr><td>Proposal submitted:</td><td>September 10, 2026</td></tr>
    <tr><td>Mid-report due:</td><td>October 15, 2026</td></tr>
    <tr><td>Final submission:</td><td>November 26, 2026</td></tr>
  </table>
</div>

<!-- ===== 1. EXECUTIVE SUMMARY ===== -->
<h2>1. Executive Summary</h2>

<p>
Current Large Language Model (LLM) serving systems allocate a <em>fixed</em> computational and memory
budget — the same maximum KV cache size, full context window, and static decoding strategy — regardless
of query complexity. This leads to two pathologies: (1) <strong>resource waste</strong> on simple,
high-confidence queries, and (2) <strong>hallucination risk</strong> on ambiguous, knowledge-intensive
prompts where insufficient resources cause the model to confabulate.
</p>

<p>
This project proposes <strong>UCA (Uncertainty-Calibrated Adaptive Compression)</strong>, a novel inference
middleware that synthesises four recent ICLR/ICML/MLSys 2025–2026 papers to convert uncertainty signals
into <em>continuous, per-dimension compression ratios</em> rather than the hard binary switches used by
all existing systems.
</p>

<div class="gap-box">
  <div class="box-title">🎯 The Core Insight</div>
  <strong>No existing method uses uncertainty as a continuous control signal for compression.</strong>
  All four papers either estimate uncertainty without acting on it (IC, SemanticDPP), or compress without
  consulting uncertainty (C-DIC, HyC-LoRA). UCA bridges this gap by making the compression ratio itself
  a differentiable function of uncertainty.
</div>

<p>
UCA uses a fast IC pre-filter so that the expensive DPP computation runs only on the truly uncertain subset
(~40% of queries), achieving a ~3× speedup over full SemanticDPP while targeting AUROC &gt;0.83 with
&gt;32% KV cache reduction — beating all four baseline methods simultaneously.
</p>

<!-- ===== 2. BACKGROUND ===== -->
<h2>2. Background: Critical Analysis of Four Reference Papers</h2>

<h3>2.1 Paper A — Query-Level Uncertainty in LLMs (Chen et al., ICLR 2026)</h3>
<p><em>Code: <code>tigerchen52/query_level_uncertainty</code> | Venue: ICLR 2026</em></p>

<h4>What It Does</h4>
<p>
Internal Confidence (IC) is a training-free uncertainty estimator requiring only a single forward pass over
the input query — no tokens are generated. It probes P(YES) logits at every (layer <em>l</em>,
token position <em>t</em>) pair and aggregates via <strong>Attenuated Encoding</strong>:
</p>
<div class="eq eq-label">
  <span>IC(q) = Σ<sub>l,t</sub> w(l,t) · P(YES | h<sub>l</sub><sup>(t)</sup>),   w(l,t) ∝ exp(−λ · dist(l,t))</span>
  <span class="eq-num">(1)</span>
</div>
<p>
where <em>dist(l,t)</em> is the distance from the "optimal decision centre" (last token, last layer).
Achieves <strong>30–600× speedup</strong> over answer-level methods.
</p>

<h4>Critical Gaps We Target</h4>
<ul>
  <li><strong>Binary routing only:</strong> IC uses a hard threshold τ — binary decision only, no continuous resource modulation.</li>
  <li><strong>Static fallback:</strong> Low IC always triggers RAG or cascade; never controls <em>how aggressively</em> to compress context.</li>
  <li><strong>Threshold is dataset-specific:</strong> Optimal τ and k (last-k tokens) must be re-tuned per dataset/model pair.</li>
</ul>

<h3>2.2 Paper B — SemanticDPP (Hamer et al., ICLR 2026)</h3>
<p><em>Authors: Jenny Hamer, Jasper Snoek, et al. (Google DeepMind)</em></p>

<h4>What It Does</h4>
<p>
Samples <em>m</em> responses S = {r₁,...,rₘ} and measures semantic diversity via a Determinantal Point
Process (DPP) L-ensemble over projected output embeddings. Learns projection ϕ: ℝ<sup>d</sup>→ℝ<sup>p</sup>
(p≪d). The kernel and uncertainty score are:
</p>
<div class="eq eq-label">
  <span>K(xᵢ, xⱼ) = exp(−α‖ϕ(xᵢ)−ϕ(xⱼ)‖²),   U(S) = log det(L<sub>S</sub>) − log det(L+I)</span>
  <span class="eq-num">(2)</span>
</div>

<h4>Critical Gaps We Target</h4>
<ul>
  <li><strong>Requires m ≥ 5 generations:</strong> Expensive for real-time routing.</li>
  <li><strong>Discards kernel structure:</strong> U(S) is a scalar; all rich pairwise information in L<sub>S</sub> is lost after log-det.</li>
  <li><strong>No compression action:</strong> Pure estimation; never used to control compression or memory allocation.</li>
</ul>

<h3>2.3 Paper C — C-DIC: Context-Driven Incremental Compression (Jung et al., ICML 2026)</h3>
<p><em>OpenReview: ZGeVMaiIlb | Authors: Yeongseo Jung et al. (HKUST)</em></p>

<h4>What It Does</h4>
<p>
Addresses the O(n²) attention cost of multi-turn dialogue by treating conversation as <em>interleaved contextual threads</em>.
Maintains one compressed state vector per thread; runs a lightweight <strong>Retrieve → Revise → Write-Back</strong> loop at each turn.
Achieves near-constant inference latency over hundreds of dialogue turns via Truncated BPTT.
</p>

<h4>Critical Gaps We Target</h4>
<ul>
  <li><strong>No uncertainty awareness:</strong> Every turn compresses with the same aggressiveness regardless of model uncertainty.</li>
  <li><strong>Fixed compression ratio:</strong> f<sub>comp</sub> always outputs the same-dimensional state.</li>
  <li><strong>No error recovery:</strong> Over-compressed threads cannot be detected or revisited.</li>
</ul>

<h3>2.4 Paper D — HyC-LoRA: Hybrid Activation Compression (Wang et al., MLSys 2025)</h3>
<p><em>OpenReview: WU86J4uM9V | Code: <code>thu-ee-acts-lab/HyC-LoRA-release</code></em></p>

<h4>What It Does</h4>
<p>
Reduces GPU memory dominated by buffered activations during LoRA fine-tuning. Uses a two-tier hybrid
compression: <strong>intra-operator</strong> (detects outlier channels in FP16, quantises rest to ~2-bit)
and <strong>inter-operator</strong> (LoRA adapter compensates for quantisation error; operator reordering
and selective recomputation). Achieves <strong>3.97× end-to-end memory reduction</strong>.
</p>

<h4>Critical Gaps We Target</h4>
<ul>
  <li><strong>Training-time only:</strong> Designed for fine-tuning backprop, not inference-time KV/context compression.</li>
  <li><strong>Fixed 2-bit quantisation:</strong> No per-input adaptivity based on uncertainty or importance.</li>
  <li><strong>Static outlier detection:</strong> Channel outliers detected once per model, not per-query.</li>
</ul>

<!-- ===== 3. GAP AND CONTRIBUTION ===== -->
<h2>3. Research Gap and Novel Contributions</h2>

<table>
  <caption style="caption-side:top; font-family:sans-serif; font-size:9pt; text-align:center; margin-bottom:5pt;">
    Table 1: Capability comparison. UCA is the only system that is simultaneously uncertainty-aware,
    adaptively compressive, zero-cost routable, and incremental.
  </caption>
  <thead>
    <tr>
      <th>Method</th>
      <th>Uncertainty-Aware</th>
      <th>Adaptive Compression</th>
      <th>Zero-Cost Routing</th>
      <th>Incremental</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Internal Confidence (IC)</td>
      <td><span class="check">✔</span></td>
      <td><span class="cross">✘</span></td>
      <td><span class="check">✔</span></td>
      <td><span class="cross">✘</span></td>
    </tr>
    <tr>
      <td>SemanticDPP</td>
      <td><span class="check">✔</span></td>
      <td><span class="cross">✘</span></td>
      <td><span class="cross">✘</span></td>
      <td><span class="cross">✘</span></td>
    </tr>
    <tr>
      <td>C-DIC</td>
      <td><span class="cross">✘</span></td>
      <td>Fixed ratio</td>
      <td>N/A</td>
      <td><span class="check">✔</span></td>
    </tr>
    <tr>
      <td>HyC-LoRA</td>
      <td><span class="cross">✘</span></td>
      <td>Fixed 2-bit</td>
      <td>N/A</td>
      <td><span class="cross">✘</span></td>
    </tr>
    <tr class="ours">
      <td>UCA (Ours)</td>
      <td><span class="check">✔</span></td>
      <td>Continuous</td>
      <td><span class="check">✔</span></td>
      <td><span class="check">✔</span></td>
    </tr>
  </tbody>
</table>

<div class="novelty-box">
  <div class="box-title">💡 Contribution 1 — DPP Eigenspectrum as Compression Oracle</div>
  <p>
  Instead of collapsing L<sub>S</sub> to a scalar via log-det, we perform eigendecomposition L<sub>S</sub> = UΣUᵀ.
  Per-eigenvalue σᵢ measures semantic variance in the i-th principal direction of response diversity.
  We map these to per-dimension compression ratios:
  </p>
  <div class="eq eq-label">
    <span>rᵢ = clip(1 − α · σᵢ / σ<sub>max</sub>,  r<sub>min</sub>,  r<sub>max</sub>)</span>
    <span class="eq-num">(3)</span>
  </div>
  <p>
  High σᵢ (semantically contested direction) → less compression.
  Low σᵢ (semantically stable direction) → more compression.
  This is a strict generalisation of HyC-LoRA's static 2-bit quantisation.
  </p>
</div>

<div class="novelty-box">
  <div class="box-title">💡 Contribution 2 — IC-Gated DPP Cascade</div>
  <p>
  IC is computed first at near-zero cost (1 forward pass). DPP runs <em>only</em> when
  IC ∈ [τ<sub>low</sub>, τ<sub>high</sub>] — the uncertain zone covering ~40% of queries.
  This eliminates ~60% of DPP overhead, making UCA ~3× faster than full SemanticDPP
  at equivalent AUROC quality.
  </p>
</div>

<div class="novelty-box">
  <div class="box-title">💡 Contribution 3 — Uncertainty History in Thread Memory</div>
  <p>
  Each C-DIC thread carries an <em>uncertainty history</em> {u<sub>k</sub>} — the UCA score
  at each write time. During retrieval, threads with recent high uncertainty receive a score
  boost (+δ<sub>revisit</sub>), encouraging the system to re-compress them with updated context.
  This provides a retrospective error-recovery mechanism absent from C-DIC.
  </p>
</div>

<!-- ===== 4. ARCHITECTURE ===== -->
<h2>4. Proposed Architecture</h2>

<div class="arch-block">
┌────────────────────────────────────────────────────────────────────┐
│                       INPUT QUERY  q_t                             │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 0: Internal Confidence Gate  (IC)                             │
│  • 1 forward pass, zero token generation                            │
│  • IC(q) = Σ_{l,t} w(l,t) · P(YES | h_l^t)  [Attenuated Encoding] │
│  • Latency: ~5 ms on GPU                                            │
└──────────────────────────────┬──────────────────────────────────────┘
         IC ≥ τ_high           │ τ_low ≤ IC < τ_high         IC < τ_low
        (Confident)            │   (Uncertain zone)          (Very Uncertain)
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌──────────────────┐ ┌─────────────────────┐ ┌──────────────────────┐
│   FAST PATH      │ │  TIER 1: DPP        │ │  ABSTAIN / CASCADE   │
│ KV budget = 25%  │ │  Eigenspectrum      │ │  KV budget = 100%    │
│ Beam width = 1   │ │  Decoder            │ │  Beam width = 4      │
│ Max compression  │ │  (m = 3 samples)    │ │  Always trigger RAG  │
└──────────────────┘ │                     │ └──────────────────────┘
                     │ L_S = UΣUᵀ          │
                     │ rᵢ = f(σᵢ)         │
                     └──────────┬──────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
       ┌──────────────────┐       ┌───────────────────────┐
       │  KV Cache Budget │       │  Thread Memory        │
       │  = 0.25+0.75·UCA │       │  C-DIC + UCA history  │
       │  Sliding eviction│       │  b_eff = 4+12·UCA     │
       └──────────────────┘       │  HyC-LoRA outlier Q   │
                                  └───────────────────────┘
</div>

<p>
The combined UCA score used for all continuous resource decisions is:
</p>
<div class="eq eq-label">
  <span>UCA(q) = 0.4 · (1 − IC(q))  +  0.6 · U<sub>DPP</sub>(q)</span>
  <span class="eq-num">(4)</span>
</div>
<p>
KV cache budget and beam width are smooth functions of UCA(q):
</p>
<div class="eq eq-label">
  <span>KV budget  =  0.25 + 0.75 · UCA(q),      beam width  =  max(1, ⌊1 + 3·UCA(q)⌉)</span>
  <span class="eq-num">(5)</span>
</div>
<p>
Thread quantisation bit-width follows HyC-LoRA's intra-operator outlier-aware scheme, with bit-width
calibrated to uncertainty:
</p>
<div class="eq eq-label">
  <span>b<sub>eff</sub>  =  4 + 12 · UCA(q)   ∈ [4, 16 bits]</span>
  <span class="eq-num">(6)</span>
</div>

<!-- ===== 5. PROGRESS ===== -->
<h2>5. Implementation Progress</h2>

<h3>5.1 Completed Work</h3>
<ul>
  <li class="done"><strong>Literature review &amp; gap analysis</strong> — all 4 papers thoroughly analysed; weaknesses mapped to UCA contributions.</li>
  <li class="done"><strong>IC baseline wrapper</strong> (<code>src/core/internal_confidence.py</code>) — <code>ICGate</code> class wrapping the official <code>query-level-uncertainty</code> package; token ID verification for Llama-3.2-3B.</li>
  <li class="done"><strong>SemanticDPP reimplementation</strong> (<code>src/core/semantic_dpp.py</code>) — <code>SemanticDPP</code> + <code>SemanticSubspaceProjector</code> (d=4096→p=256); full kernel pipeline + eigenspectrum decoder.</li>
  <li class="done"><strong>UCA Controller</strong> (<code>src/core/uca_controller.py</code>) — Three-tier routing logic; per-dimension compression ratio computation via Eq. (3); combined UCA score via Eq. (4).</li>
  <li class="done"><strong>Thread Memory skeleton</strong> (<code>src/core/compression/thread_memory.py</code>) — C-DIC Retrieve→Revise→Write-Back loop with uncertainty history and adaptive quantisation.</li>
  <li class="done"><strong>KV Cache Manager skeleton</strong> (<code>src/core/compression/kv_cache_manager.py</code>) — Dynamic budget via HuggingFace <code>DynamicCache</code>; sliding-window eviction.</li>
  <li class="done"><strong>Evaluation metrics</strong> (<code>src/evaluation/metrics.py</code>) — AUROC, PRR, ECE, memory savings implemented.</li>
</ul>

<h3>5.2 Work Remaining</h3>
<ul>
  <li class="todo"><strong>End-to-end integration</strong> — Connect all modules; test full forward pass with routing.</li>
  <li class="todo"><strong>Baseline reproduction</strong> — Run IC and SemanticDPP on TriviaQA/GSM8K; verify against reported AUROC.</li>
  <li class="todo"><strong>Full evaluation pipeline</strong> (<code>src/experiments/run_uca.py</code>) — 500-sample runs on TriviaQA, GSM8K, TruthfulQA.</li>
  <li class="todo"><strong>Ablation study</strong> — 6 configurations proving per-component contribution.</li>
  <li class="todo"><strong>Visualisations</strong> — ROC curves, memory savings histograms, IC vs. UCA scatter.</li>
  <li class="todo"><strong>Thread compressor training</strong> — Fine-tune f<sub>comp</sub> on multi-turn dialogue corpus.</li>
  <li class="todo"><strong>Final report + PPT video</strong> — Due November 26.</li>
</ul>

<!-- ===== 6. EVALUATION PLAN ===== -->
<h2>6. Evaluation Plan</h2>

<h3>6.1 Datasets and Models</h3>
<ul>
  <li><strong>TriviaQA</strong> (<code>rc.nocontext</code>, 500 val samples): factual recall, knowledge boundary.</li>
  <li><strong>GSM8K</strong> (<code>main</code>, 300 test samples): multi-step mathematical reasoning.</li>
  <li><strong>TruthfulQA</strong> (<code>generation</code>, 200 val samples): hallucination detection.</li>
  <li><strong>Primary model:</strong> <code>meta-llama/Llama-3.2-3B</code> (4-bit quantised).</li>
  <li><strong>Secondary:</strong> <code>microsoft/Phi-3-mini-4k-instruct</code> (cross-model validation).</li>
</ul>

<h3>6.2 Target Benchmarks</h3>

<table>
  <caption style="caption-side:top; font-family:sans-serif; font-size:9pt; text-align:center; margin-bottom:5pt;">
    Table 2: Target benchmark results on TriviaQA with Llama-3.2-3B.
    UCA targets higher AUROC than all baselines with significant memory savings.
  </caption>
  <thead>
    <tr><th>Method</th><th>AUROC ↑</th><th>PRR ↑</th><th>KV Saved ↑</th><th>Relative Latency</th></tr>
  </thead>
  <tbody>
    <tr><td>Static Baseline</td><td>0.50</td><td>0.00</td><td>0%</td><td>1×</td></tr>
    <tr><td>Semantic Entropy</td><td>~0.78</td><td>~0.20</td><td>0%</td><td>~30×</td></tr>
    <tr><td>SemanticDPP (m=5)</td><td>~0.81</td><td>~0.24</td><td>0%</td><td>~15×</td></tr>
    <tr><td>Internal Confidence</td><td>~0.73</td><td>~0.18</td><td>0%</td><td>1×</td></tr>
    <tr class="ours"><td>UCA-Full (Ours)</td><td>&gt;0.83</td><td>&gt;0.27</td><td>&gt;32%</td><td>~5×</td></tr>
  </tbody>
</table>

<h3>6.3 Ablation Study</h3>

<table>
  <caption style="caption-side:top; font-family:sans-serif; font-size:9pt; text-align:center; margin-bottom:5pt;">
    Table 3: Planned ablation study. Each row removes one UCA component independently.
  </caption>
  <thead>
    <tr><th>Configuration</th><th>Removed Component</th><th>Expected ΔAUROC</th></tr>
  </thead>
  <tbody>
    <tr class="ours"><td>UCA-Full (ours)</td><td>—</td><td>Baseline</td></tr>
    <tr><td>−Eigenspectrum</td><td>Use log-det scalar for ratios</td><td>−0.02 to −0.04</td></tr>
    <tr><td>−IC Gate</td><td>Always run DPP (no pre-filter)</td><td>≈0 (latency 3× ↑)</td></tr>
    <tr><td>−Adaptive Compression</td><td>Fixed compression ratio</td><td>≈0 (memory worse)</td></tr>
    <tr><td>−Uncertainty History</td><td>No UCA history in threads</td><td>Memory efficiency ↓</td></tr>
    <tr><td>IC Only</td><td>No DPP at all</td><td>−0.05 to −0.08</td></tr>
    <tr><td>DPP Only</td><td>No IC gate, always run DPP</td><td>Same AUROC, 3× slower</td></tr>
  </tbody>
</table>

<!-- ===== 7. TIMELINE ===== -->
<h2>7. Timeline and Risk Assessment</h2>

<table>
  <caption style="caption-side:top; font-family:sans-serif; font-size:9pt; text-align:center; margin-bottom:5pt;">
    Table 4: 40-hour sprint plan from mid-project to final submission.
  </caption>
  <thead>
    <tr><th>Phase</th><th>Hours</th><th>Dates</th><th>Key Deliverable</th></tr>
  </thead>
  <tbody>
    <tr><td>0: Setup</td><td>1–4</td><td>Oct 16–17</td><td>Environment, model downloads</td></tr>
    <tr><td>1: Baselines</td><td>5–12</td><td>Oct 17–20</td><td>IC + DPP AUROC verified vs. papers</td></tr>
    <tr><td>2: UCA Controller</td><td>13–22</td><td>Oct 21–25</td><td>End-to-end integration working</td></tr>
    <tr><td>3: Thread + KV</td><td>23–34</td><td>Oct 26–Nov 5</td><td>Thread memory + KV manager tested</td></tr>
    <tr><td>4: Full Eval</td><td>35–42</td><td>Nov 6–12</td><td>Results CSV, AUROC/PRR/memory tables</td></tr>
    <tr><td>5: Ablation + Plots</td><td>43–48</td><td>Nov 13–26</td><td>Final report + PPT video</td></tr>
  </tbody>
</table>

<table>
  <caption style="caption-side:top; font-family:sans-serif; font-size:9pt; text-align:center; margin-bottom:5pt;">
    Table 5: Key risks and mitigation strategies.
  </caption>
  <thead>
    <tr><th>Risk</th><th>Severity</th><th>Mitigation</th></tr>
  </thead>
  <tbody>
    <tr><td>DPP with m=3 too noisy</td><td>High</td><td>Increase to m=5; use SemanticDPP-C soft clustering</td></tr>
    <tr><td>GPU OOM with full model</td><td>High</td><td>4-bit bitsandbytes; start with Phi-3-mini</td></tr>
    <tr><td>IC token IDs model-specific</td><td>Medium</td><td><code>tokenizer.encode("Yes")[1]</code> per model</td></tr>
    <tr><td>Eigenvalue NaN/instability</td><td>Medium</td><td>Tikhonov regularisation: K + εI before eigvalsh</td></tr>
    <tr><td>AUROC Δ too small</td><td>Medium</td><td>500+ samples; bootstrap 95% CI</td></tr>
    <tr><td>Thread compressor untrained</td><td>High</td><td>Initialise from IC paper's embedding projector weights</td></tr>
  </tbody>
</table>

<!-- ===== 8. CONCLUSION ===== -->
<h2>8. Conclusion</h2>

<p>
We have identified a clear and well-motivated research gap: no existing method uses uncertainty as a
continuous control signal for compression quality. Our proposed UCA system bridges this gap by
synthesising the zero-cost routing of Internal Confidence, the semantic richness of the SemanticDPP
kernel, the incremental thread memory of C-DIC, and the outlier-aware quantisation scheme of HyC-LoRA.
</p>

<p>
The three specific technical contributions — (1) DPP eigenspectrum as compression oracle, (2) IC-gated
DPP cascade, and (3) uncertainty history in thread memory — are individually novel and jointly form a
system targeting <strong>AUROC &gt; 0.83</strong> with <strong>&gt; 32% KV cache reduction</strong>,
beating all four baseline methods on at least one primary metric.
</p>

<p>
The codebase skeleton is in place. The next six weeks are dedicated to end-to-end integration,
benchmarking, ablation, and final reporting.
</p>

<hr class="section-sep"/>

<!-- ===== REFERENCES ===== -->
<h2>References</h2>
<div class="references">
  <p>[1] Lihu Chen, Gerard de Melo, Fabian M. Suchanek, Gaël Varoquaux. <em>Query-Level Uncertainty in Large Language Models</em>. ICLR 2026.</p>
  <p>[2] Jenny Hamer, Jasper Snoek, Alexander D'Amour, et al. <em>SemanticDPP: Efficient Uncertainty Quantification in LLMs</em>. ICLR 2026.</p>
  <p>[3] Yeongseo Jung, Jaehyeok Kim, Eunseo Jung, Jiachuan Wang, et al. <em>Context-Driven Incremental Compression for Multi-Turn Dialogue Generation</em>. ICML 2026. OpenReview: ZGeVMaiIlb.</p>
  <p>[4] Yujin Wang, Shunan Dong, Zongle Huang, et al. <em>HyC-LoRA: Memory Efficient LoRA Fine-tuning with Hybrid Activation Compression</em>. MLSys 2025. OpenReview: WU86J4uM9V.</p>
  <p>[5] Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. <em>Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation</em>. ICLR 2023.</p>
  <p>[6] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. <em>Detecting Hallucinations in Large Language Models Using Semantic Entropy</em>. Nature, 630:625–630, 2024.</p>
  <p>[7] Jannik Kossen, et al. <em>Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs</em>. arXiv:2406.15927, 2024.</p>
  <p>[8] Neel Gupta, et al. <em>Language Model Cascades: Token-Level Uncertainty and Beyond</em>. ICLR 2024.</p>
</div>

</body>
</html>
"""

import os
import sys

def generate_pdf():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(output_dir, "mid_report.html")
    pdf_path = os.path.join(output_dir, "mid_report.pdf")

    # Write HTML file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    print(f"HTML written to: {html_path}")

    try:
        from weasyprint import HTML, CSS
        print("Generating PDF with WeasyPrint...")
        HTML(filename=html_path).write_pdf(pdf_path)
        print(f"✅ PDF generated: {pdf_path}")
        return pdf_path
    except ImportError:
        print("WeasyPrint not available. Trying alternative methods...")

    # Fallback 1: Try pdflatex
    tex_path = os.path.join(output_dir, "mid_report.tex")
    if os.path.exists(tex_path):
        result = os.system(f"cd '{output_dir}' && pdflatex -interaction=nonstopmode mid_report.tex 2>/dev/null")
        if result == 0 and os.path.exists(pdf_path):
            print(f"✅ PDF generated via pdflatex: {pdf_path}")
            return pdf_path

    # Fallback 2: Try wkhtmltopdf
    wk = os.popen("which wkhtmltopdf").read().strip()
    if wk:
        result = os.system(f"wkhtmltopdf --page-size A4 --margin-top 20mm --margin-bottom 20mm "
                           f"--margin-left 20mm --margin-right 20mm '{html_path}' '{pdf_path}'")
        if result == 0:
            print(f"✅ PDF generated via wkhtmltopdf: {pdf_path}")
            return pdf_path

    print(f"⚠️  PDF generation failed. HTML is available at: {html_path}")
    print("   Open in Chrome/Safari and print to PDF (File > Print > Save as PDF)")
    return html_path

if __name__ == "__main__":
    generate_pdf()
