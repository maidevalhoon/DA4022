#!/usr/bin/env python3
"""
Generate a publication-grade MID_EVALUATION.pdf using WeasyPrint.
Natural multi-page flow without forced page breaks per section,
planned ablation studies removed, target exactly 3 pages.
"""

import weasyprint

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>DA4022 Mid-Evaluation Report — Uncertainty for Optimal Memory Allocation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@400;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap');

  :root {
    --blue: #1e5ab4;
    --blue-dark: #133a75;
    --blue-light: #eef4fc;
    --navy: #0f274a;
    --orange: #d46c14;
    --orange-light: #fef5ee;
    --green: #15803d;
    --green-light: #f0fdf4;
    --red: #b91c1c;
    --red-light: #fef2f2;
    --gray-dark: #1f2937;
    --gray-mid: #4b5563;
    --gray-light: #f8fafc;
    --border: #cbd5e1;
    --border-light: #e2e8f0;
    --text: #1e293b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  @page {
    size: A4;
    margin: 10mm 15mm 10mm 15mm;
    @top-left {
      content: "DA4022 — Mid-Evaluation Report";
      font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
      font-size: 8pt;
      color: var(--gray-mid);
    }
    @top-right {
      content: "Project 7: Uncertainty for Memory Allocation";
      font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
      font-size: 8pt;
      color: var(--gray-mid);
    }
    @bottom-center {
      content: "Page " counter(page) " of " counter(pages);
      font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
      font-size: 8pt;
      color: var(--gray-mid);
    }
  }

  body {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    font-size: 9.1pt;
    line-height: 1.42;
    color: var(--text);
    background: white;
  }

  /* Document Header */
  .doc-header {
    border-bottom: 2pt solid var(--blue);
    padding-bottom: 4pt;
    margin-bottom: 6pt;
  }
  .proposal-badge {
    display: inline-block;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 7.2pt;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--blue);
    background: var(--blue-light);
    border: 1px solid #bfdbfe;
    padding: 1pt 5pt;
    border-radius: 3pt;
    margin-bottom: 2pt;
  }
  h1.main-title {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 13.5pt;
    font-weight: 700;
    color: var(--navy);
    line-height: 1.2;
    margin-bottom: 3.5pt;
  }
  .team-section {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 7.8pt;
    color: var(--gray-dark);
    background: #f1f5f9;
    border-radius: 3pt;
    padding: 3pt 6pt;
    line-height: 1.30;
  }
  .team-label {
    font-weight: 700;
    color: var(--navy);
    font-size: 7.2pt;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 1pt;
  }
  .team-members {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5pt 7pt;
  }
  .member {
    white-space: nowrap;
  }
  .member strong {
    color: #0f172a;
  }
  .member span {
    color: var(--gray-mid);
  }

  /* Headings */
  h2 {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 10.5pt;
    font-weight: 700;
    color: var(--blue);
    border-bottom: 1pt solid var(--border);
    padding-bottom: 1.2pt;
    margin-top: 7pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
  }
  h3 {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 9.1pt;
    font-weight: 700;
    color: var(--navy);
    margin-top: 5pt;
    margin-bottom: 2.5pt;
    page-break-after: avoid;
  }

  p {
    margin-bottom: 3.5pt;
    text-align: justify;
  }

  ul, ol {
    margin: 2.5pt 0 4.5pt 13pt;
  }
  li {
    margin-bottom: 1.8pt;
    padding-left: 1.5pt;
  }

  /* Core Gap Callout */
  .callout-box {
    border-left: 3pt solid var(--orange);
    background: var(--orange-light);
    border-radius: 0 3pt 3pt 0;
    padding: 4pt 7.5pt;
    margin: 4pt 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }
  .callout-title {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    color: var(--orange);
    margin-bottom: 1pt;
    font-size: 8.5pt;
  }

  /* Code & Formulas */
  code {
    font-family: 'Source Code Pro', 'Menlo', 'DejaVu Sans Mono', monospace;
    font-size: 8pt;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    padding: 0.5pt 2.5pt;
    border-radius: 2pt;
    color: #0f172a;
  }
  .formula-box {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 8.7pt;
    background: #f8fafc;
    border-left: 2.5pt solid var(--blue);
    border-top: 1px solid var(--border-light);
    border-right: 1px solid var(--border-light);
    border-bottom: 1px solid var(--border-light);
    padding: 2.5pt 6.5pt;
    margin: 3pt 0;
    border-radius: 0 2.5pt 2.5pt 0;
    color: #0f172a;
    page-break-inside: avoid;
  }

  /* Tables */
  table.comp-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 8.2pt;
    margin: 4.5pt 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    border-radius: 3pt;
    overflow: hidden;
    page-break-inside: avoid;
  }
  table.comp-table th {
    background: #1e3a8a; /* Deep rich navy blue */
    color: #ffffff !important; /* High contrast crisp white text */
    font-weight: 700;
    padding: 3.8pt 6.5pt;
    text-align: left;
    border: 1px solid #1e3a8a;
    font-size: 8.3pt;
    letter-spacing: 0.02em;
  }
  table.comp-table td {
    padding: 2.8pt 6.5pt;
    border: 1px solid var(--border-light);
    vertical-align: middle;
  }
  table.comp-table tr:nth-child(even) td {
    background: #f8fafc;
  }
  .aspect-col {
    font-weight: 600;
    color: var(--navy);
    width: 26%;
  }
  .tag-no {
    display: inline-flex;
    align-items: center;
    color: #b91c1c;
    font-weight: 600;
    background: #fef2f2;
    border: 1px solid #fecaca;
    padding: 0.8pt 4pt;
    border-radius: 2.5pt;
    font-size: 7.6pt;
  }
  .tag-no .cross-icon {
    font-weight: 800;
    margin-right: 2pt;
  }

  /* Vector Architecture Diagram */
  .diagram-container {
    margin: 5pt 0 6pt 0;
    text-align: center;
    background: #f8fafc;
    border: 1px solid var(--border-light);
    border-radius: 5pt;
    padding: 6pt 5pt 4pt 5pt;
    page-break-inside: avoid;
  }
  .diagram-title {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 8pt;
    font-weight: 700;
    color: var(--gray-mid);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 5pt;
  }

  /* Paper Decisions Box */
  .decisions-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 3.5pt;
    padding: 4.5pt 7pt;
    margin: 4.5pt 0;
    font-size: 8.2pt;
    page-break-inside: avoid;
  }
  .decisions-box-title {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 2pt;
    font-size: 8.4pt;
  }

  /* Progress Sections */
  .status-tag {
    display: inline-block;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 7.2pt;
    font-weight: 700;
    text-transform: uppercase;
    padding: 1pt 4pt;
    border-radius: 2.5pt;
    margin-left: 3pt;
    vertical-align: middle;
  }
  .tag-completed {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
  }
  .tag-progress {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
  }
  .tag-planned {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
  }

  /* Evaluation 2-column cards */
  .eval-row {
    display: flex;
    gap: 8pt;
    margin: 4.5pt 0;
    page-break-inside: avoid;
  }
  .eval-card {
    flex: 1;
    background: #f8fafc;
    border: 1px solid var(--border-light);
    border-radius: 3.5pt;
    padding: 4.5pt 7pt;
  }
  .eval-card h4 {
    font-family: 'Source Sans 3', sans-serif;
    color: var(--navy);
    font-size: 8.6pt;
    font-weight: 700;
    margin-bottom: 2pt;
  }

  /* References */
  .ref-list {
    margin-left: 0;
    list-style-type: none;
    counter-reset: ref-counter;
    page-break-inside: avoid;
  }
  .ref-list li {
    position: relative;
    padding-left: 17pt;
    margin-bottom: 3.5pt;
    font-size: 8.2pt;
    line-height: 1.38;
  }
  .ref-list li::before {
    counter-increment: ref-counter;
    content: "[" counter(ref-counter) "]";
    position: absolute;
    left: 0;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
    color: var(--blue);
  }
  .ref-link {
    color: var(--blue);
    text-decoration: none;
    word-break: break-all;
    font-size: 7.8pt;
    font-family: 'Source Code Pro', monospace;
  }

  sub, sup {
    font-size: 75%;
    line-height: 0;
    position: relative;
    vertical-align: baseline;
  }
  sup { top: -0.4em; }
  sub { bottom: -0.2em; }
</style>
</head>
<body>

<!-- ==================== HEADER ==================== -->
<header class="doc-header">
  <div class="proposal-badge">DA4022 Mid-Evaluation Report</div>
  <h1 class="main-title">Project 7: Uncertainty for Optimal Memory Allocation</h1>
  
  <div class="team-section">
    <div class="team-label">Project Team Members</div>
    <div class="team-members">
      <div class="member"><strong>Deval Singhal</strong> <span>(230101035)</span></div>
      <div class="member">&bull; <strong>Jayant Kumar</strong> <span>(230101050)</span></div>
      <div class="member">&bull; <strong>Sonal Mangesh Wagh</strong> <span>(230108054)</span></div>
      <div class="member">&bull; <strong>Sristi Vats</strong> <span>(230122057)</span></div>
      <div class="member">&bull; <strong>Shourya Veer Singh</strong> <span>(230106069)</span></div>
    </div>
  </div>
</header>

<!-- ==================== 1. BACKGROUND ==================== -->
<h2>1. Background</h2>
<p>
Large Language Models (LLMs) are increasingly deployed in settings where computational resources are limited. A key bottleneck is how memory and compute are allocated during inference &mdash; particularly Key-Value (KV) cache size and generation strategy (greedy decoding vs. beam search). Currently, these execution parameters are statically fixed regardless of how confident the model is in its generated output.
</p>
<p>
Two recent ICLR 2026 papers address uncertainty estimation in LLMs from complementary angles:
</p>
<ul>
  <li>
    <strong>Paper 1 &mdash; Query-Level Uncertainty (Internal Confidence):</strong> Proposes estimating model uncertainty <em>before</em> generating any tokens, via a single forward pass. The method, called Internal Confidence (IC), probes <code>P(YES)</code> logits at every (layer, token) position and aggregates them using Attenuated Encoding centered at the last layer and last token. This produces a cheap, training-free uncertainty scalar that predicts whether the model can correctly answer a query.
  </li>
  <li>
    <strong>Paper 2 &mdash; SemanticDPP:</strong> Proposes estimating uncertainty <em>after</em> sampling a small set of responses, by measuring their semantic diversity using a Determinantal Point Process (DPP) over internal model embeddings. A learned projection maps embeddings to a semantic subspace, and the log-determinant of the resulting kernel matrix captures how diverse (uncertain) the responses are. An extension, SemanticDPP-C, uses greedy Maximum A Posteriori (MAP) clustering to identify the number of semantically distinct clusters.
  </li>
</ul>
<p>
Crucially, <strong>neither paper connects uncertainty to downstream memory or compute allocation</strong>.
</p>

<!-- ==================== 2. RESEARCH GAP AND NOVEL CONTRIBUTION ==================== -->
<h2>2. Research Gap and Novel Contribution</h2>

<h3>2.1 What the Papers Leave Open</h3>

<table class="comp-table">
  <thead>
    <tr>
      <th style="width: 25%;">Aspect</th>
      <th style="width: 37.5%;">Paper 1 (IC)</th>
      <th style="width: 37.5%;">Paper 2 (SemanticDPP)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="aspect-col">Uncertainty type</td>
      <td>Pre-generation scalar</td>
      <td>Post-generation diversity</td>
    </tr>
    <tr>
      <td class="aspect-col">Computation cost</td>
      <td>~1 forward pass (zero decode)</td>
      <td><em>m</em> forward passes (<em>m</em> &ge; 3)</td>
    </tr>
    <tr>
      <td class="aspect-col">Output</td>
      <td>Single IC score &isin; [0, 1]</td>
      <td>log-det scalar + kernel matrix</td>
    </tr>
    <tr>
      <td class="aspect-col">Downstream use</td>
      <td>Binary: trigger RAG / abstain</td>
      <td>Binary: abstain threshold</td>
    </tr>
    <tr>
      <td class="aspect-col">Memory allocation</td>
      <td><span class="tag-no"><span class="cross-icon">&#x2717;</span> Not addressed</span></td>
      <td><span class="tag-no"><span class="cross-icon">&#x2717;</span> Not addressed</span></td>
    </tr>
    <tr>
      <td class="aspect-col">Adaptive compression</td>
      <td><span class="tag-no"><span class="cross-icon">&#x2717;</span> Not addressed</span></td>
      <td><span class="tag-no"><span class="cross-icon">&#x2717;</span> Not addressed</span></td>
    </tr>
  </tbody>
</table>

<div class="callout-box">
  <div class="callout-title">&#x1F3AF; The Core Research Gap</div>
  Both papers produce uncertainty signals but use them solely as binary on/off switches. Neither framework uses uncertainty as a <em>continuous control variable</em> to dynamically modulate how much memory or compute is allocated per query.
</div>

<h3>2.2 Our Novel Contribution</h3>
<p>
We propose <strong>Uncertainty-Calibrated Adaptive Compression (UCA)</strong>: a system that leverages IC as a zero-cost pre-filter and exploits the DPP eigenspectrum as a richer, per-dimension uncertainty signal to continuously control KV cache budget and generation strategy.
</p>
<p>Three specific contributions beyond both papers:</p>
<ol>
  <li>
    <strong>IC-Gated DPP Cascade:</strong> IC pre-filters queries. Only uncertain queries (~40&ndash;50%) proceed to the more computationally demanding DPP computation, saving roughly half the overall DPP overhead.
  </li>
  <li>
    <strong>DPP Eigenspectrum as Memory Oracle:</strong> Paper 2 computes <code>L<sub>S</sub></code> but discards all per-dimension structure, retaining only <code>log det(L<sub>S</sub>)</code>. We expose the full eigenspectrum <code>{&sigma;<sub>1</sub>, &hellip;, &sigma;<sub>m</sub>}</code> of <code>L<sub>S</sub></code> and map each eigenvalue to an individual per-dimension compression ratio:
    <div class="formula-box">
      <em>r<sub>i</sub></em> = clip(1 &minus; &alpha; &middot; &sigma;<sub>i</sub> / &sigma;<sub>max</sub>, &nbsp;<em>r</em><sub>min</sub>, &nbsp;<em>r</em><sub>max</sub>)
    </div>
    A high eigenvalue &sigma;<sub>i</sub> indicates a contested semantic dimension requiring higher fidelity (less compression), whereas a low &sigma;<sub>i</sub> permits aggressive compression.
  </li>
  <li>
    <strong>Continuous KV Budget Control:</strong> The composite UCA score drives KV cache budget and beam width continuously, replacing rigid heuristic selections:
    <div class="formula-box">
      UCA = 0.4 &middot; (1 &minus; IC) + 0.6 &middot; DPP_uncertainty &nbsp;&nbsp;|&nbsp;&nbsp; KV_budget = 0.25 + 0.75 &middot; UCA
    </div>
  </li>
</ol>

<!-- ==================== 3. PROPOSED ARCHITECTURE ==================== -->
<h2>3. Proposed Architecture</h2>

<div class="diagram-container">
  <div class="diagram-title">Figure 1: Complete UCA Inference &amp; Dynamic Allocation Pipeline</div>
  <svg viewBox="0 0 740 375" width="100%" height="auto" style="max-height: 330px; font-family: 'Source Sans 3', sans-serif;">
    <defs>
      <!-- Arrow markers -->
      <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <polygon points="0 0, 6 3, 0 6" fill="#475569"/>
      </marker>
      <marker id="arr-blue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <polygon points="0 0, 6 3, 0 6" fill="#1e5ab4"/>
      </marker>
      <marker id="arr-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <polygon points="0 0, 6 3, 0 6" fill="#166534"/>
      </marker>
      <marker id="arr-red" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <polygon points="0 0, 6 3, 0 6" fill="#991b1b"/>
      </marker>
      <filter id="shadow" x="-4%" y="-4%" width="108%" height="112%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-opacity="0.07"/>
      </filter>
    </defs>

    <!-- 1. Input Query -->
    <rect x="295" y="8" width="150" height="28" rx="14" fill="#0f274a" filter="url(#shadow)"/>
    <text x="370" y="26" fill="#ffffff" font-size="10.5" font-weight="700" text-anchor="middle" letter-spacing="0.04em">INPUT QUERY q</text>

    <!-- Arrow down to Tier 0 -->
    <line x1="370" y1="36" x2="370" y2="56" stroke="#475569" stroke-width="1.8" marker-end="url(#arr)"/>

    <!-- 2. Tier 0 Box -->
    <rect x="180" y="58" width="380" height="62" rx="5" fill="#eef4fc" stroke="#1e5ab4" stroke-width="1.4" filter="url(#shadow)"/>
    <rect x="180" y="58" width="380" height="20" rx="5" fill="#1e5ab4"/>
    <rect x="180" y="72" width="380" height="6" fill="#1e5ab4"/>
    <text x="370" y="72" fill="#ffffff" font-size="9.5" font-weight="700" text-anchor="middle">TIER 0: Internal Confidence (Paper 1 &mdash; IC)</text>
    <text x="370" y="93" fill="#1e293b" font-size="8.8" text-anchor="middle">Single forward pass &bull; Training-free logit probe</text>
    <text x="370" y="109" fill="#0f274a" font-size="8.8" font-family="'Source Code Pro', monospace" font-weight="600" text-anchor="middle">IC(q) = &Sigma; w(l,n) &middot; P(YES | h_n^l)</text>

    <!-- Routing Branch Lines from Tier 0 -->
    <!-- Left Branch: Fast Path -->
    <path d="M 230 120 L 230 155 L 105 155 L 105 180" fill="none" stroke="#166534" stroke-width="1.8" marker-end="url(#arr-green)"/>
    <rect x="52" y="134" width="106" height="18" rx="3" fill="#f0fdf4" stroke="#86efac"/>
    <text x="105" y="146" fill="#166534" font-size="7.8" font-weight="700" text-anchor="middle">IC &ge; &tau;_high (Confident)</text>

    <!-- Middle Branch: Uncertain Zone -->
    <line x1="370" y1="120" x2="370" y2="180" stroke="#1e5ab4" stroke-width="1.8" marker-end="url(#arr-blue)"/>
    <rect x="306" y="137" width="128" height="18" rx="3" fill="#eef4fc" stroke="#bfdbfe"/>
    <text x="370" y="149" fill="#1e5ab4" font-size="7.8" font-weight="700" text-anchor="middle">&tau;_low &le; IC &lt; &tau;_high (Uncertain)</text>

    <!-- Right Branch: Abstain / RAG -->
    <path d="M 510 120 L 510 155 L 635 155 L 635 180" fill="none" stroke="#991b1b" stroke-width="1.8" marker-end="url(#arr-red)"/>
    <rect x="582" y="134" width="106" height="18" rx="3" fill="#fef2f2" stroke="#fca5a5"/>
    <text x="635" y="146" fill="#991b1b" font-size="7.8" font-weight="700" text-anchor="middle">IC &lt; &tau;_low (Very Uncertain)</text>

    <!-- Left Box: FAST PATH -->
    <rect x="25" y="182" width="160" height="76" rx="5" fill="#f0fdf4" stroke="#166534" stroke-width="1.4" filter="url(#shadow)"/>
    <text x="105" y="201" fill="#166534" font-size="10" font-weight="700" text-anchor="middle">FAST PATH</text>
    <line x1="45" y1="208" x2="165" y2="208" stroke="#86efac" stroke-width="0.8"/>
    <text x="105" y="223" fill="#14532d" font-size="8.3" text-anchor="middle">&bull; Maximum compression</text>
    <text x="105" y="237" fill="#14532d" font-size="8.3" font-weight="600" text-anchor="middle">&bull; 25% KV cache budget</text>
    <text x="105" y="251" fill="#14532d" font-size="8.3" text-anchor="middle">&bull; Greedy decoding (beam = 1)</text>

    <!-- Middle Box: TIER 1 DPP -->
    <rect x="250" y="182" width="240" height="76" rx="5" fill="#fffbeb" stroke="#d97706" stroke-width="1.4" filter="url(#shadow)"/>
    <text x="370" y="199" fill="#b45309" font-size="9.5" font-weight="700" text-anchor="middle">TIER 1: SemanticDPP (Paper 2)</text>
    <line x1="270" y1="206" x2="470" y2="206" stroke="#fde68a" stroke-width="0.8"/>
    <text x="370" y="219" fill="#78350f" font-size="8.2" text-anchor="middle">Sample m=3 responses &bull; Project &phi;(x)</text>
    <text x="370" y="233" fill="#78350f" font-size="8.2" text-anchor="middle">Construct Kernel L_S &bull; Eigendecompose L_S = U&Sigma;U&#7488;</text>
    <text x="370" y="248" fill="#78350f" font-size="8.2" font-weight="600" text-anchor="middle">Extract Eigenspectrum {&sigma;&#8321;, &hellip;, &sigma;<tspan font-size="6.8" dy="1.5">m</tspan><tspan dy="-1.5">}</tspan></text>

    <!-- Right Box: ABSTAIN / RAG -->
    <rect x="555" y="182" width="160" height="76" rx="5" fill="#fef2f2" stroke="#991b1b" stroke-width="1.4" filter="url(#shadow)"/>
    <text x="635" y="201" fill="#991b1b" font-size="10" font-weight="700" text-anchor="middle">ABSTAIN / RAG</text>
    <line x1="575" y1="208" x2="695" y2="208" stroke="#fca5a5" stroke-width="0.8"/>
    <text x="635" y="223" fill="#7f1d1d" font-size="8.3" text-anchor="middle">&bull; External knowledge retrieval</text>
    <text x="635" y="237" fill="#7f1d1d" font-size="8.3" font-weight="600" text-anchor="middle">&bull; Full KV cache (100%)</text>
    <text x="635" y="251" fill="#7f1d1d" font-size="8.3" text-anchor="middle">&bull; Wide beam search (width = 4)</text>

    <!-- Arrow down from Tier 1 to Compression Controller -->
    <line x1="370" y1="258" x2="370" y2="280" stroke="#475569" stroke-width="1.6" marker-end="url(#arr)"/>

    <!-- Middle Sub-Box: Compression Controller -->
    <rect x="235" y="282" width="270" height="32" rx="4" fill="#f1f5f9" stroke="#64748b" stroke-width="1.1"/>
    <text x="370" y="296" fill="#0f172a" font-size="8.2" font-weight="700" text-anchor="middle">Per-Dimension Compression Ratio</text>
    <text x="370" y="308" fill="#1e293b" font-size="8" font-family="'Source Code Pro', monospace" text-anchor="middle">r_i = clip(1 &minus; &alpha; &middot; &sigma;_i / &sigma;_max, r_min, r_max)</text>

    <!-- Output allocation branches -->
    <path d="M 300 314 L 300 332 L 220 332 L 220 342" fill="none" stroke="#1e5ab4" stroke-width="1.6" marker-end="url(#arr-blue)"/>
    <path d="M 440 314 L 440 332 L 520 332 L 520 342" fill="none" stroke="#1e5ab4" stroke-width="1.6" marker-end="url(#arr-blue)"/>

    <rect x="130" y="344" width="180" height="23" rx="4" fill="#eef4fc" stroke="#1e5ab4"/>
    <text x="220" y="359" fill="#133a75" font-size="8.2" font-weight="700" text-anchor="middle">KV Budget = f(UCA) &isin; [0.25, 1.0]</text>

    <rect x="430" y="344" width="180" height="23" rx="4" fill="#eef4fc" stroke="#1e5ab4"/>
    <text x="520" y="359" fill="#133a75" font-size="8.2" font-weight="700" text-anchor="middle">Beam Width = f(UCA) &isin; [1, 4]</text>
  </svg>
</div>

<div class="decisions-box">
  <div class="decisions-box-title">Key Design Decisions Grounded in Literature:</div>
  <ul style="margin: 2pt 0 0 12pt;">
    <li><code>&tau;<sub>high</sub> = 0.65, &tau;<sub>low</sub> = 0.30</code> &mdash; Grounded directly in IC's reported adaptive RAG threshold experiments (Paper 1, Section 5).</li>
    <li><code>&alpha; = 1.0</code> in Attenuated Encoding &mdash; Kept fixed per Paper 1, Section 4.1 for training-free execution.</li>
    <li><code>m = 3</code> DPP responses &mdash; Paper 2 uses <em>m</em> = 5 for peak standalone AUROC; we use <em>m</em> = 3 since the IC pre-filter removes most clear queries.</li>
    <li>DPP-C cluster count <code>k = max(1, round(trace(K)))</code> &mdash; Follows Paper 2, Algorithm 1; utilized directly as an adaptive beam width floor.</li>
  </ul>
</div>

<!-- ==================== 4. IMPLEMENTATION PROGRESS ==================== -->
<h2>4. Implementation Progress</h2>

<h3>4.1 Completed <span class="status-tag tag-completed">Done</span></h3>
<ul>
  <li>
    <strong>Environment Setup:</strong> Python 3.11, PyTorch 2.3, HuggingFace Transformers, and the official <code>query-level-uncertainty</code> package installed and verified.
  </li>
  <li>
    <strong>IC Gate (<code>src/core/internal_confidence.py</code>):</strong> Implemented the <code>ICGate</code> class wrapping the official <code>tigerchen52/query_level_uncertainty</code> package. Self-evaluation prompt template and Attenuated Encoding with <code>&alpha; = 1.0</code> operational. Tested on TriviaQA with Llama-3.2-3B; scores route reliably into fast, uncertain, and abstain buckets.
  </li>
  <li>
    <strong>SemanticDPP Module (<code>src/core/semantic_dpp.py</code>):</strong> Implemented <code>SemanticSubspaceProjector</code> (orthogonal-initialized NN, <em>d</em> &rarr; 256) and kernel matrix computation:
    <div class="formula-box">
      <em>K</em>(<em>x<sub>i</sub></em>, <em>x<sub>j</sub></em>) = exp(&minus;&alpha; ||&phi;(<em>x<sub>i</sub></em>) &minus; &phi;(<em>x<sub>j</sub></em>)||<sup>2</sup>)
    </div>
    The <code>uncertainty_score()</code> returns <code>log_det</code>, normalized uncertainty, eigenspectrum, and DPP-C cluster count. Eigenvalue extraction via <code>torch.linalg.eigvalsh</code> is numerically verified.
  </li>
  <li>
    <strong>UCA Controller (<code>src/core/uca_controller.py</code>):</strong> Wires IC and DPP together. Method <code>compute_compression_config()</code> produces execution route, KV budget, beam width, and per-dimension compression ratios for any incoming query. Unit-tested on sample inputs; routing and continuous control logic verified.
  </li>
</ul>

<h3 style="page-break-before: always;">4.2 In Progress <span class="status-tag tag-progress">Active</span></h3>
<ul>
  <li>
    <strong>Projector Training:</strong> Training <code>SemanticSubspaceProjector</code> by maximizing DPP marginal likelihood:
    <div class="formula-box">
      &Delta; = log det(<em>L<sub>Y</sub></em>) &minus; log det(<em>L</em> + <em>I</em>)
    </div>
    on pseudolabeled TriviaQA clusters. Optimization loop is implemented; tuning hyperparameters for stable convergence.
  </li>
  <li>
    <strong>Baseline IC Reproduction:</strong> Running IC benchmark on TriviaQA validation split (Llama-3.2-3B) to empirically reproduce published AUROC numbers. Evaluation is currently underway.
  </li>
</ul>

<h3>4.3 Not Yet Started <span class="status-tag tag-planned">Planned</span></h3>
<ul>
  <li><strong>Full UCA End-to-End Evaluation (<code>run_uca.py</code>):</strong> Integrating routing with KV cache compression execution.</li>
  <li><strong>SemanticDPP Standalone AUROC Measurement:</strong> Running full sampling-based evaluation without the pre-filter.</li>
  <li><strong>Ablation Experiments:</strong> Isolating individual contributions of eigenspectrum and gating mechanisms.</li>
</ul>

<!-- ==================== 5. EVALUATION PLAN ==================== -->
<h2>5. Evaluation Plan</h2>
<p>
We will evaluate UCA against two standalone baselines drawn directly from the two reference papers.
</p>

<div class="eval-row">
  <div class="eval-card">
    <h4>Evaluated Baselines</h4>
    <ul style="margin: 2pt 0 0 10pt; font-size: 8.2pt;">
      <li><strong>IC Only</strong> (Paper 1 standalone): IC score alone for routing; no DPP stage.</li>
      <li><strong>SemanticDPP Only</strong> (Paper 2 standalone): DPP for every query; log-det scalar threshold for abstain.</li>
    </ul>
  </div>

  <div class="eval-card">
    <h4>Benchmark Datasets &amp; Models</h4>
    <ul style="margin: 2pt 0 0 10pt; font-size: 8.2pt;">
      <li><strong>Datasets:</strong> TriviaQA (factual retrieval), GSM8K (multi-step reasoning) &mdash; matching Paper 1 protocol.</li>
      <li><strong>Base Model:</strong> Llama-3.2-3B.</li>
    </ul>
  </div>
</div>

<h3>Primary Evaluation Metrics</h3>
<ul>
  <li><strong>AUROC:</strong> Uncertainty score vs. correctness ground truth (primary discriminative metric from both papers).</li>
  <li><strong>PRR (Prediction Rejection Ratio):</strong> Measures how rapidly error rate drops as high-uncertainty queries are rejected.</li>
  <li><strong>Average KV Tokens Used:</strong> Measures physical GPU memory savings achieved through adaptive budgeting.</li>
  <li><strong>End-to-End Latency per Query:</strong> Profiling overhead &mdash; IC adds ~1 forward pass; DPP adds ~3 generation passes.</li>
</ul>

<!-- ==================== REFERENCES ==================== -->
<h2>References</h2>
<ol class="ref-list">
  <li>
    Chen, L., de Melo, G., Suchanek, F. M., &amp; Varoquaux, G. (2026). <em>Query-Level Uncertainty in Large Language Models</em>. International Conference on Learning Representations (ICLR 2026).<br/>
    <a class="ref-link" href="https://openreview.net/pdf?id=11QZITAMUO">https://openreview.net/pdf?id=11QZITAMUO</a>
  </li>
  <li>
    Hamer, J., Snoek, J., et al. (2026). <em>SemanticDPP: Efficient Uncertainty Quantification in LLMs</em>. International Conference on Learning Representations (ICLR 2026).<br/>
    <a class="ref-link" href="https://openreview.net/pdf?id=C8yjlrx4Wo">https://openreview.net/pdf?id=C8yjlrx4Wo</a>
  </li>
</ol>

</body>
</html>
"""

def generate_pdf():
    print("Generating MID_EVALUATION.pdf using WeasyPrint...")
    doc = weasyprint.HTML(string=HTML_CONTENT)
    output_path = "/Users/devalsinghal/Desktop/sem 7/DA4022/project/DA4022/MID_EVALUATION.pdf"
    doc.write_pdf(output_path)
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_pdf()
