"""
Step 3 - turn the saved model runs into results.

    python -m src.experiments.analyze

Reads results/runs.jsonl + results/runs.npz (written by collect.py) and produces
results/summary.json and the figures in results/figures/.  No model is needed
here, so this step takes about a minute and can be re-run freely.

Everything that has to be fitted (the DPP kernel, the calibration curves, the
thresholds) is fitted on one half of the questions and evaluated on the other
half.  We repeat that for several random halvings and report the average.
"""

import argparse
import json
import os

import numpy as np

from src.core.allocator import (allocate, allocate_oracle, escalate, fit_calibrators, fit_gain_table,
                                fixed_mix_accuracy, probability_correct)
from src.core.internal_confidence import (internal_confidence, p_yes_naive_average,
                                          p_yes_top_right)
from src.core.semantic_dpp import SemanticDPP, cluster_answers, lexical_diversity
from src.evaluation.metrics import auroc, ece, prr

# "Confident enough" thresholds that we sweep.  0 = always answer at once, above 1 = never stop early.
THRESHOLDS = np.concatenate([[0.0], np.round(np.arange(0.10, 0.96, 0.05), 2), [1.01]])
HEADLINE_THRESHOLD = 0.70                  # the setting quoted in the summary table
CASCADE_BAND = (0.30, 0.70)                # closed-book IC percentiles where we also run SemanticDPP

# Confidence signals (higher = more sure).  Names are used in tables and figures.
IC, PROB, DPP, COMBINED = "Internal Confidence", "Answer probability", "SemanticDPP", "All three combined"
MAIN_SIGNALS = [IC, PROB, DPP]
ESCALATION_SIGNALS = [IC, PROB, DPP, COMBINED]
ONE_SHOT_IC, ONE_SHOT_CASCADE = "One-shot: IC before retrieval", "One-shot: IC, then DPP if unsure"
ORACLE = "Oracle (upper bound)"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_runs(prefix: str) -> dict:
    records = [json.loads(line) for line in open(prefix + ".jsonl")]
    arrays = np.load(prefix + ".npz")
    n = min(len(records), len(arrays["p_yes"]))
    records = records[:n]
    levels = [r["levels"] for r in records]
    n_checks = len(levels[0]) - 1               # confidence is checked at every level but the last

    def table(key, n_cols):
        return np.array([[lv[i][key] for i in range(n_cols)] for lv in levels], dtype=float)

    return {
        "n": n, "records": records, "n_checks": n_checks,
        "passages": [lv["passages"] for lv in levels[0]],
        "correct": table("correct", n_checks + 1).astype(bool),             # [n, levels]
        "tokens": table("prompt_tokens", n_checks + 1),
        "seconds": table("seconds", n_checks + 1),
        "answer_log_prob": table("answer_log_prob", n_checks + 1),
        "ic_tokens": table("ic_tokens", n_checks),                          # [n, checks]
        "ic_seconds": table("ic_seconds", n_checks),
        "sample_seconds": table("sample_seconds", n_checks),
        "p_yes": arrays["p_yes"][:n],                                       # [n, checks, layers, tokens]
        "embeddings": arrays["embeddings"][:n].astype(np.float32),          # [n, checks, m, hidden]
        "clusters": np.array([[cluster_answers(lv[c]["samples"]) for c in range(n_checks)] for lv in levels]),
        "disagreement": np.array([[lexical_diversity(lv[c]["samples"]) for c in range(n_checks)] for lv in levels]),
    }


def percentile_in(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Where does each value fall among the calibration values? 0 = lowest, 1 = highest."""
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


# --------------------------------------------------------------------------
# Confidence signals
# --------------------------------------------------------------------------
def raw_confidence(data: dict, cal: np.ndarray, seed: int) -> dict:
    """
    Every confidence signal for ALL questions at every check level: [n, checks].
    The DPP kernel is the only trainable part and is fitted on `cal` questions only.
    """
    n, n_checks = data["n"], data["n_checks"]
    p_yes = data["p_yes"]

    def per_cell(function):
        return np.array([[function(p_yes[i, c]) for c in range(n_checks)] for i in range(n)])

    emb, clusters = data["embeddings"], data["clusters"]
    m, d = emb.shape[2:]

    def dpp_confidence(trainable: bool):
        dpp = SemanticDPP(trainable=trainable, seed=seed)
        dpp.fit(emb[cal].reshape(-1, m, d), clusters[cal].reshape(-1, m))   # all check levels pooled
        return -dpp.score(emb.reshape(-1, m, d)).reshape(n, n_checks)       # diverse answers = unsure

    return {
        IC: per_cell(internal_confidence),
        "P(Yes), last layer and token only": per_cell(p_yes_top_right),
        "P(Yes), plain average": per_cell(p_yes_naive_average),
        PROB: data["answer_log_prob"][:, :n_checks],
        DPP: dpp_confidence(trainable=True),
        "SemanticDPP, untrained kernel": dpp_confidence(trainable=False),
        "Sampled-answer agreement (string match)": -data["disagreement"],
    }


def calibrated(raw: dict, correct: np.ndarray, cal: np.ndarray) -> dict:
    """
    Turn every main signal into P(answer at this level is correct): [n, checks].
    The curves are fitted on the calibration questions.  "All three combined" is
    the plain average of the three probabilities.
    """
    p = {name: probability_correct(fit_calibrators(raw[name][cal], correct[cal]), raw[name])
         for name in MAIN_SIGNALS}
    p[COMBINED] = np.mean([p[name] for name in MAIN_SIGNALS], axis=0)
    return p


# --------------------------------------------------------------------------
# One calibration / test split
# --------------------------------------------------------------------------
def evaluate_split(data: dict, cal: np.ndarray, test: np.ndarray, seed: int) -> dict:
    correct, tokens = data["correct"], data["tokens"]
    n_checks, n_levels = data["n_checks"], correct.shape[1]
    rows = np.arange(len(test))
    raw = raw_confidence(data, cal, seed)
    p_correct = calibrated(raw, correct, cal)

    def measure(level: np.ndarray) -> tuple:
        """(accuracy, average tokens, share of questions at each level) of an assignment on the test half."""
        return (correct[test][rows, level].mean(), tokens[test][rows, level].mean(),
                np.bincount(level, minlength=n_levels) / len(level))

    fixed_acc, fixed_tok = correct[test].mean(axis=0), tokens[test].mean(axis=0)

    def with_gain(measured: list) -> dict:
        """Stack results and add the advantage over the best uncertainty-blind mix at the same memory."""
        acc, tok, mix = (np.array(x) for x in zip(*measured))
        gain = acc - np.array([fixed_mix_accuracy(fixed_acc, fixed_tok, t) for t in tok])
        return {"accuracy": acc, "tokens": tok, "level_mix": mix, "gain": gain}

    # ---- How well does each signal tell right answers from wrong ones, at each level?
    quality = {name: [{"auroc": auroc(correct[test, c], -score[test, c]),
                       "prr": prr(correct[test, c], -score[test, c])} for c in range(n_checks)]
               for name, score in raw.items()}

    # ---- A) Escalation: sweep the confidence threshold for every maximum level.
    escalation = {name: {max_level: with_gain([measure(escalate(p_correct[name][test], t, max_level))
                                               for t in THRESHOLDS])
                         for max_level in range(1, n_levels)}
                  for name in ESCALATION_SIGNALS}

    # ---- B) One-shot allocation from closed-book confidence only, for a range of budgets.
    level_cost = tokens[cal].mean(axis=0)
    budgets = np.linspace(level_cost[0], level_cost[-1], 21)
    ic_unsure, dpp_unsure = -raw[IC][:, 0], -raw[DPP][:, 0]
    ic_rank = percentile_in(ic_unsure[cal], ic_unsure)
    dpp_rank = percentile_in(dpp_unsure[cal], dpp_unsure)
    in_band = (ic_rank > CASCADE_BAND[0]) & (ic_rank <= CASCADE_BAND[1])
    cascade = np.where(in_band, (ic_rank + dpp_rank) / 2, ic_rank)
    by_budget = {}
    for name, score in [(ONE_SHOT_IC, ic_unsure), (ONE_SHOT_CASCADE, cascade)]:
        edges, gain_table = fit_gain_table(score[cal], correct[cal])
        by_budget[name] = with_gain([measure(allocate(score[test], edges, gain_table, level_cost, b))
                                     for b in budgets])
    by_budget[ORACLE] = with_gain([measure(allocate_oracle(correct[test], level_cost, b)) for b in budgets])

    return {"quality": quality, "escalation": escalation, "by_budget": by_budget,
            "fixed_accuracy": fixed_acc, "fixed_tokens": fixed_tok,
            "cascade_dpp_fraction": float(in_band[test].mean())}


# --------------------------------------------------------------------------
# Time accounting
# --------------------------------------------------------------------------
def escalation_seconds(signal: str, max_level: int, level_mix: np.ndarray, timing: dict) -> np.ndarray:
    """
    Expected wall-clock seconds per question, counting every confidence check on
    the way up.  level_mix: [..., levels] share of questions ending at each level.
    A question ending below max_level was checked at every level up to its own;
    one that reaches max_level is answered there without a further check.
    """
    answer, ic, sampling = timing["answer"], timing["ic"], timing["sampling"]
    cost_of_ending_at = np.zeros(len(answer))
    for final in range(max_level + 1):
        checks = range(final + 1) if final < max_level else range(max_level)
        answered_unchecked = final == max_level
        if signal == IC:                 # one IC pass per check, one answer at the end
            total = sum(ic[c] for c in checks) + answer[final]
        elif signal == PROB:             # the answer itself is the check, at every visited level
            total = sum(answer[lv] for lv in range(final + 1))
        elif signal == DPP:              # answer pass + sampled answers at every check
            total = sum(answer[c] + sampling[c] for c in checks) + answered_unchecked * answer[final]
        else:                            # all three
            total = sum(ic[c] + answer[c] + sampling[c] for c in checks) + answered_unchecked * answer[final]
        cost_of_ending_at[final] = total
    return level_mix @ cost_of_ending_at


# --------------------------------------------------------------------------
# Aggregation over splits
# --------------------------------------------------------------------------
def mean_std(values) -> dict:
    values = np.array(values, dtype=float)
    return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}


def curve_summary(per_split: list, mb: float) -> dict:
    """Average a list of per-split curves (dicts of arrays) point by point."""
    acc, tok, gain = (np.array([s[k] for s in per_split]) for k in ("accuracy", "tokens", "gain"))
    return {"tokens": tok.mean(axis=0).tolist(), "kv_mb": (tok.mean(axis=0) * mb).tolist(),
            "accuracy": acc.mean(axis=0).tolist(), "accuracy_std": acc.std(axis=0).tolist(),
            "gain_over_fixed_mix": gain.mean(axis=0).tolist(), "gain_std": gain.std(axis=0).tolist(),
            "level_mix": np.mean([s["level_mix"] for s in per_split], axis=0).tolist()}


def summarise(data: dict, splits: list, kv_bytes_per_token: int) -> dict:
    mb = kv_bytes_per_token / 1e6
    passages = data["passages"]
    fixed_acc = np.array([s["fixed_accuracy"] for s in splits])              # [splits, levels]
    fixed_tok = np.array([s["fixed_tokens"] for s in splits])
    timing = {"answer": np.median(data["seconds"], axis=0),
              "ic": np.median(data["ic_seconds"], axis=0),
              "sampling": np.median(data["sample_seconds"], axis=0)}
    headline = int(np.argmin(np.abs(THRESHOLDS - HEADLINE_THRESHOLD)))

    summary = {
        "n_questions": data["n"], "n_splits": len(splits), "kv_bytes_per_token": kv_bytes_per_token,
        "fixed_levels": [{"passages": passages[i], "accuracy": float(fixed_acc[:, i].mean()),
                          "tokens": float(fixed_tok[:, i].mean()), "kv_mb": float(fixed_tok[:, i].mean() * mb),
                          "seconds": float(timing["answer"][i])} for i in range(len(passages))],
        "check_seconds": {"passages": passages[:-1], "internal_confidence": timing["ic"].round(3).tolist(),
                          "sampling_5_answers": timing["sampling"].round(3).tolist()},
        "ic_prompt_extra_tokens": float((data["ic_tokens"] - data["tokens"][:, :-1]).mean()),
        "cascade_dpp_fraction": float(np.mean([s["cascade_dpp_fraction"] for s in splits])),
        "thresholds": THRESHOLDS.tolist(), "headline_threshold": HEADLINE_THRESHOLD,
        "signal_quality": {
            name: [{"passages": passages[c],
                    "auroc": mean_std([s["quality"][name][c]["auroc"] for s in splits]),
                    "prr": mean_std([s["quality"][name][c]["prr"] for s in splits])}
                   for c in range(data["n_checks"])]
            for name in splits[0]["quality"]},
        "escalation": {}, "headline": [], "by_budget": {},
    }

    for name in ESCALATION_SIGNALS:
        summary["escalation"][name] = {}
        for max_level in range(1, len(passages)):
            per_split = [s["escalation"][name][max_level] for s in splits]
            curve = curve_summary(per_split, mb)
            curve["seconds"] = escalation_seconds(name, max_level, np.array(curve["level_mix"]), timing).tolist()
            summary["escalation"][name][str(passages[max_level])] = curve

            # The headline row: this signal, this maximum level, threshold 0.70.
            acc = np.array([s["accuracy"][headline] for s in per_split])
            tok = np.array([s["tokens"][headline] for s in per_split])
            gain = np.array([s["gain"][headline] for s in per_split])
            summary["headline"].append({
                "signal": name, "max_passages": passages[max_level],
                "accuracy": mean_std(acc), "tokens": float(tok.mean()), "kv_mb": float(tok.mean() * mb),
                "fixed_accuracy": float(fixed_acc[:, max_level].mean()),
                "fixed_tokens": float(fixed_tok[:, max_level].mean()),
                "accuracy_vs_fixed_level": mean_std(acc - fixed_acc[:, max_level]),
                "memory_saved": mean_std(1 - tok / fixed_tok[:, max_level]),
                "gain_over_fixed_mix_at_same_memory": mean_std(gain),
                "seconds": curve["seconds"][headline], "fixed_seconds": float(timing["answer"][max_level]),
            })

    for name in splits[0]["by_budget"]:
        per_split = [s["by_budget"][name] for s in splits]
        curve = curve_summary(per_split, mb)
        curve["mean_gain_over_fixed_mix"] = mean_std([s["gain"].mean() for s in per_split])
        summary["by_budget"][name] = curve
    return summary


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#0d366b"]
VIOLET = "#6250d6"
SIGNAL_STYLE = [(IC, BLUE), (PROB, ORANGE), (DPP, AQUA), (COMBINED, VIOLET)]


def _axes(title: str, xlabel: str, ylabel: str, figsize=(7.6, 4.6), ncols=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xlabel(xlabel, color=SECONDARY)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel(ylabel, color=SECONDARY)
    if ncols == 1:
        first.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    else:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    return fig, axes


def _save(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def figure_accuracy_vs_memory(summary: dict, path: str):
    """Left: escalation up to 2 passages.  Right: up to 8.  Each line sweeps the confidence threshold."""
    fig, axes = _axes("Accuracy against KV-cache memory: escalation vs. the same memory for everyone",
                      "Average KV-cache memory per question (MB)", "Accuracy (%)", figsize=(11, 4.8), ncols=2)
    fixed = summary["fixed_levels"]
    for ax, max_passages in zip(axes, (2, 8)):
        shown = [f for f in fixed if f["passages"] <= max_passages]
        ax.plot([f["kv_mb"] for f in shown], [100 * f["accuracy"] for f in shown], color=INK,
                linewidth=1.5, linestyle="--", label="Same for everyone (best fixed mix)")
        for name, color in SIGNAL_STYLE:
            curve = summary["escalation"][name][str(max_passages)]
            ax.plot(curve["kv_mb"], 100 * np.array(curve["accuracy"]), color=color, linewidth=2,
                    label=f"Escalation: {name}")
        ax.scatter([f["kv_mb"] for f in shown], [100 * f["accuracy"] for f in shown], s=64, color=INK,
                   zorder=5, edgecolor=SURFACE, linewidth=2)
        for f in shown:
            ax.annotate(f"{f['passages']} passages" if f["passages"] != 1 else "1 passage",
                        (f["kv_mb"], 100 * f["accuracy"]), textcoords="offset points",
                        xytext=(7, -13), color=SECONDARY, fontsize=9)
        ax.set_title(f"At most {max_passages} passages per question", loc="left", color=SECONDARY, fontsize=10)
        ax.margins(x=0.14)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9, labelcolor=SECONDARY)
    _save(fig, path)


def figure_signal_quality(summary: dict, path: str):
    """AUROC of each main signal at each memory level (grouped bars)."""
    fig, ax = _axes("Can the model tell when its answer is wrong?",
                    "Passages in the prompt when confidence is read",
                    "AUROC (0.5 = chance, 1.0 = perfect)")
    quality = summary["signal_quality"]
    passages = [q["passages"] for q in quality[IC]]
    x = np.arange(len(passages))
    width = 0.26
    for k, (name, color) in enumerate([(IC, BLUE), (PROB, ORANGE), (DPP, AQUA)]):
        values = [q["auroc"]["mean"] for q in quality[name]]
        errors = [q["auroc"]["std"] for q in quality[name]]
        bars = ax.bar(x + (k - 1) * width, values, width - 0.03, color=color, label=name,
                      yerr=errors, error_kw={"ecolor": MUTED, "lw": 1})
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.405, f"{value:.2f}", ha="center",
                    color="#ffffff", fontsize=8)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(x, [str(p) for p in passages])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9,
              labelcolor=SECONDARY, ncols=3)
    _save(fig, path)


def figure_ic_heatmap(data: dict, path: str) -> dict:
    """AUROC of P(Yes) at every (layer, token) cell, closed book - cf. Paper A, Figure 3b."""
    p_yes, correct = data["p_yes"][:, 0], data["correct"][:, 0]
    n_layers, n_tokens = p_yes.shape[1:]
    grid = np.array([[auroc(correct, 1 - p_yes[:, layer, t]) for t in range(n_tokens)]
                     for layer in range(n_layers)])
    fig, ax = _axes("Where inside the model is the 'I know this' signal?",
                    "Position among the last 6 prompt tokens", "Layer", figsize=(5.4, 6.2))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="Blues", vmin=0.5,
                      vmax=max(0.7, float(grid.max())))
    ax.grid(visible=False)
    ax.set_xticks(range(n_tokens), [str(t - n_tokens + 1) if t < n_tokens - 1 else "last"
                                    for t in range(n_tokens)])
    ax.set_yticks(range(0, n_layers, 3), [str(layer + 1) for layer in range(0, n_layers, 3)])
    best = np.unravel_index(np.argmax(grid), grid.shape)
    ax.scatter([best[1]], [best[0]], s=110, facecolor="none", edgecolor=ORANGE, linewidth=2)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("AUROC", color=SECONDARY)
    bar.outline.set_visible(False)
    _save(fig, path)
    return {"best_layer": int(best[0]) + 1, "best_token_from_end": int(n_tokens - 1 - best[1]),
            "best_auroc": float(grid[best]), "last_layer_last_token_auroc": float(grid[-1, -1])}


def figure_where_questions_stop(summary: dict, path: str) -> dict:
    """How escalation (answer probability, threshold 0.70) spreads memory, next to the fixed levels."""
    passages = [f["passages"] for f in summary["fixed_levels"]]
    index = summary["thresholds"].index(HEADLINE_THRESHOLD)
    rows = [("Everyone gets 2", np.eye(len(passages))[2]),
            ("Escalation, at most 2", np.array(summary["escalation"][PROB]["2"]["level_mix"][index])),
            ("Everyone gets 8", np.eye(len(passages))[4]),
            ("Escalation, at most 8", np.array(summary["escalation"][PROB]["8"]["level_mix"][index]))]
    fig, ax = _axes("Where questions stop (answer probability, threshold 0.70)",
                    "Share of questions (%)", "", figsize=(7.6, 3.4))
    table = {}
    for y, (label, mix) in enumerate(rows[::-1]):
        table[label] = (100 * mix).round(1).tolist()
        left = 0.0
        for level, share in enumerate(100 * mix):
            ax.barh(y, share, left=left, height=0.6, color=BLUE_RAMP[level], edgecolor=SURFACE, linewidth=2,
                    label=(f"{passages[level]} passage" + ("" if passages[level] == 1 else "s")) if y == 0 else None)
            if share >= 6:
                ax.text(left + share / 2, y, f"{share:.0f}", ha="center", va="center", fontsize=9,
                        color=INK if level < 2 else "#ffffff")
            left += share
    ax.set_yticks(range(len(rows)), [label for label, _ in rows[::-1]])
    ax.tick_params(axis="y", colors=INK)
    ax.grid(visible=False)
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncols=5, fontsize=8.5, labelcolor=SECONDARY, loc="upper center",
              bbox_to_anchor=(0.45, -0.22))
    _save(fig, path)
    return table


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--kv_bytes_per_token", type=int, default=28672,
                        help="Qwen2.5-1.5B in float16: 2 x 28 layers x 2 KV heads x 128 dims x 2 bytes")
    args = parser.parse_args()

    data = load_runs(args.runs)
    n = data["n"]
    print(f"{n} questions")

    splits = []
    for seed in range(args.splits):
        order = np.random.default_rng(seed).permutation(n)
        splits.append(evaluate_split(data, order[: n // 2], order[n // 2:], seed))
    summary = summarise(data, splits, args.kv_bytes_per_token)

    # ---- A few descriptive numbers over all questions
    correct = data["correct"]
    closed_book_ic = np.array([internal_confidence(p) for p in data["p_yes"][:, 0]])
    summary["ic_calibration_error_closed_book"] = ece(correct[:, 0], closed_book_ic)
    summary["mean_distinct_sampled_answers"] = (data["clusters"].max(axis=2) + 1).mean(axis=0).tolist()
    summary["right_closed_book_stays_right"] = correct[correct[:, 0]].mean(axis=0).tolist()
    summary["wrong_closed_book_becomes_right"] = correct[~correct[:, 0]].mean(axis=0).tolist()
    summary["right_at_some_level"] = float(correct.any(axis=1).mean())

    figures = os.path.join(args.out, "figures")
    os.makedirs(figures, exist_ok=True)
    figure_accuracy_vs_memory(summary, f"{figures}/accuracy_vs_memory.png")
    figure_signal_quality(summary, f"{figures}/signal_quality.png")
    summary["ic_heatmap"] = figure_ic_heatmap(data, f"{figures}/ic_heatmap.png")
    summary["where_questions_stop"] = figure_where_questions_stop(summary, f"{figures}/where_questions_stop.png")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---- Console report
    print("\nSame memory for everyone:")
    for f in summary["fixed_levels"]:
        print(f"  {f['passages']} passages: accuracy {100 * f['accuracy']:.1f}%   {f['tokens']:.0f} tokens "
              f"= {f['kv_mb']:.1f} MB KV cache   {f['seconds']:.2f}s")
    print("\nAUROC of each confidence signal, by passages in the prompt "
          f"{[q['passages'] for q in summary['signal_quality'][IC]]}:")
    for name, per_level in summary["signal_quality"].items():
        print(f"  {name:42s} " + "  ".join(f"{q['auroc']['mean']:.3f}" for q in per_level))
    print(f"\nEscalation at threshold {HEADLINE_THRESHOLD}:")
    for row in summary["headline"]:
        if row["max_passages"] in (2, 8):
            print(f"  {row['signal']:20s} at most {row['max_passages']}: "
                  f"{100 * row['accuracy']['mean']:.1f}% with {row['kv_mb']:.1f} MB "
                  f"(everyone gets {row['max_passages']}: {100 * row['fixed_accuracy']:.1f}%) | "
                  f"memory saved {100 * row['memory_saved']['mean']:.0f}% | accuracy "
                  f"{100 * row['accuracy_vs_fixed_level']['mean']:+.1f} | vs fixed mix at same memory "
                  f"{100 * row['gain_over_fixed_mix_at_same_memory']['mean']:+.1f} ± "
                  f"{100 * row['gain_over_fixed_mix_at_same_memory']['std']:.1f} | "
                  f"{row['seconds']:.2f}s vs {row['fixed_seconds']:.2f}s")
    print("\nDeciding once, before retrieval (average gain over the fixed mix across budgets):")
    for name, curve in summary["by_budget"].items():
        g = curve["mean_gain_over_fixed_mix"]
        print(f"  {name:36s} {100 * g['mean']:+.1f} ± {100 * g['std']:.1f} pts")
    print(f"\nwrote {args.out}/summary.json and figures to {figures}/")


if __name__ == "__main__":
    main()
