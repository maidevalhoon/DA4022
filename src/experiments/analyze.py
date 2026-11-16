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

from src.core.allocator import (Calibrator, allocate, allocate_oracle, best_fixed_mix, escalate,
                                fit_gain_table, threshold_for_budget)
from src.core.internal_confidence import (internal_confidence, p_yes_naive_average,
                                          p_yes_top_right)
from src.core.semantic_dpp import SemanticDPP, cluster_answers, lexical_diversity
from src.evaluation.metrics import auroc, ece, prr

THRESHOLDS = np.linspace(0.0, 1.01, 102)   # candidate "confident enough" thresholds
CASCADE_BAND = (0.30, 0.70)                # closed-book IC percentiles where we also run SemanticDPP

# Confidence signals (higher = more sure).  Names are used in tables and figures.
IC, PROB, DPP = "Internal Confidence", "Answer probability", "SemanticDPP"
IC_PROB, ALL = "IC + answer probability", "All three combined"
MAIN_SIGNALS = [IC, PROB, DPP]
ESCALATION_POLICIES = [IC, PROB, DPP, IC_PROB, ALL]
# Policies
FIXED, ORACLE = "Best fixed mix (no uncertainty)", "Oracle (upper bound)"
ONE_SHOT_IC, ONE_SHOT_CASCADE = "One-shot: IC before retrieval", "One-shot: IC then DPP cascade"


def esc(signal: str) -> str:
    return f"Escalation: {signal}"


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


def calibrated_probabilities(raw: dict, correct: np.ndarray, cal: np.ndarray) -> dict:
    """P(answer at this level is correct), one calibration curve per signal and level."""
    n_checks = raw[IC].shape[1]
    prob = {}
    for name in MAIN_SIGNALS:
        prob[name] = np.column_stack([
            Calibrator().fit(raw[name][cal, c], correct[cal, c]).predict(raw[name][:, c])
            for c in range(n_checks)])
    prob[IC_PROB] = (prob[IC] + prob[PROB]) / 2
    prob[ALL] = (prob[IC] + prob[PROB] + prob[DPP]) / 3
    return prob


# --------------------------------------------------------------------------
# One calibration / test split
# --------------------------------------------------------------------------
def evaluate_split(data: dict, cal: np.ndarray, test: np.ndarray, seed: int) -> dict:
    correct, tokens = data["correct"], data["tokens"]
    n_checks, n_levels = data["n_checks"], correct.shape[1]
    rows = np.arange(len(test))
    raw = raw_confidence(data, cal, seed)
    prob = calibrated_probabilities(raw, correct, cal)

    # ---- How well does each signal tell right answers from wrong ones, at each level?
    quality = {name: [{"auroc": auroc(correct[test, c], -score[test, c]),
                       "prr": prr(correct[test, c], -score[test, c])} for c in range(n_checks)]
               for name, score in raw.items()}

    level_cost = tokens[cal].mean(axis=0)
    budgets = np.unique(np.concatenate([np.linspace(level_cost[0], level_cost[-1], 21), level_cost]))
    final_levels = {}                                   # policy -> list over budgets of level arrays

    # ---- A) Escalation: threshold chosen on calibration questions for each budget.
    for name in ESCALATION_POLICIES:
        chosen = [threshold_for_budget(prob[name][cal], correct[cal], tokens[cal], b, THRESHOLDS)
                  for b in budgets]
        final_levels[esc(name)] = [escalate(prob[name][test], t) for t in chosen]

    # ---- B) One-shot allocation from closed-book confidence only.
    ic_unsure, dpp_unsure = -raw[IC][:, 0], -raw[DPP][:, 0]
    ic_rank = percentile_in(ic_unsure[cal], ic_unsure)
    dpp_rank = percentile_in(dpp_unsure[cal], dpp_unsure)
    in_band = (ic_rank > CASCADE_BAND[0]) & (ic_rank <= CASCADE_BAND[1])
    cascade = np.where(in_band, (ic_rank + dpp_rank) / 2, ic_rank)
    for name, score in [(ONE_SHOT_IC, ic_unsure), (ONE_SHOT_CASCADE, cascade)]:
        edges, gain_table = fit_gain_table(score[cal], correct[cal])
        final_levels[name] = [allocate(score[test], edges, gain_table, level_cost, b) for b in budgets]

    final_levels[ORACLE] = [allocate_oracle(correct[test], level_cost, b) for b in budgets]

    out = {"quality": quality, "budgets": budgets, "level_cost": level_cost,
           "cascade_dpp_fraction": float(in_band[test].mean()),
           "uniform_accuracy": correct[test].mean(axis=0), "uniform_tokens": tokens[test].mean(axis=0),
           "accuracy": {}, "tokens": {}, "level_mix": {}}
    for name, levels in final_levels.items():
        out["accuracy"][name] = np.array([correct[test][rows, lv].mean() for lv in levels])
        out["tokens"][name] = np.array([tokens[test][rows, lv].mean() for lv in levels])
        out["level_mix"][name] = np.array([np.bincount(lv, minlength=n_levels) / len(lv) for lv in levels])

    # ---- Reference: the best policy that ignores uncertainty (random mix of two fixed levels).
    mixes = [best_fixed_mix(correct[cal].mean(axis=0), level_cost, b) for b in budgets]
    mix_matrix = np.zeros((len(budgets), n_levels))
    for j, mix in enumerate(mixes):
        for level, share in mix:
            mix_matrix[j, level] = share
    out["accuracy"][FIXED] = mix_matrix @ out["uniform_accuracy"]
    out["tokens"][FIXED] = mix_matrix @ out["uniform_tokens"]
    out["level_mix"][FIXED] = mix_matrix
    return out


# --------------------------------------------------------------------------
# Time accounting
# --------------------------------------------------------------------------
def seconds_per_question(policy: str, level_mix: np.ndarray, timing: dict, dpp_fraction: float) -> np.ndarray:
    """
    Expected wall-clock seconds per question for a policy, including every
    confidence check it makes on the way.  level_mix: [budgets, levels].
    """
    answer, ic, sampling = timing["answer"], timing["ic"], timing["sampling"]
    n_levels = len(answer)
    cost_of_ending_at = np.zeros(n_levels)
    for final in range(n_levels):
        checks = range(min(final + 1, n_levels - 1))        # check levels visited on the way
        reaches_last = final == n_levels - 1                # last level is answered without a check
        if policy == esc(IC):                               # IC pass at each check, one answer at the end
            total = sum(ic[c] for c in checks) + answer[final]
        elif policy == esc(PROB):                           # an answer at every visited level
            total = sum(answer[lv] for lv in range(final + 1))
        elif policy == esc(DPP):                            # answer pass + sampling at each check
            total = sum(answer[c] + sampling[c] for c in checks) + reaches_last * answer[final]
        elif policy == esc(IC_PROB):
            total = sum(ic[c] + answer[c] for c in checks) + reaches_last * answer[final]
        elif policy == esc(ALL):
            total = sum(ic[c] + answer[c] + sampling[c] for c in checks) + reaches_last * answer[final]
        elif policy == ONE_SHOT_IC:
            total = ic[0] + answer[final]
        elif policy == ONE_SHOT_CASCADE:
            total = ic[0] + dpp_fraction * (answer[0] + sampling[0]) + answer[final]
        else:                                               # fixed mix, oracle: no checks
            total = answer[final]
        cost_of_ending_at[final] = total
    return level_mix @ cost_of_ending_at


# --------------------------------------------------------------------------
# Aggregation over splits
# --------------------------------------------------------------------------
def mean_std(values) -> dict:
    values = np.array(values, dtype=float)
    return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}


def summarise(data: dict, splits: list, kv_bytes_per_token: int) -> dict:
    mb = kv_bytes_per_token / 1e6
    passages = data["passages"]
    budgets = np.mean([s["budgets"] for s in splits], axis=0)
    uniform_acc = np.mean([s["uniform_accuracy"] for s in splits], axis=0)
    uniform_tok = np.mean([s["uniform_tokens"] for s in splits], axis=0)
    best = int(np.argmax(uniform_acc))
    dpp_fraction = float(np.mean([s["cascade_dpp_fraction"] for s in splits]))
    timing = {"answer": np.median(data["seconds"], axis=0),
              "ic": np.median(data["ic_seconds"], axis=0),
              "sampling": np.median(data["sample_seconds"], axis=0)}

    summary = {
        "n_questions": data["n"], "n_splits": len(splits), "kv_bytes_per_token": kv_bytes_per_token,
        "uniform": [{"passages": passages[i], "accuracy": float(uniform_acc[i]),
                     "tokens": float(uniform_tok[i]), "kv_mb": float(uniform_tok[i] * mb),
                     "answer_seconds": float(timing["answer"][i])} for i in range(len(passages))],
        "best_uniform_passages": passages[best],
        "check_seconds": {"passages": passages[:-1], "internal_confidence": timing["ic"].round(3).tolist(),
                          "dpp_sampling": timing["sampling"].round(3).tolist()},
        "ic_prompt_extra_tokens": float((data["ic_tokens"] - data["tokens"][:, :-1]).mean()),
        "cascade_dpp_fraction": dpp_fraction,
        "budgets_tokens": budgets.tolist(),
        "signal_quality": {
            name: [{"passages": passages[c],
                    "auroc": mean_std([s["quality"][name][c]["auroc"] for s in splits]),
                    "prr": mean_std([s["quality"][name][c]["prr"] for s in splits])}
                   for c in range(data["n_checks"])]
            for name in splits[0]["quality"]},
        "policies": {},
    }

    fixed_acc = np.array([s["accuracy"][FIXED] for s in splits])            # [splits, budgets]
    level_budget_index = [int(np.argmin(np.abs(budgets - c))) for c in uniform_tok]
    for name in splits[0]["accuracy"]:
        acc = np.array([s["accuracy"][name] for s in splits])
        tok = np.array([s["tokens"][name] for s in splits])
        mix = np.mean([s["level_mix"][name] for s in splits], axis=0)
        mean_acc, mean_tok = acc.mean(axis=0), tok.mean(axis=0)
        near_best = np.flatnonzero(mean_acc >= uniform_acc[best] - 0.01)
        tokens_needed = float(mean_tok[near_best[0]]) if len(near_best) else float("nan")
        summary["policies"][name] = {
            "tokens": mean_tok.tolist(), "kv_mb": (mean_tok * mb).tolist(),
            "accuracy": mean_acc.tolist(), "accuracy_std": acc.std(axis=0).tolist(),
            "seconds": seconds_per_question(name, mix, timing, dpp_fraction).tolist(),
            "level_mix": mix.tolist(),
            # Average accuracy advantage over the best uncertainty-blind policy at the same budget.
            "gain_over_fixed": mean_std((acc - fixed_acc).mean(axis=1)),
            "at_budget_of_fixed_level": {
                str(passages[i]): {"accuracy": float(mean_acc[j]), "tokens": float(mean_tok[j]),
                                   "gain": mean_std(acc[:, j] - fixed_acc[:, j])}
                for i, j in enumerate(level_budget_index)},
            "tokens_within_1pt_of_best_fixed_level": tokens_needed,
            "memory_saved_at_that_point": 1 - tokens_needed / float(uniform_tok[best]),
        }
    return summary


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#0d366b"]
POLICY_STYLE = [                       # (policy, colour, legend label, line style)
    (ORACLE, MUTED, "Oracle (knows the answers)", ":"),
    (FIXED, INK, "Best fixed mix, no uncertainty", "--"),
    (ONE_SHOT_IC, YELLOW, "One-shot: IC before retrieval", "-"),
    (esc(DPP), AQUA, "Escalation with SemanticDPP", "-"),
    (esc(PROB), ORANGE, "Escalation with answer probability", "-"),
    (esc(IC), BLUE, "Escalation with Internal Confidence", "-"),
]


def _axes(title: str, xlabel: str, ylabel: str, figsize=(7.6, 4.6)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=SECONDARY)
    ax.set_ylabel(ylabel, color=SECONDARY)
    return fig, ax


def _save(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _policy_lines(ax, summary: dict, x_key: str, skip=()):
    for name, color, label, style in POLICY_STYLE:
        if name in skip:
            continue
        policy = summary["policies"][name]
        ax.plot(policy[x_key], 100 * np.array(policy["accuracy"]), color=color, linewidth=2,
                linestyle=style, label=label)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=SECONDARY)


def figure_accuracy_vs_memory(summary: dict, path: str):
    fig, ax = _axes("Accuracy for a given KV-cache memory budget",
                    "Average KV-cache memory per question (MB)", "Accuracy (%)")
    _policy_lines(ax, summary, "kv_mb")
    uniform = summary["uniform"]
    ax.scatter([u["kv_mb"] for u in uniform], [100 * u["accuracy"] for u in uniform],
               s=64, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
    for u in uniform:
        label = f"everyone gets {u['passages']}" if u["passages"] == 0 else f"{u['passages']}"
        ax.annotate(label, (u["kv_mb"], 100 * u["accuracy"]), textcoords="offset points",
                    xytext=(7, -13), color=SECONDARY, fontsize=9)
    _save(fig, path)


def figure_accuracy_vs_time(summary: dict, path: str):
    fig, ax = _axes("Accuracy against time per question (confidence checks included)",
                    "Seconds per question on the M1 laptop", "Accuracy (%)")
    _policy_lines(ax, summary, "seconds", skip=(ORACLE,))
    uniform = summary["uniform"]
    ax.scatter([u["answer_seconds"] for u in uniform], [100 * u["accuracy"] for u in uniform],
               s=64, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
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
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=SECONDARY, ncols=3)
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
    """At the budget of 'everyone gets 2 passages': how each policy spreads the memory."""
    passages = [u["passages"] for u in summary["uniform"]]
    budget_index = int(np.argmin(np.abs(np.array(summary["budgets_tokens"]) - summary["uniform"][2]["tokens"])))
    rows = [("Everyone gets 2", None), ("Escalation with IC", esc(IC)),
            ("Escalation with answer prob.", esc(PROB)), ("Oracle", ORACLE)]
    fig, ax = _axes("Same memory budget, spent differently",
                    "Share of questions (%)", "", figsize=(7.6, 3.4))
    table = {}
    for y, (label, name) in enumerate(rows[::-1]):
        if name is None:
            mix = np.eye(len(passages))[2]
        else:
            mix = np.array(summary["policies"][name]["level_mix"][budget_index])
        table[label] = (100 * mix).round(1).tolist()
        left = 0.0
        for level, share in enumerate(100 * mix):
            ax.barh(y, share, left=left, height=0.6, color=BLUE_RAMP[level], edgecolor=SURFACE, linewidth=2,
                    label=f"{passages[level]} passages" if y == 0 else None)
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

    closed_book_ic = np.array([internal_confidence(p) for p in data["p_yes"][:, 0]])
    summary["ic_calibration_error_closed_book"] = ece(data["correct"][:, 0], closed_book_ic)
    summary["mean_distinct_sampled_answers"] = (data["clusters"].max(axis=2) + 1).mean(axis=0).tolist()
    # How often does a question that is right with fewer passages stay right with more?
    correct = data["correct"]
    summary["closed_book_right_still_right_with_passages"] = correct[correct[:, 0]].mean(axis=0).tolist()
    summary["closed_book_wrong_fixed_by_passages"] = correct[~correct[:, 0]].mean(axis=0).tolist()
    summary["right_at_some_level"] = float(correct.any(axis=1).mean())

    figures = os.path.join(args.out, "figures")
    os.makedirs(figures, exist_ok=True)
    figure_accuracy_vs_memory(summary, f"{figures}/accuracy_vs_memory.png")
    figure_accuracy_vs_time(summary, f"{figures}/accuracy_vs_time.png")
    figure_signal_quality(summary, f"{figures}/signal_quality.png")
    summary["ic_heatmap"] = figure_ic_heatmap(data, f"{figures}/ic_heatmap.png")
    summary["level_mix_at_2_passage_budget"] = figure_where_questions_stop(
        summary, f"{figures}/where_questions_stop.png")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---- Console report
    print("\nSame memory for everyone:")
    for u in summary["uniform"]:
        print(f"  {u['passages']} passages: accuracy {100 * u['accuracy']:.1f}%   {u['tokens']:.0f} tokens "
              f"= {u['kv_mb']:.1f} MB KV cache   {u['answer_seconds']:.2f}s")
    print("\nAUROC of each confidence signal, by passages in the prompt "
          f"{[q['passages'] for q in summary['signal_quality'][IC]]}:")
    for name, per_level in summary["signal_quality"].items():
        print(f"  {name:42s} " + "  ".join(f"{q['auroc']['mean']:.3f}" for q in per_level))
    print("\nPolicies: accuracy at the budget of each fixed level, gain over the best fixed mix, memory saved")
    for name, p in summary["policies"].items():
        cells = "  ".join(f"{100 * v['accuracy']:.1f}" for v in p["at_budget_of_fixed_level"].values())
        print(f"  {name:42s} {cells}   gain {100 * p['gain_over_fixed']['mean']:+.1f} ± "
              f"{100 * p['gain_over_fixed']['std']:.1f} pts   saves {100 * p['memory_saved_at_that_point']:.0f}%")
    print(f"\nwrote {args.out}/summary.json and figures to {figures}/")


if __name__ == "__main__":
    main()
