"""
Live demo - the whole method on fresh questions the experiments never used.

    python -m src.experiments.demo --n 10
    python -m src.experiments.demo --n 10 --max_passages 8 --threshold 0.8
    python -m src.experiments.demo --question "Who wrote the novel Dracula?"

What it does for each question:
  1. answer with no retrieved passages and read how sure the model is
     (the probability it gave its own answer - this costs nothing extra),
  2. if it is confident enough, keep that answer;
     otherwise add passages (1, then 2, 4, 8) and try again,
  3. stop at --max_passages at the latest.
The raw confidence is turned into "probability of being right" with calibration
curves fitted on the saved runs (results/runs.*).
"""

import argparse

import numpy as np

from src.core.allocator import fit_calibrators
from src.core.internal_confidence import internal_confidence
from src.core.llm import LLM, qa_prompt
from src.evaluation.data import is_correct, load_subset
from src.experiments.analyze import load_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of fresh questions")
    parser.add_argument("--max_passages", type=int, default=2, choices=[1, 2, 4, 8])
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="stop adding memory once P(answer is correct) reaches this")
    parser.add_argument("--question", default=None, help="just show the confidence for one question")
    parser.add_argument("--data", default="data/triviaqa_rag.json")
    parser.add_argument("--runs", default="results/runs")
    args = parser.parse_args()

    # Calibration curves, one per memory level, from the saved runs.
    runs = load_runs(args.runs)
    levels, n_checks = runs["passages"], runs["n_checks"]
    calibrators = fit_calibrators(runs["answer_log_prob"][:, :n_checks], runs["correct"])

    llm = LLM()
    mb = llm.kv_bytes_per_token / 1e6

    if args.question:
        p_yes, _, seconds = llm.yes_no_confidence(args.question)
        out = llm.answer(qa_prompt(args.question))
        p = float(calibrators[0].predict(np.array([out["log_prob"]]))[0])
        print(f"\nQuestion: {args.question}")
        print(f"Closed-book answer: {out['text']}")
        print(f"Internal Confidence (before answering, {seconds:.2f}s): {internal_confidence(p_yes):.3f}")
        print(f"Probability the model gave its own answer: {np.exp(out['log_prob']):.3f}")
        print(f"Calibrated chance that this answer is correct: {100 * p:.0f}%  -> "
              + ("no passages needed" if p >= args.threshold else "would retrieve passages"))
        return

    print(f"\nStop when P(correct) >= {args.threshold}, at most {args.max_passages} passages\n")
    print(f"{'question':50s} {'passages':>8s} {'KV MB':>6s}  {'P(correct) at each try':24s} answer")
    questions = load_subset(args.data)[runs["n"]: runs["n"] + args.n]     # never seen by collect.py
    used, n_right = 0, 0
    for q in questions:
        trace = []
        for c, k in enumerate(levels):
            out = llm.answer(qa_prompt(q["question"], q["passages"][:k]))
            if k >= args.max_passages:
                break                                      # memory limit reached: keep this answer
            p = float(calibrators[c].predict(np.array([out["log_prob"]]))[0])
            trace.append(f"{p:.2f}")
            if p >= args.threshold:
                break                                      # confident: keep this answer
        right = is_correct(out["text"], q["aliases"])
        used += out["prompt_tokens"]
        n_right += right
        verdict = "[correct]" if right else "[wrong, gold: " + q["answer"] + "]"
        print(f"{q['question'][:48]:50s} {k:8d} {out['prompt_tokens'] * mb:6.1f}  "
              f"{' '.join(trace):24s} {out['text'][:24]}  {verdict}")
        llm.free_cache()

    n = len(questions)
    everyone = runs["tokens"][:, levels.index(args.max_passages)].mean() * n
    print(f"\n{n_right}/{n} correct using {used * mb:.0f} MB of KV cache in total "
          f"({used / n:.0f} tokens per question).")
    print(f"Giving every question {args.max_passages} passages would need about {everyone * mb:.0f} MB; "
          f"this run used {100 * used / everyone:.0f}% of that.")


if __name__ == "__main__":
    main()
