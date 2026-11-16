"""
Step 2 - run the real model once over every question and save everything.

    python -m src.experiments.collect --limit 400

The memory level of a question is the number of retrieved passages in its
prompt: 0, 1, 2, 4 or 8.  For every question and every level we record

  * the greedy answer, whether it is correct, the prompt length (= KV-cache
    entries), the time taken, and the probability the model gave its own answer;

and at every level except the last (after the last there is nothing left to
decide) we also record the two papers' confidence signals:

  * Internal Confidence inputs: P(Yes) per layer / last tokens   (1 forward pass)
  * m sampled answers and their embeddings, for SemanticDPP      (m generations)

Because every (question, level) pair is stored, the analysis step can evaluate
any allocation policy afterwards by table lookup, without touching the model.
The run is resumable: re-running continues where it stopped.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from src.core.llm import DEFAULT_MODEL, LLM, qa_prompt
from src.evaluation.data import is_correct, load_subset

LEVELS = [0, 1, 2, 4, 8]      # number of retrieved passages = memory level
M_SAMPLES = 5                 # sampled answers per confidence check (Paper B uses 5-10)


def run_question(llm: LLM, q: dict):
    """Everything we need from the model for one question."""
    levels, p_yes_per_level, embeddings_per_level = [], [], []
    for k in LEVELS:
        passages = q["passages"][:k]
        prompt = qa_prompt(q["question"], passages)
        last_level = k == LEVELS[-1]
        # Paper B needs sampled answers; they are decoded from the same cached prompt.
        out = llm.answer(prompt, n_samples=0 if last_level else M_SAMPLES)
        record = {"passages": k, "answer": out["text"],
                  "correct": bool(is_correct(out["text"], q["aliases"])),
                  "prompt_tokens": out["prompt_tokens"], "seconds": round(out["seconds"], 3),
                  "answer_log_prob": round(out["log_prob"], 4)}

        if not last_level:
            # Paper A: can the model answer with the memory it has now?  (no generation)
            p_yes, ic_tokens, ic_seconds = llm.yes_no_confidence(q["question"], passages)
            record.update({"ic_tokens": ic_tokens, "ic_seconds": round(ic_seconds, 3),
                           "samples": out["samples"], "sample_seconds": round(out["sample_seconds"], 3)})
            p_yes_per_level.append(p_yes)
            embeddings_per_level.append(out["embeddings"])
        levels.append(record)
        llm.free_cache()
    return levels, np.stack(p_yes_per_level), np.stack(embeddings_per_level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/triviaqa_rag.json")
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    jsonl_path, npz_path = args.out + ".jsonl", args.out + ".npz"

    questions = load_subset(args.data)[: args.limit]
    records, p_yes_all, emb_all = [], [], []
    if os.path.exists(jsonl_path) and os.path.exists(npz_path):       # resume
        saved = np.load(npz_path)
        n_done = len(saved["p_yes"])
        records = [json.loads(line) for line in open(jsonl_path)][:n_done]
        p_yes_all, emb_all = list(saved["p_yes"]), list(saved["embeddings"])
        print(f"resuming after {n_done} questions", flush=True)

    def checkpoint():
        # Write to temporary files first so a crash can never leave half a file.
        with open(jsonl_path + ".tmp", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        np.savez_compressed(npz_path + ".tmp.npz", p_yes=np.stack(p_yes_all).astype(np.float32),
                            embeddings=np.stack(emb_all).astype(np.float16))
        os.replace(npz_path + ".tmp.npz", npz_path)
        os.replace(jsonl_path + ".tmp", jsonl_path)

    torch.manual_seed(len(records))
    llm = LLM(args.model)
    print(f"model {args.model} on {llm.device}; KV cache = {llm.kv_bytes_per_token} bytes/token", flush=True)

    started, n_start = time.time(), len(records)
    for i in range(n_start, len(questions)):
        q = questions[i]
        levels, p_yes, embeddings = run_question(llm, q)
        records.append({"id": q["id"], "question": q["question"], "gold": q["answer"], "levels": levels})
        p_yes_all.append(p_yes)
        emb_all.append(embeddings)

        if (i + 1) % 10 == 0 or i + 1 == len(questions):
            checkpoint()
            accuracy = np.mean([[lv["correct"] for lv in r["levels"]] for r in records], axis=0)
            per_question = (time.time() - started) / (i + 1 - n_start)
            remaining = per_question * (len(questions) - i - 1) / 60
            print(f"{i + 1}/{len(questions)}  accuracy per level {np.round(accuracy, 3).tolist()}  "
                  f"{per_question:.1f}s per question, about {remaining:.0f} min left", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
