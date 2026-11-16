"""
Step 1 - download a small TriviaQA subset and retrieve passages for it.

    python -m src.experiments.prepare_data --n 600

Writes data/triviaqa_rag.json (question, accepted answers, top-8 BM25 passages).
"""

import argparse
import os

from src.evaluation.data import build_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600, help="number of questions")
    parser.add_argument("--out", default="data/triviaqa_rag.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    subset = build_subset(args.n, args.out)
    print(f"saved {len(subset)} questions to {args.out}")
    os._exit(0)   # the streaming reader keeps background threads alive; exit hard


if __name__ == "__main__":
    main()
