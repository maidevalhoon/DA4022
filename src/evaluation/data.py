"""
Dataset helpers: TriviaQA questions + retrieved Wikipedia passages.

Why TriviaQA?  Both reference papers evaluate on it, and Paper A uses it for
its adaptive-RAG experiment.  We use the "rc.wikipedia" config, which ships the
Wikipedia pages that are evidence for each question.  We cut those pages into
~100-word passages and rank them against the question with BM25, exactly like
a small retrieval-augmented-generation (RAG) pipeline would.

The number of passages we put in the prompt is our memory knob:
more passages -> longer prompt -> bigger KV cache.
"""

import json
import math
import re
import string
from collections import Counter
from typing import Dict, List

PASSAGE_WORDS = 100      # size of one retrieved passage
MAX_PASSAGES = 8         # we keep the top-8 passages per question


# --------------------------------------------------------------------------
# Text normalisation and answer checking
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lower-case, drop punctuation and articles, squeeze spaces (SQuAD style)."""
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, aliases: List[str]) -> bool:
    """An answer counts as correct if it contains any accepted alias."""
    pred = f" {normalize(prediction)} "
    return any(f" {normalize(a)} " in pred for a in aliases if normalize(a))


# --------------------------------------------------------------------------
# Tiny BM25 retriever (no external dependency)
# --------------------------------------------------------------------------
def split_into_passages(pages: List[str], words_per_passage: int = PASSAGE_WORDS) -> List[str]:
    passages = []
    for page in pages:
        words = page.split()
        for start in range(0, len(words), words_per_passage):
            chunk = words[start:start + words_per_passage]
            if len(chunk) >= 20:                      # skip tiny tail fragments
                passages.append(" ".join(chunk))
    return passages


def bm25_rank(question: str, passages: List[str], k1: float = 1.5, b: float = 0.75) -> List[int]:
    """Return passage indices sorted from most to least relevant to the question."""
    docs = [normalize(p).split() for p in passages]
    query = normalize(question).split()
    n_docs = len(docs)
    avg_len = sum(len(d) for d in docs) / max(1, n_docs)
    doc_freq = Counter(term for d in docs for term in set(d))

    scores = []
    for d in docs:
        term_freq = Counter(d)
        score = 0.0
        for term in query:
            if term not in term_freq:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = term_freq[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(score)
    return sorted(range(n_docs), key=lambda i: -scores[i])


# --------------------------------------------------------------------------
# Building / loading the evaluation subset
# --------------------------------------------------------------------------
def build_subset(n_questions: int, out_path: str) -> List[Dict]:
    """Stream TriviaQA from the HuggingFace hub and save a small RAG-ready subset."""
    from datasets import load_dataset

    stream = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia",
                          split="validation", streaming=True)
    subset = []
    for example in stream:
        passages = split_into_passages(example["entity_pages"]["wiki_context"])
        if len(passages) < MAX_PASSAGES:
            continue                                   # not enough evidence text
        order = bm25_rank(example["question"], passages)[:MAX_PASSAGES]
        subset.append({
            "id": example["question_id"],
            "question": example["question"],
            "answer": example["answer"]["value"],
            "aliases": sorted(set(example["answer"]["aliases"]
                                  + example["answer"]["normalized_aliases"])),
            "passages": [passages[i] for i in order],  # best passage first
        })
        if len(subset) % 50 == 0:
            print(f"  collected {len(subset)} questions", flush=True)
        if len(subset) >= n_questions:
            break

    with open(out_path, "w") as f:
        json.dump(subset, f)
    return subset


def load_subset(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)
