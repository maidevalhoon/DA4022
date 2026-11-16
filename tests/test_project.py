"""
Unit tests for the maths.  They use small synthetic arrays, never a fake model:
anything that needs the real LLM is checked by running the experiments.

    python -m pytest tests -q
"""

import numpy as np

from src.core.allocator import (Calibrator, allocate, allocate_oracle, best_fixed_mix, escalate,
                                escalation_curve, fit_gain_table, threshold_for_budget)
from src.core.internal_confidence import (attenuated_weights, internal_confidence,
                                          p_yes_naive_average, p_yes_top_right)
from src.core.semantic_dpp import SemanticDPP, cluster_answers, lexical_diversity
from src.evaluation.data import bm25_rank, is_correct, normalize, split_into_passages
from src.evaluation.metrics import auroc, ece, prr


# ----------------------------------------------------------- Internal Confidence
def test_attenuated_weights_focus_on_the_last_position():
    w = attenuated_weights(6, alpha=1.0)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.diff(w) > 0)                 # grows towards the end
    assert np.isclose(w[-2] / w[-1], np.exp(-1))  # Eq. 4 with distance 1


def test_internal_confidence_is_a_weighted_average():
    assert np.isclose(internal_confidence(np.full((28, 6), 0.7)), 0.7)
    p = np.zeros((28, 6))
    p[-1, -1] = 1.0                               # only the top-right cell says "Yes"
    assert np.isclose(internal_confidence(p), 0.72 * 0.72, atol=0.01)   # last layer x last token
    assert p_yes_top_right(p) == 1.0
    assert np.isclose(p_yes_naive_average(p), 1 / (28 * 6))


# ------------------------------------------------------------------ SemanticDPP
def test_cluster_answers_groups_same_meaning_strings():
    assert cluster_answers(["Paris", "paris.", "Paris, France", "Lyon", "The Lyon"]) == [0, 0, 0, 1, 1]
    assert lexical_diversity(["Paris"] * 5) == 0.0
    assert lexical_diversity(["a1", "b2", "c3", "d4", "e5"]) > lexical_diversity(["a1", "a1", "a1", "b2", "b2"])


def _synthetic_answer_sets(n=120, m=5, d=40, seed=0):
    """Each 'answer' embedding = a meaning vector + small wording noise."""
    rng = np.random.default_rng(seed)
    embeddings, clusters = [], []
    for _ in range(n):
        n_meanings = rng.integers(1, 4)
        meanings = rng.normal(size=(n_meanings, d))
        ids = np.sort(rng.integers(0, n_meanings, size=m))
        ids = np.unique(ids, return_inverse=True)[1]
        embeddings.append(meanings[ids] + 0.05 * rng.normal(size=(m, d)))
        clusters.append(ids)
    return np.array(embeddings), np.array(clusters)


def test_dpp_scores_diverse_answer_sets_as_more_uncertain():
    embeddings, clusters = _synthetic_answer_sets()
    dpp = SemanticDPP(pca_dim=16, steps=150)
    losses = dpp.fit(embeddings[:80], clusters[:80])
    assert losses[-1] < losses[0]                 # training reduces the loss

    n_clusters = clusters[80:].max(axis=1) + 1
    scores = dpp.score(embeddings[80:])
    assert scores[n_clusters == 3].mean() > scores[n_clusters == 1].mean()
    expected = dpp.expected_clusters(embeddings[80:])
    assert expected[n_clusters == 3].mean() > expected[n_clusters == 1].mean()


def test_dpp_single_set_and_untrained_variant():
    embeddings, clusters = _synthetic_answer_sets(n=40)
    dpp = SemanticDPP(pca_dim=16, trainable=False)
    dpp.fit(embeddings, clusters)
    assert dpp.score(embeddings[0]).shape == ()   # one set -> one number
    same = np.repeat(embeddings[0][:1], 5, axis=0)
    assert dpp.score(same) < dpp.score(embeddings[clusters.max(axis=1) == 2][0])


# -------------------------------------------------------------------- Allocator
def test_calibrator_turns_scores_into_probabilities():
    rng = np.random.default_rng(0)
    score = rng.uniform(size=2000)
    correct = rng.uniform(size=2000) < score          # P(correct) equals the score
    p = Calibrator().fit(score, correct).predict(np.array([0.1, 0.5, 0.9]))
    assert np.all(np.diff(p) >= 0)                    # never decreasing
    assert np.allclose(p, [0.1, 0.5, 0.9], atol=0.1)


def test_escalation_stops_at_the_first_confident_level():
    p_correct = np.array([[0.9, 0.9, 0.9],            # confident straight away -> level 0
                          [0.2, 0.8, 0.1],            # confident after one upgrade -> level 1
                          [0.1, 0.2, 0.3]])           # never confident -> last level (3)
    assert escalate(p_correct, 0.7).tolist() == [0, 1, 3]
    assert escalate(p_correct, 0.0).tolist() == [0, 0, 0]      # threshold 0: nobody gets memory
    assert escalate(p_correct, 1.01).tolist() == [3, 3, 3]     # impossible threshold: everybody gets all


def _toy_escalation_problem(n=600, seed=0):
    """Each question has a hidden 'needs k passages'; confidence is a noisy view of 'is it right now'."""
    rng = np.random.default_rng(seed)
    needed = rng.integers(0, 4, size=n)                        # level at which it becomes right
    correct = np.arange(4)[None, :] >= needed[:, None]
    p_correct = np.clip(correct[:, :3] * 0.8 + 0.1 + 0.1 * rng.normal(size=(n, 3)), 0, 1)
    tokens = np.tile(np.array([50.0, 200.0, 350.0, 650.0]), (n, 1))
    return p_correct, correct, tokens


def test_escalation_beats_fixed_levels_on_the_same_budget():
    p_correct, correct, tokens = _toy_escalation_problem()
    thresholds = np.linspace(0, 1.01, 102)
    budget = 350.0                                              # what "everyone gets level 2" costs
    t = threshold_for_budget(p_correct[:300], correct[:300], tokens[:300], budget, thresholds)
    memory, accuracy = escalation_curve(p_correct[300:], correct[300:], tokens[300:], np.array([t]))
    assert memory[0] <= budget * 1.1
    assert accuracy[0] > correct[300:, 2].mean() + 0.05         # clearly better than fixed level 2

    mix = best_fixed_mix(correct.mean(axis=0), tokens[0], budget)
    assert np.isclose(sum(share for _, share in mix), 1.0)
    assert sum(share * tokens[0][level] for level, share in mix) <= budget + 1e-9


def _toy_allocation_problem(n=400, seed=0):
    """Confident questions are right without passages; unsure ones need 2 passages."""
    rng = np.random.default_rng(seed)
    uncertainty = rng.uniform(size=n)
    needs_help = rng.uniform(size=n) < uncertainty
    correct = np.stack([~needs_help, ~needs_help, np.ones(n, bool), np.ones(n, bool)], axis=1)
    cost = np.array([50.0, 200.0, 350.0, 650.0])
    return uncertainty, correct, cost


def test_one_shot_allocation_respects_budget_and_uses_uncertainty():
    u, correct, cost = _toy_allocation_problem()
    edges, table = fit_gain_table(u[:200], correct[:200])
    assert table.shape == (5, 4)
    assert table[-1, 0] < table[0, 0]             # unsure bin is worse without passages

    budget = 200.0
    level = allocate(u[200:], edges, table, cost, budget)
    assert cost[level].mean() <= budget + 1e-9
    assert u[200:][level > 0].mean() > u[200:][level == 0].mean()   # memory goes to the unsure

    idx = np.arange(200)
    ours = correct[200:][idx, level].mean()
    shuffled = np.random.default_rng(1).permutation(u[200:])
    random_level = allocate(shuffled, edges, table, cost, budget)
    assert ours > correct[200:][idx, random_level].mean()

    oracle_level = allocate_oracle(correct[200:], cost, budget)
    assert cost[oracle_level].mean() <= budget + 1e-9
    assert correct[200:][idx, oracle_level].mean() >= ours

    assert allocate(u, edges, table, cost, cost[0]).max() == 0   # no spare memory -> nobody upgraded
    assert allocate(u, edges, table, cost, cost[-1]).max() <= 2  # level 3 never helps, never bought


def test_gain_table_handles_tied_scores():
    u = np.array([0.0] * 60 + [1.0] * 40)          # many identical scores (as with sampled answers)
    correct = np.stack([u < 0.5, np.ones(100, bool)], axis=1)
    edges, table = fit_gain_table(u, correct)
    sure_bin, unsure_bin = np.digitize([0.0, 1.0], edges, right=True)
    assert sure_bin != unsure_bin
    assert table[sure_bin, 0] > 0.8 and table[unsure_bin, 0] < 0.2


# ---------------------------------------------------------------------- Metrics
def test_metrics_on_known_cases():
    correct = np.array([1, 1, 1, 0, 0, 0])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(correct, perfect) == 1.0
    assert auroc(correct, -perfect) == 0.0
    assert np.isclose(prr(correct, perfect), 1.0)
    assert prr(correct, -perfect) < 0
    assert np.isnan(auroc(np.ones(4), np.arange(4)))
    assert ece(np.array([1, 1, 0, 0]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.0
    assert np.isclose(ece(np.array([0, 0]), np.array([0.9, 0.9])), 0.9)


# ------------------------------------------------------------------------- Data
def test_answer_matching_and_retrieval():
    assert normalize("The  Eiffel-Tower!") == "eiffel tower"
    assert is_correct("It is Canberra.", ["Canberra"])
    assert not is_correct("Sydney", ["Canberra"])
    assert not is_correct("Nigeria", ["Niger"])            # whole-word match only

    page = ("filler words here " * 40) + "the capital of australia is canberra " + ("more filler " * 60)
    passages = split_into_passages([page], words_per_passage=30)
    best = bm25_rank("What is the capital of Australia?", passages)[0]
    assert "canberra" in passages[best]
