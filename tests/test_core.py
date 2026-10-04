import math

from jevbench import metrics
from jevbench.labels import label_question, normalize
from jevbench.rerankers.base import order_by_scores
from jevbench.retrieve import candidates_hash, rrf_fuse


def test_rrf_fuse_prefers_items_ranked_high_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == 1  # rank1 + rank2 beats rank3 + rank1 (1/61+1/62 > 1/63+1/61)
    assert set(ids) == {1, 2, 3, 4}
    assert abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-12


def test_rrf_is_deterministic_on_ties():
    a = rrf_fuse([[5, 6], [6, 5]])
    b = rrf_fuse([[5, 6], [6, 5]])
    assert a == b and [i for i, _ in a] == [5, 6]


def test_metrics_basic():
    ranked = ["a", "b", "c", "d"]
    per_quote = [["c"]]
    gold = {"c"}
    assert metrics.hit_at_k(ranked, gold, 1) == 0.0
    assert metrics.hit_at_k(ranked, gold, 3) == 1.0
    assert metrics.mrr(ranked, gold) == 1 / 3
    assert math.isclose(metrics.ndcg_at_k(ranked, per_quote, "any", 10), 1 / math.log2(4))
    assert metrics.ndcg_at_k(["c", "a"], per_quote, "any", 10) == 1.0


def test_ndcg_all_mode_needs_every_quote():
    per_quote = [["g1"], ["g2"]]
    ranked = ["g1", "x", "g2"]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(metrics.ndcg_at_k(ranked, per_quote, "all", 10), dcg / ideal)


def test_ndcg_any_mode_counts_only_one_hit():
    per_quote = [["g1"], ["g2"]]  # alternative evidence
    assert metrics.ndcg_at_k(["g1", "g2"], per_quote, "any", 10) == 1.0
    assert metrics.ndcg_at_k(["x", "g2"], per_quote, "any", 10) == 1 / math.log2(3)


def test_ndcg_overlapping_chunks_do_not_double_count():
    # the same quote sits in two overlapping chunks; one quote = one unit of gain
    per_quote = [["a", "b"]]
    assert metrics.ndcg_at_k(["a", "b"], per_quote, "all", 10) == 1.0
    assert metrics.ndcg_at_k(["x", "a", "b"], per_quote, "all", 10) == 1 / math.log2(3)


def test_coverage_modes():
    per_quote = [["a", "b"], ["z"]]
    assert metrics.coverage_at_k(["b", "x"], per_quote, "all", 2) == 0.5
    assert metrics.coverage_at_k(["b", "z"], per_quote, "all", 2) == 1.0
    assert metrics.coverage_at_k(["b", "x"], per_quote, "any", 2) == 1.0
    assert metrics.coverage_at_k(["x", "y"], per_quote, "any", 2) == 0.0


def test_auroc():
    assert metrics.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert metrics.auroc([0.5], [0.5]) == 0.5
    assert metrics.auroc([0.1], [0.9]) == 0.0


def test_bootstrap_ci_contains_mean():
    m, lo, hi = metrics.bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], n_boot=500)
    assert lo <= m <= hi


def test_order_by_scores_breaks_ties_by_original_rank():
    scores = [0.5, 0.9, 0.5, 0.9]
    rr = [1, 2, 3, 4]
    assert order_by_scores(scores, rr) == [1, 3, 0, 2]


def test_normalize_and_label_question():
    chunks = [
        {"chunk_id": "d:p1:c0", "doc_id": "d", "text": "The **total** was $1,234 million in 2023."},
        {"chunk_id": "d:p2:c0", "doc_id": "d", "text": "Unrelated text."},
        {"chunk_id": "e:p1:c0", "doc_id": "e", "text": "The total was $1,234 million in 2023."},
    ]
    assert normalize("A | b -- C") == "a b c"
    q = {"gold": [{"doc_id": "d", "quote": "total was $1,234 million"}]}
    lab = label_question(q, chunks)
    assert lab["gold_chunk_ids"] == ["d:p1:c0"]  # doc_id filter excludes e:p1:c0
    assert lab["missing"] == []
    q2 = {"gold": [{"doc_id": "d", "quote": "not present anywhere"}]}
    assert label_question(q2, chunks)["missing"] == ["not present anywhere"]
    assert label_question({"gold": []}, chunks)["gold_chunk_ids"] == []


def test_candidates_hash_changes_with_order_or_text():
    a = [{"chunk_id": "1", "text": "x"}, {"chunk_id": "2", "text": "y"}]
    b = list(reversed(a))
    assert candidates_hash(a) != candidates_hash(b)
    assert candidates_hash(a) == candidates_hash([dict(c) for c in a])
