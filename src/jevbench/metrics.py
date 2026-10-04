"""Ranking metrics over a ranked list of chunk ids, plus paired bootstrap helpers.

A question has gold *quotes*; each quote maps to the set of chunks containing it (`per_quote`).
  * mode "any": the quotes are alternative evidence; finding one is enough.
  * mode "all": multi-passage question; every quote's evidence is needed.
Gain is quote-level (a chunk earns gain only for quotes not yet covered higher in the ranking), so
overlapping chunks that repeat the same text do not inflate nDCG.
"""

from __future__ import annotations

import math
import random
from statistics import mean


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return float(any(c in gold for c in ranked[:k]))


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, c in enumerate(ranked, start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def _needed(per_quote: list[list[str]], mode: str) -> int:
    return 1 if mode == "any" else len(per_quote)


def ndcg_at_k(
    ranked: list[str], per_quote: list[list[str]], mode: str = "any", k: int = 10
) -> float:
    need = _needed(per_quote, mode)
    if need == 0:
        return 0.0
    covered: set[int] = set()
    dcg = 0.0
    for i, c in enumerate(ranked[:k], start=1):
        new = {qi for qi, ids in enumerate(per_quote) if c in ids and qi not in covered}
        if new and len(covered) < need:
            covered.add(min(new))  # one unit of gain per chunk position
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(need, k) + 1))
    return dcg / ideal


def coverage_at_k(ranked: list[str], per_quote: list[list[str]], mode: str, k: int) -> float:
    """'any': 1 if some quote is covered in the top-k; 'all': fraction of quotes covered."""
    if not per_quote:
        return 0.0
    top = set(ranked[:k])
    flags = [float(any(c in top for c in ids)) for ids in per_quote]
    return max(flags) if mode == "any" else mean(flags)


def question_metrics(
    ranked: list[str], per_quote: list[list[str]], mode: str = "any"
) -> dict[str, float]:
    gold = {c for ids in per_quote for c in ids}
    return {
        "hit@1": hit_at_k(ranked, gold, 1),
        "hit@3": hit_at_k(ranked, gold, 3),
        "hit@5": hit_at_k(ranked, gold, 5),
        "recall@5": coverage_at_k(ranked, per_quote, mode, 5),
        "mrr": mrr(ranked, gold),
        "ndcg@10": ndcg_at_k(ranked, per_quote, mode, 10),
    }


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over questions."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return mean(values), means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def paired_bootstrap_diff(
    a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% bootstrap CI; pairs are resampled together."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci([x - y for x, y in zip(a, b, strict=True)], n_boot=n_boot, seed=seed)


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score_pos > score_neg) with ties counted as half (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))
