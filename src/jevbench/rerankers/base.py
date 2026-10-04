"""Common reranker interface: score candidates for a query; higher = more relevant."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RerankResult:
    scores: list[float]  # aligned with the candidate list passed in
    latency_s: float = 0.0
    input_tokens: int = 0
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)


class Reranker:
    name: str = "base"

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        raise NotImplementedError


def order_by_scores(scores: list[float], tiebreak_rank: list[int]) -> list[int]:
    """Indices sorted by score desc; ties broken by the original (RRF) rank, ascending."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], tiebreak_rank[i]))
