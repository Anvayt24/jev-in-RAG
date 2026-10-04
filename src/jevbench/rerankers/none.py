"""Baseline that applies no reranking."""

from .base import Reranker, RerankResult


class NoRerank(Reranker):
    """Baseline: keep the hybrid + RRF order."""

    name = "none"

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        return RerankResult(scores=[-float(c["rrf_rank"]) for c in candidates])
