import time

from .base import Reranker, RerankResult


class FlashRankReranker(Reranker):
    """Simple, very common local cross-encoder (ms-marco MiniLM via ONNX, CPU)."""

    name = "flashrank"

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        from flashrank import Ranker

        from .. import config

        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name, cache_dir=str(config.CACHE_DIR / "flashrank"))

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
        t0 = time.perf_counter()
        out = self.ranker.rerank(RerankRequest(query=query, passages=passages))
        dt = time.perf_counter() - t0
        scores = [0.0] * len(candidates)
        for r in out:
            scores[int(r["id"])] = float(r["score"])
        return RerankResult(scores=scores, latency_s=dt, meta={"model": self.model_name})
