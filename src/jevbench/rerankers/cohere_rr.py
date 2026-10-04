import os
import time

from .base import Reranker, RerankResult


class CohereReranker(Reranker):
    """Strong hosted reranker: Cohere Rerank 4 (fast or pro)."""

    name = "cohere"

    def __init__(
        self,
        model: str = "rerank-v4.0-fast",
        min_interval_s: float = 6.5,  # trial keys are limited to ~10 rerank calls/min
        max_retries: int = 6,
    ) -> None:
        import cohere

        self.model = model
        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._last_call = 0.0

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        docs = [c["text"] for c in candidates]
        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            self._last_call = time.monotonic()
            try:
                resp = self.client.rerank(model=self.model, query=query, documents=docs, top_n=len(docs))
            except Exception as e:  # rate limit / transient
                status = getattr(e, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(min(60, 8 * 2**attempt))
                    continue
                raise
            dt = time.perf_counter() - t0
            scores = [0.0] * len(docs)
            for r in resp.results:
                scores[r.index] = float(r.relevance_score)
            units = None
            try:
                units = resp.meta.billed_units.search_units
            except AttributeError:
                pass
            return RerankResult(
                scores=scores,
                latency_s=dt,
                meta={"model": self.model, "search_units": units},
            )
        raise RuntimeError("unreachable")
