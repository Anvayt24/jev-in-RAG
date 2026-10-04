"""Hybrid retrieval (dense + BM25) fused with RRF; candidates are cached once per question."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from . import config
from .ingest import bm25_tokens


def rrf_fuse(rankings: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over ranked lists of item ids. Returns [(id, score)] best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # stable tie-break on id for determinism
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def load_chunks() -> list[dict]:
    with config.CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class HybridRetriever:
    def __init__(self) -> None:
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = load_chunks()
        self.index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))
        self.bm25 = BM25Okapi([bm25_tokens(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    def dense_rank(self, query: str, k: int) -> list[int]:
        q = self.model.encode([config.QUERY_PREFIX + query], normalize_embeddings=True).astype(
            "float32"
        )
        _, idx = self.index.search(q, k)
        return [int(i) for i in idx[0] if i >= 0]

    def sparse_rank(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [int(i) for i in order]

    def retrieve(self, query: str, n: int = config.N_CANDIDATES) -> list[dict]:
        k = config.RETRIEVE_PER_RETRIEVER
        dense = self.dense_rank(query, k)
        sparse = self.sparse_rank(query, k)
        d_rank = {i: r for r, i in enumerate(dense, start=1)}
        s_rank = {i: r for r, i in enumerate(sparse, start=1)}
        fused = rrf_fuse([dense, sparse])[:n]
        out = []
        for rank, (i, score) in enumerate(fused, start=1):
            c = self.chunks[i]
            out.append(
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "page": c["page"],
                    "text": c["text"],
                    "rrf_rank": rank,
                    "rrf_score": score,
                    "dense_rank": d_rank.get(i),
                    "bm25_rank": s_rank.get(i),
                }
            )
        return out


def candidates_hash(cands: list[dict]) -> str:
    h = hashlib.sha256()
    for c in cands:
        h.update(c["chunk_id"].encode())
        h.update(c["text"].encode())
    return h.hexdigest()[:16]


def cache_candidates(questions: list[dict], force: bool = False) -> None:
    """Retrieve once per question and freeze the result to cache/candidates/{qid}.json."""
    config.CAND_DIR.mkdir(parents=True, exist_ok=True)
    todo = [q for q in questions if force or not (config.CAND_DIR / f"{q['qid']}.json").exists()]
    if not todo:
        print("all candidates already cached")
        return
    r = HybridRetriever()
    for q in todo:
        cands = r.retrieve(q["question"])
        payload = {
            "qid": q["qid"],
            "question": q["question"],
            "hash": candidates_hash(cands),
            "candidates": cands,
        }
        (config.CAND_DIR / f"{q['qid']}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    print(f"cached candidates for {len(todo)} questions")


def load_candidates(qid: str) -> dict:
    return json.loads((config.CAND_DIR / f"{qid}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    from .labels import load_questions

    cache_candidates(load_questions())
