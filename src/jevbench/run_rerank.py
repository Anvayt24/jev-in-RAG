"""Score the frozen candidate lists with each reranker. Resumable: finished (reranker, qid) pairs are skipped.

  uv run python -m jevbench.run_rerank --rerankers none flashrank --split dev
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from . import config
from .labels import load_questions
from .retrieve import load_candidates


def make_reranker(name: str):
    if name == "none":
        from .rerankers.none import NoRerank
        return NoRerank()
    if name == "flashrank":
        from .rerankers.flashrank_rr import FlashRankReranker
        return FlashRankReranker()
    if name == "cohere":
        from .rerankers.cohere_rr import CohereReranker
        return CohereReranker()
    if name == "cohere_pro":
        from .rerankers.cohere_rr import CohereReranker
        return CohereReranker(model="rerank-v4.0-pro")
    if name == "jev_pair":
        from .rerankers.jev_rr import JevPair
        return JevPair()
    if name == "jev_pack":
        from .rerankers.jev_rr import JevPack
        return JevPack()
    if name == "jev_pack_shuffled":  # order-sensitivity probe
        from .rerankers.jev_rr import JevPack
        return JevPack(shuffle_seed=1, cache_tag="shuf1")
    if name == "jev_pair_rerun":  # run-to-run stability probe
        from .rerankers.jev_rr import JevPair
        return JevPair(cache_tag="rerun1")
    raise ValueError(f"unknown reranker {name}")


def result_path(name: str):
    return config.RESULTS_DIR / "scores" / f"{name}.jsonl"


def load_done(name: str) -> dict[str, dict]:
    p = result_path(name)
    if not p.exists():
        return {}
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {r["qid"]: r for r in map(json.loads, lines)}


def run(names: list[str], split: str | None, limit: int | None) -> None:
    load_dotenv()
    questions = load_questions(split)
    if limit:
        questions = questions[:limit]
    (config.RESULTS_DIR / "scores").mkdir(parents=True, exist_ok=True)
    for name in names:
        done = load_done(name)
        todo = [q for q in questions if q["qid"] not in done]
        print(f"[{name}] {len(done)} done, {len(todo)} to run", flush=True)
        if not todo:
            continue
        rr = make_reranker(name)
        with result_path(name).open("a", encoding="utf-8") as f:
            for n, q in enumerate(todo, start=1):
                cand = load_candidates(q["qid"])
                res = rr.rerank(q["question"], cand["candidates"])
                rec = {
                    "qid": q["qid"],
                    "reranker": name,
                    "cand_hash": cand["hash"],
                    "chunk_ids": [c["chunk_id"] for c in cand["candidates"]],
                    "scores": res.scores,
                    "latency_s": res.latency_s,
                    "input_tokens": res.input_tokens,
                    "cost_usd": res.cost_usd,
                    "meta": res.meta,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(todo):
                    print(f"[{name}] {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerankers", nargs="+", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.rerankers, a.split, a.limit)
