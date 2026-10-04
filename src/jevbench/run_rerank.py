"""Score the frozen candidate lists with each reranker.

Resumable: (reranker, question) pairs that already have a result are skipped.

  uv run python -m jevbench.run_rerank --rerankers none flashrank --split dev
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .labels import load_questions
from .rerankers.base import Reranker
from .retrieve import load_candidates


def _lazy(module: str, cls: str, **kwargs) -> Callable[[], Reranker]:
    """A constructor that imports the reranker class only when it is needed.

    Rerankers pull in heavy or optional libraries (ONNX, cohere, the Jev SDK), so none of them
    should be imported just to run a different one.
    """

    def make() -> Reranker:
        mod = importlib.import_module(f".rerankers.{module}", package=__package__)
        return getattr(mod, cls)(**kwargs)

    return make


REGISTRY: dict[str, Callable[[], Reranker]] = {
    "none": _lazy("none", "NoRerank"),
    "flashrank": _lazy("flashrank_rr", "FlashRankReranker"),
    "cohere": _lazy("cohere_rr", "CohereReranker"),
    "cohere_pro": _lazy("cohere_rr", "CohereReranker", model="rerank-v4.0-pro"),
    "jev_pair": _lazy("jev_rr", "JevPair"),
    "jev_pack": _lazy("jev_rr", "JevPack"),
    # probes: same models, but fresh requests so repeat runs and candidate order can be compared
    "jev_pair_rerun": _lazy("jev_rr", "JevPair", cache_tag="rerun1"),
    "jev_pack_shuffled": _lazy("jev_rr", "JevPack", shuffle_seed=1, cache_tag="shuf1"),
}


def make_reranker(name: str) -> Reranker:
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown reranker {name!r}; choose from {sorted(REGISTRY)}") from None
    return factory()


def result_path(name: str) -> Path:
    return config.RESULTS_DIR / "scores" / f"{name}.jsonl"


def load_done(name: str) -> dict[str, dict]:
    """Stored result records for a reranker, keyed by question id."""
    path = result_path(name)
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {r["qid"]: r for r in map(json.loads, lines)}


def run(
    names: list[str],
    split: str | None,
    limit: int | None,
    make: Callable[[str], Reranker] = make_reranker,
) -> None:
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
        reranker = make(name)
        with result_path(name).open("a", encoding="utf-8") as f:
            for n, q in enumerate(todo, start=1):
                cand = load_candidates(q["qid"])
                res = reranker.rerank(q["question"], cand["candidates"])
                record = {
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
                f.write(json.dumps(record) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(todo):
                    print(f"[{name}] {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rerankers", nargs="+", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = ap.parse_args()
    run(args.rerankers, args.split, args.limit)
