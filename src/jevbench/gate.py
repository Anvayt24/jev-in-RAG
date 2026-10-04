"""Answerability gate: for each reranker's top-k passages ask Jev 'can the query be answered from these?'.

  uv run python -m jevbench.gate --rerankers flashrank cohere cohere_pro jev_pair --split test

Writes results/gate/{reranker}.jsonl with p(answerable) per question.
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from . import config
from .labels import load_questions
from .rerankers.base import order_by_scores
from .retrieve import load_candidates
from .run_rerank import load_done


def gate_path(name: str):
    return config.RESULTS_DIR / "gate" / f"{name}.jsonl"


def load_gate(name: str) -> dict[str, dict]:
    p = gate_path(name)
    if not p.exists():
        return {}
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {r["qid"]: r for r in map(json.loads, lines)}


def run(names: list[str], split: str | None, top_k: int) -> None:
    from .rerankers.jev_rr import jev_answerable

    load_dotenv()
    questions = load_questions(split)
    (config.RESULTS_DIR / "gate").mkdir(parents=True, exist_ok=True)
    for name in names:
        scores = load_done(name)
        done = load_gate(name)
        todo = [q for q in questions if q["qid"] in scores and q["qid"] not in done]
        print(f"[gate:{name}] {len(done)} done, {len(todo)} to run", flush=True)
        with gate_path(name).open("a", encoding="utf-8") as f:
            for n, q in enumerate(todo, start=1):
                cand = load_candidates(q["qid"])["candidates"]
                s = scores[q["qid"]]["scores"]
                order = order_by_scores(s, [c["rrf_rank"] for c in cand])[:top_k]
                texts = [cand[i]["text"] for i in order]
                out = jev_answerable(q["question"], texts)
                rec = {
                    "qid": q["qid"],
                    "reranker": name,
                    "top_chunk_ids": [cand[i]["chunk_id"] for i in order],
                    "p_answerable": out["p"],
                    "max_rerank_score": max(s),
                    "latency_s": out["latency_s"],
                    "input_tokens": out["tokens"],
                    "cost_usd": out["cost"],
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(todo):
                    print(f"[gate:{name}] {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerankers", nargs="+", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()
    run(a.rerankers, a.split, a.top_k)
