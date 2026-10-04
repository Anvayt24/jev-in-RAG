"""Shared fixtures: a tiny synthetic corpus in a temporary workspace, plus a stub reranker."""

import json
from types import SimpleNamespace

import pytest

from jevbench import config
from jevbench.rerankers.base import Reranker, RerankResult
from jevbench.retrieve import candidates_hash

CHUNKS = [
    {
        "chunk_id": "d:p1:c0",
        "doc_id": "d",
        "page": 1,
        "text": "Rayleigh scattering makes the sky look blue.",
    },
    {"chunk_id": "d:p2:c0", "doc_id": "d", "page": 2, "text": "Paris is the capital of France."},
    {
        "chunk_id": "d:p3:c0",
        "doc_id": "d",
        "page": 3,
        "text": "The Pico board has 2 MB of flash memory.",
    },
    {
        "chunk_id": "d:p4:c0",
        "doc_id": "d",
        "page": 4,
        "text": "Boil pasta in salted water for ten minutes.",
    },
]

QUESTIONS = [
    {
        "qid": "q1",
        "question": "What is the capital of France?",
        "type": "fact",
        "answerable": True,
        "mode": "any",
        "gold": [{"doc_id": "d", "quote": "Paris is the capital of France"}],
        "split": "test",
    },
    {
        "qid": "q2",
        "question": "How much flash memory does the Pico have?",
        "type": "table",
        "answerable": True,
        "mode": "any",
        "gold": [{"doc_id": "d", "quote": "2 MB of flash memory"}],
        "split": "test",
    },
    {
        "qid": "q3",
        "question": "What is the airspeed of an unladen swallow?",
        "type": "unanswerable",
        "answerable": False,
        "mode": "any",
        "gold": [],
        "split": "test",
    },
    {
        "qid": "q4",
        "question": "Why is the sky blue?",
        "type": "fact",
        "answerable": True,
        "mode": "any",
        "gold": [{"doc_id": "d", "quote": "Rayleigh scattering makes the sky look blue"}],
        "split": "dev",
    },
]

# every question gets the same candidate order, so the gold chunk sits at rank 3 (q1), 4 (q2)
# and 2 (q4) before any reranking
CANDIDATE_ORDER = ["d:p4:c0", "d:p1:c0", "d:p2:c0", "d:p3:c0"]


class KeywordReranker(Reranker):
    """Stand-in for a real reranker: scores a passage by how many query words it contains."""

    name = "keyword"

    def rerank(self, query, candidates):
        words = set(query.lower().replace("?", "").split())
        scores = [
            float(len(words & set(c["text"].lower().rstrip(".").split()))) for c in candidates
        ]
        return RerankResult(scores=scores, latency_s=0.01, input_tokens=5, cost_usd=0.001)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every project path at a temporary directory (and never read a real .env)."""
    layout = {
        "ROOT": tmp_path,
        "RAW_DIR": tmp_path / "data" / "raw",
        "EVAL_DIR": tmp_path / "data" / "eval",
        "CHUNKS_PATH": tmp_path / "data" / "chunks.jsonl",
        "CACHE_DIR": tmp_path / "cache",
        "PARSED_DIR": tmp_path / "cache" / "parsed",
        "INDEX_DIR": tmp_path / "cache" / "index",
        "CAND_DIR": tmp_path / "cache" / "candidates",
        "JEV_RAW_DIR": tmp_path / "cache" / "jev_raw",
        "RESULTS_DIR": tmp_path / "results",
    }
    for attr, value in layout.items():
        monkeypatch.setattr(config, attr, value)
    monkeypatch.setattr("jevbench.run_rerank.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("jevbench.gate.load_dotenv", lambda *a, **k: None)
    return tmp_path


@pytest.fixture
def corpus(workspace):
    """Write chunks, questions and frozen candidates for the synthetic corpus."""
    by_id = {c["chunk_id"]: c for c in CHUNKS}
    write_jsonl(config.CHUNKS_PATH, CHUNKS)
    write_jsonl(config.EVAL_DIR / "questions.jsonl", QUESTIONS)
    config.CAND_DIR.mkdir(parents=True, exist_ok=True)
    for q in QUESTIONS:
        cands = [
            {**by_id[cid], "rrf_rank": rank, "rrf_score": 1 / (60 + rank)}
            for rank, cid in enumerate(CANDIDATE_ORDER, start=1)
        ]
        payload = {
            "qid": q["qid"],
            "question": q["question"],
            "hash": candidates_hash(cands),
            "candidates": cands,
        }
        (config.CAND_DIR / f"{q['qid']}.json").write_text(json.dumps(payload), encoding="utf-8")
    return SimpleNamespace(root=workspace, chunks=CHUNKS, questions=QUESTIONS)
