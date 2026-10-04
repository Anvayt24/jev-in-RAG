"""Map gold evidence quotes to the chunk ids that contain them (robust to re-chunking)."""

from __future__ import annotations

import json
import re

from . import config


def normalize(text: str) -> str:
    """Lowercase, drop markdown/table punctuation, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_questions(split: str | None = None) -> list[dict]:
    path = config.EVAL_DIR / "questions.jsonl"
    qs = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [q for q in qs if split is None or q["split"] == split]


def label_question(q: dict, chunks: list[dict]) -> dict:
    """Return {'gold_chunk_ids': [...], 'per_quote': [[ids]...], 'missing': [quote...]}.

    A chunk is gold if it contains (after normalization) any gold quote of the question.
    Unanswerable questions have no gold quotes and therefore no gold chunks.
    """
    norm_chunks = [(c["chunk_id"], c["doc_id"], normalize(c["text"])) for c in chunks]
    per_quote: list[list[str]] = []
    missing: list[str] = []
    for g in q.get("gold", []):
        nq = normalize(g["quote"])
        ids = [cid for cid, doc, ntext in norm_chunks if doc == g["doc_id"] and nq and nq in ntext]
        per_quote.append(ids)
        if not ids:
            missing.append(g["quote"])
    gold = sorted({cid for ids in per_quote for cid in ids})
    return {
        "gold_chunk_ids": gold,
        "per_quote": per_quote,
        "missing": missing,
        "mode": q.get("mode", "any"),
    }
