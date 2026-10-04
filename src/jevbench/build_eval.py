"""Merge per-document question files into data/eval/questions.jsonl, validate gold quotes against
the real chunks, and assign a deterministic dev/test split (dev ~30%, used only to tune Jev wording).

  uv run python -m jevbench.build_eval
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from . import config
from .labels import label_question
from .retrieve import load_chunks


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 10 < 3 else "test"


def main() -> int:
    chunks = load_chunks()
    src_dir = config.EVAL_DIR / "src"
    questions: list[dict] = []
    for p in sorted(src_dir.glob("*.jsonl")):
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                q = json.loads(line)
                q["_src"] = f"{p.name}:{ln}"
                questions.append(q)

    qids = [q["qid"] for q in questions]
    dup = [k for k, v in Counter(qids).items() if v > 1]
    problems: list[str] = [f"duplicate qid {d}" for d in dup]
    multi_chunk = 0
    for q in questions:
        lab = label_question(q, chunks)
        if q["answerable"] and not q["gold"]:
            problems.append(f"{q['qid']}: answerable but no gold quotes")
        if not q["answerable"] and q["gold"]:
            problems.append(f"{q['qid']}: unanswerable but has gold quotes")
        for m in lab["missing"]:
            problems.append(f"{q['qid']}: quote not found in any chunk of its doc: {m[:80]!r}")
        multi_chunk += sum(1 for ids in lab["per_quote"] if len(ids) > 1)
        q["split"] = split_of(q["qid"])
        q.pop("_src", None)

    out = config.EVAL_DIR / "questions.jsonl"
    out.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")

    print(f"{len(questions)} questions -> {out}")
    by = Counter((q["split"], q["type"]) for q in questions)
    for split in ("dev", "test"):
        row = {t: by[(split, t)] for t in sorted({q['type'] for q in questions})}
        print(f"  {split}: {sum(row.values())}  {row}")
    print(f"  quotes landing in >1 chunk (chunk overlap): {multi_chunk}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for pr in problems:
            print("  -", pr)
        return 1
    print("all gold quotes matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
