"""Write data/eval/spot_check.md: 20 seeded-random questions with their gold evidence for human review."""
import random

from jevbench import config
from jevbench.labels import label_question, load_questions
from jevbench.retrieve import load_chunks

chunks = {c["chunk_id"]: c for c in load_chunks()}
qs = load_questions()
rng = random.Random(7)

# stratified: a few of every type so the reviewer sees each kind
by_type: dict[str, list[dict]] = {}
for q in qs:
    by_type.setdefault(q["type"], []).append(q)
picked: list[dict] = []
for t, n in (("fact", 5), ("table", 5), ("paraphrase", 5), ("multi", 3), ("unanswerable", 2)):
    picked += rng.sample(by_type[t], n)

out = ["# Eval-set spot check (20 questions)\n",
       "For each item: is the question sensible, and does the quoted evidence actually answer it? "
       "For unanswerable items: confirm the answer is NOT in the documents.\n"]
for i, q in enumerate(picked, start=1):
    lab = label_question(q, list(chunks.values()))
    out.append(f"\n## {i}. `{q['qid']}`  ({q['type']}, mode={q['mode']}, split={q['split']})\n")
    out.append(f"**Q:** {q['question']}\n")
    if not q["answerable"]:
        out.append("**Expected: unanswerable** (no gold evidence)\n")
        continue
    for g, ids in zip(q["gold"], lab["per_quote"]):
        out.append(f"- **Gold quote** ({g['doc_id']}): `{g['quote']}`")
        for cid in ids[:2]:
            c = chunks[cid]
            out.append(f"  - found in `{cid}` (page {c['page']})")
    out.append("")

path = config.EVAL_DIR / "spot_check.md"
path.write_text("\n".join(out), encoding="utf-8")
print("wrote", path)
