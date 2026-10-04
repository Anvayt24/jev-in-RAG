"""Compute metrics + Jev-specific analyses from results/ and write results/report.md.

  uv run python -m jevbench.evaluate --split test
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from statistics import mean, median

from . import config, metrics
from .gate import load_gate
from .labels import label_question, load_questions
from .rerankers.base import order_by_scores
from .retrieve import load_candidates, load_chunks
from .run_rerank import load_done

MAIN = ["none", "flashrank", "cohere", "cohere_pro", "jev_pair", "jev_pack"]
KEYS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@10"]


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def ci_str(vals: list[float]) -> str:
    m, lo, hi = metrics.bootstrap_ci(vals)
    return f"{pct(m)} [{pct(lo)}, {pct(hi)}]"


def ranked_ids(rec: dict, cand: list[dict]) -> list[str]:
    order = order_by_scores(rec["scores"], [c["rrf_rank"] for c in cand])
    return [rec["chunk_ids"][i] for i in order]


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def calibration(points: list[tuple[float, int]], bins: int = 10) -> tuple[float, list[tuple[float, float, int]]]:
    """ECE and reliability rows (mean_pred, frac_pos, n) for (p, label) pairs."""
    rows, ece, n = [], 0.0, len(points)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        pts = [(p, y) for p, y in points if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not pts:
            continue
        mp, fp = mean(p for p, _ in pts), mean(y for _, y in pts)
        rows.append((mp, fp, len(pts)))
        ece += len(pts) / n * abs(mp - fp)
    return ece, rows


def main(split: str | None, only: list[str] | None = None, tag: str | None = None) -> None:
    global MAIN
    if only:
        MAIN = [n for n in MAIN if n in only]
    chunks = load_chunks()
    questions = load_questions(split)
    results = {n: load_done(n) for n in MAIN}
    for n in ("jev_pair", "jev_pack"):
        results.setdefault(n, {})
    # score only questions that every selected reranker has covered, so tables are comparable
    present = [n for n in MAIN if results[n]]
    common = set.intersection(*(set(results[n]) for n in present)) if present else set()
    questions = [q for q in questions if q["qid"] in common]
    labels = {q["qid"]: label_question(q, chunks) for q in questions}
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    extra = {n: load_done(n) for n in ("jev_pack_shuffled", "jev_pair_rerun")}
    out: list[str] = []
    w = out.append

    w(f"# Jev reranking benchmark report (split: {split or 'all'})\n")
    w(f"Questions: {len(questions)} ({len(answerable)} answerable, {len(unanswerable)} unanswerable). "
      f"Model: `{config.JEV_MODEL}`; candidates per question: {config.N_CANDIDATES}.\n")

    # --- candidate recall (ceiling for every reranker) ---
    in_cand = []
    for q in answerable:
        cids = {c["chunk_id"] for c in load_candidates(q["qid"])["candidates"]}
        in_cand.append(float(bool(cids & set(labels[q["qid"]]["gold_chunk_ids"]))))
    w(f"## Candidate recall@{config.N_CANDIDATES} (ceiling for all rerankers)\n")
    w(f"Gold chunk present in the candidate list for **{pct(mean(in_cand))}%** of answerable questions "
      f"({int(sum(in_cand))}/{len(in_cand)}).\n")
    addressable = {q["qid"] for q, f in zip(answerable, in_cand) if f}

    # --- hash check: identical candidates for every reranker ---
    hashes = defaultdict(set)
    for n, d in results.items():
        for qid, r in d.items():
            hashes[qid].add(r["cand_hash"])
    bad = [qid for qid, h in hashes.items() if len(h) > 1]
    w(f"Candidate-list integrity: {'OK, every reranker saw byte-identical candidates' if not bad else 'MISMATCH on ' + ', '.join(bad)}\n")

    # --- main ranking table ---
    def collect(name: str, subset: set[str] | None):
        per = defaultdict(list)
        types = defaultdict(lambda: defaultdict(list))
        for q in answerable:
            r = results[name].get(q["qid"])
            if r is None or (subset is not None and q["qid"] not in subset):
                continue
            cand = load_candidates(q["qid"])["candidates"]
            lab = labels[q["qid"]]
            m = metrics.question_metrics(ranked_ids(r, cand), lab["per_quote"], lab["mode"])
            for k, v in m.items():
                per[k].append(v)
                types[q["type"]][k].append(v)
        return per, types

    for title, subset in (("All answerable questions", None), ("Reranking-addressable subset (gold in candidates)", addressable)):
        w(f"## Ranking quality: {title}\n")
        w("Mean % with 95% bootstrap CI over questions.\n")
        w("| reranker | n | " + " | ".join(KEYS) + " |")
        w("|---|---|" + "---|" * len(KEYS))
        for n in MAIN:
            per, _ = collect(n, subset)
            if not per["mrr"]:
                continue
            w(f"| {n} | {len(per['mrr'])} | " + " | ".join(ci_str(per[k]) for k in KEYS) + " |")
        w("")

    # --- paired differences vs flashrank and none on nDCG@10 ---
    w("## Paired difference in nDCG@10 (points, 95% CI), addressable subset\n")
    w("| A | B | A - B |")
    w("|---|---|---|")
    base = {n: collect(n, addressable)[0]["ndcg@10"] for n in MAIN if results[n]}
    for a in ("jev_pair", "jev_pack"):
        for b in ("none", "flashrank", "cohere", "cohere_pro"):
            if a in base and b in base and len(base[a]) == len(base[b]) and base[a]:
                m, lo, hi = metrics.paired_bootstrap_diff(base[a], base[b])
                w(f"| {a} | {b} | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    w("")

    # --- by question type (nDCG@10, addressable) ---
    w("## nDCG@10 by question type (addressable subset)\n")
    all_types = sorted({q["type"] for q in answerable})
    w("| reranker | " + " | ".join(all_types) + " |")
    w("|---|" + "---|" * len(all_types))
    for n in MAIN:
        if not results[n]:
            continue
        _, types = collect(n, addressable)
        w(f"| {n} | " + " | ".join(f"{pct(mean(types[t]['ndcg@10']))} (n={len(types[t]['ndcg@10'])})" if types[t]["ndcg@10"] else "n/a" for t in all_types) + " |")
    w("")

    # --- ops ---
    w("## Latency and cost per query\n")
    w("Jev pair latency is the slowest single request (all pair calls assumed concurrent); "
      "sequential sum is shown too. Cohere/FlashRank latency is the rerank call only.\n")
    w("| reranker | p50 latency (s) | p95 latency (s) | mean input tokens | mean cost / query (USD) |")
    w("|---|---|---|---|---|")
    for n in MAIN:
        rs = [r for qid, r in results[n].items() if qid in common]
        if not rs:
            continue
        lat = sorted(r["latency_s"] for r in rs)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        w(f"| {n} | {median(lat):.3f} | {p95:.3f} | {mean(r['input_tokens'] for r in rs):.0f} | {mean(r['cost_usd'] for r in rs):.6f} |")
    if results["jev_pair"]:
        seq = [sum(r["meta"].get("per_request_latency_s", [0])) for r in results["jev_pair"].values()]
        w(f"\nJev pair sequential-sum latency per query: median {median(seq):.2f}s "
          f"({mean(r['meta'].get('n_requests', 0) for r in results['jev_pair'].values()):.0f} requests/query).")
    w("")

    # --- Jev-specific ---
    w("## Jev-specific analyses\n")
    for n in ("jev_pair", "jev_pack"):
        rs = results[n]
        if not rs:
            continue
        allp = [s for r in rs.values() for s in r["scores"]]
        sat_hi = mean(float(s >= 0.99) for s in allp)
        sat_lo = mean(float(s <= 0.01) for s in allp)
        ties = []
        for r in rs.values():
            top = max(r["scores"])
            ties.append(float(sum(1 for s in r["scores"] if abs(s - top) < 1e-9) > 1))
        w(f"**{n}**: {pct(sat_hi)}% of candidate scores >= 0.99, {pct(sat_lo)}% <= 0.01; "
          f"top score tied within a query in {pct(mean(ties))}% of queries (ties broken by RRF rank).")
        pts = []
        for q in answerable:
            r = rs.get(q["qid"])
            if r is None:
                continue
            gold = set(labels[q["qid"]]["gold_chunk_ids"])
            pts += [(s, int(cid in gold)) for s, cid in zip(r["scores"], r["chunk_ids"])]
        if pts:
            ece, rows = calibration(pts)
            w(f"\nCalibration of P(yes) against 'chunk contains gold quote' (ECE {ece:.3f}; label noise caveat: other chunks may also answer):\n")
            w("| mean P(yes) | fraction gold | n |")
            w("|---|---|---|")
            for mp, fp, c in rows:
                w(f"| {mp:.2f} | {fp:.3f} | {c} |")
        w("")

    for variant in ("evidence", "relevant", "mean"):
        per = []
        for q in answerable:
            r = results["jev_pair"].get(q["qid"])
            if r is None or q["qid"] not in addressable or "variants" not in r["meta"]:
                continue
            cand = load_candidates(q["qid"])["candidates"]
            tmp = dict(r, scores=r["meta"]["variants"][variant])
            lab = labels[q["qid"]]
            per.append(metrics.question_metrics(ranked_ids(tmp, cand), lab["per_quote"], lab["mode"])["ndcg@10"])
        if per:
            w(f"- jev_pair scoring variant `{variant}`: nDCG@10 {ci_str(per)} (n={len(per)})")
    w("")

    if extra["jev_pack_shuffled"] and results["jev_pack"]:
        rho = [spearman(results["jev_pack"][qid]["scores"], r["scores"])
               for qid, r in extra["jev_pack_shuffled"].items() if qid in results["jev_pack"]]
        same_top = [float(max(range(30), key=lambda i: results["jev_pack"][qid]["scores"][i]) ==
                          max(range(30), key=lambda i: r["scores"][i]))
                    for qid, r in extra["jev_pack_shuffled"].items() if qid in results["jev_pack"]]
        w(f"Candidate-order sensitivity (jev_pack, original vs shuffled order, n={len(rho)}): "
          f"mean Spearman {mean(rho):.3f}; same top-1 candidate in {pct(mean(same_top))}% of queries.\n")
    if extra["jev_pair_rerun"] and results["jev_pair"]:
        diffs = [mean(abs(a - b) for a, b in zip(results["jev_pair"][qid]["scores"], r["scores"]))
                 for qid, r in extra["jev_pair_rerun"].items() if qid in results["jev_pair"]]
        rho = [spearman(results["jev_pair"][qid]["scores"], r["scores"])
               for qid, r in extra["jev_pair_rerun"].items() if qid in results["jev_pair"]]
        w(f"Run-to-run stability (jev_pair, identical inputs, n={len(rho)}): mean |delta P| {mean(diffs):.4f}; mean Spearman {mean(rho):.3f}.\n")
    if results["jev_pair"] and results["jev_pack"]:
        both = [q["qid"] for q in questions if q["qid"] in results["jev_pair"] and q["qid"] in results["jev_pack"]]
        rho = [spearman(results["jev_pair"][i]["scores"], results["jev_pack"][i]["scores"]) for i in both]
        rho = [r for r in rho if r == r]  # drop queries where all scores tie (correlation undefined)
        if rho:
            w(f"Pair vs packed agreement: mean Spearman {mean(rho):.3f} over {len(rho)} queries.\n")

    # --- answerability gate ---
    w("## Answerability gate (AUROC: answerable vs unanswerable)\n")
    gate_rows = []
    for n in MAIN:
        g = load_gate(n)
        pos = [g[q["qid"]] for q in answerable if q["qid"] in g]
        neg = [g[q["qid"]] for q in unanswerable if q["qid"] in g]
        if pos and neg:
            a1 = metrics.auroc([r["p_answerable"] for r in pos], [r["p_answerable"] for r in neg])
            a2 = metrics.auroc([r["max_rerank_score"] for r in pos], [r["max_rerank_score"] for r in neg])
            gate_rows.append(f"| {n} | {a1:.3f} | {a2:.3f} | {len(pos)} / {len(neg)} |")
    if gate_rows:
        w("Signal 1 = Jev P(answerable) over the reranker's top-5 passages. Signal 2 = the reranker's own top score "
          "(the only gate-like signal a cross-encoder offers; AUROC is rank-based so scales do not matter).\n")
        w("| reranker | AUROC Jev P(answerable) top-5 | AUROC reranker max score | n answerable / unanswerable |")
        w("|---|---|---|---|")
        out.extend(gate_rows)
    else:
        w("**Not run.** The gate needs extra Jev calls and the OpenRouter account ran out of credits; "
          "the Jev probes (run-to-run stability, candidate-order sensitivity) were skipped for the same reason, "
          "and `jev_pack` covers only 56 of the 107 test questions. Conclusions about Jev as a gate are therefore untested here.")
    w("")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"report_{split or 'all'}{'_' + tag if tag else ''}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--rerankers", nargs="+", default=None, help="restrict the report to these rerankers")
    ap.add_argument("--tag", default=None, help="suffix for the report file name")
    a = ap.parse_args()
    main(a.split, a.rerankers, a.tag)
