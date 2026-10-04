"""Compute ranking metrics and Jev diagnostics from results/ and write a markdown report.

  uv run python -m jevbench.evaluate --split test

The report is built from small section functions that each take an `Evaluation` (the compared
questions, their labels and frozen candidates, and every reranker's stored scores) and return
markdown lines, so each piece can be tested on synthetic data.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from statistics import mean, median

from . import config, metrics
from .gate import load_gate
from .labels import label_question, load_questions
from .rerankers.base import order_by_scores
from .retrieve import candidates_hash, load_candidates, load_chunks
from .run_rerank import load_done

RERANKERS = ["none", "flashrank", "cohere", "cohere_pro", "jev_pair", "jev_pack"]
BASELINES = ("none", "flashrank", "cohere", "cohere_pro")
JEV_RERANKERS = ("jev_pair", "jev_pack")
PROBES = ("jev_pack_shuffled", "jev_pair_rerun")
METRIC_KEYS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@10"]


# --- small formatting and statistics helpers -------------------------------------------------


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def ci_str(values: list[float]) -> str:
    m, lo, hi = metrics.bootstrap_ci(values)
    return f"{pct(m)} [{pct(lo)}, {pct(hi)}]"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties; nan if either side is constant."""

    def ranks(x: list[float]) -> list[float]:
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
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def calibration(
    points: list[tuple[float, int]], bins: int = 10
) -> tuple[float, list[tuple[float, float, int]]]:
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


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


# --- the data a report is computed from ------------------------------------------------------


@dataclass
class Evaluation:
    """Compared questions plus everything needed to score them."""

    split: str | None
    names: list[str]  # rerankers in the report, in display order
    questions: list[dict]  # questions covered by every selected reranker (and in the split)
    labels: dict[str, dict]  # qid -> label_question() output
    candidates: dict[str, list[dict]]  # qid -> frozen candidate list
    results: dict[str, dict[str, dict]]  # reranker -> qid -> stored result record
    probes: dict[str, dict[str, dict]] = field(default_factory=dict)
    gate: dict[str, dict[str, dict]] = field(default_factory=dict)
    split_questions: list[dict] | None = None  # every question in the split, scored or not

    @classmethod
    def load(cls, split: str | None, only: list[str] | None = None) -> Evaluation:
        names = [n for n in RERANKERS if not only or n in only]
        results = {n: load_done(n) for n in names}
        # compare only questions every selected reranker has covered, so the tables are comparable
        present = [n for n in names if results[n]]
        covered = set.intersection(*(set(results[n]) for n in present)) if present else set()
        split_questions = load_questions(split)
        questions = [q for q in split_questions if q["qid"] in covered]
        chunks = load_chunks()
        return cls(
            split=split,
            names=names,
            questions=questions,
            labels={q["qid"]: label_question(q, chunks) for q in questions},
            candidates={q["qid"]: load_candidates(q["qid"])["candidates"] for q in questions},
            results=results,
            probes={n: load_done(n) for n in PROBES},
            gate={n: load_gate(n) for n in names},
            split_questions=split_questions,
        )

    @cached_property
    def compared(self) -> set[str]:
        return {q["qid"] for q in self.questions}

    @property
    def all_questions(self) -> list[dict]:
        return self.questions if self.split_questions is None else self.split_questions

    def coverage(self, name: str) -> int:
        """How many questions of the split this reranker has been scored on."""
        scored = self.results.get(name, {})
        return sum(1 for q in self.all_questions if q["qid"] in scored)

    def records(self, name: str) -> dict[str, dict]:
        """Stored results for a reranker or probe, restricted to the compared questions."""
        stored = self.results[name] if name in self.results else self.probes.get(name, {})
        return {qid: record for qid, record in stored.items() if qid in self.compared}

    @cached_property
    def answerable(self) -> list[dict]:
        return [q for q in self.questions if q["answerable"]]

    @cached_property
    def unanswerable(self) -> list[dict]:
        return [q for q in self.questions if not q["answerable"]]

    @cached_property
    def candidate_hits(self) -> dict[str, bool]:
        """qid -> does the frozen candidate list contain a gold chunk (answerable questions)?"""
        hits = {}
        for q in self.answerable:
            gold = set(self.labels[q["qid"]]["gold_chunk_ids"])
            hits[q["qid"]] = any(c["chunk_id"] in gold for c in self.candidates[q["qid"]])
        return hits

    @cached_property
    def addressable(self) -> set[str]:
        """Questions a reranker can actually fix: the gold chunk is among the candidates."""
        return {qid for qid, hit in self.candidate_hits.items() if hit}

    def ranked_ids(self, record: dict, qid: str) -> list[str]:
        order = order_by_scores(record["scores"], [c["rrf_rank"] for c in self.candidates[qid]])
        return [record["chunk_ids"][i] for i in order]

    def question_metrics(self, record: dict, qid: str) -> dict[str, float]:
        label = self.labels[qid]
        return metrics.question_metrics(
            self.ranked_ids(record, qid), label["per_quote"], label["mode"]
        )

    def metric_values(
        self, name: str, subset: set[str] | None = None
    ) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
        """Per-question metric values for one reranker: overall and grouped by question type."""
        overall: dict[str, list[float]] = defaultdict(list)
        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for q in self.answerable:
            record = self.results[name].get(q["qid"])
            if record is None or (subset is not None and q["qid"] not in subset):
                continue
            for key, value in self.question_metrics(record, q["qid"]).items():
                overall[key].append(value)
                by_type[q["type"]][key].append(value)
        return overall, by_type

    def candidate_mismatches(self) -> list[str]:
        """qids whose stored scores were not all computed on the candidate list frozen now.

        A mismatch means two rerankers saw different candidates, or the frozen candidates changed
        (for example after re-ingesting a revised document) since the scores were stored.
        """
        hashes: dict[str, set[str]] = defaultdict(set)
        for records in self.results.values():
            for qid, record in records.items():
                hashes[qid].add(record["cand_hash"])
        for qid, candidates in self.candidates.items():
            hashes[qid].add(candidates_hash(candidates))
        return sorted(qid for qid, seen in hashes.items() if len(seen) > 1)


# --- report sections: each returns markdown lines --------------------------------------------


def section_header(ev: Evaluation) -> list[str]:
    scored = ", ".join(f"{name} {ev.coverage(name)}" for name in ev.names)
    return [
        f"# Jev reranking benchmark report (split: {ev.split or 'all'})\n",
        f"Questions: {len(ev.questions)} ({len(ev.answerable)} answerable, "
        f"{len(ev.unanswerable)} unanswerable). "
        f"Model: `{config.JEV_MODEL}`; candidates per question: {config.N_CANDIDATES}.\n",
        f"Questions scored per reranker (of {len(ev.all_questions)} in this split): {scored}. "
        f"Every table below compares only the {len(ev.questions)} questions that all of them "
        "have scored.\n",
    ]


def section_candidate_recall(ev: Evaluation) -> list[str]:
    hits = [float(ev.candidate_hits[q["qid"]]) for q in ev.answerable]
    lines = [f"## Candidate recall@{config.N_CANDIDATES} (ceiling for all rerankers)\n"]
    if hits:
        lines.append(
            f"Gold chunk present in the candidate list for **{pct(mean(hits))}%** of answerable "
            f"questions ({int(sum(hits))}/{len(hits)}).\n"
        )
    else:
        lines.append("There are no answerable questions to compare.\n")
    bad = ev.candidate_mismatches()
    verdict = (
        "OK, every reranker was scored on byte-identical candidates that match the frozen lists"
        if not bad
        else "MISMATCH on " + ", ".join(bad)
    )
    lines.append(f"Candidate-list integrity: {verdict}\n")
    return lines


def section_ranking(ev: Evaluation) -> list[str]:
    lines: list[str] = []
    subsets = (
        ("All answerable questions", None),
        ("Reranking-addressable subset (gold in candidates)", ev.addressable),
    )
    for title, subset in subsets:
        rows = []
        for name in ev.names:
            overall, _ = ev.metric_values(name, subset)
            if not overall["mrr"]:
                continue
            rows.append(
                [name, str(len(overall["mrr"])), *(ci_str(overall[k]) for k in METRIC_KEYS)]
            )
        lines += [
            f"## Ranking quality: {title}\n",
            "Mean % with 95% bootstrap CI over questions.\n",
            *md_table(["reranker", "n", *METRIC_KEYS], rows),
            "",
        ]
    return lines


def section_paired(ev: Evaluation) -> list[str]:
    ndcg = {n: ev.metric_values(n, ev.addressable)[0]["ndcg@10"] for n in ev.names if ev.results[n]}
    rows = []
    for a in JEV_RERANKERS:
        for b in BASELINES:
            if a in ndcg and b in ndcg and len(ndcg[a]) == len(ndcg[b]) and ndcg[a]:
                m, lo, hi = metrics.paired_bootstrap_diff(ndcg[a], ndcg[b])
                rows.append([a, b, f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"])
    return [
        "## Paired difference in nDCG@10 (points, 95% CI), addressable subset\n",
        *md_table(["A", "B", "A - B"], rows),
        "",
    ]


def section_by_type(ev: Evaluation) -> list[str]:
    types = sorted({q["type"] for q in ev.answerable})
    rows = []
    for name in ev.names:
        if not ev.results[name]:
            continue
        _, by_type = ev.metric_values(name, ev.addressable)
        cells = []
        for t in types:
            values = by_type[t]["ndcg@10"]
            cells.append(f"{pct(mean(values))} (n={len(values)})" if values else "n/a")
        rows.append([name, *cells])
    return [
        "## nDCG@10 by question type (addressable subset)\n",
        *md_table(["reranker", *types], rows),
        "",
    ]


def section_latency(ev: Evaluation) -> list[str]:
    rows = []
    for name in ev.names:
        records = list(ev.records(name).values())
        if not records:
            continue
        latency = sorted(r["latency_s"] for r in records)
        p95 = latency[min(len(latency) - 1, int(0.95 * len(latency)))]
        rows.append(
            [
                name,
                f"{median(latency):.3f}",
                f"{p95:.3f}",
                f"{mean(r['input_tokens'] for r in records):.0f}",
                f"{mean(r['cost_usd'] for r in records):.6f}",
            ]
        )
    headers = [
        "reranker",
        "p50 latency (s)",
        "p95 latency (s)",
        "mean input tokens",
        "mean cost / query (USD)",
    ]
    lines = [
        "## Latency and cost per query\n",
        "Jev pair latency is the slowest single request (all pair calls assumed concurrent); "
        "sequential sum is shown too. Cohere/FlashRank latency is the rerank call only.\n",
        *md_table(headers, rows),
    ]
    pair = ev.records("jev_pair")
    if pair:
        sequential = [sum(r["meta"].get("per_request_latency_s", [0])) for r in pair.values()]
        n_requests = mean(r["meta"].get("n_requests", 0) for r in pair.values())
        lines.append(
            f"\nJev pair sequential-sum latency per query: median {median(sequential):.2f}s "
            f"({n_requests:.0f} requests/query)."
        )
    lines.append("")
    return lines


def jev_score_summary(ev: Evaluation, name: str, records: dict[str, dict]) -> list[str]:
    """Score saturation, ties and calibration for one Jev reranker."""
    scores = [s for r in records.values() for s in r["scores"]]
    saturated_high = mean(float(s >= 0.99) for s in scores)
    saturated_low = mean(float(s <= 0.01) for s in scores)
    ties = []
    for r in records.values():
        top = max(r["scores"])
        ties.append(float(sum(1 for s in r["scores"] if abs(s - top) < 1e-9) > 1))
    lines = [
        f"**{name}**: {pct(saturated_high)}% of candidate scores >= 0.99, "
        f"{pct(saturated_low)}% <= 0.01; top score tied within a query in {pct(mean(ties))}% "
        "of queries (ties broken by RRF rank)."
    ]
    points = []
    for q in ev.answerable:
        r = records.get(q["qid"])
        if r is None:
            continue
        gold = set(ev.labels[q["qid"]]["gold_chunk_ids"])
        points += [
            (s, int(cid in gold)) for s, cid in zip(r["scores"], r["chunk_ids"], strict=True)
        ]
    if points:
        ece, rows = calibration(points)
        lines += [
            f"\nCalibration of P(yes) against 'chunk contains gold quote' (ECE {ece:.3f}; "
            "label noise caveat: other chunks may also answer):\n",
            *md_table(
                ["mean P(yes)", "fraction gold", "n"],
                [[f"{mp:.2f}", f"{fp:.3f}", str(c)] for mp, fp, c in rows],
            ),
        ]
    lines.append("")
    return lines


def jev_variant_lines(ev: Evaluation) -> list[str]:
    """nDCG@10 of the three ways of scoring a Jev pair request (stored side by side)."""
    lines = []
    pair = ev.records("jev_pair")
    for variant in ("evidence", "relevant", "mean"):
        values = []
        for q in ev.answerable:
            record = pair.get(q["qid"])
            if record is None or q["qid"] not in ev.addressable or "variants" not in record["meta"]:
                continue
            rescored = dict(record, scores=record["meta"]["variants"][variant])
            values.append(ev.question_metrics(rescored, q["qid"])["ndcg@10"])
        if values:
            lines.append(
                f"- jev_pair scoring variant `{variant}`: nDCG@10 {ci_str(values)} "
                f"(n={len(values)})"
            )
    lines.append("")
    return lines


def jev_probe_lines(ev: Evaluation) -> list[str]:
    """Order sensitivity, run-to-run stability and pair-vs-packed agreement (when available)."""
    lines = []
    pair = ev.records("jev_pair")
    pack = ev.records("jev_pack")
    shuffled = ev.records("jev_pack_shuffled")
    rerun = ev.records("jev_pair_rerun")
    if shuffled and pack:
        pairs = [(pack[qid]["scores"], r["scores"]) for qid, r in shuffled.items() if qid in pack]
        rho = [spearman(a, b) for a, b in pairs]
        same_top = [float(argmax(a) == argmax(b)) for a, b in pairs]
        lines.append(
            f"Candidate-order sensitivity (jev_pack, original vs shuffled order, n={len(rho)}): "
            f"mean Spearman {mean(rho):.3f}; same top-1 candidate in {pct(mean(same_top))}% "
            "of queries.\n"
        )
    elif "jev_pack" in ev.names:
        lines.append("Candidate-order sensitivity: not run (no shuffled-order results found).\n")
    if rerun and pair:
        pairs = [(pair[qid]["scores"], r["scores"]) for qid, r in rerun.items() if qid in pair]
        diffs = [mean(abs(x - y) for x, y in zip(a, b, strict=True)) for a, b in pairs]
        rho = [spearman(a, b) for a, b in pairs]
        lines.append(
            f"Run-to-run stability (jev_pair, identical inputs, n={len(rho)}): "
            f"mean |delta P| {mean(diffs):.4f}; mean Spearman {mean(rho):.3f}.\n"
        )
    elif "jev_pair" in ev.names:
        lines.append("Run-to-run stability: not run (no repeated-request results found).\n")
    if pair and pack:
        both = [q["qid"] for q in ev.questions if q["qid"] in pair and q["qid"] in pack]
        rho = [spearman(pair[i]["scores"], pack[i]["scores"]) for i in both]
        rho = [r for r in rho if r == r]  # drop queries where all scores tie (undefined)
        if rho:
            lines.append(
                f"Pair vs packed agreement: mean Spearman {mean(rho):.3f} over {len(rho)} "
                "queries.\n"
            )
    return lines


def section_jev(ev: Evaluation) -> list[str]:
    lines = ["## Jev-specific analyses\n"]
    for name in JEV_RERANKERS:
        records = ev.records(name)
        if records:
            lines += jev_score_summary(ev, name, records)
    return lines + jev_variant_lines(ev) + jev_probe_lines(ev)


def section_gate(ev: Evaluation) -> list[str]:
    rows = []
    for name in ev.names:
        gate = ev.gate.get(name, {})
        pos = [gate[q["qid"]] for q in ev.answerable if q["qid"] in gate]
        neg = [gate[q["qid"]] for q in ev.unanswerable if q["qid"] in gate]
        if pos and neg:
            by_jev = metrics.auroc(
                [r["p_answerable"] for r in pos], [r["p_answerable"] for r in neg]
            )
            by_score = metrics.auroc(
                [r["max_rerank_score"] for r in pos], [r["max_rerank_score"] for r in neg]
            )
            rows.append([name, f"{by_jev:.3f}", f"{by_score:.3f}", f"{len(pos)} / {len(neg)}"])
    lines = ["## Answerability gate (AUROC: answerable vs unanswerable)\n"]
    if rows:
        lines += [
            "Signal 1 = Jev P(answerable) over the reranker's top-5 passages. Signal 2 = the "
            "reranker's own top score (the only gate-like signal a cross-encoder offers; AUROC is "
            "rank-based so scales do not matter).\n",
            *md_table(
                [
                    "reranker",
                    "AUROC Jev P(answerable) top-5",
                    "AUROC reranker max score",
                    "n answerable / unanswerable",
                ],
                rows,
            ),
        ]
    else:
        lines.append(
            "**Not run.** No gate results were found in `results/gate/` for these questions, so "
            "this report makes no claim about Jev as an answerability gate."
        )
    lines.append("")
    return lines


SECTIONS = (
    section_header,
    section_candidate_recall,
    section_ranking,
    section_paired,
    section_by_type,
    section_latency,
    section_jev,
    section_gate,
)


def build_report(ev: Evaluation) -> str:
    return "\n".join(line for section in SECTIONS for line in section(ev))


def report_path(split: str | None, tag: str | None = None) -> Path:
    return config.RESULTS_DIR / f"report_{split or 'all'}{'_' + tag if tag else ''}.md"


def main(split: str | None, only: list[str] | None = None, tag: str | None = None) -> None:
    report = build_report(Evaluation.load(split, only))
    path = report_path(split, tag)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--rerankers", nargs="+", default=None, help="restrict the report to these")
    ap.add_argument("--tag", default=None, help="suffix for the report file name")
    args = ap.parse_args()
    main(args.split, args.rerankers, args.tag)
