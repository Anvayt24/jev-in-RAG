"""Report statistics and the generated report, checked against hand-computed values."""

import json
import math

import pytest

from jevbench import config, evaluate, run_rerank
from jevbench.evaluate import (
    Evaluation,
    argmax,
    build_report,
    calibration,
    ci_str,
    md_table,
    pct,
    spearman,
)
from jevbench.rerankers.none import NoRerank

from .conftest import KeywordReranker


def make_stub(name: str):
    return NoRerank() if name == "none" else KeywordReranker()


# --- helpers ----------------------------------------------------------------------------------


def test_spearman_perfect_inverse_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)  # average ranks for ties


def test_spearman_is_nan_when_one_side_is_constant():
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_calibration_of_a_perfectly_calibrated_set_has_zero_error():
    points = [(0.1, 0)] * 9 + [(0.1, 1)] + [(0.9, 1)] * 9 + [(0.9, 0)]
    ece, rows = calibration(points)
    assert ece == pytest.approx(0.0)
    assert [n for _, _, n in rows] == [10, 10]


def test_calibration_counts_probability_one_in_the_top_bin():
    ece, rows = calibration([(1.0, 1), (1.0, 0)])
    assert rows == [(1.0, 0.5, 2)]
    assert ece == pytest.approx(0.5)


def test_calibration_penalises_overconfidence():
    ece, _ = calibration([(0.95, 0)] * 10)
    assert ece == pytest.approx(0.95)


def test_formatting_helpers():
    assert pct(0.1234) == "12.3"
    assert pct(float("nan")) == "n/a"
    assert ci_str([1.0, 1.0, 1.0]) == "100.0 [100.0, 100.0]"
    assert md_table(["a", "b"], [["1", "2"]]) == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_argmax_returns_the_first_maximum():
    assert argmax([0.1, 0.9, 0.9, 0.2]) == 1


# --- the report on the synthetic corpus ---------------------------------------------------------


@pytest.fixture
def scored(corpus):
    """`none` keeps the fused order (gold at rank 3 and 4); `flashrank` is the keyword stub."""
    run_rerank.run(["none", "flashrank"], split=None, limit=None, make=make_stub)
    return corpus


def table_row(report: str, name: str, header: str) -> list[str]:
    """Cells of the first table row for `name` after the section titled `header`."""
    section = report.split(header, 1)[1]
    row = next(line for line in section.splitlines() if line.startswith(f"| {name} |"))
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_ranking_table_matches_hand_computed_metrics(scored):
    ev = Evaluation.load("test", ["none", "flashrank"])
    report = build_report(ev)
    # only the two answerable test questions are ranked; q4 is dev, q3 is unanswerable
    none = table_row(report, "none", "## Ranking quality: All answerable questions")
    keyword = table_row(report, "flashrank", "## Ranking quality: All answerable questions")
    assert none[1] == keyword[1] == "2"
    assert none[2] == "0.0 [0.0, 0.0]"  # hit@1: gold is at rank 3 and 4
    assert keyword[2] == "100.0 [100.0, 100.0]"
    mrr = (1 / 3 + 1 / 4) / 2  # hand-computed from the candidate order
    assert none[6].split()[0] == f"{100 * mrr:.1f}"


def test_only_questions_every_reranker_scored_are_compared(corpus):
    run_rerank.run(["flashrank"], split=None, limit=None, make=make_stub)
    run_rerank.run(["none"], split="test", limit=None, make=make_stub)
    ev = Evaluation.load(None, ["none", "flashrank"])
    assert sorted(q["qid"] for q in ev.questions) == ["q1", "q2", "q3"]  # q4 has no `none` score
    assert ev.coverage("flashrank") == 4 and ev.coverage("none") == 3
    report = build_report(ev)
    assert "none 3, flashrank 4" in report
    assert "compares only the 3 questions" in report


def test_candidate_recall_and_addressable_subset(scored):
    ev = Evaluation.load("test", ["none", "flashrank"])
    assert ev.addressable == {"q1", "q2"}
    report = build_report(ev)
    assert "**100.0%** of answerable questions (2/2)" in report


def test_a_reranker_given_different_candidates_is_flagged(scored):
    path = run_rerank.result_path("flashrank")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.replace('"cand_hash": "', '"cand_hash": "x') for line in lines))
    report = build_report(Evaluation.load("test", ["none", "flashrank"]))
    assert "MISMATCH on" in report
    assert "byte-identical" not in report


def test_consistent_data_passes_the_integrity_check(scored):
    ev = Evaluation.load("test", ["none", "flashrank"])
    assert ev.candidate_mismatches() == []
    assert "OK, every reranker was scored on byte-identical candidates" in build_report(ev)


def test_candidates_changed_after_scoring_are_flagged(scored):
    path = config.CAND_DIR / "q1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidates"][0]["text"] += " (the document was revised)"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ev = Evaluation.load("test", ["none", "flashrank"])
    assert ev.candidate_mismatches() == ["q1"]
    assert "MISMATCH on q1" in build_report(ev)


def test_report_says_when_the_gate_and_probes_were_not_run(scored):
    report = build_report(Evaluation.load("test", ["none", "flashrank"]))
    assert "**Not run.**" in report
    assert "results/gate" in report


def test_report_without_answerable_questions_does_not_crash(corpus):
    run_rerank.run(["none"], split="test", limit=None, make=make_stub)
    ev = Evaluation.load("test", ["none"])
    ev.questions[:] = [q for q in ev.questions if not q["answerable"]]
    assert "no answerable questions" in build_report(ev)


def test_report_path_includes_split_and_tag():
    assert evaluate.report_path("test").name == "report_test.md"
    assert evaluate.report_path(None, "partial").name == "report_all_partial.md"
