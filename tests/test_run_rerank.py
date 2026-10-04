"""The runner, candidate caching, the gate runner and the eval-set builder, on a tiny corpus."""

import json

import pytest

from jevbench import build_eval, config, gate, retrieve, run_rerank
from jevbench.labels import load_questions
from jevbench.rerankers.none import NoRerank
from jevbench.retrieve import candidates_hash, load_candidates

from .conftest import CHUNKS, KeywordReranker, write_jsonl


def make_stub(name: str):
    return NoRerank() if name == "none" else KeywordReranker()


# --- runner -----------------------------------------------------------------------------------


def test_run_writes_one_record_per_question(corpus):
    run_rerank.run(["keyword"], split=None, limit=None, make=make_stub)
    done = run_rerank.load_done("keyword")
    assert set(done) == {"q1", "q2", "q3", "q4"}
    record = done["q1"]
    assert record["reranker"] == "keyword"
    assert len(record["scores"]) == len(record["chunk_ids"]) == 4
    assert record["cand_hash"] == load_candidates("q1")["hash"]
    assert record["input_tokens"] == 5 and record["cost_usd"] == 0.001


def test_run_is_resumable(corpus):
    run_rerank.run(["keyword"], split=None, limit=2, make=make_stub)
    assert len(run_rerank.load_done("keyword")) == 2

    run_rerank.run(["keyword"], split=None, limit=None, make=make_stub)
    assert len(run_rerank.load_done("keyword")) == 4

    def must_not_be_called(name):
        raise AssertionError("a finished reranker should not be constructed again")

    run_rerank.run(["keyword"], split=None, limit=None, make=must_not_be_called)
    lines = run_rerank.result_path("keyword").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4, "no duplicate records after resuming"


def test_run_respects_the_split(corpus):
    run_rerank.run(["keyword"], split="dev", limit=None, make=make_stub)
    assert set(run_rerank.load_done("keyword")) == {"q4"}


def test_load_done_ignores_blank_lines_and_missing_files(workspace):
    assert run_rerank.load_done("nothing") == {}
    path = run_rerank.result_path("x")
    path.parent.mkdir(parents=True)
    path.write_text('{"qid": "a"}\n\n{"qid": "b"}\n', encoding="utf-8")
    assert set(run_rerank.load_done("x")) == {"a", "b"}


def test_unknown_reranker_name_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown reranker 'bge'"):
        run_rerank.make_reranker("bge")


def test_registry_covers_every_reranker_the_report_expects():
    from jevbench import evaluate

    assert set(evaluate.RERANKERS) | set(evaluate.PROBES) <= set(run_rerank.REGISTRY)


def test_none_reranker_through_the_registry_keeps_the_fused_order(corpus):
    run_rerank.run(["none"], split="test", limit=None)
    record = run_rerank.load_done("none")["q1"]
    assert record["scores"] == [-1.0, -2.0, -3.0, -4.0]


# --- candidate caching ------------------------------------------------------------------------


class FakeRetriever:
    calls = 0

    def retrieve(self, query, n=30):
        FakeRetriever.calls += 1
        return [
            {**c, "rrf_rank": i, "rrf_score": 1.0, "dense_rank": i, "bm25_rank": None}
            for i, c in enumerate(CHUNKS, start=1)
        ]


def test_candidates_are_retrieved_once_and_frozen(workspace, monkeypatch):
    monkeypatch.setattr(retrieve, "HybridRetriever", FakeRetriever)
    FakeRetriever.calls = 0
    questions = [{"qid": "a", "question": "first?"}, {"qid": "b", "question": "second?"}]

    retrieve.cache_candidates(questions)
    assert FakeRetriever.calls == 2
    frozen = load_candidates("a")
    assert frozen["hash"] == candidates_hash(frozen["candidates"])
    assert [c["rrf_rank"] for c in frozen["candidates"]] == [1, 2, 3, 4]

    retrieve.cache_candidates(questions)  # already cached: no new retrieval
    assert FakeRetriever.calls == 2


def test_candidates_hash_depends_on_order_and_text():
    a = [{"chunk_id": "1", "text": "x"}, {"chunk_id": "2", "text": "y"}]
    assert candidates_hash(a) != candidates_hash(list(reversed(a)))
    assert candidates_hash(a) != candidates_hash([{"chunk_id": "1", "text": "x2"}, a[1]])


# --- answerability gate -----------------------------------------------------------------------


def test_gate_scores_each_rerankers_top_passages(corpus, monkeypatch):
    seen = {}

    def fake_answerable(query, passages, **kwargs):
        seen[query] = passages
        p = 0.1 if "airspeed" in query else 0.9
        return {"p": p, "cost": 0.0, "tokens": 10, "latency_s": 0.1}

    monkeypatch.setattr("jevbench.rerankers.jev_rr.jev_answerable", fake_answerable)
    run_rerank.run(["keyword"], split="test", limit=None, make=make_stub)
    gate.run(["keyword"], split="test", top_k=1)

    records = gate.load_gate("keyword")
    assert set(records) == {"q1", "q2", "q3"}
    assert records["q1"]["p_answerable"] == 0.9 and records["q3"]["p_answerable"] == 0.1
    assert records["q1"]["top_chunk_ids"] == ["d:p2:c0"]  # keyword reranker puts the answer first
    assert len(seen["What is the capital of France?"]) == 1

    gate.run(["keyword"], split="test", top_k=1)  # resumable
    assert len(gate.gate_path("keyword").read_text(encoding="utf-8").splitlines()) == 3


# --- eval-set builder -------------------------------------------------------------------------


def test_split_is_deterministic_and_about_thirty_percent_dev():
    qids = [f"q-{i}" for i in range(2000)]
    splits = [build_eval.split_of(q) for q in qids]
    assert splits == [build_eval.split_of(q) for q in qids]
    assert 0.25 < splits.count("dev") / len(splits) < 0.35


def make_source(rows):
    write_jsonl(config.EVAL_DIR / "src" / "d.jsonl", rows)


def question(qid, quote):
    return {
        "qid": qid,
        "question": "q?",
        "type": "fact",
        "answerable": True,
        "mode": "any",
        "gold": [{"doc_id": "d", "quote": quote}],
    }


def test_builder_merges_sources_and_assigns_splits(corpus):
    make_source([question("a", "Paris is the capital of France"), question("b", "2 MB of flash")])
    assert build_eval.main() == 0
    built = load_questions()
    assert {q["qid"] for q in built} == {"a", "b"}
    assert all(q["split"] in ("dev", "test") for q in built)


def test_builder_fails_on_a_quote_that_is_not_in_the_document(corpus, capsys):
    make_source([question("a", "this sentence is nowhere in the corpus")])
    assert build_eval.main() == 1
    assert "quote not found" in capsys.readouterr().out


def test_builder_rejects_duplicate_ids_and_inconsistent_answerability(corpus, capsys):
    unanswerable_with_gold = {
        **question("b", "Paris is the capital of France"),
        "answerable": False,
    }
    make_source(
        [
            question("a", "Paris is the capital of France"),
            question("a", "2 MB"),
            unanswerable_with_gold,
        ]
    )
    assert build_eval.main() == 1
    out = capsys.readouterr().out
    assert "duplicate qid a" in out
    assert "unanswerable but has gold quotes" in out


def test_questions_file_roundtrips(corpus):
    assert [q["qid"] for q in load_questions("test")] == ["q1", "q2", "q3"]
    assert (
        json.loads((config.EVAL_DIR / "questions.jsonl").read_text().splitlines()[0])["qid"] == "q1"
    )
