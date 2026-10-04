"""Jev adapters against a fake async client (no network)."""

import hashlib
from types import SimpleNamespace

from jevbench.rerankers import jev_questions as jq
from jevbench.rerankers.jev_rr import JevPack, JevPair, jev_answerable


def _p(text: str) -> float:
    """Deterministic fake probability derived from the passage text."""
    return int(hashlib.md5(text.encode()).hexdigest()[:4], 16) / 0xFFFF


class FakeClient:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def system_one(self, state, questions):
        self.log.append((state, questions))
        answers = {}
        if "passage" in state:  # pair mode
            for name in questions:
                answers[name] = SimpleNamespace(noul=_p(name + state["passage"]))
        elif list(questions) == ["answerable"]:
            answers["answerable"] = SimpleNamespace(noul=0.42)
        else:  # pack mode: question id -> passage id
            for pid in questions:
                answers[pid] = SimpleNamespace(noul=_p("evidence" + state["passages"][pid]))
        raw = SimpleNamespace(json=lambda: {"usage": {"cost": 1e-5}})
        return SimpleNamespace(
            answers=answers,
            model="typesafe/jev-1.13-test",
            usage=SimpleNamespace(input_tokens=100, output_tokens=5),
            raw_http_response=raw,
        )


def _cands(n=7):
    return [{"chunk_id": f"d:p{i}:c0", "text": f"passage number {i}"} for i in range(n)]


def test_pair_scores_align_with_candidates_and_record_cost():
    log = []
    rr = JevPair(use_cache=False, client_factory=lambda: FakeClient(log))
    cands = _cands()
    res = rr.rerank("what?", cands)
    assert len(res.scores) == len(cands) == len(log)
    assert res.scores == [_p("evidence" + c["text"]) for c in cands]
    assert res.cost_usd == 1e-5 * len(cands)
    assert res.input_tokens == 100 * len(cands)
    assert set(res.meta["variants"]) == {"evidence", "relevant", "mean"}
    # the query and passage are sent as named state fields, with the frozen question wording
    state, questions = log[0]
    assert state == {"query": "what?", "passage": cands[0]["text"]} or state["query"] == "what?"
    assert questions == jq.pair_questions()


def test_pair_variant_selection():
    log = []
    cands = _cands(3)
    rel = JevPair(
        variant="relevant", use_cache=False, client_factory=lambda: FakeClient(log)
    ).rerank("q", cands)
    assert rel.scores == rel.meta["variants"]["relevant"]
    mean = JevPair(variant="mean", use_cache=False, client_factory=lambda: FakeClient(log)).rerank(
        "q", cands
    )
    assert mean.scores == mean.meta["variants"]["mean"]


def test_pack_single_request_maps_scores_back():
    log = []
    cands = _cands(5)
    res = JevPack(use_cache=False, client_factory=lambda: FakeClient(log)).rerank("q", cands)
    assert len(log) == 1
    _, questions = log[0]
    assert list(questions) == [f"C{i}" for i in range(1, 6)]
    assert res.scores == [_p("evidence" + c["text"]) for c in cands]


def test_pack_shuffle_changes_order_in_request_but_not_score_alignment():
    log = []
    cands = _cands(8)
    res = JevPack(shuffle_seed=3, use_cache=False, client_factory=lambda: FakeClient(log)).rerank(
        "q", cands
    )
    state, _ = log[0]
    sent = [state["passages"][f"C{i}"] for i in range(1, 9)]
    assert sent != [c["text"] for c in cands]  # order actually shuffled
    assert sorted(sent) == sorted(c["text"] for c in cands)
    assert res.scores == [_p("evidence" + c["text"]) for c in cands]  # still aligned to input order


def test_pack_batches():
    log = []
    cands = _cands(7)
    res = JevPack(batch_size=3, use_cache=False, client_factory=lambda: FakeClient(log)).rerank(
        "q", cands
    )
    assert len(log) == 3 and res.meta["n_requests"] == 3
    assert res.scores == [_p("evidence" + c["text"]) for c in cands]


def test_answerable_gate():
    out = jev_answerable("q", ["a", "b"], use_cache=False, client_factory=lambda: FakeClient([]))
    assert out["p"] == 0.42 and out["tokens"] == 100
