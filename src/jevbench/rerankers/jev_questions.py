"""Frozen Jev question wording. Tuned on the dev split only; bump WORDING_VERSION on any change
(the version is part of every cache key so old responses are never reused for new wording)."""
from __future__ import annotations

WORDING_VERSION = "v1"

EVIDENCE_CRITERIA = {
    "true": "The passage states the specific facts, numbers, or definitions needed to answer the query.",
    "false": "The passage does not state what is needed to answer the query, even if it covers a related topic.",
}
RELEVANT_CRITERIA = {
    "true": "The passage discusses the specific subject the query asks about.",
    "false": "The passage is about a different subject than the query.",
}
ANSWERABLE_CRITERIA = {
    "true": "Together the passages state everything needed to answer the query.",
    "false": "The passages are missing information needed to answer the query.",
}


def pair_state(query: str, passage: str) -> dict:
    return {"query": query, "passage": passage}


def pair_questions() -> dict:
    return {
        "evidence": {
            "type": "noul",
            "instructions": "Does `passage` contain information that directly answers `query`?",
            "criteria": EVIDENCE_CRITERIA,
        },
        "relevant": {
            "type": "noul",
            "instructions": "Is `passage` about the same specific subject as `query`?",
            "criteria": RELEVANT_CRITERIA,
        },
    }


def pack_state(query: str, passages: dict[str, str]) -> dict:
    return {"query": query, "passages": passages}


def pack_questions(ids: list[str]) -> dict:
    return {
        pid: {
            "type": "noul",
            "instructions": f"Does `passages.{pid}` contain information that directly answers `query`?",
            "criteria": EVIDENCE_CRITERIA,
        }
        for pid in ids
    }


def answerable_questions() -> dict:
    return {
        "answerable": {
            "type": "noul",
            "instructions": "Can `query` be answered using only the information in `passages`?",
            "criteria": ANSWERABLE_CRITERIA,
        }
    }
