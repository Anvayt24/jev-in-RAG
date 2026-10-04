"""Jev as a reranker, reached through OpenRouter with the official typesafe-sdk.

Two modes (see plan):
  * JevPair  - one request per (query, passage); the vendor cookbook's reranking recipe.
  * JevPack  - one request per query; one noul question per candidate (the article's method).
Plus `jev_answerable` for the "can the retained passages answer this?" gate.

Every raw response is cached under cache/jev_raw/ keyed by (mode, model, wording, inputs, cache_tag).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time

from .. import config
from . import jev_questions as jq
from .base import Reranker, RerankResult


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


class _Cache:
    def __init__(self, mode: str, enabled: bool = True) -> None:
        self.dir = config.JEV_RAW_DIR / mode
        self.enabled = enabled
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        p = self.dir / f"{key}.json"
        if self.enabled and p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, payload: dict) -> None:
        if self.enabled:
            (self.dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


def make_async_client(api_key: str | None = None):
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    return AsyncTypeSafeClient(
        api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        base_url=config.JEV_BASE_URL,
        model=config.JEV_MODEL,
        retry=RetryPolicy(
            max_retries=6, backoff_initial=1.0, backoff_max=30.0, timeout=120.0,
            http_statuses={429, 500, 502, 503, 504, 529},
        ),
    )


def _cost_and_tokens(resp) -> tuple[float, int]:
    tokens = int(getattr(resp.usage, "input_tokens", 0) or 0)
    try:
        cost = float(resp.raw_http_response.json()["usage"]["cost"])
    except Exception:
        cost = tokens * config.JEV_PRICE_PER_INPUT_TOKEN
    return cost, tokens


async def _call(client, state, questions) -> dict:
    """One System One request -> {'noul': {name: p}, 'cost', 'tokens', 'latency_s', 'model'}."""
    t0 = time.perf_counter()
    resp = await client.system_one(state, questions)
    dt = time.perf_counter() - t0
    cost, tokens = _cost_and_tokens(resp)
    return {
        "noul": {k: float(v.noul) for k, v in resp.answers.items()},
        "cost": cost,
        "tokens": tokens,
        "latency_s": dt,
        "model": resp.model,
    }


class JevPair(Reranker):
    """Pointwise: score = P(yes) of the `evidence` question for each (query, passage) pair."""

    name = "jev_pair"

    def __init__(self, variant: str = "evidence", concurrency: int = 8, cache_tag: str = "",
                 use_cache: bool = True, client_factory=make_async_client) -> None:
        assert variant in ("evidence", "relevant", "mean")
        self.variant = variant
        self.concurrency = concurrency
        self.cache_tag = cache_tag
        self.cache = _Cache("pair", use_cache)
        self.client_factory = client_factory

    async def _run(self, query: str, candidates: list[dict]) -> list[dict]:
        sem = asyncio.Semaphore(self.concurrency)
        questions = jq.pair_questions()
        async with self.client_factory() as client:

            async def one(c: dict) -> dict:
                key = _key("pair", config.JEV_MODEL, jq.WORDING_VERSION, query, c["text"], self.cache_tag)
                hit = self.cache.get(key)
                if hit is not None:
                    return hit
                async with sem:
                    out = await _call(client, jq.pair_state(query, c["text"]), questions)
                self.cache.put(key, out)
                return out

            return await asyncio.gather(*(one(c) for c in candidates))

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        t0 = time.perf_counter()
        outs = asyncio.run(self._run(query, candidates))
        wall = time.perf_counter() - t0
        ev = [o["noul"]["evidence"] for o in outs]
        rel = [o["noul"]["relevant"] for o in outs]
        mean = [(a + b) / 2 for a, b in zip(ev, rel)]
        scores = {"evidence": ev, "relevant": rel, "mean": mean}[self.variant]
        return RerankResult(
            scores=scores,
            # latency = what a live system would see: all pair calls run concurrently
            latency_s=max(o["latency_s"] for o in outs) if outs else 0.0,
            input_tokens=sum(o["tokens"] for o in outs),
            cost_usd=sum(o["cost"] for o in outs),
            meta={
                "model": outs[0]["model"] if outs else None,
                "variants": {"evidence": ev, "relevant": rel, "mean": mean},
                "n_requests": len(outs),
                "wall_s_this_run": wall,
                "per_request_latency_s": [o["latency_s"] for o in outs],
            },
        )


class JevPack(Reranker):
    """Packed: one request per query, one noul question per candidate id."""

    name = "jev_pack"

    def __init__(self, batch_size: int | None = None, shuffle_seed: int | None = None,
                 cache_tag: str = "", use_cache: bool = True, client_factory=make_async_client) -> None:
        self.batch_size = batch_size  # None = all candidates in one request
        self.shuffle_seed = shuffle_seed
        self.cache_tag = cache_tag
        self.cache = _Cache("pack", use_cache)
        self.client_factory = client_factory

    async def _run(self, query: str, order: list[int], candidates: list[dict]) -> list[dict]:
        size = self.batch_size or len(order)
        batches = [order[i : i + size] for i in range(0, len(order), size)]
        async with self.client_factory() as client:

            async def one(batch: list[int]) -> dict:
                ids = [f"C{n + 1}" for n in range(len(batch))]
                passages = {pid: candidates[i]["text"] for pid, i in zip(ids, batch)}
                key = _key("pack", config.JEV_MODEL, jq.WORDING_VERSION, query,
                           "|".join(candidates[i]["chunk_id"] for i in batch), self.cache_tag)
                hit = self.cache.get(key)
                if hit is None:
                    hit = await _call(client, jq.pack_state(query, passages), jq.pack_questions(ids))
                    self.cache.put(key, hit)
                hit = dict(hit)
                hit["mapping"] = {pid: i for pid, i in zip(ids, batch)}
                return hit

            return await asyncio.gather(*(one(b) for b in batches))

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        order = list(range(len(candidates)))
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(order)
        outs = asyncio.run(self._run(query, order, candidates))
        scores = [0.0] * len(candidates)
        for o in outs:
            for pid, i in o["mapping"].items():
                scores[i] = o["noul"][pid]
        return RerankResult(
            scores=scores,
            latency_s=max(o["latency_s"] for o in outs) if outs else 0.0,
            input_tokens=sum(o["tokens"] for o in outs),
            cost_usd=sum(o["cost"] for o in outs),
            meta={"model": outs[0]["model"] if outs else None, "n_requests": len(outs)},
        )


def jev_answerable(query: str, passages: list[str], cache_tag: str = "", use_cache: bool = True,
                   client_factory=make_async_client) -> dict:
    """P(query is answerable from the given passages) -> {'p', 'cost', 'tokens', 'latency_s'}."""
    cache = _Cache("answerable", use_cache)
    key = _key("answerable", config.JEV_MODEL, jq.WORDING_VERSION, query, "\x1e".join(passages), cache_tag)
    hit = cache.get(key)
    if hit is not None:
        return hit

    async def run() -> dict:
        async with client_factory() as client:
            state = jq.pack_state(query, {f"P{i + 1}": t for i, t in enumerate(passages)})
            return await _call(client, state, jq.answerable_questions())

    out = asyncio.run(run())
    out["p"] = out["noul"]["answerable"]
    cache.put(key, out)
    return out
