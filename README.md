# jevbench: is Jev useful as a reranker in RAG?

A small, repeatable benchmark. One conventional hybrid-retrieval pipeline produces a **frozen** list of 30
candidate passages per question; each reranker then re-orders the *same* candidates, and we score them
against labeled evidence. Jev (TypeSafe AI's typed decision model, reached through OpenRouter) is compared
with conventional rerankers, and also tested as an "is this answerable from these passages?" gate.

## Rerankers compared

| id | what | tier |
|---|---|---|
| `none` | hybrid BM25 + dense, fused with RRF; no reranker | baseline |
| `flashrank` | FlashRank `ms-marco-MiniLM-L-12-v2` cross-encoder (local ONNX, CPU) | simple / very common |
| `cohere` | Cohere Rerank 4 **Fast** (`rerank-v4.0-fast`) | mid |
| `cohere_pro` | Cohere Rerank 4 **Pro** (`rerank-v4.0-pro`) | strong hosted |
| `jev_pair` | Jev, one request per (query, passage): `evidence` + `relevant` yes/no questions; score = P(evidence) | under test (primary) |
| `jev_pack` | Jev, one request per query with one yes/no question per candidate (the article's method) | under test (secondary) |

`BAAI/bge-reranker-v2-m3` was originally planned as the "medium" tier. It was **dropped** (2.2 GB weights, the
Hugging Face download repeatedly stalled and the model is too heavy for this machine). Cohere Fast now fills the
middle tier.

## Pipeline (kept deliberately simple)

PDF -> `pymupdf4llm` markdown (keeps tables) -> per-page recursive chunks (<=480 tokens, 50-token overlap) ->
dense `BAAI/bge-small-en-v1.5` (FAISS) + BM25 -> Reciprocal Rank Fusion (k=60) -> top 30 candidates, cached in
`cache/candidates/`. 7 public documents (arXiv papers, NIST AI RMF and CSF 2.0, Raspberry Pi Pico datasheet,
a 33-page Berkshire Hathaway 2023 annual-report excerpt), 580 chunks.

## Documents

The PDFs are not committed (third-party material). Put them in `data/raw/` under these names, then run ingest:

| file | source |
|---|---|
| `rag_lewis_2020.pdf` | https://arxiv.org/abs/2005.11401 |
| `lost_in_the_middle_2023.pdf` | https://arxiv.org/abs/2307.03172 |
| `self_rag_2023.pdf` | https://arxiv.org/abs/2310.11511 |
| `nist_ai_rmf_1_0.pdf` | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf |
| `nist_csf_2_0.pdf` | https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf |
| `raspberry_pi_pico_datasheet.pdf` | https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf |
| `berkshire_2023_excerpt.pdf` | pages 5-19, 59-63, 66-67, 81-82, 94-98 and 136-139 of https://www.berkshirehathaway.com/2023ar/2023ar.pdf |

## Eval set

`data/eval/questions.jsonl`: 162 questions (55 dev / 107 test, hash-assigned), written against the documents with
an exact gold evidence quote each. Types: fact, table lookup, paraphrase, multi-passage (`mode: all`), and
unanswerable (24, no gold). Gold chunks are found by quote matching (`labels.py`), so labels survive re-chunking.
`data/eval/spot_check.md` lists 20 items for human review. The dev split was reserved for tuning Jev's
question wording; in the end the wording (`rerankers/jev_questions.py`, version `v1`) was **not tuned at all**.

## Metrics

Hit@1/3/5, Recall@5, MRR, nDCG@10 (quote-level gain, so overlapping chunks do not inflate scores), 95%
bootstrap CIs over questions, paired bootstrap differences, per-question-type breakdown, latency and cost per
query, plus Jev-specific checks (score saturation/ties, calibration, order sensitivity, run-to-run stability,
answerability AUROC). Everything is reported on the reranking-addressable subset (gold present in the 30
candidates) as well as on all answerable questions, because no reranker can recover a passage retrieval missed.

## Run it

```bash
uv sync                                    # everything installs into .venv
uv run python -m jevbench.ingest           # parse, chunk, embed, index (once)
uv run python -m jevbench.build_eval       # validate gold quotes, assign splits
uv run python -m jevbench.retrieve         # freeze candidates (once)
uv run python -m jevbench.run_rerank --rerankers none flashrank cohere cohere_pro jev_pair jev_pack
uv run python -m jevbench.run_rerank --rerankers jev_pair_rerun jev_pack_shuffled --split test   # probes
uv run python -m jevbench.gate --rerankers flashrank cohere cohere_pro jev_pair --split test
uv run python -m jevbench.evaluate --split test
uv run pytest
```

Keys go in `.env` (see `.env.example`): `OPENROUTER_API_KEY` (Jev via `https://openrouter.ai/api`,
model `typesafe/jev-1.13`) and `COHERE_API_KEY`. All runs are resumable; raw Jev responses are cached under
`cache/jev_raw/`.

## Status and results

Final report: `results/report_test.md` (107 test questions; 92 answerable). Side report with the partial
packed run: `results/report_test_with_pack_partial.md`.

**Completed:** all rerankers except `jev_pack` on every question; `jev_pair` on all 162.
**Not run (stopped by the user after the OpenRouter account ran out of credits):** `jev_pack` on 75 questions
(it covers 87 of 162, 56 of 107 test), the Jev stability and order-sensitivity probes, and the answerability gate.
Those commands in "Run it" above are therefore untested in practice, and nothing here says whether Jev works as a gate.

nDCG@10, % (reranking-addressable subset: gold chunk is among the 30 candidates, n=83):

| reranker | nDCG@10 | hit@1 | table lookups (n=16) |
|---|---|---|---|
| none (RRF) | 68.5 | 53.0 | 50.5 |
| FlashRank MiniLM | 74.4 | 61.4 | 47.5 |
| Cohere Rerank 4 Fast | 89.5 | 85.5 | 79.8 |
| Cohere Rerank 4 Pro | 90.1 | 86.7 | 80.6 |
| **Jev pair** | **92.8** | **91.6** | **95.4** |

Paired nDCG@10 difference for Jev pair (95% CI): +18.4 [+11.7, +26.0] vs FlashRank, +3.3 [-0.4, +7.3] vs Cohere
Fast, +2.7 [-0.6, +6.5] vs Cohere Pro. So Jev is clearly better than the simple cross-encoder and *not
distinguishably* different from Cohere overall; its advantage is concentrated in table lookups (n=16, suggestive only).

On the 56-question overlap, `jev_pack` (one request per query) scored 92.1 vs 92.9 for `jev_pair`, at 0.74 s and
$0.00058 per query against 1.5 s (slowest of 30 concurrent requests; 19 s if sequential) and $0.00097.

Outcome against the pre-registered hypotheses: H1 (Jev about FlashRank, below Cohere) was **not** supported, Jev
matched or beat both; H2 (packed worse than pair) was **not** supported on the small overlap; H3 (answerability
gate) is **untested**; H4 (latency competitive with Cohere) holds only for the packed mode or with concurrent requests.

## Caveats

- One corpus, questions authored by the benchmark builder, n=92 answerable test questions: CIs are wide.
- Jev's question asks whether a passage *contains the facts needed to answer*, which matches how gold is
  defined; cross-encoders are trained for relevance. A relevance-only Jev variant is reported alongside.
- Candidate recall@30 is 90% overall but only ~72% on table lookups; that retrieval ceiling applies to all rerankers.
