# Methodology

How the benchmark is built, what each number means, and where it can mislead.

## Design

The comparison isolates one variable. A single retrieval pipeline produces 30 candidate passages
per question, **once**, and writes them to disk with a hash. Every reranker then re-orders those
same candidates, and the result is scored against labeled evidence. Nothing upstream of the
reranker differs between systems, and the report refuses to run if a stored score belongs to a
different candidate list than the one frozen now.

## Documents and chunking

Seven public documents with deliberately different shapes: three arXiv papers (prose, results
tables), NIST AI RMF 1.0 and CSF 2.0 (regulatory text with many near-duplicate subcategories),
the Raspberry Pi Pico datasheet (measurement tables), and 33 pages of Berkshire Hathaway's 2023
annual report (shareholder letter plus financial statements). `python -m jevbench.download_docs`
fetches them.

`pymupdf4llm` converts each page to markdown so tables survive. Each page is split with a
recursive splitter (paragraph, line, sentence, word) into chunks of at most 480 tokens, measured
with the embedding model's own tokenizer, with a 50-token overlap. That yields 580 chunks.

## Retrieval

BM25 and dense retrieval (`BAAI/bge-small-en-v1.5`, exact inner-product search in FAISS) each
return their top 50. The two rankings are merged with Reciprocal Rank Fusion (k = 60) and cut to
the top 30. The dense and BM25 ranks of every candidate are stored alongside it.

## Evaluation set

162 questions: 50 fact, 39 paraphrase, 32 table lookup, 17 multi-passage and 24 unanswerable.

- **Gold evidence is a verbatim quote** from the document, not a chunk id. Gold chunks are found
  by matching the quote (lower-cased, punctuation removed) against the chunks of that document,
  so the labels survive a change of chunking. `build_eval` fails if any quote matches nothing.
- **`mode: any`** marks alternative evidence (finding one quote is enough). **`mode: all`** marks
  multi-passage questions that need every quote.
- **Unanswerable questions** are on topic but have no answer in the corpus. They are used only by
  the answerability gate.
- **Split:** a hash of the question id assigns about 30% to `dev` and 70% to `test` (55 / 107).
  `dev` was set aside for tuning Jev's question wording. In the end the wording (version `v1`,
  `src/jevbench/rerankers/jev_questions.py`) was not changed after the first draft. All reported
  results use `test`.
- **Authorship:** the questions and quotes were drafted with an LLM assistant and then checked
  mechanically (the verbatim-quote rule above). Beyond that, label quality has not been audited
  independently; `data/eval/spot_check.md` is a 20-question sheet for doing so.

## Rerankers

| id | implementation |
|---|---|
| `none` | The fused RRF order, unchanged. |
| `flashrank` | FlashRank `ms-marco-MiniLM-L-12-v2`, an ONNX cross-encoder on CPU. |
| `cohere`, `cohere_pro` | Cohere Rerank 4, models `rerank-v4.0-fast` and `rerank-v4.0-pro`. |
| `jev_pair` | One Jev request per (query, passage). |
| `jev_pack` | One Jev request per query, with one question per candidate. |

Jev is reached through OpenRouter (`https://openrouter.ai/api`) with the official `typesafe-sdk`.
The model is pinned to `typesafe/jev-1.13` (the API reports `typesafe/jev-1.13-20260917`).

Jev answers *noul* questions, yes/no questions that return a probability. For `jev_pair` the state
is `{query, passage}` and two questions are asked in the same request:

- `evidence`: "Does `passage` contain information that directly answers `query`?" The criteria say
  that "true" means the passage states the specific facts, numbers or definitions needed, and
  "false" means it does not, even if it covers a related topic.
- `relevant`: "Is `passage` about the same specific subject as `query`?"

The ranking score is P(`evidence`). The two other ways of scoring a pair (P(`relevant`) and the mean
of both) are stored and reported as variants. `jev_pack` sends all 30 passages as `C1`..`C30` in one
state and asks one `evidence`-style question per id. Ties are broken by the RRF rank, and every raw
response is cached so a run can be audited or repeated without new requests.

A larger open-weight cross-encoder (`BAAI/bge-reranker-v2-m3`) was considered as the middle tier
and left out because of its 2.2 GB weights; Cohere Fast fills that tier.

## Metrics

All metrics are computed per question and averaged.

- **Hit@k:** whether any gold chunk is in the top k.
- **MRR:** reciprocal rank of the first gold chunk.
- **Recall@5:** for `any` questions, whether some quote is covered in the top 5; for `all`
  questions, the fraction of quotes covered.
- **nDCG@10:** gain is per *quote*, not per chunk. A chunk earns gain only for a quote not yet
  covered higher in the ranking, so overlapping chunks that repeat the same text cannot inflate the
  score. `any` questions need one unit of gain, `all` questions need one per quote.
- **Intervals:** percentile bootstrap over questions (2000 resamples, fixed seed). Differences
  between rerankers use a *paired* bootstrap, resampling the same questions for both.
- **Reranking-addressable subset:** the questions whose gold chunk is among the 30 candidates. No
  reranker can recover a passage retrieval missed, so the headline numbers use this subset. Tables
  for all answerable questions are in the report too.
- **Latency:** for Cohere and FlashRank, the rerank call. For `jev_pair`, the slowest of the 30
  requests (what a system sending them concurrently would see); the sequential sum is reported
  as well. Cohere cost is not recorded (the runs used a free trial key). Jev cost is the
  `usage.cost` that OpenRouter returns, from input tokens only (output is free).

## Jev diagnostics

The report also contains checks specific to a probability-valued model: score saturation and
ties, calibration of P(yes) against the gold labels, the three scoring variants, pair-versus-packed
agreement, run-to-run stability, sensitivity to candidate order (`jev_pack` with shuffled input),
and an answerability gate (AUROC of Jev's P("answerable") over a reranker's top-5 passages,
answerable versus unanswerable questions). Stability, order sensitivity and the gate are implemented
and unit-tested but have **not been run** (see the README), and the report says so instead of
omitting them.

## Reproducibility

- Candidate retrieval and scoring write to disk and resume where they stopped.
- `results/scores/*.jsonl` hold every stored score; `python -m jevbench.evaluate` rebuilds the
  report from them, and rebuilding the candidates requires only the local embedding model.
- Reruns call hosted APIs whose models can change. Two identical Jev requests in a smoke test
  returned 0.75 and 0.76, so small run-to-run differences are expected.

## Threats to validity

- **Size and scope.** Seven documents and 92 answerable test questions. The intervals are wide,
  and the by-question-type numbers (n = 10 to 30) are descriptive only.
- **Label alignment.** Gold means "this chunk contains the quoted evidence". Jev is asked whether a
  passage contains the facts needed to answer, which mirrors that definition, while cross-encoders
  are trained for relevance. The relevance-only Jev variant (91.0 against 92.8 nDCG@10) suggests
  the wording is not what drives the result.
- **Question origin.** LLM-drafted questions can share phrasing with their source passage, which
  helps lexical methods, and may be easier or harder for an LLM-style reranker than human ones.
- **Retrieval ceiling.** 92% of answerable questions have their gold chunk among the candidates,
  but only 72% of table lookups do, because both retrievers handle numeric tables poorly.
- **One run, one model version.** No repeated trials, and a closed model reached through a gateway.
