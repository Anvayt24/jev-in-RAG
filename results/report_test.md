# Jev reranking benchmark report (split: test)

Questions: 107 (92 answerable, 15 unanswerable). Model: `typesafe/jev-1.13`; candidates per question: 30.

## Candidate recall@30 (ceiling for all rerankers)

Gold chunk present in the candidate list for **90.2%** of answerable questions (83/92).

Candidate-list integrity: OK, every reranker saw byte-identical candidates

## Ranking quality: All answerable questions

Mean % with 95% bootstrap CI over questions.

| reranker | n | hit@1 | hit@3 | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| none | 92 | 47.8 [38.0, 57.6] | 66.3 [56.5, 75.0] | 73.9 [64.1, 82.6] | 73.9 [64.1, 82.6] | 59.3 [51.1, 67.5] | 61.8 [54.2, 69.5] |
| flashrank | 92 | 55.4 [45.7, 66.3] | 72.8 [63.0, 81.5] | 78.3 [69.6, 87.0] | 78.3 [69.6, 87.0] | 65.5 [57.1, 74.2] | 67.1 [59.2, 75.2] |
| cohere | 92 | 77.2 [68.5, 85.9] | 89.1 [82.6, 94.6] | 89.1 [82.6, 94.6] | 89.1 [82.6, 94.6] | 82.7 [75.2, 89.3] | 80.8 [73.8, 87.0] |
| cohere_pro | 92 | 78.3 [69.6, 85.9] | 89.1 [82.6, 94.6] | 89.1 [82.6, 94.6] | 89.1 [82.6, 94.6] | 83.4 [76.2, 89.7] | 81.3 [74.3, 87.5] |
| jev_pair | 92 | 82.6 [73.9, 90.2] | 90.2 [83.7, 95.7] | 90.2 [83.7, 95.7] | 89.7 [83.2, 95.7] | 86.4 [79.9, 92.4] | 83.7 [77.2, 89.7] |

## Ranking quality: Reranking-addressable subset (gold in candidates)

Mean % with 95% bootstrap CI over questions.

| reranker | n | hit@1 | hit@3 | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| none | 83 | 53.0 [42.2, 63.9] | 73.5 [63.9, 81.9] | 81.9 [72.3, 89.2] | 81.9 [72.3, 89.2] | 65.7 [57.2, 73.8] | 68.5 [61.0, 75.7] |
| flashrank | 83 | 61.4 [50.6, 72.3] | 80.7 [71.1, 89.2] | 86.7 [79.5, 94.0] | 86.7 [79.5, 94.0] | 72.6 [64.5, 80.3] | 74.4 [66.9, 81.4] |
| cohere | 83 | 85.5 [78.3, 92.8] | 98.8 [96.4, 100.0] | 98.8 [96.4, 100.0] | 98.8 [96.4, 100.0] | 91.6 [87.1, 95.8] | 89.5 [85.3, 93.6] |
| cohere_pro | 83 | 86.7 [79.5, 94.0] | 98.8 [96.4, 100.0] | 98.8 [96.4, 100.0] | 98.8 [96.4, 100.0] | 92.4 [87.9, 96.4] | 90.1 [85.9, 94.1] |
| jev_pair | 83 | 91.6 [85.5, 96.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.4 [98.2, 100.0] | 95.8 [92.8, 98.2] | 92.8 [89.5, 95.9] |

## Paired difference in nDCG@10 (points, 95% CI), addressable subset

| A | B | A - B |
|---|---|---|
| jev_pair | none | +24.3 [+17.6, +31.8] |
| jev_pair | flashrank | +18.4 [+11.7, +26.0] |
| jev_pair | cohere | +3.3 [-0.4, +7.3] |
| jev_pair | cohere_pro | +2.7 [-0.6, +6.5] |

## nDCG@10 by question type (addressable subset)

| reranker | fact | multi | paraphrase | table |
|---|---|---|---|---|
| none | 76.8 (n=30) | 51.4 (n=10) | 76.3 (n=27) | 50.5 (n=16) |
| flashrank | 86.8 (n=30) | 55.5 (n=10) | 83.7 (n=27) | 47.5 (n=16) |
| cohere | 96.3 (n=30) | 64.1 (n=10) | 97.3 (n=27) | 79.8 (n=16) |
| cohere_pro | 96.3 (n=30) | 64.1 (n=10) | 98.6 (n=27) | 80.6 (n=16) |
| jev_pair | 95.1 (n=30) | 66.3 (n=10) | 98.6 (n=27) | 95.4 (n=16) |

## Latency and cost per query

Jev pair latency is the slowest single request (all pair calls assumed concurrent); sequential sum is shown too. Cohere/FlashRank latency is the rerank call only.

| reranker | p50 latency (s) | p95 latency (s) | mean input tokens | mean cost / query (USD) |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0 | 0.000000 |
| flashrank | 2.730 | 4.672 | 0 | 0.000000 |
| cohere | 1.116 | 3.795 | 0 | 0.000000 |
| cohere_pro | 1.345 | 6.595 | 0 | 0.000000 |
| jev_pair | 1.538 | 6.271 | 23115 | 0.000971 |

Jev pair sequential-sum latency per query: median 19.22s (30 requests/query).

## Jev-specific analyses

**jev_pair**: 0.3% of candidate scores >= 0.99, 10.3% <= 0.01; top score tied within a query in 12.3% of queries (ties broken by RRF rank).

Calibration of P(yes) against 'chunk contains gold quote' (ECE 0.080; label noise caveat: other chunks may also answer):

| mean P(yes) | fraction gold | n |
|---|---|---|
| 0.03 | 0.000 | 2210 |
| 0.13 | 0.005 | 206 |
| 0.24 | 0.013 | 79 |
| 0.34 | 0.000 | 49 |
| 0.45 | 0.000 | 23 |
| 0.54 | 0.059 | 17 |
| 0.64 | 0.000 | 13 |
| 0.73 | 0.050 | 20 |
| 0.85 | 0.219 | 32 |
| 0.97 | 0.748 | 111 |

- jev_pair scoring variant `evidence`: nDCG@10 92.8 [89.5, 95.9] (n=83)
- jev_pair scoring variant `relevant`: nDCG@10 91.0 [87.2, 94.6] (n=83)
- jev_pair scoring variant `mean`: nDCG@10 93.2 [89.9, 96.2] (n=83)

## Answerability gate (AUROC: answerable vs unanswerable)

**Not run.** The gate needs extra Jev calls and the OpenRouter account ran out of credits; the Jev probes (run-to-run stability, candidate-order sensitivity) were skipped for the same reason, and `jev_pack` covers only 56 of the 107 test questions. Conclusions about Jev as a gate are therefore untested here.
