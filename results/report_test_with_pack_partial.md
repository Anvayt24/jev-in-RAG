# Jev reranking benchmark report (split: test)

Questions: 56 (46 answerable, 10 unanswerable). Model: `typesafe/jev-1.13`; candidates per question: 30.

Questions scored per reranker (of 107 in this split): none 107, flashrank 107, cohere 107, cohere_pro 107, jev_pair 107, jev_pack 56. Every table below compares only the 56 questions that all of them have scored.

## Candidate recall@30 (ceiling for all rerankers)

Gold chunk present in the candidate list for **93.5%** of answerable questions (43/46).

Candidate-list integrity: OK, every reranker saw byte-identical candidates

## Ranking quality: All answerable questions

Mean % with 95% bootstrap CI over questions.

| reranker | n | hit@1 | hit@3 | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| none | 46 | 47.8 [34.8, 60.9] | 67.4 [52.2, 80.4] | 73.9 [60.9, 87.0] | 73.9 [60.9, 87.0] | 59.9 [47.9, 70.9] | 62.2 [51.1, 72.3] |
| flashrank | 46 | 50.0 [34.8, 63.0] | 76.1 [63.0, 87.0] | 80.4 [69.6, 91.3] | 80.4 [69.6, 91.3] | 63.8 [52.7, 74.9] | 65.5 [55.0, 75.5] |
| cohere | 46 | 73.9 [60.9, 84.8] | 91.3 [82.6, 97.8] | 91.3 [82.6, 97.8] | 91.3 [82.6, 97.8] | 81.6 [71.7, 90.2] | 79.8 [70.7, 88.0] |
| cohere_pro | 46 | 80.4 [67.4, 91.3] | 91.3 [82.6, 97.8] | 91.3 [82.6, 97.8] | 91.3 [82.6, 97.8] | 85.2 [75.7, 93.6] | 82.5 [73.6, 90.8] |
| jev_pair | 46 | 87.0 [76.1, 95.7] | 93.5 [84.8, 100.0] | 93.5 [84.8, 100.0] | 93.5 [84.8, 100.0] | 90.2 [81.5, 96.7] | 86.9 [78.6, 93.7] |
| jev_pack | 46 | 84.8 [73.9, 93.5] | 93.5 [84.8, 100.0] | 93.5 [84.8, 100.0] | 93.5 [84.8, 100.0] | 89.1 [80.4, 95.7] | 86.1 [77.9, 93.4] |

## Ranking quality: Reranking-addressable subset (gold in candidates)

Mean % with 95% bootstrap CI over questions.

| reranker | n | hit@1 | hit@3 | hit@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| none | 43 | 51.2 [34.9, 67.4] | 72.1 [58.1, 83.7] | 79.1 [65.1, 90.7] | 79.1 [65.1, 90.7] | 64.0 [52.0, 75.5] | 66.5 [55.9, 76.6] |
| flashrank | 43 | 53.5 [37.2, 67.4] | 81.4 [69.8, 93.0] | 86.0 [74.4, 95.3] | 86.0 [74.4, 95.3] | 68.2 [56.5, 79.1] | 70.1 [59.7, 80.0] |
| cohere | 43 | 79.1 [65.1, 90.7] | 97.7 [93.0, 100.0] | 97.7 [93.0, 100.0] | 97.7 [93.0, 100.0] | 87.3 [78.8, 95.0] | 85.4 [78.1, 91.9] |
| cohere_pro | 43 | 86.0 [74.4, 95.3] | 97.7 [93.0, 100.0] | 97.7 [93.0, 100.0] | 97.7 [93.0, 100.0] | 91.2 [83.6, 97.7] | 88.3 [81.3, 94.4] |
| jev_pair | 43 | 93.0 [83.7, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 96.5 [91.9, 100.0] | 92.9 [88.5, 97.3] |
| jev_pack | 43 | 90.7 [81.4, 97.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 95.3 [90.7, 98.8] | 92.1 [87.7, 96.5] |

## Paired difference in nDCG@10 (points, 95% CI), addressable subset

| A | B | A - B |
|---|---|---|
| jev_pair | none | +26.4 [+17.6, +36.8] |
| jev_pair | flashrank | +22.9 [+13.2, +33.1] |
| jev_pair | cohere | +7.5 [+1.7, +14.5] |
| jev_pair | cohere_pro | +4.7 [-0.6, +11.3] |
| jev_pack | none | +25.5 [+15.5, +37.0] |
| jev_pack | flashrank | +22.0 [+11.7, +32.7] |
| jev_pack | cohere | +6.7 [+0.9, +13.6] |
| jev_pack | cohere_pro | +3.8 [-1.7, +10.5] |

## nDCG@10 by question type (addressable subset)

| reranker | fact | multi | paraphrase | table |
|---|---|---|---|---|
| none | 87.2 (n=13) | 48.2 (n=5) | 68.1 (n=12) | 51.5 (n=13) |
| flashrank | 88.2 (n=13) | 64.5 (n=5) | 85.5 (n=12) | 39.8 (n=13) |
| cohere | 94.3 (n=13) | 61.3 (n=5) | 93.8 (n=12) | 77.9 (n=13) |
| cohere_pro | 97.2 (n=13) | 61.3 (n=5) | 100.0 (n=12) | 78.9 (n=13) |
| jev_pair | 100.0 (n=13) | 61.3 (n=5) | 96.9 (n=12) | 94.3 (n=13) |
| jev_pack | 94.3 (n=13) | 61.3 (n=5) | 96.9 (n=12) | 97.2 (n=13) |

## Latency and cost per query

Jev pair latency is the slowest single request (all pair calls assumed concurrent); sequential sum is shown too. Cohere/FlashRank latency is the rerank call only.

| reranker | p50 latency (s) | p95 latency (s) | mean input tokens | mean cost / query (USD) |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0 | 0.000000 |
| flashrank | 2.531 | 3.263 | 0 | 0.000000 |
| cohere | 1.184 | 5.276 | 0 | 0.000000 |
| cohere_pro | 1.516 | 9.905 | 0 | 0.000000 |
| jev_pair | 1.574 | 4.747 | 23732 | 0.000997 |
| jev_pack | 0.765 | 1.520 | 13793 | 0.000579 |

Jev pair sequential-sum latency per query: median 19.40s (30 requests/query).

## Jev-specific analyses

**jev_pair**: 0.4% of candidate scores >= 0.99, 10.3% <= 0.01; top score tied within a query in 17.9% of queries (ties broken by RRF rank).

Calibration of P(yes) against 'chunk contains gold quote' (ECE 0.081; label noise caveat: other chunks may also answer):

| mean P(yes) | fraction gold | n |
|---|---|---|
| 0.04 | 0.000 | 1104 |
| 0.13 | 0.000 | 101 |
| 0.24 | 0.000 | 46 |
| 0.34 | 0.000 | 26 |
| 0.45 | 0.000 | 8 |
| 0.56 | 0.000 | 6 |
| 0.64 | 0.000 | 8 |
| 0.73 | 0.100 | 10 |
| 0.84 | 0.263 | 19 |
| 0.97 | 0.788 | 52 |

**jev_pack**: 0.1% of candidate scores >= 0.99, 0.9% <= 0.01; top score tied within a query in 8.9% of queries (ties broken by RRF rank).

Calibration of P(yes) against 'chunk contains gold quote' (ECE 0.068; label noise caveat: other chunks may also answer):

| mean P(yes) | fraction gold | n |
|---|---|---|
| 0.04 | 0.000 | 1172 |
| 0.13 | 0.000 | 76 |
| 0.24 | 0.000 | 27 |
| 0.35 | 0.000 | 19 |
| 0.43 | 0.000 | 7 |
| 0.54 | 0.000 | 12 |
| 0.66 | 0.000 | 3 |
| 0.76 | 0.250 | 4 |
| 0.86 | 0.400 | 10 |
| 0.96 | 0.840 | 50 |

- jev_pair scoring variant `evidence`: nDCG@10 92.9 [88.5, 97.3] (n=43)
- jev_pair scoring variant `relevant`: nDCG@10 90.1 [84.7, 95.3] (n=43)
- jev_pair scoring variant `mean`: nDCG@10 92.1 [87.6, 96.5] (n=43)

Candidate-order sensitivity: not run (no shuffled-order results found).

Run-to-run stability: not run (no repeated-request results found).

Pair vs packed agreement: mean Spearman 0.745 over 55 queries.

## Answerability gate (AUROC: answerable vs unanswerable)

**Not run.** No gate results were found in `results/gate/` for these questions, so this report makes no claim about Jev as an answerability gate.
