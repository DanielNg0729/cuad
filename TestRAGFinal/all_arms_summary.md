# All arms — 102 CUAD contracts, gpt-4.1

## Under ContractEval's protocol (comparable to the paper)

| System | P | R | F1 | F2 | Jaccard | False rate | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ContractEval GPT-4.1 (full document, published)** | 0.595* | 0.694* | **0.641** | 0.672 | 0.472 | 0.071 | ~$50 |
| RAG RRF n20 top-8 | 0.649 | 0.625 | 0.637 | 0.629 | 0.488 | 0.049 | $18.20 |
| RAG RRF n10 top-5 | 0.666 | 0.571 | 0.615 | 0.588 | 0.469 | 0.054 | $12.68 |

\* derived from the paper's published F1/F2.

## Under this project's native scorer (per gold span, Jaccard >= 0.5)

| Arm | TP | FP | FN | P | R | F1 | F2 | AUPR | Jaccard | R@k | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RRF n20 top-8 | 1357 | 1720 | 1286 | 0.441 | 0.513 | 0.474 | 0.497 | 0.370 | 0.476 | 0.769 | $18.20 |
| RRF n10 top-5 | 1269 | 1449 | 1374 | 0.467 | 0.480 | 0.473 | 0.477 | 0.352 | 0.411 | 0.693 | $12.68 |

## Where the false negatives come from (ContractEval protocol)

| Arm | Total FN | Retrieval | Extraction | Retrieval share |
|---|---:|---:|---:|---:|
| RRF n20 top-8 | 467 | 147 | 320 | 31.5% |
| RRF n10 top-5 | 534 | 230 | 304 | 43.1% |
