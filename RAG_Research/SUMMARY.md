# RAG pipeline research — 4 retrieval methods on 3 CUAD contracts

Same 3 contracts, chunks, span-validation and scorer as every other experiment; only the retrieval strategy (and one optional verification pass) changes. Metrics via the project's `evaluate.evaluate`, so numbers are directly comparable to the full-scan `results/*__3contracts.json` baselines.

## Methods

- **M1_top2_cosine** — Label-centric RAG: top-2 chunks per category by cosine similarity (text-embedding-3-small), one LLM call per category. Matches the web UI.
- **M2_top1_cosine** — Label-centric RAG: single best chunk per category by cosine similarity.
- **M3_top1_cosine_llmcheck** — M2 (top-1 cosine) plus one LLM verification pass per contract over the full contract + all 41 categories + candidate answers, reasoning to prune hallucinations and recover clear misses.
- **M4_top1_bm25** — Label-centric RAG: single best chunk per category by BM25 (lexical) retrieval instead of cosine similarity.

## Results (micro over all 41 categories × 3 contracts)

`Prec/Rec/F1/F2` are micro over tp/fp/fn. `Jac-LG` = Jaccard on License Grant (the reference complex category); `Jac-macro` = mean Jaccard across categories with ground truth. `baseline_fullscan` = existing all-41-per-chunk run.

| method                   | model    | TP | FP | FN | Prec | Rec |  F1  |  F2  | AUPR | bestF1 | Jac-LG | Jac-macro |  cost  |
|--------------------------|----------|----|----|----|------|------|------|------|------|--------|--------|-----------|--------|
| baseline_fullscan        | gpt-5.4  | 50 | 73 | 44 | 0.407 | 0.532 | 0.461 | 0.501 | 0.374 | 0.461 | 0.604 | 0.531   |   n/a  |
| M1_top2_cosine           | gpt-5.4  | 36 | 54 | 58 | 0.400 | 0.383 | 0.391 | 0.386 | 0.268 | 0.391 | 0.089 | 0.384   | $0.662 |
| M2_top1_cosine           | gpt-5.4  | 28 | 39 | 66 | 0.418 | 0.298 | 0.348 | 0.316 | 0.211 | 0.348 | 0.000 | 0.303   | $0.330 |
| M3_top1_cosine_llmcheck  | gpt-5.4  | 32 | 26 | 62 | 0.552 | 0.340 | 0.421 | 0.369 | 0.264 | 0.421 | 0.203 | 0.360   | $0.496 |
| M4_top1_bm25             | gpt-5.4  | 24 | 45 | 70 | 0.348 | 0.255 | 0.294 | 0.270 | 0.172 | 0.294 | 0.276 | 0.254   | $0.491 |
| baseline_fullscan        | gpt-5.5  | 42 | 39 | 52 | 0.519 | 0.447 | 0.480 | 0.460 | 0.339 | 0.480 | 0.238 | 0.402   |   n/a  |
| M1_top2_cosine           | gpt-5.5  | 33 | 45 | 61 | 0.423 | 0.351 | 0.384 | 0.363 | 0.250 | 0.384 | 0.000 | 0.360   | $1.008 |
| M2_top1_cosine           | gpt-5.5  | 30 | 33 | 64 | 0.476 | 0.319 | 0.382 | 0.342 | 0.236 | 0.382 | 0.000 | 0.291   | $0.529 |
| M3_top1_cosine_llmcheck  | gpt-5.5  | 57 | 47 | 37 | 0.548 | 0.606 | 0.576 | 0.594 | 0.469 | 0.576 | 0.644 | 0.644   | $0.823 |
| M4_top1_bm25             | gpt-5.5  | 26 | 34 | 68 | 0.433 | 0.277 | 0.338 | 0.298 | 0.198 | 0.338 | 0.311 | 0.245   | $0.696 |

## Findings

- **gpt-5.4**: best RAG method is **M3_top1_cosine_llmcheck** (F1 0.421, P 0.552, R 0.340) — below full-scan baseline (F1 0.461) by -0.040.
    - top-2 vs top-1 cosine: F1 0.391 vs 0.348 (more chunks → recall 0.383 vs 0.298).
    - cosine vs BM25 (both top-1): F1 0.348 vs 0.294.
    - the LLM check pass (M3 vs M2): F1 0.348 → 0.421, FP 39 → 26, TP 28 → 32 (prunes false positives; recovers misses from the full contract).
- **gpt-5.5**: best RAG method is **M3_top1_cosine_llmcheck** (F1 0.576, P 0.548, R 0.606) — beats full-scan baseline (F1 0.480) by +0.096.
    - top-2 vs top-1 cosine: F1 0.384 vs 0.382 (more chunks → recall 0.351 vs 0.319).
    - cosine vs BM25 (both top-1): F1 0.382 vs 0.338.
    - the LLM check pass (M3 vs M2): F1 0.382 → 0.576, FP 33 → 47, TP 30 → 57 (recovers misses from the full contract).

## Per-category F1 by method — gpt-5.5

| category                           |  top2_cosine |  top1_cosine | top1_cosine_ |    top1_bm25 |
|------------------------------------|--------------|--------------|--------------|--------------|
| Affiliate License-Licensee         |         0.00 |         0.00 |         1.00 |         0.00 |
| Agreement Date                     |         0.80 |         0.80 |         1.00 |         0.00 |
| Anti-Assignment                    |         0.80 |         0.80 |         1.00 |         0.80 |
| Audit Rights                       |         0.33 |         0.75 |         0.75 |         0.50 |
| Cap On Liability                   |         0.48 |         0.32 |         0.78 |         0.42 |
| Change Of Control                  |         0.50 |         0.67 |         0.67 |         0.00 |
| Covenant Not To Sue                |         0.00 |         0.00 |         0.40 |         0.00 |
| Document Name                      |         0.50 |         0.00 |         1.00 |         0.00 |
| Effective Date                     |         0.33 |         0.33 |         0.86 |         0.33 |
| Exclusivity                        |         0.50 |         0.33 |         0.67 |         0.00 |
| Expiration Date                    |         0.00 |         0.00 |         0.00 |         0.00 |
| Governing Law                      |         0.50 |         0.00 |         0.80 |         0.50 |
| Insurance                          |         0.80 |         0.57 |         0.50 |         0.50 |
| License Grant                      |         0.00 |         0.00 |         0.60 |         0.20 |
| Minimum Commitment                 |         0.50 |         0.40 |         0.40 |         0.40 |
| Most Favored Nation                |         1.00 |         0.00 |         1.00 |         0.00 |
| No-Solicit Of Customers            |         0.00 |         0.00 |         0.67 |         0.67 |
| Non-Compete                        |         0.29 |         0.29 |         0.67 |         0.33 |
| Non-Transferable License           |         0.00 |         0.00 |         0.67 |         0.67 |
| Parties                            |         0.84 |         0.90 |         0.80 |         0.80 |
| Post-Termination Services          |         0.11 |         0.25 |         0.33 |         0.00 |
| Renewal Term                       |         0.00 |         0.00 |         0.00 |         0.00 |
| Revenue/Profit Sharing             |         0.00 |         0.00 |         0.00 |         0.00 |
| Termination For Convenience        |         1.00 |         1.00 |         1.00 |         1.00 |
| Unlimited/All-You-Can-Eat-License  |         0.00 |         0.00 |         1.00 |         0.00 |
| Volume Restriction                 |         0.00 |         0.00 |         0.00 |         0.00 |
| Warranty Duration                  |         0.25 |         0.33 |         0.25 |         0.00 |
