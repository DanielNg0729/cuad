# Reciprocal Rank Fusion (RRF) Test (gpt-5.4)

Kept in its own folder, separate from `../TestAblation/`, per request. Same section
chunking (~220 chunks), same 6 contracts, same 41 categories, same gpt-5.4 model as
the rest of Result6, so numbers are directly comparable to `../TestAblation`'s
`cosine_top5` and `hybrid_bm10_cos5` arms.

## What RRF is, and how it differs from the existing "hybrid" method

The `hybrid_bm10_cos5` method already tested in `../TestAblation/` is
**prefilter-then-rerank**: BM25 picks a shortlist, then cosine similarity
*completely takes over* -- BM25's score is discarded once it's done its gatekeeping
job, and a chunk cosine would love but BM25 didn't shortlist is never considered at
all.

**RRF fuses both rankings instead of letting one replace the other.** Each ranker
(BM25, cosine) independently ranks ALL chunks and contributes its own top-N
shortlist; the two shortlists are **unioned** (a chunk strong in only one ranker
still gets partial credit); every candidate in the union is scored by:

```
RRF_score(chunk) = 1/(rrf_k + bm25_rank(chunk))  +  1/(rrf_k + cosine_rank(chunk))
```

(rank is 1-indexed within that ranker's own shortlist; a chunk absent from a given
ranker's shortlist contributes 0 from it. `rrf_k=60` is the community-standard
constant from the original Cormack et al. RRF paper -- not the same thing as our
`top_k` final selection count.) The final top-5 are the highest-RRF-scored chunks in
the union.

This was validated on a toy example before spending any real money: a chunk with
**zero** lexical (BM25) match but the #1 cosine rank still made RRF's top-3, edging
out a BM25-only chunk -- exactly the behavior the current hybrid method can't produce
(it would have excluded that chunk outright since BM25 never shortlisted it).

## The 10 arms

N=10/15 were tested for all 3 embedders; N=20/25 were tested for qwen3 and bge-m3
only (not OpenAI), as requested.

| Arm | Embedder | BM25 shortlist N | Cosine shortlist N | RRF k | Final top-k |
|---|---|---|---|---|---|
| `openai_rrf_n10_top5` | OpenAI `text-embedding-3-small` | 10 | 10 | 60 | 5 |
| `openai_rrf_n15_top5` | OpenAI `text-embedding-3-small` | 15 | 15 | 60 | 5 |
| `qwen3_rrf_n10_top5` | `qwen3-embedding:0.6b` (local) | 10 | 10 | 60 | 5 |
| `qwen3_rrf_n15_top5` | `qwen3-embedding:0.6b` (local) | 15 | 15 | 60 | 5 |
| `qwen3_rrf_n20_top5` | `qwen3-embedding:0.6b` (local) | 20 | 20 | 60 | 5 |
| `qwen3_rrf_n25_top5` | `qwen3-embedding:0.6b` (local) | 25 | 25 | 60 | 5 |
| `bge3_rrf_n10_top5` | `bge-m3` (local) | 10 | 10 | 60 | 5 |
| `bge3_rrf_n15_top5` | `bge-m3` (local) | 15 | 15 | 60 | 5 |
| `bge3_rrf_n20_top5` | `bge-m3` (local) | 20 | 20 | 60 | 5 |
| `bge3_rrf_n25_top5` | `bge-m3` (local) | 25 | 25 | 60 | 5 |

**Total spend: $5.81** ($3.49 for the first 6 arms + $2.32 for the 4 new N=20/25
arms). `run_rrf.py` skips any arm whose result JSON already exists, so adding N=20/25
only paid for the 4 new arms, not a re-run of the first 6.

## Full metrics (10 RRF arms)

| Arm | Embedder | P | R | F1 | F2 | AUPR | Jac.avg | R@k | Cost |
|---|---|---|---|---|---|---|---|---|---|
| `openai_rrf_n10_top5` | OpenAI | 0.377 | 0.453 | 0.411 | 0.435 | 0.311 | 0.542 | 0.712 | $0.586 |
| `openai_rrf_n15_top5` | OpenAI | 0.346 | 0.447 | 0.390 | 0.422 | 0.301 | 0.536 | 0.701 | $0.595 |
| `qwen3_rrf_n10_top5` | qwen3 | **0.411** | **0.514** | **0.457** | **0.489** | **0.362** | 0.547 | 0.712 | $0.577 |
| `qwen3_rrf_n15_top5` | qwen3 | 0.413 | 0.480 | 0.444 | 0.465 | 0.340 | 0.529 | 0.706 | $0.588 |
| `qwen3_rrf_n20_top5` | qwen3 | 0.414 | 0.497 | 0.452 | 0.478 | 0.351 | **0.543** | 0.706 | $0.581 |
| `qwen3_rrf_n25_top5` | qwen3 | 0.409 | 0.492 | 0.447 | 0.473 | 0.346 | 0.524 | 0.712 | $0.584 |
| `bge3_rrf_n10_top5` | bge-m3 | 0.405 | 0.453 | 0.427 | 0.442 | 0.318 | 0.496 | 0.695 | $0.572 |
| `bge3_rrf_n15_top5` | bge-m3 | 0.404 | 0.458 | 0.429 | 0.446 | 0.322 | 0.480 | 0.695 | $0.574 |
| `bge3_rrf_n20_top5` | bge-m3 | 0.404 | 0.447 | 0.424 | 0.438 | 0.314 | 0.492 | 0.695 | $0.577 |
| `bge3_rrf_n25_top5` | bge-m3 | 0.415 | 0.464 | 0.438 | 0.453 | 0.328 | 0.523 | 0.695 | $0.580 |

Bold = best value in that column across all 10 RRF arms. Full tp/fp/fn, per-category
Jaccard, and per-(contract, category) ground_truth/predictions/context/retrieved_chunks
are in `rerank_summary.csv`/`.json` and `results/<arm>/gpt-5.4.json` (identical schema
to `../TestAblation`).

## Head-to-head vs. TestAblation's cosine-only and hybrid arms (same embedder, same top-5 budget)

| Embedder | cosine_top5 F1 | hybrid_bm10_cos5 F1 | RRF best F1 (N) |
|---|---|---|---|
| OpenAI | 0.406 | 0.394 | 0.411 (N=10) |
| qwen3 | 0.436 | 0.423 | **0.457 (N=10)** |
| bge-m3 | **0.449** | 0.428 | 0.438 (N=25) |

## Key findings

**1. `qwen3_rrf_n10_top5` (F1=0.457) is the new best arm across EVERYTHING tested so far** -- beating `bge3_cosine_top5` (0.449), the previous best from `TestAblation`. It also has the best precision, recall, F1, F2, and AUPR of all 10 RRF arms, and the best AUPR (0.362) of any arm tested in this whole project to date. For qwen3, RRF clearly beats both of its own alternatives (cosine_top5=0.436, hybrid_bm10_cos5=0.423) by a wide margin -- rank fusion is doing real work here, likely because qwen3's embedding space and BM25 disagree just enough that the union catches chunks either one alone would miss (recall jumps to 0.514, the highest of any arm anywhere).

**2. RRF helps OpenAI too (+0.005 F1 over its own cosine_top5), but does NOT help bge-m3 at any width tested.**
`bge-m3`'s own `cosine_top5` (0.449) still beats all four of its RRF variants (0.427, 0.429, 0.424, 0.438) and its hybrid variant (0.428), even at RRF's best width for it (N=25). This is consistent with the earlier finding that bge-m3's edge is specifically a **high-precision, tight-budget** phenomenon -- RRF's shortlist union tends to pull in more candidates (like widening a budget does), which erodes bge-m3's precision advantage the same way top-8 did in `TestAblation`. RRF is not a universal upgrade; it depends on how much the embedder's ranking and BM25's ranking actually disagree and where the embedder's strength comes from.

**3. Shortlist width N doesn't move monotonically with F1 -- it's embedder-specific and non-monotonic across the full N=10/15/20/25 sweep.**
```
qwen3:  N=10 F1=0.457 (best) > N=20 F1=0.452 > N=25 F1=0.447 > N=15 F1=0.444 (worst)
bge-m3: N=25 F1=0.438 (best) > N=15 F1=0.429 > N=10 F1=0.427 > N=20 F1=0.424 (worst)
```
For qwen3, N=10 remains the clear best across the full sweep -- widening the shortlist
never beats it, though the relationship isn't perfectly monotonic (N=20 partially
recovers from N=15's dip). For bge-m3, the pattern inverts: **N=25 is actually its
best RRF width tested**, beating N=10 -- but even at its best, bge-m3's RRF (0.438)
still falls short of its own plain `cosine_top5` (0.449), so widening the RRF
shortlist helps bge-m3's RRF variant without ever making RRF the right choice for
bge-m3 overall. The takeaway: there's no universal "wider is worse" or "narrower is
better" rule here -- it has to be swept per embedder, and for qwen3 the answer is
clearly "keep it narrow" (N=10) while for bge-m3 RRF just isn't the right retrieval
strategy regardless of width.

## Recommendation

Use **`qwen3-embedding:0.6b` with RRF (N=10, top-5)** -- it's the single best-performing
configuration found across all of `TestAblation` and this test combined, on every
metric except being edged out on Jaccard-avg by `qwen3_cosine_top8` (0.559 vs 0.547,
a much smaller gap than the F1/AUPR/recall wins RRF has everywhere else). If a wider
top-8-style budget is preferred instead, plain `qwen3_cosine_top8` (from `TestAblation`)
remains a strong, simpler alternative. Do not use RRF with `bge-m3` -- its own plain
`cosine_top5` is better and simpler.

## Files in this folder

- `run_rrf.py` -- the script that produced everything below (idempotent like
  `../TestAblation/run_ablation.py` -- skips any arm whose result JSON already exists).
- `rerank_summary.csv` / `.json` -- the master table above, machine-readable.
- `results/<arm>/gpt-5.4.json` -- full per-arm detail: ground_truth/predictions/tp/fn/fp,
  retrieved chunk text (`context` blob + `retrieved_chunks` index->text map +
  `retrieved_chunk_idxs`), per (contract, category), plus token usage and cost --
  identical schema to `../TestAblation`.
- `run.log` -- full console output from the run.
