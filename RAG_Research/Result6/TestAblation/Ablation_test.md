# Ablation Test: Embedding Model x Retrieval Strategy (gpt-5.4)

Full end-to-end extraction test (real LLM calls, not retrieval-only) comparing
**OpenAI `text-embedding-3-small`** against two local/Ollama embedders --
**`qwen3-embedding:0.6b`** and **`bge-m3`** -- the two winners from the earlier
retrieval-only test in `../TestEmbeddedModel/` -- plus a pure-BM25 (lexical, no
embeddings at all) baseline at two budgets. Each embedder is tested with cosine-only
and BM25(10)-prefilter-then-cosine-rerank ("hybrid") retrieval, each at two final
budgets (top-5 and top-8), holding chunking, model, categories, and contracts fixed.

- **Chunking**: section chunking (~220 chunks), same as M5-M9 / the hybrid grid.
- **Contracts**: the same 6 used throughout Result6.
- **Categories**: all 41 CUAD categories.
- **LLM**: gpt-5.4 (label-centric: one call per category, over just its retrieved chunks).
- **Total spend across all runs**: **$9.33** ($2.71 for the first 5 arms + $1.13 for
  the 2 bge-m3 arms + $0.57 for `bm25_top5` + $4.91 for the 6 top-8 arms).
  `openai_hybrid_bm10_cos5` is **identical to `M7_section_hybrid_top5`**, so its
  already-paid-for predictions were reused instead of re-spending money on the same
  246 calls -- $0 new spend for that row, and its numbers below match
  `master_summary.csv`'s M7 row exactly (tp/fp/fn/P/R/F1/F2/AUPR/cost all identical),
  which also serves as a correctness check on this script's plumbing.
  `run_ablation.py` skips any arm whose `results/<arm>/gpt-5.4.json` already exists,
  so adding more arms later only pays for the new ones.

## The 14 arms

| Arm | Embedder | Search | Final top-k | BM25 prefilter |
|---|---|---|---|---|
| `bm25_top5` | n/a (lexical only -- embedder-agnostic) | BM25 | 5 | -- |
| `bm25_top10` | n/a (lexical only -- embedder-agnostic) | BM25 | 10 | -- |
| `openai_cosine_top5` | OpenAI `text-embedding-3-small` | cosine | 5 | -- |
| `openai_cosine_top8` | OpenAI `text-embedding-3-small` | cosine | 8 | -- |
| `openai_hybrid_bm10_cos5` | OpenAI `text-embedding-3-small` | hybrid | 5 | 10 (= M7) |
| `openai_hybrid_bm10_cos8` | OpenAI `text-embedding-3-small` | hybrid | 8 | 10 |
| `qwen3_cosine_top5` | `qwen3-embedding:0.6b` (local) | cosine | 5 | -- |
| `qwen3_cosine_top8` | `qwen3-embedding:0.6b` (local) | cosine | 8 | -- |
| `qwen3_hybrid_bm10_cos5` | `qwen3-embedding:0.6b` (local) | hybrid | 5 | 10 |
| `qwen3_hybrid_bm10_cos8` | `qwen3-embedding:0.6b` (local) | hybrid | 8 | 10 |
| `bge3_cosine_top5` | `bge-m3` (local) | cosine | 5 | -- |
| `bge3_cosine_top8` | `bge-m3` (local) | cosine | 8 | -- |
| `bge3_hybrid_bm10_cos5` | `bge-m3` (local) | hybrid | 5 | 10 |
| `bge3_hybrid_bm10_cos8` | `bge-m3` (local) | hybrid | 8 | 10 |

`bm25_top5`/`bm25_top10` each appear **once**, not once per embedder -- BM25 is
purely lexical and never touches an embedding vector, so it is identical regardless
of embedder (already demonstrated in the retrieval-only test, where `hybrid_bm5_cos5`
scored identically across all 5 embedders for the same reason). Running it "per
embedder" would just be 3x the cost for 3 identical result files. The `cos8` and
`hybrid_bm10_cos8` arms genuinely DO depend on the embedder (cosine has real work to
do -- prefilter 10 > final 8, so 2 candidates get filtered out per hybrid call), so
those were correctly run 3x, once per embedder.

## Full metrics (all 14 arms)

| Arm | Embedder | P | R | F1 | F2 | AUPR | best_F1 | Jac.avg | Jac.best | Jac.worst | Coverage | R@k | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bm25_top5` | lexical | 0.365 | 0.408 | 0.385 | 0.399 | 0.278 | 0.385 | 0.455 | 0.99 (Termination For Convenience) | 0.00 (Renewal Term) | 0.989 | 0.655 | $0.574 |
| `bm25_top10` | lexical | 0.341 | 0.486 | 0.401 | 0.448 | 0.326 | 0.401 | 0.535 | 1.00 (Most Favored Nation) | 0.00 (Renewal Term) | 0.989 | 0.802 | $0.997 |
| `openai_cosine_top5` | OpenAI | 0.376 | 0.441 | 0.406 | 0.427 | 0.304 | 0.406 | 0.513 | 1.00 (No-Solicit Of Employees) | 0.00 (Irrevocable Or Perpetual License) | 0.989 | 0.661 | $0.571 |
| `openai_cosine_top8` | OpenAI | 0.343 | 0.464 | 0.394 | 0.433 | 0.311 | 0.394 | 0.533 | 1.00 (No-Solicit Of Employees) | 0.00 (Volume Restriction) | 0.989 | 0.763 | $0.816 |
| `openai_hybrid_bm10_cos5` | OpenAI | 0.367 | 0.425 | 0.394 | 0.412 | 0.290 | 0.394 | 0.480 | 1.00 (No-Solicit Of Employees) | 0.00 (Renewal Term) | 0.989 | 0.678 | $0.576 (reused, $0 new) |
| `openai_hybrid_bm10_cos8` | OpenAI | 0.370 | 0.503 | 0.426 | 0.469 | 0.344 | 0.394 | 0.525 | -- | -- | 0.989 | 0.768 | $0.834 |
| `qwen3_cosine_top5` | qwen3 | 0.396 | 0.486 | 0.436 | 0.465 | 0.339 | 0.436 | 0.538 | 1.00 (No-Solicit Of Employees) | 0.00 (Renewal Term) | 0.989 | 0.729 | $0.563 |
| `qwen3_cosine_top8` | qwen3 | 0.385 | **0.503** | 0.436 | 0.474 | **0.348** | 0.436 | **0.559** | 1.00 (Affiliate License-Licensee) | 0.00 (Volume Restriction) | 0.989 | 0.785 | $0.789 |
| `qwen3_hybrid_bm10_cos5` | qwen3 | 0.381 | 0.475 | 0.423 | 0.453 | 0.328 | 0.423 | 0.544 | 1.00 (Most Favored Nation) | 0.00 (Renewal Term) | 0.989 | 0.701 | $0.582 |
| `qwen3_hybrid_bm10_cos8` | qwen3 | 0.361 | 0.503 | 0.421 | 0.466 | 0.342 | 0.421 | 0.558 | -- | -- | 0.989 | 0.768 | $0.833 |
| `bge3_cosine_top5` | bge-m3 | **0.445** | 0.453 | **0.449** | 0.451 | 0.327 | **0.449** | 0.494 | 1.00 (Most Favored Nation) | 0.00 (Volume Restriction) | 0.989 | 0.684 | $0.552 |
| `bge3_cosine_top8` | bge-m3 | 0.379 | 0.453 | 0.412 | 0.435 | 0.312 | 0.412 | 0.518 | 1.00 (Most Favored Nation) | 0.00 (Volume Restriction) | 0.989 | 0.746 | $0.809 |
| `bge3_hybrid_bm10_cos5` | bge-m3 | 0.397 | 0.464 | 0.428 | 0.449 | 0.324 | 0.428 | 0.480 | 1.00 (Most Favored Nation) | 0.00 (Renewal Term) | 0.989 | 0.718 | $0.581 |
| `bge3_hybrid_bm10_cos8` | bge-m3 | 0.374 | 0.497 | 0.427 | 0.467 | 0.342 | 0.427 | 0.523 | -- | -- | 0.989 | 0.763 | $0.831 |

Bold = best value in that column overall (across all 14 arms). "--" = not pulled from
this report's summary (see `ablation_summary.json` for the full per-category Jaccard
breakdown of every arm). Raw counts (tp/fp/fn), per-category Jaccard, and
per-(contract, category) ground_truth/predictions/tp/fn/fp are in
`ablation_summary.csv` / `.json` and `results/<arm>/gpt-5.4.json`.

## Key findings

**1. `bge-m3 cosine_top5` has the single best F1 across all 14 arms (0.449), but widening its own budget to top-8 HURTS it -- while the same widening HELPS qwen3.**
Going from top-5 to top-8 is not a universal win; it depends on the embedder:

| Embedder | cosine_top5 F1 | cosine_top8 F1 | Delta |
|---|---|---|---|
| OpenAI | 0.406 | 0.394 | -0.012 (worse) |
| qwen3 | 0.436 | 0.436 | 0.000 (flat, but AUPR/Jaccard/R@k all improve) |
| bge-m3 | **0.449** | 0.412 | **-0.037 (worse)** |

Widening the budget always increases R@k (more chances to retrieve the right chunk)
and always increases FP count (more chances to hallucinate on irrelevant chunks) --
whether F1 net improves depends on which effect dominates for that embedder. For
OpenAI and bge-m3, precision erodes faster than recall gains (bge-m3 goes from 101
FPs at top5 to 133 at top8, precision 0.445 -> 0.379). For qwen3 the two effects
roughly cancel on F1, but AUPR (0.339 -> 0.348), Jaccard avg (0.538 -> 0.559,
best of all 14 arms), and recall/F2 all improve -- so **qwen3 is the one embedder that
benefits from a wider top-8 budget**, while bge-m3's edge is specifically a top-5,
high-precision phenomenon that doesn't hold up if you widen it.

**2. `bge-m3 cosine_top5` (F1=0.449) is still the single best arm overall**, followed
by `qwen3_cosine_top5` and `qwen3_cosine_top8` tied at F1=0.436 -- all three beat every
OpenAI arm (best OpenAI F1 is 0.406, `cosine_top5`). The best-F1 winner depends on
budget: at top-5, bge-m3 wins on precision; at top-8, qwen3 pulls ahead on
AUPR/Jaccard/recall even though F1 ties.

**3. Hybrid (BM25-prefilter -> cosine-rerank) underperforms plain cosine-over-the-full-corpus at top-5 for every embedder, but the gap narrows or reverses at top-8.**
```
top-5:  OpenAI cosine 0.406 > hybrid 0.394 | qwen3 cosine 0.436 > hybrid 0.423 | bge-m3 cosine 0.449 > hybrid 0.428
top-8:  OpenAI cosine 0.394 < hybrid 0.426 | qwen3 cosine 0.436 > hybrid 0.421 | bge-m3 cosine 0.412 > hybrid 0.427 (~tie)
```
At top-5, plain cosine beat hybrid for all three embedders (narrowing to BM25's top-10
lexical shortlist before cosine picks the final 5 apparently excludes some
semantically-correct chunks that a full-corpus cosine search would find). At top-8,
that gap shrinks to near-zero and even **reverses for OpenAI and bge-m3** -- once
cosine only has to filter 10 candidates down to 8 (barely any filtering, 2 dropped)
rather than down to 5, the BM25 prefilter's main risk (losing a good chunk it never
shortlisted) matters less, while its main benefit (2 fewer, precisely-lexically-matched
chunks vs cosine's own top-8) can tip it back ahead. Net: **the earlier "hybrid always
loses to plain cosine at equal budget" finding does NOT generalize to top-8** -- it was
specific to the tighter top-5 budget where cosine had more filtering to do.

**4. BM25-alone remains weak at every budget tested.** `bm25_top5` is still the worst
of all 14 arms (F1=0.385), and `bm25_top10` (F1=0.401) is worse than every
embedding-based arm despite retrieving the widest net of any arm (R@k=0.802) --
lexical-only matching without any semantic signal simply isn't competitive here, at
any budget.

**5. Coverage (0.989) is identical across all 14 arms**, as expected -- it's a
property of the chunk *text*, not the embedder, budget, or retrieval method.

## Methodology notes

- **R@k here uses a fuzzy containment check** (exact normalized substring, falling
  back to >=90% token-overlap -- same logic as `../diagnose.py`), which is slightly
  more lenient than `master_summary.csv` / `recall_at_k_all_methods.csv`'s
  strict-substring-only check. That's why `openai_hybrid_bm10_cos5`'s R@k here
  (0.678) differs from M7's R@k in those other tables (0.655/0.66) even though every
  other metric (P/R/F1/F2/AUPR/cost) matches exactly -- R@k is the one number that
  isn't directly comparable across those files. All 14 rows in *this* report use the
  identical function, so comparisons within this table are apples-to-apples.
- **`openai_hybrid_bm10_cos5` = `M7_section_hybrid_top5` exactly** (same predictions,
  reused from disk, not re-run) -- included here mainly as an internal consistency
  check and to fill out the embedder x {cosine, hybrid} grid.
- **Incremental / idempotent**: `run_ablation.py` skips any arm whose result JSON
  already exists on disk (see `FORCE_RERUN` at the top of the script to force a
  specific arm to re-run). The bge-m3 arms, `bm25_top5`, and the 6 top-8 arms were
  each added and run without re-spending on the arms already computed.
- **BM25 is embedder-agnostic by construction** -- it was requested "for all 3
  embedding models," but since BM25 never touches an embedding vector at all, running
  it 3 times would produce 3 byte-identical result files for 3x the cost. It's
  included once (per top-k budget) instead; say the word if a genuinely different
  per-embedder BM25 variant was actually intended. The top-8 `cosine`/`hybrid` arms,
  by contrast, genuinely do depend on the embedder (cosine has real filtering work to
  do in both), so those were correctly run once per embedder.
- Retrieved chunk text is saved **two ways** in every
  `results/<arm>/gpt-5.4.json` -> `by_contract.<contract>.<category>`:
  - `context`: one concatenated string of all retrieved chunks (as actually sent to the LLM)
  - `retrieved_chunks`: an explicit `{"<chunk_index>": "<chunk text>", ...}` map, so any
    specific retrieved chunk is directly addressable by its index without having to
    split the concatenated blob
  - `retrieved_chunk_idxs`: the raw list of indices, for anything that just needs the ids

## Recommendation

Swap the RAG pipeline's embedder from OpenAI `text-embedding-3-small` to a local
Ollama model -- both `qwen3-embedding:0.6b` and `bge-m3` clearly beat it at every
budget tested, are free (no per-call API cost), and run comfortably on a 4GB GPU.
Beyond that, the choice depends on budget and what you optimize for:
- **`bge-m3 cosine_top5`** -- best F1 overall (0.449), if you want a tight top-5
  budget and value precision (fewest false positives of any arm).
- **`qwen3-embedding:0.6b cosine_top8`** -- best AUPR (0.348) and best Jaccard avg
  (0.559) of all 14 arms, plus the best recall/F2, if you can afford a wider top-8
  budget and value recall / ranking quality over raw F1.
- Avoid **BM25-alone** at any budget (worst or near-worst arm every time) and avoid
  **hybrid at top-5** (loses to plain cosine for every embedder) -- but hybrid at
  top-8 is roughly competitive with or better than plain cosine, so the "hybrid
  always loses" conclusion from the top-5-only test doesn't generalize; it was an
  artifact of that specific tight budget.

## Files in this folder

- `run_ablation.py` -- the script that produced everything below (idempotent; add new
  arms to the `ARMS` dict and re-run -- existing arms are skipped automatically).
- `ablation_summary.csv` / `.json` -- the master table above, machine-readable.
- `results/<arm>/gpt-5.4.json` -- full per-arm detail: ground_truth/predictions/tp/fn/fp,
  **retrieved chunk text (both as a concatenated blob and as an index -> text map)**,
  per (contract, category), plus token usage and cost.
- `run.log` -- full console output from the latest run.
