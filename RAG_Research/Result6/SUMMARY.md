# Result6 — 6 RAG methods × 6 contracts × gpt-5.4

Same harness, span-validation and scorer as the 3-contract study; the contract set is
3 original + 3 new, the model is fixed to **gpt-5.4**, and metrics come from the
project's `evaluate.evaluate` so they are directly comparable to every other run here.

Methods M1–M4 use the **markdown** chunking (60 chunks); M5–M6 use a finer **section**
chunking (1.1/1.2 reading-order split, ~1500 chars, **220 chunks**) to test whether more,
smaller chunks help retrieval.

## Contracts (6)

| # | contract | chunks |
|---|----------|-------:|
| 1 | BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT | 7 |
| 2 | BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement | 12 |
| 3 | AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement | 6 |
| 4 | AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT | 13 |
| 5 | Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT | 9 |
| 6 | DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT | 13 |

## Ground-truth coverage (see `ground_truth_category_coverage.csv`)

- 41 categories asked, **179 gold answers** in total.
- **31 / 41** categories have at least one gold answer somewhere in the 6 contracts.
- **6 / 41** have a gold answer in *all* 6 contracts (Parties, Anti-Assignment, Effective
  Date, Agreement Date, Document Name, Expiration Date).
- **10 / 41** have **no gold answer anywhere** — the pipeline is still asked about them
  on every contract, so they can only ever produce false positives.

## Results (micro over 41 categories × 6 contracts)

| method | chunking | retrieval | TP | FP | FN | Prec | Rec | F1 | F2 | AUPR | Jac(LicGrant) | cost |
|--------|----------|-----------|---:|---:|---:|-----:|----:|---:|---:|-----:|--------------:|-----:|
| **M1 top-2 cosine**   | markdown (60) | cosine top-2 | 78 | 109 | 101 | 0.417 | 0.436 | **0.426** | 0.432 | 0.309 | 0.175 | $1.18 |
| M2 top-1 cosine       | markdown (60) | cosine top-1 | 60 |  79 | 119 | 0.432 | 0.335 | 0.377 | 0.350 | 0.240 | 0.046 | $0.62 |
| M3 top-1 + LLM check  | markdown (60) | cosine top-1 +check | 48 | 34 | 131 | **0.585** | 0.268 | 0.368 | 0.297 | 0.213 | 0.201 | $0.94 |
| M4 top-1 BM25         | markdown (60) | bm25 top-1 | 56 |  91 | 123 | 0.381 | 0.313 | 0.344 | 0.323 | 0.216 | 0.309 | $1.01 |
| **M5 section top-2 cosine** | section (220) | cosine top-2 | 56 | 80 | 123 | 0.412 | 0.313 | 0.356 | 0.328 | 0.221 | 0.221 | $0.31 |
| **M6 section hybrid top-3** | section (220) | bm25(10)→cosine top-3 | 68 | 97 | 111 | 0.412 | 0.380 | 0.395 | 0.384 | 0.268 | 0.310 | $0.41 |
| **M7 section hybrid top-5** | section (220) | bm25(10)→cosine top-5 | 76 | 131 | 103 | 0.367 | 0.425 | 0.394 | 0.411 | 0.290 | 0.421 | $0.58 |

Total API cost across all runs into Result6: **~$5.04**.

**M7 (hybrid, wider k=5) is the cleanest confirmation of the diagnosis.** Widening the final
budget from 3→5 lifted retrieval to the **best R@k of any method (0.66 > M1's 0.62)** and
recall to **0.425** (near M1's 0.436) — so the earlier section methods really were
context-starved. But precision fell 0.412→0.367 (FP 97→131), because with **no abstention**
the extra retrieved chunks became extra false positives on absent categories (45 of the 59
new FPs are zero-gold or boundary). Net F1 is flat (0.394 ≈ M6's 0.395). Retrieval budget was
the recall bottleneck; **abstention is now clearly the missing piece that would convert M7's
recall into F1.**

Reading the table:
- **M1 (markdown, top-2 cosine) still wins on F1 (0.426).** But not because markdown chunks
  are "better" — markdown chunks are huge (up to 7,600 tokens), so top-2 hands the model
  **3,127 tokens** on average. M5's section top-2 hands it only **479**. M1 wins by seeing
  ~6× more text, not by better retrieval (see `ANALYSIS.md` → "the chunking experiment").
- **The finer section chunking did NOT beat markdown at the same k** — because at the same k
  it shows far less context. M5 (section top-2) = 0.356; M6 (section hybrid top-3) = 0.395.
- **The hybrid retriever (M6) is the best thing about the section runs**: BM25-prefilter →
  cosine-rerank lifts F1 from 0.356 (M5, plain cosine) to 0.395 and beats M2/M3/M4. On the
  section chunks it wins or ties at every k in the R@k table.
- **M3 keeps the best precision (0.585)** but worst recall (0.268): its check pass prunes
  hard and drops candidates for any category it fails to return (TP 60 → 48).
- Every method is still below the **full-scan baseline** (F1 0.461, sees all chunks). RAG is
  *losing* answers, not finding them.

### Hybrid retrieval grid — BM25 prefilter {5,10,15} × cosine-rerank k {1,2,3,5,8}

Full grid on the section chunks (gpt-5.4). Three pivot tables live in the Excel
(`Hybrid_grid` sheet); the machine-readable form is `master_summary.csv`. k=8 for
prefilter=5 is capped at 5 candidates, so it repeats the k=5 row.

Best cells by F1:

| config | F1 | P | R | Jac.avg | R@k |
|--------|---:|--:|--:|--------:|----:|
| **H_bm5_cos5** (BM25→5, cosine→5) | **0.430** | 0.406 | 0.458 | 0.527 | 0.64 |
| M1 markdown top-2 (reference) | 0.426 | 0.417 | 0.436 | 0.435 | 0.62 |
| H_bm15_cos8 | 0.417 | 0.356 | 0.503 | 0.539 | 0.76 |
| H_bm10_cos8 | 0.413 | 0.356 | 0.492 | 0.549 | 0.75 |

Takeaways:
- **`H_bm5_cos5` is the new overall best (F1 0.430), edging out M1** — and it does so on the
  section chunks at 5 chunks of context, i.e. a *narrow* BM25 net (top-5) reranked by cosine.
  A tight prefilter keeps precision up while cosine picks the best 5; wider prefilters
  (10/15) add recall but bleed precision.
- **Within every prefilter, raising k trades precision for recall** exactly as M7 showed:
  R@k climbs monotonically (bm15: 0.31→0.76 from k=1→8) and recall with it, but precision
  falls and F1 plateaus around 0.40–0.43. Without abstention you cannot convert the extra
  recall into F1.
- **Prefilter size barely matters at fixed k** (bm5/bm10/bm15 give near-identical F1 per k) —
  because with cosine reranking, what reaches the model is the cosine-best k regardless of
  how wide the BM25 net was. The prefilter only bounds the maximum recall.

### Did more/smaller chunks help? (retrieval only, `recall_at_k_section.csv`)

| | chunks | gold coverage | R@2 cosine | R@3 hybrid |
|---|---:|---:|---:|---:|
| markdown | 60 | 93.3% | 0.62 | 0.69 |
| section  | 220 | **98.9%** | 0.43 | 0.55 |

At **fixed k**, section chunking *lowers* R@k (you must find 1 needle among 220, not 60).
But at **equal token budget** it is competitive-to-better: section top-8 reaches R@0.74 on
**1,929 tokens**, vs markdown top-2's R@0.62 on **3,127**. Finer chunking also lifts coverage
99% vs 93%. **Conclusion: the fine chunks are fine — they were starved at k=2/3.** To use
them you must raise k (≈6–8) or budget by tokens, not by k.

## Files

| file | what |
|------|------|
| `results/<method>/gpt-5.4.json` | full per-method output (all 6 methods): metrics, per-category counts, per-contract tp/fn/fp (+ M3 reasoning) |
| `results/summary.json` | machine-readable result table (6 methods) |
| `section_chunking.json` | the finer 1.1/1.2 section chunks (220) used by M5/M6 |
| `ground_truth_category_coverage.csv` | per-category gold coverage across the 6 contracts |
| `ground_truth_per_contract.csv` | per-contract: how many of the 41 categories have an answer |
| `evaluation_by_category.xlsx` | per-category evaluation (6 methods) + F1/Jaccard charts sorted worst-first |
| `master_summary.py` / `master_summary.csv` / `.json` | **all methods** (M1–M7 + the hybrid grid): F1/P/R/F2/AUPR, Jaccard avg+best+worst category, coverage, R@k, cost |
| `best_per_category.py` / `best_per_category.csv` / `.json` | per-category F1 & Jaccard winner across all methods, with win tally |
| `error_breakdown.py` / `error_breakdown.csv` | IOU-threshold sweep + FP/FN composition (boundary vs genuine vs zero-gold) |
| `recall_at_k.py` / `recall_at_k.csv` | R@k for the markdown chunks (cosine vs bm25) |
| `recall_at_k_section.py` / `recall_at_k_section.csv` | R@k markdown vs section + equal-token-budget comparison |
| `diagnose.py` / `diagnosis.json` | stage-by-stage root-cause decomposition (coverage × R@k × extraction) |
| `ANALYSIS.md` | **why** performance is bad, incl. the chunking experiment |
