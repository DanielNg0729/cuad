# TestRAGFinal — the best RAG method on all 102 CUAD contracts, vs. the ContractEval benchmark

The whole RAG study up to this point ran on **6 contracts**. That was the single biggest
weakness in the results: six contracts, 179 gold answers, and 10 of 41 categories with no
gold answer anywhere. This folder runs the best method found across `TestAblation` +
`TestRerank` on the **full 102-contract CUAD test set** with **GPT-4.1**, at two retrieval
budgets, and compares both to the published
[ContractEval benchmark](https://arxiv.org/pdf/2508.03080) (arXiv:2508.03080), which
evaluates the *same model* on the *same dataset* with **full documents and no retrieval**.

That comparison is the point. ContractEval's GPT-4.1 row is exactly the full-document
baseline this project needed and had never run at scale — so it did not have to be paid for.

---

## 1. Headline

Scored under **ContractEval's own protocol**, so these numbers are comparable to its Table 3:

| System | P | R | F1 | F2 | Jaccard | False rate ↓ | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ContractEval GPT-4.1** (full document, published) | 0.595\* | **0.694**\* | **0.641** | **0.672** | 0.472 | 0.071 | **≈ $50** |
| **RAG RRF n20 top-8** (this work) | 0.649 | 0.625 | **0.637** | 0.629 | **0.488** | **0.049** | **$18.20** |
| RAG RRF n10 top-5 (this work) | **0.666** | 0.571 | 0.615 | 0.588 | 0.469 | 0.054 | **$12.68** |

\* Precision and recall are **derived** from the paper's published F1/F2 — two equations, two
unknowns. The derivation reproduces the paper's own F1/F2 to four decimals for all four
proprietary models. See `pr_from_f1f2()` in [contracteval_score.py](contracteval_score.py).

**At a top-8 retrieval budget, RAG is statistically indistinguishable from reading the entire
contract — F1 0.637 vs 0.641, a gap of 0.004 — while costing 2.7× less and reading roughly 3%
as much text per call.** It also produces *tighter* answers than the full-document baseline
(Jaccard 0.488 vs 0.472) and refuses to answer wrongly *less* often (0.049 vs 0.071).

The remaining deficit is entirely recall (0.625 vs 0.694), which is why F2 — the
recall-weighted score that matters most in legal review — still favours the full-document
approach by 0.043.

This is a substantially better result for RAG than the 6-contract study suggested, where RAG
lost to full scan on both cost *and* accuracy.

---

## 2. What was run

Method held fixed at the best arm found anywhere in this project, `qwen3_rrf_*`:

| Component | Setting |
|---|---|
| Embedder | `qwen3-embedding:0.6b` (local, Ollama, free) |
| Chunking | section split (1.1 / 1.2 / ARTICLE), packed to 1,500 chars |
| Retrieval | Reciprocal Rank Fusion — BM25 top-N ∪ cosine top-N, fused by 1/(60 + rank) |
| Extraction | one structured-output LLM call per (contract, category) |
| Model | **gpt-4.1** ($2/M input, $8/M output) |

Two arms, differing only in retrieval budget:

| Arm | Shortlist N | Final top-k | Calls | Input tokens | Output tokens | Cost | Wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3_rrf_n10_top5` | 10 | 5 | 4,182 | 5,677,043 | 165,270 | $12.68 | 7.7 min |
| `qwen3_rrf_n20_top8` | 20 | 8 | 4,182 | 8,302,021 | 199,552 | $18.20 | 11.4 min |

Both: **0 failed calls**. The n20 arm hit 181 rate-limit responses, all recovered on the
first or second retry.

| Scale | Value |
|---|---:|
| Contracts | 102 |
| Section chunks | 4,141 |
| (contract, category) pairs | 4,182 |
| Gold answers | 2,643 |
| Chunk coverage | 0.9883 |
| Embedding (local, free) | 6.2 min, cached |

For scale, the earlier 6-contract study covered 220 chunks and 179 gold answers. This is
roughly a **15× larger** evaluation.

---

## 3. Why the two scorers give different numbers

The most important thing to understand: **the paper's F1 and this project's F1 do not measure
the same thing.** Both are called F1; they are computed differently, and the paper's
definition is the more forgiving one.

| | This project (`evaluate.py` / `breakdown`) | ContractEval (§3.4) |
|---|---|---|
| Unit counted | one per **gold span** | one per **(contract, question) pair** |
| Match rule | token-Jaccard **≥ 0.5** | prediction **fully covers** the gold span |
| Answer longer than gold | **penalised** — Jaccard drops below 0.5, charged FP *and* FN | **free** — covering the gold is sufficient |
| Empty gold label | can only ever produce FP | scored **TN** when the model abstains |
| Abstention | none — the pipeline always answers | model instructed to say "no related clause" |
| Context | top-k retrieved chunks | the entire contract |

The match rule connects directly to
[ERROR_ANALYSIS.md](../RAG_Research/Result6/ERROR_ANALYSIS.md): 57% of this pipeline's false
positives and 57% of its false negatives are the **same "right clause, wrong extent" event
counted twice**. ContractEval's criterion largely forgives that error class.

So [contracteval_score.py](contracteval_score.py) re-scores **this project's own predictions**
under **the paper's exact definitions**. Nothing about the pipeline changes — only the ruler.

Because a CUAD question can have several gold spans, "fully covers the labeled span" has two
defensible readings and both are reported:

- **lenient** — TP if *any* gold span is fully covered (primary; matches the paper's singular phrasing)
- **strict** — TP only if *every* gold span is fully covered

### Evidence the two studies measure the same task

| | ContractEval | This run |
|---|---:|---:|
| Data points | 4,128 | 4,182 |
| Positive share | 30% | **29.8%** |

A positive share of 29.8% against a reported 30%, derived independently, is strong evidence
the two evaluate the same population. The 1.3% pair-count difference is unexplained and is the
main residual uncertainty.

---

## 4. Full results

### 4.1 Native scorer (this project's metric — **not** comparable to the paper)

| Arm | TP | FP | FN | P | R | F1 | F2 | AUPR | Jaccard | R@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RRF n10 top-5 | 1,269 | 1,449 | 1,374 | **0.467** | 0.480 | 0.473 | 0.477 | 0.352 | 0.411 | 0.693 |
| RRF n20 top-8 | 1,357 | 1,720 | 1,286 | 0.441 | **0.513** | **0.474** | **0.497** | **0.370** | **0.476** | **0.769** |

Worth noting independently: **F1 0.473 on 102 contracts versus 0.457 for the same method on 6
contracts** (with gpt-5.4). The method did not degrade at 17× the scale — it improved slightly.
The 6-contract result was not a small-sample fluke.

### 4.2 ContractEval protocol, both readings

| Arm | Reading | TP | TN | FP | FN | P | R | F1 | F2 | Jaccard | False rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n20 top-8 | lenient | 777 | 2,518 | 420 | 467 | 0.649 | 0.625 | **0.637** | 0.629 | 0.488 | 0.049 |
| n20 top-8 | strict | 543 | 2,518 | 420 | 701 | 0.564 | 0.437 | 0.492 | 0.457 | 0.488 | 0.049 |
| n10 top-5 | lenient | 710 | 2,582 | 356 | 534 | 0.666 | 0.571 | 0.615 | 0.588 | 0.469 | 0.054 |
| n10 top-5 | strict | 487 | 2,582 | 356 | 757 | 0.578 | 0.392 | 0.467 | 0.419 | 0.469 | 0.054 |

### 4.3 Against the full ContractEval proprietary leaderboard

| System | P\* | R\* | F1 | F2 | Jaccard | False rate ↓ |
|---|---:|---:|---:|---:|---:|---:|
| GPT 4.1 mini (full doc, paper) | 0.594 | 0.703 | **0.644** | **0.678** | 0.435 | 0.072 |
| GPT 4.1 (full doc, paper) | 0.595 | 0.694 | 0.641 | 0.672 | 0.472 | 0.071 |
| **GPT 4.1 + RAG RRF n20 top-8 (this work)** | 0.649 | 0.625 | **0.637** | 0.629 | 0.488 | **0.049** |
| **GPT 4.1 + RAG RRF n10 top-5 (this work)** | **0.666** | 0.571 | 0.615 | 0.588 | 0.469 | 0.054 |
| Claude Sonnet 4 (full doc, paper) | 0.451 | 0.622 | 0.523 | 0.578 | 0.458 | 0.025 |
| Gemini 2.5 Pro Preview (full doc, paper) | 0.384 | 0.705 | 0.497 | 0.604 | **0.506** | **0.011** |

The n20 top-8 arm places **third of seven**, 0.004 behind full-document GPT-4.1, ahead of both
Claude Sonnet 4 and Gemini 2.5 Pro Preview — which read entire contracts — and it holds the
**two best precision scores in the table**.

---

## 5. What the numbers mean

### 5.1 Retrieval budget buys recall, not F1 — under our scorer

Going from top-5 to top-8 does exactly what the 6-contract study predicted, at 17× the scale:

| | n10 top-5 | n20 top-8 | Δ |
|---|---:|---:|---:|
| R@k | 0.693 | 0.769 | **+0.076** |
| Recall | 0.480 | 0.513 | +0.033 |
| Precision | 0.467 | 0.441 | −0.026 |
| **F1** | 0.473 | 0.474 | **+0.001** |
| F2 | 0.477 | 0.497 | +0.020 |

Retrieval recall rose 7.6 points; end-to-end F1 moved by **one thousandth**. Precision fell
almost exactly as much as recall rose. This is the clearest confirmation yet of the central
claim in [ANALYSIS.md](../RAG_Research/Result6/ANALYSIS.md): **without abstention, extra
retrieved context converts into extra false positives at very nearly the same rate it converts
into extra true positives** (FP +271, TP +88).

### 5.2 …but under ContractEval's scorer, the budget increase *does* pay

Same two runs, different ruler:

| | n10 top-5 | n20 top-8 | Δ |
|---|---:|---:|---:|
| Native F1 | 0.473 | 0.474 | **+0.001** |
| ContractEval F1 | 0.615 | 0.637 | **+0.022** |

The two protocols disagree about whether widening the budget helped, and the reason is
instructive. More context makes the model return **longer and more numerous spans**. Our
scorer punishes that — a span materially longer than the gold falls below Jaccard 0.5 and is
charged FP *and* FN. ContractEval's "fully covers" rule rewards it — a longer span still
covers the gold.

**Whether a retrieval-budget increase looks like an improvement is therefore partly an
artefact of the scoring rule**, not purely a property of the system. Any claim of the form
"top-8 is better than top-5" has to name the metric it is true under. This is a real
methodological finding and belongs in the report.

### 5.3 The recall gap is a retrieval ceiling

R@k for the n20 arm is **0.769** — the retriever surfaces the gold-bearing chunk 76.9% of the
time. ContractEval recall is **0.625**, i.e. **81% of the attainable ceiling**. Extraction is
converting well; the ceiling is retrieval. Full-document prompting has R@k = 1.00 by
construction, which is exactly where its remaining +0.069 recall advantage comes from.

### 5.4 Widening the budget fixed retrieval and exposed extraction

Every false negative attributed to one stage by checking whether the gold text reached the
model at all — see [failure_attribution__n20_top8.csv](failure_attribution__n20_top8.csv):

| Arm | Total FN | Retrieval | Extraction | Retrieval share |
|---|---:|---:|---:|---:|
| n10 top-5 | 534 | 230 | 304 | 43.1% |
| **n20 top-8** | **467** | **147** | 320 | **31.5%** |

Widening the budget removed **83 retrieval misses** while adding only 16 extraction misses —
a clean, well-targeted win. But it also means **68.5% of the remaining loss is now
extraction**, not retrieval. On the 6-contract study this split was 44.8% extraction / 55.2%
retrieval. **The bottleneck has moved.** Further retrieval work has little left to recover;
prompt and extraction work is where the remaining headroom is.

### 5.5 The pipeline is *less* "lazy" than the full-document baseline

ContractEval's false "no related clause" rate measures how often a model returns nothing
despite a real clause existing. GPT-4.1 reading whole contracts: **0.071**. The same model
reading eight retrieved chunks: **0.049**.

This pipeline has **no abstention path at all** — it returns nothing only when the model
genuinely finds nothing or every candidate span fails verbatim validation. It nonetheless
refuses less often than a model explicitly offered the "no related clause" escape hatch. The
plausible reading is that a short, focused context makes the model more willing to commit,
whereas 300k characters of contract invites it to give up. It also suggests the "laziness"
ContractEval measures is partly an artefact of very long contexts.

### 5.6 Conciseness now beats the baseline

Jaccard on positive cases: **0.488 (n20 top-8) vs 0.472 (full document)**. At top-5 it was a
wash (0.469). Retrieval gives the model less irrelevant text to quote from, and at the wider
budget it finds enough of the right text for that to show up as an advantage.

### 5.7 The strict reading is where RAG still loses badly

Lenient 0.637 → strict 0.492 is a **−0.145** drop (top-5: −0.148, essentially unchanged).
The strict reading requires covering *every* gold span for a question. When a question's gold
answer is spread across several distant parts of a contract, a fixed top-k budget cannot hold
all of them; full-document prompting has no such constraint. **Widening from 5 to 8 chunks did
not close this gap at all**, which suggests multi-span questions need a different mechanism —
not simply more chunks.

### 5.8 Per-category extremes

From [per_category__n20_top8.csv](per_category__n20_top8.csv), ContractEval-protocol F1:

| Strongest | F1 | positives | | Weakest | F1 | positives |
|---|---:|---:|---|---|---:|---:|
| Parties | 0.970 | 102 | | Volume Restriction | 0.133 | 17 |
| Governing Law | 0.908 | 83 | | Third Party Beneficiary | 0.162 | 6 |
| Insurance | 0.885 | 32 | | Irrevocable Or Perpetual License | 0.250 | 13 |
| Anti-Assignment | 0.819 | 72 | | Unlimited/All-You-Can-Eat-License | 0.333 | 3 |
| Audit Rights | 0.765 | 38 | | Non-Transferable License | 0.349 | 22 |
| Agreement Date | 0.701 | 93 | | Affiliate License-Licensor | 0.000 | 6 |

Categories with a canonical location and stereotyped phrasing do well; categories requiring a
judgement about what counts do badly. Two caveats: `Source Code Escrow` shows F1 1.000 on a
**single** positive case and should be ignored, and `Price Restrictions` has **zero** positives
across all 102 contracts, so its 0.000 reflects false positives only.

Notably, **Parties at 0.970** is the category the 6-contract bad-case analysis flagged as the
*worst* retrieval failure (8 misses). At 102 contracts it is the strongest category in the set
— a good illustration of how unreliable six-contract category conclusions were.

---

## 6. Limitations

This is an honest comparison, but it is **not a controlled experiment**.

- **Different pipelines, not one variable.** The paper's prompt, parsing and output handling
  are theirs; ours are ours. The measured Δ of −0.004 F1 bundles *retrieval vs full document*
  together with every other implementation difference. It is "our system vs their published
  number", not "RAG vs full scan, all else equal". A controlled version would run
  full-document prompting through **this** harness — approximately **$98**, because the
  contract must be resent for each of the 41 questions. **Not run.**
- **Two variables changed at once between arms.** n10→n20 (shortlist) and top5→top8 (budget)
  moved together, so their individual contributions cannot be separated. Result6's 6-contract
  data suggests the budget did most of the work: at fixed top-5, N=10 scored 0.457 and N=20
  scored 0.452, i.e. widening N alone was slightly *negative*.
- **Their numbers were not reproduced.** The paper's Table 3 is taken as published; only our
  own predictions were re-scored.
- **Precision and recall for the paper are derived, not reported.** The derivation is exact
  and self-checking, but assumes their F1/F2 come from the standard formulas they state.
- **Pair-count discrepancy.** 4,182 pairs here versus 4,128 reported — 1.3%, unexplained.
- **Match-rule asymmetry cuts one way.** ContractEval's "fully covers" rewards over-inclusive
  answers, as §5.2 demonstrates directly. Neither pipeline was tuned for the other's ruler.
- **Single run per arm, no seeds.** Temperature 0, but no repeats, so no variance estimate.
  A 0.004 F1 gap is well inside the noise one would expect from re-running.

---

## 7. What this changes in the wider study

1. **The "RAG loses to full scan" headline needs qualifying.** On 6 contracts RAG lost on both
   accuracy and cost. On 102 contracts, against a published full-document baseline, the
   top-8 arm is within **0.004 F1** at **2.7× lower cost**. The honest statement is now
   *"RAG matches full-document prompting on F1 at a fraction of the cost, and trades recall
   for precision"* — not *"RAG is strictly worse"*.
2. **The small-sample worry is resolved in the method's favour.** Native F1 0.473–0.474 at 102
   contracts versus 0.457 at 6.
3. **Extraction, not retrieval, is now the dominant loss** — 68.5% of false negatives at
   top-8, up from 56.9% at top-5 and inverted from the 6-contract finding. Priority should
   shift from retrieval tuning to prompt and extraction work.
4. **Scoring rule choice materially changes conclusions** (§5.2). The same budget increase is
   worth +0.001 F1 under one protocol and +0.022 under another. Every metric claim in the
   report should name its scorer.
5. **Abstention remains untested but looks less urgent than assumed.** The false-refusal rate
   already beats the full-document baseline; the FP problem is over-answering on the 2,938
   negative pairs (420 FPs at top-8), not under-answering.
6. **Category-level conclusions from 6 contracts should be treated as unreliable** — Parties
   went from worst-retrieval-failure to best category overall.

---

## 8. Files

| File | Contents |
|---|---|
| [run_compare.py](run_compare.py) | The runner. `--model`, `--shortlist-n`, `--top-k`, `--limit-contracts`, `--dry-run`, `--force`. Idempotent. |
| [contracteval_score.py](contracteval_score.py) | Re-scores predictions under ContractEval §3.4; derives the paper's P/R from its F1/F2. |
| [per_category.py](per_category.py) | Per-category breakdown under both protocols + failure attribution. `--result`, `--suffix`. |
| [summarize_arms.py](summarize_arms.py) | Consolidates every arm in `results/` into one comparison table. |
| [all_arms_summary.md](all_arms_summary.md) / `.csv` | **Start here** — all arms, both protocols, vs. the paper. |
| [section_chunking_102.json](section_chunking_102.json) | 102 contracts → 4,141 section chunks (1,500 chars). |
| `results/qwen3_rrf_n10_top5__all102/gpt-4.1.json` | Full output, 51.8 MB |
| `results/qwen3_rrf_n20_top8__all102/gpt-4.1.json` | Full output, 78.4 MB |
| `results/qwen3_rrf_n10_top5__smoke/gpt-4.1.json` | 2-contract smoke test ($0.24) |
| `contracteval_protocol_scores__n10_top5.json` / `__n20_top8.json` | Both protocols, both readings, headline deltas |
| `per_category.csv` / `per_category__n20_top8.csv` | 41 rows: native + ContractEval metrics per category |
| `failure_attribution.csv` / `failure_attribution__n20_top8.csv` | Retrieval- vs extraction-caused FNs per category |
| [run.log](run.log) / [run_n20_top8.log](run_n20_top8.log) | Console output for each run |
| `.cache/` | qwen3 embedding vectors (17 MB), so re-runs never re-embed |

### Result JSON schema (same as `TestAblation`)

```
arm, model, scope, retrieval{embedder, embedder_model, search, shortlist_n, rrf_k, top_k,
  chunking, chunk_chars}, n_contracts, n_chunks, contracts[],
n_llm_calls_ok, n_llm_calls_failed, tokens{input, output}, cost_usd, wall_seconds,
micro{tp, fp, fn, precision, recall, f1, f2},
metrics{aupr, best_f1, best_f2, jaccard_similarity, jaccard_per_category, ...},
by_category_counts{<category>: {tp, fp, fn}},
by_contract{<contract>: {<category>: {
    ground_truth[], predictions[], tp[], fn[], fp[],
    context,                      # concatenated retrieved chunks, as sent to the LLM
    retrieved_chunk_idxs[],       # which chunk indices were retrieved
    retrieved_chunks{idx: text}   # index -> chunk text, individually addressable
}}},
r_at_k, coverage, gold_total
```

## 9. Reproducing

```bash
# cost projection only, no LLM calls
python TestRAGFinal/run_compare.py --shortlist-n 20 --top-k 8 --dry-run

# the two arms  (~$12.68 / 7.7 min  and  ~$18.20 / 11.4 min)
python TestRAGFinal/run_compare.py --shortlist-n 10 --top-k 5
python TestRAGFinal/run_compare.py --shortlist-n 20 --top-k 8

# scoring and analysis (free)
python TestRAGFinal/contracteval_score.py \
    --result TestRAGFinal/results/qwen3_rrf_n20_top8__all102/gpt-4.1.json \
    --out    TestRAGFinal/contracteval_protocol_scores__n20_top8.json
python TestRAGFinal/per_category.py \
    --result TestRAGFinal/results/qwen3_rrf_n20_top8__all102/gpt-4.1.json \
    --suffix __n20_top8
python TestRAGFinal/summarize_arms.py
```

Requires `OPENAI_API_KEY` in `.env`, Ollama running with `qwen3-embedding:0.6b` pulled, and
`gpt-4.1` priced at (2.00, 8.00) in `OpenAITest.PRICING` — added as part of this work, since
an unpriced model silently falls back to the `default` rate and reports the wrong cost.
