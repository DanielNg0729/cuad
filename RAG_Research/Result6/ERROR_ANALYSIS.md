# Why does it still "not work"? — code + method error analysis

The headline F1 (~0.40) makes the pipeline look broken. It mostly isn't. When you look at
*what kind* of errors it makes (`error_breakdown.py`), the failure is not "can't find the
clause" and not "hallucinates" — it is **span-boundary disagreement with the gold, plus a
forced-answer setup, plus two real code quirks.** Below, reproducible from the result JSONs.

## Finding 1 — most "misses" are the RIGHT clause with the WRONG extent (scoring, not skill)

The scorer counts a prediction correct only at token-Jaccard ≥ 0.5 vs the gold span.
Re-score the same predictions at a looser bar:

| method | F1 @0.5 | F1 @0.3 | F1 @0.1 (any overlap) | lift |
|--------|--------:|--------:|----------------------:|-----:|
| M1 markdown top-2 | 0.426 | 0.523 | **0.730** | **+0.30** |
| M2 markdown top-1 | 0.377 | 0.456 | 0.677 | +0.30 |
| M4 markdown bm25  | 0.344 | 0.422 | 0.598 | +0.26 |
| M5 section top-2  | 0.356 | 0.422 | 0.644 | +0.29 |
| M6 section hybrid | 0.395 | 0.487 | 0.650 | +0.26 |

At "any overlap" (IOU 0.1 = the prediction touches the right clause at all), M1 hits
**0.73**. So ~30 F1 points are the model landing on the correct clause but quoting a
different *extent* than the CUAD annotator (too long, too short, or offset), and being
charged a **false positive AND a false negative** for it.

Confirmed by the error composition (M1):

| false positives (109) | | false negatives (101) | |
|---|---:|---|---:|
| genuine hallucination | **3 (3%)** | genuinely missed (nothing near it) | 43 (43%) |
| right clause, wrong extent | **62 (57%)** | found it, extent off | **58 (57%)** |
| on a zero-gold category | 44 (40%) | | |

**Only 3% of false positives are real hallucinations. 57% of both FP and FN are the same
boundary-disagreement events**, each double-counted. This is the single biggest reason the
number looks bad.

## Finding 2 — no abstention: 40–47% of FPs are on categories with no answer at all

10 of 41 categories have zero gold in these contracts, and ~25 of 41 are absent on any
given contract. The label-centric prompt still asks about every one and hands the model its
"most relevant" chunk, i.e. a leading question. Result: **44/109 (M1) to 50/91 (M4) of all
false positives are on categories that simply are not there.** `retrieve_idxs` has no score
threshold and no "not present" path, and the prompt ("return every substring that matches")
presupposes a match exists.

## Finding 3 — CODE: `validate_span` silently drops whitespace-normalized spans

```python
# OpenAITest.validate_span / extract._validate_span
if span in chunk: return span
if span.lower() in chunk.lower(): ...      # only case-folds; does NOT normalize whitespace
return None
```

docling markdown is **2.9% non-breaking spaces** (`\xa0`); the raw-text section chunks are
**0.0%**. If the model copies a clause but renders the gaps as ordinary spaces (models
routinely normalize whitespace), `span in chunk` is False, the case-fold fallback is also
False, and the span is **thrown away before it is ever scored** — even though the scorer
itself (`get_jaccard`, `_norm_ws`) *is* whitespace-robust. So a correct extraction can die
in validation on the markdown methods (M1–M4) for a reason that has nothing to do with the
model. One-line fix: normalize whitespace (and NBSP) on both sides before the containment
test. (This also partly explains why the finer section chunks, with 0% NBSP, held up.)

## Finding 4 — CODE/CONFIG: the hybrid prefilter is a no-op on the markdown chunks

M6's hybrid is "BM25 → top-`hybrid_n` (10) → cosine-rerank → top-k". But the markdown
contracts have **6–13 chunks**, so "BM25 top-10" keeps **71–100%** of them — the lexical
prefilter selects nothing, and hybrid **degenerates to plain cosine**. On the markdown R@k
table, hybrid@2 = cosine@2 = 0.62 (identical), while *pure* BM25@2 = 0.72. Hybrid only does
real work on the section chunks (220), where top-10 is an actual filter — which is exactly
where it helped (F1 0.356 → 0.395). Lesson: `hybrid_n` must scale with the chunk count, and
hybrid is pointless unless there are many chunks.

## So — is it "RAG that sucks"? No. Ranked by how much it actually costs:

| cause | kind | cost | fix |
|-------|------|------|-----|
| **span-boundary vs IOU 0.5 scorer** | method + scoring | ~+0.30 F1 latent | ask for whole sentences/clauses; or score at clause level, not token-IOU |
| **no abstention** | method | ~40% of FPs | similarity threshold + "not present" option |
| **`validate_span` whitespace drop** | **code bug** | silent, markdown-only | normalize NBSP/whitespace before containment |
| retrieval budget (small k) | config | recall ceiling | budget by tokens, k≈6–8 (see ANALYSIS.md) |
| `hybrid_n` fixed at 10 | **code/config** | hybrid inert on few-chunk docs | scale `hybrid_n` to chunk count |
| the genuine 43% of FN | real | the actual hard part | better prompt / retrieval — the only part that is "model quality" |

**Bottom line:** the model finds the correct clause ~73% of the time (IOU 0.1) and almost
never hallucinates (3%). The pipeline is not failing to *understand* the contracts — it is
losing points to (a) a strict boundary scorer, (b) being forced to answer absent categories,
and (c) a whitespace-fragile validation step. Those are prompt/scoring/code fixes, not a
verdict that "RAG doesn't work here". The one genuinely-model-limited bucket — the 43% of
false negatives with nothing near them — is smaller than any of the above.
