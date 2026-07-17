# Why is the performance bad? — root-cause analysis

Model: **gpt-5.4**. Data: **6 contracts, 41 categories, 179 gold answers**.
Most numbers below come from `diagnose.py` → `diagnosis.json`; the chunking experiment
comes from `recall_at_k_section.py`. Not from intuition.

Best method (M1, top-2 cosine over the markdown chunks) scores **F1 0.426 / precision
0.417 / recall 0.436**. The full-scan baseline — same model, same prompt, but every chunk
shown — scores **F1 0.461**. So *RAG is losing to simply showing the model everything*,
while making **41 LLM calls instead of 6–13**. That is the fact the analysis has to explain.

> **Update — this study now includes two extra methods (M5, M6)** that swap in a finer
> "section" chunking (1.1/1.2 reading-order split, 220 chunks vs 60). Short version: finer
> chunking did **not** raise F1 at the same k (M5 section top-2 = 0.356, M6 section hybrid
> top-3 = 0.395, both below M1's 0.426) — but only because a section chunk is ~6× smaller,
> so "top-2" starves the model of context. The full write-up is the "chunking experiment"
> section below; the stage decomposition that follows is on the markdown methods (M1–M4).

---

## The method: decompose recall into the three stages where it can die

A gold answer can only be missed in one of three places, and they multiply:

```
recall  =  coverage  ×  retrieval R@k  ×  extraction X|R
```

| stage | question | if it fails, whose fault? |
|-------|----------|---------------------------|
| **Coverage** | is the answer even present in the chunked markdown? | docling / chunking |
| **Retrieval R@k** | given it's in a chunk, did top-k pick that chunk? | the embedding / retriever |
| **Extraction X\|R** | given the right chunk was retrieved, did the LLM return the span? | the prompt / model |

Measured:

| method | coverage | R@k | X\|R | product | actual recall |
|--------|---------:|----:|-----:|--------:|--------------:|
| M1 top-2 cosine | 0.99 | **0.64** | **0.68** | 0.44 | 0.436 |
| M2 top-1 cosine | 0.99 | **0.49** | 0.67 | 0.32 | 0.335 |
| M4 top-1 BM25   | 0.99 | 0.53 | 0.59 | 0.31 | 0.313 |

The predicted product matches the observed recall almost exactly, so the decomposition
is sound. Now read it.

---

## Verdict 1 — It is NOT the chunking or the markdown conversion

**Coverage = 99.4%** (178 of 179 gold answers are present in some chunk; strict
verbatim containment still gives 93.3%). Exactly **one** gold answer is unreachable.

So the chunker and the stored docling markdown are essentially innocent. Chunk sizes
are also fine — 60 chunks, **median 414 tokens**, max 7,610, and **zero** chunks exceed
the 8,000-token embedding cap, so nothing is being silently truncated before embedding.

> (The docling `std::bad_alloc` bug I found earlier affects the *web UI's live
> re-conversion*, not `test_chunking.json`, which is what this study reads. It is a real
> bug but it is not what is causing these scores.)

**Chunking is not the bottleneck. Stop looking here.**

---

## Verdict 2 — Retrieval is the single biggest loss, and the EMBEDDING is the weak part

At top-1, cosine retrieval finds the chunk holding the answer **only 49% of the time**.
And remember: these contracts have just **6–13 chunks**. The retriever is choosing 1
document out of ~10 and gets it wrong half the time.

Recall of the retriever alone, as a function of k:

| k | cosine (embedding) | BM25 (lexical) |
|---|-------------------:|---------------:|
| 1 | 0.49 | **0.52** |
| 2 | 0.62 | **0.72** |
| 3 | 0.69 | **0.86** |
| 5 | 0.86 | **0.93** |
| 8 | 0.94 | **0.99** |

**A bag-of-words BM25 beats `text-embedding-3-small` at every single k** (0.86 vs 0.69 at
k=3) — on *recall*. This survives every artifact check: restricted to gold answers that
occur in exactly one chunk (n=131, so no retriever can get lucky), BM25 still wins
0.84 vs 0.66 at k=3.

> ### ⚠️ But BM25 still LOSES end-to-end. Read this before concluding "switch to BM25".
>
> Cosine beats BM25 on F1 in **both** studies (3 contracts: 0.348 vs 0.294; 6 contracts:
> 0.377 vs 0.344). Higher retrieval recall did **not** translate into a better score. Why:
>
> | at top-1 | count | extraction rate |
> |---|---:|---:|
> | both retrievers hit the gold chunk | 63 | cosine **68%**, bm25 60% |
> | only cosine hit | 18 | 67% |
> | only BM25 hit (its *extra* recall) | 24 | **67%** — converts fine, not junk |
>
> Net true positives are a wash (**cosine 55, BM25 54**). The decider is false positives:
>
> | | FP total | **FP on zero-gold categories** |
> |---|---:|---:|
> | M2 cosine | 79 | **29** |
> | M4 BM25 | 91 | **50** |
>
> **Lexical matching always finds a keyword hit.** Ask BM25 for "Source Code Escrow" in a
> contract that has none and it still returns a chunk containing escrow-ish tokens — which
> then baits the LLM into extracting something. Cosine, for an absent category, returns a
> semantically generic chunk and the model more often correctly stays silent.
>
> **BM25 is better at finding a clause that exists, and much worse at not finding one that
> doesn't.** Since 10/41 categories have zero gold — and ~25 of 41 are absent on any given
> contract — precision is the binding constraint, so BM25's recall win is more than eaten
> by its hallucination bait. The correct read is *not* "the embedding is useless"; it is
> "**the embedding under-retrieves, BM25 over-retrieves, and neither can abstain**".

With that caveat established, the embedding's recall weakness is still real, and here is
why it under-retrieves:

1. **Query/document asymmetry.** We embed a *category definition* ("Audit Rights — does
   the contract give a party the right to audit the books and records…?") and cosine it
   against *raw legalese*. Those are different registers. The embedding model is a
   general-purpose symmetric encoder; nobody trained it to match "a definition of a legal
   concept" to "the clause that instantiates it". So the nearest chunk is frequently the
   one that is *topically* generic (the preamble, the definitions section) rather than the
   one containing the operative clause.

2. **CUAD clauses are lexically signposted, and the embedding throws that signal away.**
   Governing-law clauses literally contain "governed by the laws of"; insurance clauses
   say "insurance"; audit clauses say "audit". BM25 keys on exactly those tokens and wins.
   Compressing a chunk to a single 1,536-dim vector discards the rare, high-information
   tokens that actually identify the clause.

3. **One vector per chunk, and the chunks are wildly uneven** (414-token median but a
   7,610-token max — an 18× spread, because the split follows whatever headings docling
   happened to find). A 7,600-token chunk that contains one 2-line audit clause has that
   clause's signal averaged into oblivion, while a short heading-ish chunk gets an
   artificially sharp vector. The retriever is therefore biased toward short, generic
   chunks — precisely the wrong bias.

**This is the #1 problem — but state it precisely:** the embedding retriever has poor
*recall* (a 1970s keyword algorithm finds the right chunk more often). It is nonetheless
the better end-to-end choice today because it hallucinates less on absent categories.
Neither retriever can say "not here", and that is the gap worth closing.

---

## Verdict 3 — The prompt is the second, independent loss (~a third of what survives)

Even when retrieval hands the model the correct chunk, it extracts the gold span only
**X|R ≈ 0.67** of the time. A third of the answers die *with the right text already in the
context window*. Retrieval cannot be blamed for those.

Why:

- **The category descriptions are CUAD's yes/no *classification* questions, being used as
  *extraction* instructions.** "Is there a covenant not to sue?" is not "return the exact
  substring that constitutes the covenant not to sue." (The earlier `experiment_prompt.py`
  ablation already pointed at this; it is confirmed here.)
- **Verbatim validation is brutally strict.** We keep a span only if it literally appears
  in the chunk (`validate_span`). Docling markdown is riddled with non-breaking spaces
  (`\xa0`); a reasoning model that normalizes whitespace or smart-quotes produces a
  semantically perfect span that gets **silently dropped**. Evidence for how fiddly the
  text is: even the *gold* answers only verbatim-match a chunk 93.3% of the time.
- **Span-boundary disagreement.** The scorer needs token-Jaccard ≥ 0.5 against the gold.
  The model routinely finds the right clause with the wrong extent (too short, or the
  whole section) → it is charged a **false positive AND a false negative** for one answer.
  Evidence: **65 of M1's 109 false positives are on categories that DO have gold** — it was
  in the right neighbourhood and still missed.

Consequence: even with a *perfect* retriever, recall is capped at ≈ 0.99 × 1.00 × 0.67 ≈
**0.66**. Fixing retrieval alone cannot get you past two-thirds.

---

## Verdict 4 — The false positives are structural: there is no abstention

`retrieve_idxs` **always returns k chunks**, no matter how low the similarity. There is no
threshold and no "this category isn't here" escape hatch.

Now recall the ground truth: **10 of the 41 categories have no gold answer anywhere in
these 6 contracts.** We nevertheless ask about them on every contract, hand the model the
"most relevant" chunk, and effectively say *"here is the text for Volume Restriction —
extract it."* That is a **leading question**, and the model obliges.

| method | FP total | FP on categories with **zero** gold | FP on real categories |
|--------|---------:|------------------------------------:|----------------------:|
| M1 | 109 | **44 (40%)** | 65 |
| M2 |  79 | 29 (37%) | 50 |
| M4 |  91 | **50 (55%)** | 41 |

**~40% of all noise is manufactured by asking about clauses that aren't in the contract.**

This is also why label-centric RAG is *structurally* more FP-prone than the full scan:
full-scan asks "*which of these 41 categories appear in this chunk?*" — the model can
answer "none". RAG asks "*extract the Audit Rights clause from this, its most relevant
text*" — which presupposes the clause exists.

---

## Verdict 5 — For documents this size, RAG is the wrong architecture

The premise of retrieval is *"the corpus is too big to read, so we must select."* These
contracts are **6–13 chunks**. That premise is false here.

| | LLM calls / contract | sees | F1 |
|---|---:|---|---:|
| Full scan | 6–13 | **everything** (R = 1.00) | **0.461** |
| RAG top-2 | 41 (one per category) | 2 chunks per category | 0.426 |

RAG here is **more expensive AND less accurate**. It pays for a selection step that we do
not need and that is only 49–64% accurate. The information was already affordable to show
in full.

---

## The chunking experiment (M5, M6) — does finer chunking help?

Hypothesis under test: the markdown chunks are coarse and wildly uneven (median 414
tokens, max 7,610). Splitting on section numbering (1.1, 1.2, ARTICLE, …) into ~1,500-char
pieces gives **220 chunks instead of 60** — smaller, more uniform, one clause-ish idea
each. Does that improve retrieval, and does it improve the end-to-end score?

**Coverage goes UP.** Gold answers present in some chunk: **98.9%** for section vs **93.3%**
for markdown. (Section chunks are cut from the raw CUAD text, so gold spans match verbatim;
markdown carries docling's NBSP noise. That is a real, if incidental, win — and a
confound: M5/M6 change both granularity *and* text substrate.)

**But R@k at fixed k goes DOWN — a lot:**

| R@k | markdown (60 chunks) | section (220 chunks) |
|-----|---------------------:|---------------------:|
| cosine @2 | 0.62 | **0.43** |
| bm25 @3   | 0.86 | **0.56** |
| hybrid @3 | 0.69 | **0.55** |

This looks damning until you see *why*: **"top-2 of 220 tiny chunks" hands the LLM far less
text than "top-2 of 60 fat chunks."** Measured average context per category call:

| method | avg tokens shown to the LLM | F1 |
|--------|----------------------------:|---:|
| M1 markdown cosine top-2 | **3,127** | 0.426 |
| M5 section cosine top-2   | **479**   | 0.356 |
| M6 section hybrid top-3   | **744**   | 0.395 |

M5 was fed **6.5× less context** than M1. That, not chunk quality, is why it scored lower.
The fair comparison is at **equal token budget**:

| retrieval | tokens | R@k |
|-----------|-------:|----:|
| markdown cosine top-2 | 3,127 | 0.62 |
| section cosine top-5 | 1,204 | 0.63 |
| section cosine top-8 | 1,929 | **0.74** |

At **~60% of the token budget**, section top-8 already **beats** markdown top-2 on retrieval
recall (0.74 vs 0.62). Per token of context, the fine chunks are *more* efficient, and they
carry less irrelevant text around each hit.

**So the honest verdict on chunking:**
1. Finer chunking is **not worse** — it is neutral-to-better on retrieval *per token* and
   clearly better on coverage (99% vs 93%).
2. It **loses at fixed small k** purely because each chunk is smaller. M5/M6 used k=2/3,
   which starved the model. To exploit fine chunks you must raise k (≈6–8) or, better,
   retrieve to a fixed **token budget** instead of a fixed chunk count.
3. **Hybrid retrieval is the clear winner among the section runs.** BM25-prefilter →
   cosine-rerank lifts F1 from 0.356 (M5 plain cosine) to 0.395 (M6), and on the section
   R@k table hybrid ≥ cosine at every k. This is the one config change here that helped.

The experiment did not raise the headline F1, but it was still informative: it proves the
retrieval loss in Verdict 2 is a **budget/k problem, not a chunk-quality problem**, and it
shows hybrid retrieval is the right direction.

**M7 (section hybrid, wider k=5) closes the loop.** Take M6's hybrid and widen the final
budget from top-3 to top-5:

| method | R@k | recall | precision | F1 |
|--------|----:|-------:|----------:|---:|
| M6 section hybrid top-3 | 0.55 | 0.380 | 0.412 | 0.395 |
| **M7 section hybrid top-5** | **0.66** | **0.425** | 0.367 | 0.394 |

Widening k did exactly what the budget argument predicted: retrieval jumped to the **best
R@k of any method (0.66 > M1's 0.62)** and recall rose to 0.425 (near M1's 0.436). The fine
chunks were genuinely starved, and hybrid+budget fixes retrieval. **But F1 did not move** —
because precision fell in lockstep (0.412 → 0.367, FP 97 → 131), and 45 of those extra FPs
are on zero-gold or boundary. With **no abstention**, every extra chunk you retrieve is
another chance to answer a category that isn't there. M7 is the proof that **retrieval is now
solved and abstention is the binding constraint**: you cannot buy F1 with more recall until
the model is allowed to say "not present".

---

## Summary — apportioning the blame

| cause | severity | evidence |
|-------|----------|----------|
| **Too little context retrieved (small k / small budget)** | 🔴 **primary** | cosine R@2 = 0.62; M1 wins only by showing 3,127 tok vs M5's 479 |
| **Prompt is classification-shaped, validation too strict** | 🔴 **primary** | X\|R = 0.67 even with the right chunk in context |
| **No abstention → manufactured FPs** | 🟠 major | 40–55% of FPs are on zero-gold categories; hybrid/BM25 make this worse |
| **RAG architecture unjustified at this doc size** | 🟠 major | full-scan is cheaper *and* better (0.461 vs 0.426) |
| Chunk **granularity** (markdown vs section) | 🟢 not the issue | equal-budget retrieval is a wash; coverage 93→99% |
| Chunking / markdown conversion (coverage) | 🟢 not the issue | 99.4% of gold is reachable |

**So: it is not "RAG sucks" in the abstract, and it is not the chunking. It is that (a) the
retriever is fed too little context to hit the right chunk (a k/budget problem, confirmed by
the section experiment), and (b) the prompt then loses a third of the answers it *is* handed.
Those two are multiplicative, which is why the end-to-end numbers look so poor. The one
retriever change that clearly helped was hybrid (M6): lexical recall + dense rerank.**

---

## What to do, ranked by expected gain per unit of effort

1. **Retrieve to a token budget, not a fixed k — or just stop using RAG.** M5/M6 proved
   fixed small k starves fine chunks. On markdown, k=5 takes cosine to 0.86; on section,
   match ~2–3k tokens (k≈6–8). Since contracts are only 6–13 markdown chunks, `k = all`
   (full scan) is both affordable and optimal. **If you keep RAG, budget by tokens.**
2. **Adopt hybrid retrieval (M6's recipe): BM25 prefilter → dense rerank.** It was the only
   retriever change in this study that helped (F1 0.356 → 0.395 on section chunks; hybrid ≥
   cosine at every k). Keep the finer section chunks — they lift coverage 93→99% and are
   more efficient per token — but feed them at k≈6–8.
3. **Abstention is the highest-value fix, and it must come *with* any retriever change.**
   If the best chunk's score is below τ, return "not present" instead of forcing an
   extraction. This directly attacks the 29–50 FPs on zero-gold categories — and it is
   *precisely* what prevents BM25/hybrid's better recall from paying off. Do not add lexical
   recall without it: you trade extra false positives for the recall gain and end up worse,
   which is exactly what M4 (bm25) did.
4. **Rewrite the category prompts as extraction instructions with a positive example**
   (the ArmB treatment from `experiment_prompt.py`), and **normalize whitespace/NBSP before
   `validate_span`** so semantically-correct spans stop being silently discarded. (Section
   chunks already dodge most of the NBSP problem — another reason to keep them.)
5. **Ask for whole clauses/sentences** so span boundaries clear the Jaccard ≥ 0.5 bar,
   killing the paired FP+FN on the 65 "right area, wrong extent" cases.

Fixes 1–3 attack retrieval (budget + hybrid + abstention); fixes 4+5 attack extraction (the
0.67 ceiling). **You need both halves** — they multiply.
