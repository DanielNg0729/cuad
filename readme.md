# LLMs for Legal Contract Clause Extraction — CUAD

Can instruction-following LLMs extract legal clauses from commercial contracts without
task-specific training, and is retrieval-augmented generation (RAG) the right architecture
for it?

This repository is the full record of that investigation: ~138 experimental configurations
across nine studies, all scored by one harness so every number is mutually comparable,
validated on the **complete 102-contract CUAD test set** against a published external
benchmark — plus a working contract-review web application.

Built on the [CUAD](https://www.atticusprojectai.org/cuad) benchmark (Hendrycks et al.,
NeurIPS 2021): 41 clause categories, expert-annotated verbatim spans.

---

## Headline result

Scored under the protocol of [**ContractEval**](https://arxiv.org/pdf/2508.03080)
(arXiv:2508.03080), which evaluates the *same model* on the *same dataset* using **full
documents and no retrieval** — so these rows are directly comparable:

| System | P | R | F1 | F2 | Jaccard | False refusal ↓ | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ContractEval GPT-4.1** — full document, published | 0.595\* | **0.694**\* | **0.641** | **0.672** | 0.472 | 0.071 | **≈ $50** |
| **This repo — RAG, RRF n20 top-8** | 0.649 | 0.625 | **0.637** | 0.629 | **0.488** | **0.049** | **$18.20** |
| This repo — RAG, RRF n10 top-5 | **0.666** | 0.571 | 0.615 | 0.588 | 0.469 | 0.054 | **$12.68** |

<sub>\* The paper prints only F1/F2; P and R are derived algebraically, and the derivation
reproduces its published F1/F2 to four decimals for all four proprietary models.</sub>

**RAG matches full-document prompting on F1 — 0.637 vs 0.641, a gap of 0.004 — at a quarter
to a third of the cost, while reading ~3% as much text per call.** It produces *tighter*
answers (Jaccard 0.488 vs 0.472) and wrongly refuses to answer *less* often (0.049 vs 0.071).
The entire deficit is recall; precision more than compensates.

It does not beat reading the whole contract. It matches it, cheaply, by trading a little
recall for precision.

---

## Pipeline

```
PDF ─docling─▶ Markdown ─chunking─▶ chunks
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │ FULL SCAN                    │ RAG (label-centric)         │
        │ every chunk → all 41 fields  │ embed chunks + categories   │
        │ N chunks → N LLM calls       │ retrieve top-k per category │
        │                              │ → 1 LLM call per category   │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
              verbatim span validation ─▶ char offsets ─▶ scoring / highlights
                                      │
                       (optional) LLM verification pass
```

**Best configuration found:** section chunking (1,500 chars) → `qwen3-embedding:0.6b` (local,
free) → Reciprocal Rank Fusion (BM25 top-N ∪ cosine top-N, fused by 1/(60+rank)) → top-5/8
chunks → one structured-output call per category.

---

## What was learned

| # | Question | Answer |
|---|---|---|
| 1 | Best chunking? | **Section-based**, retrieved to a *token budget* not a fixed k. Fixed-k comparisons are confounded — a Markdown chunk is ~6× larger, so "top-2" means different things. At equal tokens, section chunks win (R@k 0.74 vs 0.62) and lift coverage 93.3% → 98.9%. |
| 2 | Best retrieval? | **Reciprocal Rank Fusion** (F1 0.457) > plain cosine > hybrid prefilter > BM25 alone. BM25 retrieves *better* (R@3 0.86 vs 0.69) but scores *worse* end-to-end — it always finds a keyword hit, so it hallucinates on absent categories. |
| 3 | Does LLM-as-verifier work? | **Yes — the single highest-value addition.** F1 0.382 → **0.576** on GPT-5.5, the only method in the study to beat a full-scan baseline (0.480). Extract with a cheap model, verify with a strong one. |
| 4 | Best embedding model? | **Free local models beat the paid API.** `qwen3-embedding:0.6b` (0.457) and `bge-m3` (0.449) both beat OpenAI `text-embedding-3-small` (0.406), at zero marginal cost on a 4 GB GPU. |
| 5 | Category analysis | Categories with a canonical location and stereotyped phrasing do well (Parties 0.970, Governing Law 0.908); those needing a judgement about what counts do badly (Volume Restriction 0.133). |
| 6 | Is the low score real? | **Largely a scoring artefact.** Only **3%** of false positives are genuine hallucinations; **57%** of *both* error types are the same span-boundary disagreement counted twice. At any-overlap scoring, F1 rises 0.426 → **0.730**. |
| 7 | Does prompt design help? | **Depends on the model.** Richer prompts moved GPT-5.5 from 0.48 → 0.54 but degraded GPT-5.4-mini from 0.38 → 0.29. The prompt belongs to the model config, not the pipeline. |
| 8 | Which LLM? | Accuracy scales sub-linearly with price. GPT-5.4-mini 0.377 @ $0.05, GPT-5.4 0.461 @ $0.26, GPT-5.5 0.480 @ $0.46. |
| 9 | What's the bottleneck? | **Abstention and extraction — not retrieval.** A free 60-config retrieval sweep projects F1 0.472–0.476 across the *entire* parameter space (measured runs: 0.473, 0.474). Abstention is worth an estimated **+0.06 to +0.13** by comparison. |
| 10 | Does it hold at full scale? | **Yes** — see the headline table. 102 contracts, 4,182 questions, 2,643 gold answers. |

### Two findings worth singling out

**Retrieval budget buys recall, not F1.** Widening top-5 → top-8 lifted R@k by 7.6 points and
moved end-to-end F1 by **+0.001**. Every extra true positive cost ~3 false positives, because
the pipeline answers every question whether or not the clause exists.

**The scoring rule can reverse the conclusion.** The same budget increase is worth +0.001 F1
under this repo's scorer (token-Jaccard ≥ 0.5) and **+0.022** under ContractEval's (prediction
must *fully cover* the gold). Ours punishes over-long answers; theirs rewards them. Every
metric claim here names its scorer.

---

## Web application

A contract-review UI that turns the research into something a reviewer can operate.
See [webui/README.md](webui/README.md).

```bash
pip install -r webui/requirements.txt
cp .env.example .env        # add your keys
cd webui && python app.py   # http://127.0.0.1:5000
```

| Capability | Detail |
|---|---|
| **Dual extraction modes** | Full scan and RAG behind a dropdown — the central research question is demonstrable in the product |
| **True PDF rendering** | Actual PDF via PDF.js (vendored locally, works offline); spans located in the text layer with whitespace-tolerant search that matches across line breaks, tinted per category |
| **Review workflow** | Approve / reject / edit / add answers; click to scroll-and-flash on the PDF; state persists server-side per document |
| **Per-category AI verification** | Audit a category with a model chosen *independently* of the extractor → correct / incorrect / unsure + reason. This is finding #3 shipped as a feature |
| **User-defined categories** | Add a clause type with a label + description; sent to the model exactly like the 41 built-ins — extends beyond CUAD with no code changes |
| **Excel export** | Category · Answer · Status · AI verdict · Page. Rejected answers excluded |
| **Live progress** | Background extraction with a polled progress bar, so a slow reasoning model looks *running* rather than stuck |

---

## Repository layout

### Core pipeline
| Path | Purpose |
|---|---|
| [`chunking.py`](chunking.py) | Four chunking strategies (fixed / recursive / section / markdown) as a separate deterministic stage |
| [`PdfToMarkdownBatch.py`](PdfToMarkdownBatch.py) | Resumable docling conversion of all 102 contract PDFs |
| [`OpenAITest.py`](OpenAITest.py) | Full-scan extraction (OpenAI), structured output, span validation, cost accounting |
| [`Groq.py`](Groq.py) / [`GroqTest.py`](GroqTest.py) | Same pipeline on Groq-hosted open models |
| [`evaluate.py`](evaluate.py) | **The scorer.** P / R / F1 / F2 / AUPR / Jaccard — used by every experiment |
| [`experiment_prompt.py`](experiment_prompt.py) | Three-arm prompt ablation across 41 categories |

### Studies
| Path | Contents |
|---|---|
| [`RAG_Research/`](RAG_Research/) | RAG study 1 — 4 retrieval methods × 2 models × 3 contracts |
| [`RAG_Research/Result6/`](RAG_Research/Result6/) | RAG study 2 — 7 methods + 15-cell hybrid grid, 6 contracts |
| [`.../Result6/ANALYSIS.md`](RAG_Research/Result6/ANALYSIS.md) | **Why performance is what it is** — recall decomposed into coverage × retrieval × extraction |
| [`.../Result6/ERROR_ANALYSIS.md`](RAG_Research/Result6/ERROR_ANALYSIS.md) | Error composition, IOU threshold sweep, two silent code defects |
| [`.../TestEmbeddedModel/`](RAG_Research/Result6/TestEmbeddedModel/) | 5 embedders, retrieval-only (cost $0) |
| [`.../TestAblation/`](RAG_Research/Result6/TestAblation/) | 14 arms — embedder × retrieval strategy × budget, end-to-end |
| [`.../TestRerank/`](RAG_Research/Result6/TestRerank/) | 10 rank-fusion arms + per-category bad-case attribution |
| **[`TestRAGFinal/`](TestRAGFinal/)** | **Full-scale validation** — 102 contracts, GPT-4.1, ContractEval comparison, free 60-config sweep |

### Reports
| File | Audience |
|---|---|
| [`ReportInternshipFinal.docx`](ReportInternshipFinal.docx) | Internship report (A*STAR-internal version) |
| [`Internship_Report.docx`](Internship_Report.docx) | Longer variant with academic background sections |
| [`TestRAGFinal/README.md`](TestRAGFinal/README.md) | Full-scale results write-up |
| [`TestRAGFinal/all_arms_summary.md`](TestRAGFinal/all_arms_summary.md) | All arms, both protocols, vs. the paper |

---

## Reproducing

```bash
# 1. chunk
python chunking.py --strategy section --chunk_chars 1500

# 2. a full-scan baseline
python OpenAITest.py --model gpt-4.1
python evaluate.py --model_path trained_models/gpt-4.1__openai

# 3. the full-scale RAG run  (~$12.68 / 7.7 min, or ~$18.20 / 11.4 min at top-8)
python TestRAGFinal/run_compare.py --dry-run          # cost projection, no LLM calls
python TestRAGFinal/run_compare.py --shortlist-n 20 --top-k 8

# 4. scoring + analysis (all free)
python TestRAGFinal/contracteval_score.py --result TestRAGFinal/results/<arm>/gpt-4.1.json
python TestRAGFinal/per_category.py      --result TestRAGFinal/results/<arm>/gpt-4.1.json
python TestRAGFinal/summarize_arms.py
python TestRAGFinal/sweep_retrieval.py                # 60-config retrieval sweep, $0
```

**Requirements:** `OPENAI_API_KEY` (and optionally `GROQ_API_KEY`) in `.env` — copy from
[`.env.example`](.env.example); `.env` is gitignored and has never been committed.
For the local embedders, [Ollama](https://ollama.com) with `qwen3-embedding:0.6b` and
`bge-m3` pulled.

### What is and isn't in the repo

- **`dataset/`** (~98 MB of source contract PDFs) is gitignored. It is public CUAD data —
  download from [the Atticus Project](https://www.atticusprojectai.org/cuad).
- **Full-scale result JSONs** are 50–75 MB each and are stored **gzipped** (3–5 MB, identical
  content). Run `gunzip TestRAGFinal/results/<arm>/gpt-4.1.json.gz` to restore. Each contains,
  per (contract, category): ground truth, predictions, TP/FN/FP, the retrieved chunk text as
  both a concatenated blob and an index→text map, plus token usage and cost.
- Every runner is **idempotent** — it skips any arm whose result file exists, so adding arms
  only pays for the new ones.

---

## Cost discipline

Total metered API spend across all nine studies: **≈ $60.5**.

| Study | Arms | Spend |
|---|---:|---:|
| Prompt ablation | 9 | $4.42 |
| RAG study 1 | 8 | $5.04 |
| RAG study 2 + hybrid grid | ~20 | $5.04 |
| Embedder retrieval test | 15 | **$0.00** |
| Embedder × retrieval ablation | 14 | $9.33 |
| Rank-fusion sweep | 10 | $5.81 |
| **Full-scale validation** | 2 | **$30.88** |
| Retrieval parameter sweep | 60 | **$0.00** |

The two $0.00 rows are retrieval-only — R@k needs no LLM calls, so large parts of the search
space were ruled out before paying for anything. Provably identical configurations reuse
predictions from disk rather than re-purchasing them.

---

## Upstream contributions

Two pull requests to [docling](https://github.com/docling-project/docling) (IBM Research),
both reviewed and merged. PDF headings were being emitted at a single level, flattening
document hierarchy — which is also an upstream cause of the uneven-chunk problem measured in
[`ANALYSIS.md`](RAG_Research/Result6/ANALYSIS.md).

- [**#3633**](https://github.com/docling-project/docling/pull/3633) — infer heading levels from
  numbering (PART I / Section 2 / 1.1.1 / (a) / (i)) with a font-size fallback.
  Merged 23 Jun 2026.
- [**#3688**](https://github.com/docling-project/docling/pull/3688) — infer heading levels from
  PDF bookmarks/ToC, confidence-gated. Precedence: bookmarks > numbering > style.
  Merged 1 Jul 2026.

---

## References

1. Hendrycks, Burns, Chen & Ball (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal
   Contract Review.* NeurIPS Datasets & Benchmarks.
   [arXiv:2103.06268](https://arxiv.org/abs/2103.06268)
2. Liu, Li, Ma, Zhao & Du (2025). *ContractEval: Benchmarking LLMs for Clause-Level Legal Risk
   Identification in Commercial Contracts.* [arXiv:2508.03080](https://arxiv.org/abs/2508.03080)
3. Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion Outperforms Condorcet and
   Individual Rank Learning Methods.* SIGIR.
4. Chen et al. (2024). *BGE M3-Embedding.* [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
