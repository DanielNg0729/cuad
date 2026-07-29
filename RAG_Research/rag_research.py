"""
RAG pipeline research on the SAME 3 CUAD contracts used everywhere else.

Compares FOUR retrieval strategies, all label-centric (one LLM call per clause
category over just the chunks retrieved for that category -- the pipeline the web
UI uses), holding model / chunks / span-validation / scorer constant and varying
only how (and how many) chunks are retrieved, plus one optional verification pass:

  M1  top2_cosine            : embed chunks + category, cosine-rank, keep top-2
                               chunks per category (this is exactly the UI default).
  M2  top1_cosine            : same, but keep only the single best chunk.
  M3  top1_cosine_llmcheck   : M2, then ONE extra LLM call per contract that sees
                               the WHOLE contract + all 41 categories + M2's
                               candidate answers and reasons over them to prune
                               hallucinations / add clear misses (reduce noise).
  M4  top1_bm25              : M2, but retrieve the single best chunk by BM25
                               (lexical) instead of cosine similarity.

Every method is scored with the project's own evaluate.evaluate (the exact metric
suite used for the existing results/*__3contracts.json baselines, so numbers are
directly comparable) AND broken down per contract / per category the way
experiment_prompt.py reports.

Output -> RAG_Research/results/<method>/<model>.json
Summary -> RAG_Research/results/summary.json   (+ printed table)

Run (defaults to gpt-5.4 and gpt-5.5, all 4 methods):
    python RAG_Research/rag_research.py
    python RAG_Research/rag_research.py --models gpt-5.4
    python RAG_Research/rag_research.py --models gpt-4o-mini --limit-cats 3 --contracts BIOPURECORP   # cheap smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from pydantic import Field, create_model

# --- make the repo root importable and the cwd, so the project's relative-path
#     helpers (category_descriptions.csv, test.json, ...) resolve unchanged. -----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from langchain_core.prompts import ChatPromptTemplate       # noqa: E402
from langchain_openai import ChatOpenAI                      # noqa: E402

from OpenAITest import (                                     # noqa: E402
    validate_span, call_with_retry, load_categories, estimate_cost, CATEGORY_CSV,
)
from evaluate import get_answers, evaluate as eval_metrics   # noqa: E402
from experiment_prompt import CONTRACTS, breakdown          # noqa: E402

# --- config -----------------------------------------------------------------

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_1M = 0.02                 # USD / 1M tokens
DEFAULT_MODELS = ["gpt-5.4", "gpt-5.5"]
CONCURRENCY = 12                          # per-category calls are small + independent
_MAX_EMBED_TOKENS = 8_000                 # text-embedding-3-small hard-caps at 8192

OUT_DIR = ROOT / "RAG_Research" / "results"
CACHE_DIR = ROOT / "RAG_Research" / ".cache"

METHODS = {
    "M1_top2_cosine": {
        "search": "cosine", "top_k": 2, "check": False,
        "desc": "Label-centric RAG: top-2 chunks per category by cosine similarity "
                "(text-embedding-3-small), one LLM call per category. Matches the web UI.",
    },
    "M2_top1_cosine": {
        "search": "cosine", "top_k": 1, "check": False,
        "desc": "Label-centric RAG: single best chunk per category by cosine similarity.",
    },
    "M3_top1_cosine_llmcheck": {
        "search": "cosine", "top_k": 1, "check": True,
        "desc": "M2 (top-1 cosine) plus one LLM verification pass per contract over the "
                "full contract + all 41 categories + candidate answers, reasoning to "
                "prune hallucinations and recover clear misses.",
    },
    "M4_top1_bm25": {
        "search": "bm25", "top_k": 1, "check": False,
        "desc": "Label-centric RAG: single best chunk per category by BM25 (lexical) "
                "retrieval instead of cosine similarity.",
    },
    # M5/M6 use the finer SECTION chunking (1.1/1.2 reading-order split, ~1500 chars,
    # ~220 chunks vs 60 for markdown). Point the runner at the section chunk file with
    # --chunk-file for these; the retrieval config is what differs between them.
    "M5_section_top2_cosine": {
        "search": "cosine", "top_k": 2, "check": False, "chunking": "section",
        "desc": "Section chunking (1.1/1.2 reading order, ~1500 chars) + top-2 chunks per "
                "category by cosine similarity. Tests whether finer chunks help retrieval.",
    },
    "M6_section_hybrid_bm25_cosine": {
        "search": "hybrid", "top_k": 3, "hybrid_n": 10, "check": False, "chunking": "section",
        "desc": "Section chunking + HYBRID retrieval: BM25 prefilter to top-10 chunks, then "
                "cosine-rerank those to top-3, and answer from those 3 chunks.",
    },
    "M7_section_hybrid_top5": {
        "search": "hybrid", "top_k": 5, "hybrid_n": 10, "check": False, "chunking": "section",
        "desc": "Section chunking + HYBRID retrieval: BM25 prefilter to top-10 chunks, then "
                "cosine-rerank to top-5 (wider final budget than M6's top-3).",
    },
    # Ablation: same final budget (3 chunks/category) on the SAME section chunking,
    # isolating what the hybrid rerank in H_bm5_cos3 actually buys over either
    # single-scorer retriever alone.
    "M8_section_bm25_top3": {
        "search": "bm25", "top_k": 3, "check": False, "chunking": "section",
        "desc": "Section chunking + BM25-only top-3 chunks per category (lexical, no "
                "cosine rerank). Ablation baseline for H_bm5_cos3 at equal top-3 budget.",
    },
    "M9_section_cosine_top3": {
        "search": "cosine", "top_k": 3, "check": False, "chunking": "section",
        "desc": "Section chunking + cosine-only top-3 chunks per category (dense, no BM25 "
                "prefilter). Ablation baseline for H_bm5_cos3 at equal top-3 budget.",
    },
}

# Hybrid GRID on the section chunks: BM25 prefilter in {5, 10, 15} x cosine-rerank k in
# {1,2,3,5,8,10}. Combos where k > prefilter are skipped (cosine cannot pick more than
# BM25 shortlisted). (bm=10, k=3) and (bm=10, k=5) already exist as M6 and M7, so they
# are skipped here to avoid re-running the same config.
_EXISTING_HYBRID = {(10, 3), (10, 5)}   # M6, M7
for _bm in (5, 10, 15):
    for _cos in (1, 2, 3, 5, 8, 10):
        if _cos > _bm or (_bm, _cos) in _EXISTING_HYBRID:
            continue
        METHODS[f"H_bm{_bm}_cos{_cos}"] = {
            "search": "hybrid", "top_k": _cos, "hybrid_n": _bm, "check": False,
            "chunking": "section",
            "desc": f"Section chunking + HYBRID: BM25 prefilter top-{_bm} -> cosine-rerank "
                    f"top-{_cos}.",
        }


# --- per-category extraction schema + prompts (mirrors webui/rag.py) ----------

LabelExtraction = create_model(
    "LabelExtraction",
    spans=(Optional[list[str]], Field(
        default=None,
        description="Every EXACT verbatim substring from the provided sections that "
                    "matches the requested clause category. Empty list if none apply.")),
)

RAG_SYSTEM = (
    "You are a legal contract analyst. You are given a few sections of a contract "
    "that were retrieved as the most relevant to ONE clause category. Return every "
    "EXACT verbatim substring from those sections that matches the category's "
    "description. Copy character-by-character; never paraphrase or invent. Return "
    "an empty list if the category does not appear."
)
RAG_USER = (
    "Clause category: {label}\n"
    "What counts as this category: {description}\n\n"
    'Retrieved contract sections:\n"""\n{chunk}\n"""'
)


# --- M3 verification schema + prompts ----------------------------------------

CheckedCat = create_model(
    "CheckedCat",
    category=(str, Field(description="The exact clause-category label being judged.")),
    reasoning=(str, Field(description="One or two sentences: are the candidate answers "
                                      "genuinely this category, per the full contract? "
                                      "Note any dropped hallucinations or added misses.")),
    spans=(Optional[list[str]], Field(
        default=None,
        description="Final EXACT verbatim substrings from the contract that truly match "
                    "this category. Empty list if the category does not appear.")),
)
CheckResult = create_model(
    "CheckResult",
    results=(list[CheckedCat], Field(
        description="Exactly one entry per category given, in the same order.")),
)

CHECK_SYSTEM = (
    "You are a senior legal contract analyst running a FINAL verification pass over an "
    "automated extraction. You get the FULL text of one contract and, for each of the 41 "
    "clause categories, the candidate answers a first-pass RAG system produced (which may "
    "contain hallucinations, mislabeled spans, or misses). For EACH category: reason "
    "briefly about whether each candidate truly matches the category as defined, using the "
    "FULL contract as the source of truth; then return the final list of EXACT verbatim "
    "substrings from the contract that genuinely match. Drop candidates that are "
    "hallucinated or mislabeled. You MAY add a clause the first pass missed if it is "
    "clearly present in the contract. Copy spans character-by-character from the contract; "
    "never paraphrase. Return an empty list for a category that does not appear."
)
CHECK_USER = (
    'FULL CONTRACT:\n"""\n{contract}\n"""\n\n'
    "CATEGORIES (definition) AND FIRST-PASS CANDIDATE ANSWERS:\n{payload}\n\n"
    "Return one entry per category with your reasoning and the final verbatim spans."
)


# --- embeddings (OpenAI, L2-normalised so dot product == cosine) --------------

_ENC = None


def _truncate(text: str) -> str:
    global _ENC
    try:
        if _ENC is None:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        toks = _ENC.encode(text or "", disallowed_special=())
        return _ENC.decode(toks[:_MAX_EMBED_TOKENS]) if len(toks) > _MAX_EMBED_TOKENS else (text or "")
    except Exception:  # noqa: BLE001
        return (text or "")[:16_000]


def embed_texts(texts: list[str]) -> tuple[np.ndarray, int]:
    """Return (n, d) float32 L2-normalised matrix + tokens billed."""
    if not texts:
        return np.zeros((0, 0), dtype="float32"), 0
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[_truncate(t) for t in texts])
    vecs = np.asarray([d.embedding for d in resp.data], dtype="float32")
    tokens = getattr(resp.usage, "total_tokens", 0) or 0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms, tokens


def _cache_key(texts: list[str]) -> str:
    h = hashlib.sha1((" ".join(texts)).encode("utf-8")).hexdigest()[:16]
    return h


def embed_cached(texts: list[str], tag: str) -> tuple[np.ndarray, int]:
    """Disk-cached embeddings (model-independent), so re-runs don't re-embed and
    the cosine methods share the same vectors. Returns (matrix, tokens_billed).
    tokens_billed is 0 on a cache hit."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{tag}_{_cache_key(texts)}.npy"
    if path.exists():
        return np.load(path), 0
    mat, tok = embed_texts(texts)
    np.save(path, mat)
    return mat, tok


# --- BM25 (dependency-free) ---------------------------------------------------

def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


class BM25:
    """Textbook Okapi BM25 over a small chunk corpus (a handful of docs)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = corpus, k1, b
        self.N = len(corpus)
        self.avgdl = (sum(len(d) for d in corpus) / self.N) if self.N else 0.0
        df: dict[str, int] = {}
        for d in corpus:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [{} for _ in corpus]
        for i, d in enumerate(corpus):
            for w in d:
                self.tf[i][w] = self.tf[i].get(w, 0) + 1

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype="float32")
        for i in range(self.N):
            dl = len(self.docs[i])
            s = 0.0
            for w in query:
                f = self.tf[i].get(w, 0)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[i] = s
        return out


# --- LLM chains ---------------------------------------------------------------

def build_label_chain(model: str):
    llm = ChatOpenAI(model=model, temperature=0, timeout=120, max_retries=2)
    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM), ("human", RAG_USER)])
    return prompt | llm.with_structured_output(LabelExtraction, include_raw=True)


def build_check_chain(model: str):
    llm = ChatOpenAI(model=model, temperature=0, timeout=300, max_retries=2)
    prompt = ChatPromptTemplate.from_messages([("system", CHECK_SYSTEM), ("human", CHECK_USER)])
    return prompt | llm.with_structured_output(CheckResult, include_raw=True)


# --- retrieval + per-category answering --------------------------------------

def retrieve_idxs(search: str, top_k: int, n_chunks: int, ci: int,
                  sims: np.ndarray, bm25: Optional[BM25],
                  query_tokens: list[list[str]], hybrid_n: int = 10) -> list[int]:
    """Indices of the top_k chunks for category `ci` under the chosen scorer.

    'hybrid' = BM25 prefilter to the top `hybrid_n` chunks (lexical recall), then
    cosine-rerank those candidates and keep the top_k (dense precision). This is the
    two-stage retrieve-then-rerank pattern: lexical casts a wide net, dense picks the
    best of it."""
    k = min(top_k, n_chunks)
    if k == 0:
        return []
    if search == "hybrid":
        bm_row = bm25.scores(query_tokens[ci])
        cand = [int(x) for x in np.argsort(-bm_row)[:min(hybrid_n, n_chunks)]]
        cos = sims[ci] if sims.size else np.zeros(n_chunks)
        cand.sort(key=lambda i: -float(cos[i]))       # rerank the BM25 shortlist by cosine
        return cand[:k]
    if search == "bm25":
        row = bm25.scores(query_tokens[ci])
    else:
        row = sims[ci] if sims.size else np.zeros(n_chunks)
    return [int(x) for x in np.argsort(-row)[:k]]


async def answer_categories(model: str, tasks: list[dict], concurrency: int):
    """One structured-output call per (contract, category) task; keep validated
    verbatim spans. tasks: [{key, label, description, context}]."""
    chain = build_label_chain(model)
    sem = asyncio.Semaphore(concurrency)
    prog = {"n": 0}
    total = len(tasks)

    async def one(t: dict):
        async with sem:
            try:
                res = await call_with_retry(chain, {
                    "label": t["label"], "description": t["description"], "chunk": t["context"]})
            except Exception as e:  # noqa: BLE001
                res = e
            prog["n"] += 1
            if prog["n"] % 20 == 0 or prog["n"] == total:
                print(f"      {prog['n']}/{total} category calls done", flush=True)
            return t["key"], t["context"], res

    results = await asyncio.gather(*(one(t) for t in tasks))

    preds: dict[str, list[str]] = {t["key"]: [] for t in tasks}
    usage = {"input": 0, "output": 0}
    n_ok = n_failed = 0
    for key, ctx, res in results:
        if isinstance(res, Exception) or res is None:
            n_failed += 1
            continue
        n_ok += 1
        raw, parsed = res.get("raw"), res.get("parsed")
        um = getattr(raw, "usage_metadata", None) if raw is not None else None
        if um:
            usage["input"] += um.get("input_tokens", 0)
            usage["output"] += um.get("output_tokens", 0)
        if parsed is None:
            continue
        for span in (parsed.spans or []):
            v = validate_span(span, ctx)
            if v:
                preds[key].append(v)
    for k in preds:
        preds[k] = list(dict.fromkeys(preds[k]))
    return preds, usage, {"n_ok": n_ok, "n_failed": n_failed}


# --- M3 verification pass -----------------------------------------------------

async def check_contract(model: str, contract_text: str, categories: list[dict],
                         cand_by_label: dict[str, list[str]]):
    """One LLM call over the whole contract + all 41 categories + candidate answers.
    Returns ({label: [validated spans]}, {label: reasoning}, token usage, ok).
    The reasoning is kept for EVERY category (empty string if the model returned
    no entry for it), so the check pass's rationale is fully auditable."""
    chain = build_check_chain(model)
    payload = "\n\n".join(
        f"[{c['label']}] definition: {c['description']}\n"
        f"  candidates: {cand_by_label.get(c['label']) or 'none'}"
        for c in categories
    )
    try:
        res = await call_with_retry(chain, {"contract": contract_text, "payload": payload})
    except Exception as e:  # noqa: BLE001
        print(f"      check pass FAILED: {str(e)[:140]}", flush=True)
        # fall back to the candidates unchanged so M3 never regresses to empty
        reasons = {lbl: f"check pass failed: {str(e)[:120]}" for lbl in cand_by_label}
        return ({lbl: list(v) for lbl, v in cand_by_label.items()}, reasons,
                {"input": 0, "output": 0}, False)

    usage = {"input": 0, "output": 0}
    raw, parsed = res.get("raw"), res.get("parsed")
    um = getattr(raw, "usage_metadata", None) if raw is not None else None
    if um:
        usage["input"] += um.get("input_tokens", 0)
        usage["output"] += um.get("output_tokens", 0)

    by_label = {c["label"]: [] for c in categories}
    reason_by_label = {c["label"]: "" for c in categories}
    lut = {c["label"].lower(): c["label"] for c in categories}
    if parsed is not None:
        for r in (parsed.results or []):
            lbl = lut.get((r.category or "").strip().lower())
            if not lbl:
                continue
            reason_by_label[lbl] = (r.reasoning or "").strip()
            for span in (r.spans or []):
                v = validate_span(span, contract_text)
                if v:
                    by_label[lbl].append(v)
    for lbl in by_label:
        by_label[lbl] = list(dict.fromkeys(by_label[lbl]))
    return by_label, reason_by_label, usage, True


# --- one method for one model ------------------------------------------------

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return p, r, f1, f2


def run_method(model: str, method_key: str, cfg: dict, categories: list[dict],
               chunkmap: dict[str, list[str]], embeds: dict, gt: dict,
               m2_cache: dict, concurrency: int, contracts: list[str] | None = None) -> dict:
    """Execute one method for one model; write the result JSON; return a summary row."""
    print(f"\n  [{method_key}] model={model}", flush=True)
    t0 = time.time()
    embed_tokens = 0
    usage = {"input": 0, "output": 0}
    n_ok = n_failed = 0
    reasoning_by_key: dict[str, str] = {}   # M3 only: per (contract, category) rationale

    if cfg["check"]:
        # M3 = reuse M2's top-1 cosine answers, then one verification call per contract.
        base_preds, base_usage, base_stats = m2_cache[model]
        usage["input"] += base_usage["input"]
        usage["output"] += base_usage["output"]
        n_ok += base_stats["n_ok"]
        n_failed += base_stats["n_failed"]
        preds = {}
        for cid in chunkmap:
            contract_text = "\n\n".join(chunkmap[cid])
            cand = {c["label"]: base_preds.get(f"{cid}__{c['label']}", []) for c in categories}
            by_label, reason_by_label, cu, ok = asyncio.run(
                check_contract(model, contract_text, categories, cand))
            usage["input"] += cu["input"]
            usage["output"] += cu["output"]
            n_ok += int(ok)
            n_failed += int(not ok)
            for c in categories:
                key = f"{cid}__{c['label']}"
                preds[key] = by_label.get(c["label"], [])
                reasoning_by_key[key] = reason_by_label.get(c["label"], "")
            print(f"      checked {cid[:45]}", flush=True)
    else:
        # Build one task per (contract, category) with its retrieved context.
        tasks = []
        for cid, chunks in chunkmap.items():
            sims = embeds["sims"][cid]                      # (n_cats, n_chunks)
            bm25 = embeds["bm25"][cid]
            qtok = embeds["query_tokens"]
            for ci, c in enumerate(categories):
                idxs = retrieve_idxs(cfg["search"], cfg["top_k"], len(chunks), ci, sims, bm25,
                                     qtok, cfg.get("hybrid_n", 10))
                ctx = "\n\n---\n\n".join(chunks[i] for i in idxs)
                tasks.append({"key": f"{cid}__{c['label']}", "label": c["label"],
                              "description": c["description"], "context": ctx})
        preds, usage, stats = asyncio.run(answer_categories(model, tasks, concurrency))
        n_ok, n_failed = stats["n_ok"], stats["n_failed"]

    # --- score: project metric suite (comparable to results/*__3contracts.json) ---
    pred_nbest = {
        k: [{"text": s, "probability": 1.0} for s in preds.get(k, [])]
           + [{"text": "", "probability": 0.0}]
        for k in gt
    }
    metrics = eval_metrics(pred_nbest, gt, category=None)
    by_contract, agg = breakdown(preds, gt)
    # M3: attach the check pass's rationale to every (contract, category) entry.
    if reasoning_by_key:
        for cid, cats in by_contract.items():
            for cat, entry in cats.items():
                entry["reasoning"] = reasoning_by_key.get(f"{cid}__{cat}", "")
    totals = {k: sum(agg[c][k] for c in agg) for k in ("tp", "fp", "fn")}
    p, r, f1, f2 = prf(totals["tp"], totals["fp"], totals["fn"])

    llm_cost = estimate_cost(model, usage["input"], usage["output"])
    embed_cost = embed_tokens / 1e6 * EMBED_PRICE_PER_1M
    cost = llm_cost + embed_cost

    out = {
        "method": method_key,
        "method_description": cfg["desc"],
        "model": model,
        "retrieval": {"search": cfg["search"], "top_k": cfg["top_k"],
                      "hybrid_n": cfg.get("hybrid_n") if cfg["search"] == "hybrid" else None,
                      "chunking": cfg.get("chunking", "markdown"),
                      "llm_check": cfg["check"], "embed_model": EMBED_MODEL},
        "contracts": contracts or list(chunkmap),
        "n_llm_calls_ok": n_ok,
        "n_llm_calls_failed": n_failed,
        "tokens": {"input": usage["input"], "output": usage["output"], "embed": embed_tokens},
        "cost_usd": round(cost, 6),
        # micro P/R/F1/F2 straight off the tp/fp/fn counts
        "micro": {"tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
                  "precision": round(p, 4), "recall": round(r, 4),
                  "f1": round(f1, 4), "f2": round(f2, 4)},
        # full project metric suite (AUPR, prec@recall, best F1/F2, Jaccard, per-cat)
        "metrics": metrics,
        "by_category_counts": agg,
        "by_contract": by_contract,
    }
    mdir = OUT_DIR / method_key
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{model}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"    TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} | "
          f"P={p:.3f} R={r:.3f} F1={f1:.3f} | AUPR={metrics['aupr']:.3f} "
          f"bestF1={metrics['best_f1']:.3f} | ${cost:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)

    return {"method": method_key, "model": model, **out["micro"],
            "aupr": round(metrics["aupr"], 4), "best_f1": round(metrics["best_f1"], 4),
            "best_f2": round(metrics["best_f2"], 4),
            "jaccard_license_grant": round(metrics["jaccard_similarity"], 4),
            "cost_usd": round(cost, 6)}


# --- driver ------------------------------------------------------------------

def load_m2_base_from_disk(model: str):
    """Reconstruct M2's per-category answers + token usage from its result JSON, so
    an M3-only re-run reuses them as the check-pass base instead of re-firing the
    123 top-1 retrieval calls. Returns (preds, usage, stats) or None if absent."""
    p = OUT_DIR / "M2_top1_cosine" / f"{model}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    preds = {}
    for cid, cats in d.get("by_contract", {}).items():
        for cat, entry in cats.items():
            preds[f"{cid}__{cat}"] = list(entry.get("predictions", []))
    usage = {"input": d["tokens"]["input"], "output": d["tokens"]["output"]}
    stats = {"n_ok": d.get("n_llm_calls_ok", 0), "n_failed": d.get("n_llm_calls_failed", 0)}
    return preds, usage, stats


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing -- add it to .env or your shell.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    help="Subset of method keys to run (default: all 4).")
    ap.add_argument("--contracts", nargs="+", default=None,
                    help="Contract IDs (exact or substring), resolved against "
                         "test_chunking.json. Default: the 3 baseline contracts.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write results (default RAG_Research/results).")
    ap.add_argument("--chunk-file", default="test_chunking.json",
                    help="Chunk source (contract_id + chunks per contract). Default is the "
                         "markdown chunking; pass a section chunk file for M5/M6.")
    ap.add_argument("--limit-cats", type=int, default=None,
                    help="Only use the first N categories (cheap smoke tests).")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir

    # categories: label (.title(), matching the CUAD qa-ids) + description
    cats_raw = load_categories(CATEGORY_CSV)
    categories = [{"label": l.title(), "description": d} for l, d in cats_raw.items()]
    if args.limit_cats:
        categories = categories[:args.limit_cats]

    chunk_path = Path(args.chunk_file) if Path(args.chunk_file).is_absolute() else ROOT / args.chunk_file
    chunkdata = {c["contract_id"]: c["chunks"]
                 for c in json.loads(chunk_path.read_text(encoding="utf-8"))["data"]}
    print(f"Chunk source: {chunk_path.name}")

    # Resolve requested contracts against the FULL corpus (exact id, else substring),
    # preserving the order they were given. Defaults to the 3 baseline contracts.
    if args.contracts:
        contracts, seen = [], set()
        for want in args.contracts:
            matches = ([want] if want in chunkdata
                       else [cid for cid in chunkdata if want.lower() in cid.lower()])
            if not matches:
                raise SystemExit(f"No contract matched {want!r} in test_chunking.json")
            for cid in matches:
                if cid not in seen:
                    seen.add(cid)
                    contracts.append(cid)
    else:
        contracts = list(CONTRACTS)

    chunkmap = {cid: chunkdata[cid] for cid in contracts}
    print(f"Contracts ({len(contracts)}):")
    for cid in contracts:
        print(f"  - {cid[:70]}  ({len(chunkmap[cid])} chunks)")

    gt_all = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                         contract_ids=contracts)
    kept = {c["label"] for c in categories}
    gt = {k: v for k, v in gt_all.items() if k.rsplit("__", 1)[1] in kept}

    # --- shared retrieval index: cosine sims + BM25 per contract, category texts ---
    print(f"Embedding {len(categories)} categories + chunks for {len(contracts)} contract(s)...", flush=True)
    total_embed_tokens = 0
    cat_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    cat_emb, ct = embed_cached(cat_texts, "cats")
    total_embed_tokens += ct
    query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]

    embeds = {"sims": {}, "bm25": {}, "query_tokens": query_tokens}
    for cid, chunks in chunkmap.items():
        ch_emb, ct = embed_cached(chunks, "chunks_" + _cache_key([cid]))
        total_embed_tokens += ct
        if ch_emb.size and cat_emb.size:
            embeds["sims"][cid] = cat_emb @ ch_emb.T          # (n_cats, n_chunks)
        else:
            embeds["sims"][cid] = np.zeros((len(categories), len(chunks)), dtype="float32")
        embeds["bm25"][cid] = BM25([_tok(c) for c in chunks])
    print(f"  embeddings ready ({total_embed_tokens:,} tokens billed this run)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    grand_cost = 0.0
    for model in args.models:
        m2_cache: dict = {}
        # If M3 is requested but M2 isn't, we still need M2's answers as its base.
        need_m2 = any(METHODS[m]["check"] for m in args.methods if m in METHODS)
        if need_m2 and "M2_top1_cosine" not in args.methods:
            # Prefer the already-computed M2 answers on disk (free); only recompute
            # them if no M2 result JSON exists yet.
            disk = None if args.limit_cats else load_m2_base_from_disk(model)
            if disk is not None:
                print(f"\n  (reusing M2 top-1 cosine answers from disk as the check-pass base)", flush=True)
                m2_cache[model] = disk
            else:
                print(f"\n  (pre-computing M2 top-1 cosine as the base for the check pass)", flush=True)
                tasks = []
                for cid, chunks in chunkmap.items():
                    for ci, c in enumerate(categories):
                        idxs = retrieve_idxs("cosine", 1, len(chunks), ci, embeds["sims"][cid], None, query_tokens)
                        ctx = "\n\n---\n\n".join(chunks[i] for i in idxs)
                        tasks.append({"key": f"{cid}__{c['label']}", "label": c["label"],
                                      "description": c["description"], "context": ctx})
                m2_cache[model] = asyncio.run(answer_categories(model, tasks, args.concurrency))

        for mk in args.methods:
            if mk not in METHODS:
                print(f"  (skipping unknown method {mk})")
                continue
            cfg = METHODS[mk]
            # cache M2 so M3 reuses it (and so we never run it twice)
            if mk == "M2_top1_cosine" and model not in m2_cache:
                tasks = []
                for cid, chunks in chunkmap.items():
                    for ci, c in enumerate(categories):
                        idxs = retrieve_idxs("cosine", 1, len(chunks), ci, embeds["sims"][cid], None, query_tokens)
                        ctx = "\n\n---\n\n".join(chunks[i] for i in idxs)
                        tasks.append({"key": f"{cid}__{c['label']}", "label": c["label"],
                                      "description": c["description"], "context": ctx})
                m2_cache[model] = asyncio.run(answer_categories(model, tasks, args.concurrency))
            row = run_method(model, mk, cfg, categories, chunkmap, embeds, gt,
                             m2_cache, args.concurrency, contracts)
            summary_rows.append(row)
            grand_cost += row["cost_usd"]

    # --- summary ------------------------------------------------------------
    # Merge with any existing summary so a partial run (e.g. just M3) updates only
    # its own rows and leaves the other methods' rows intact.
    spath = OUT_DIR / "summary.json"
    all_rows = list(summary_rows)
    if spath.exists() and not args.limit_cats:
        try:
            prev = json.loads(spath.read_text(encoding="utf-8")).get("rows", [])
        except Exception:  # noqa: BLE001
            prev = []
        done = {(r["method"], r["model"]) for r in summary_rows}
        all_rows = [r for r in prev if (r["method"], r["model"]) not in done] + summary_rows
    order = {m: i for i, m in enumerate(METHODS)}
    all_rows.sort(key=lambda r: (r["model"], order.get(r["method"], 99)))
    grand_all = sum((r.get("cost_usd") or 0) for r in all_rows)
    summary = {"models": sorted({r["model"] for r in all_rows}),
               "methods": {k: METHODS[k]["desc"] for k in METHODS},
               "contracts": contracts, "n_categories": len(categories),
               "grand_cost_usd": round(grand_all, 6), "rows": all_rows}
    spath.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"{'method':<26}{'model':<12}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>7}{'R':>7}"
          f"{'F1':>7}{'AUPR':>7}{'bF1':>7}{'Jac-LG':>8}{'cost':>9}")
    print("=" * 92)
    for r in summary_rows:
        print(f"{r['method']:<26}{r['model']:<12}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
              f"{r['aupr']:>7.3f}{r['best_f1']:>7.3f}{r['jaccard_license_grant']:>8.3f}"
              f"{('$%.3f' % r['cost_usd']):>9}")
    print("=" * 92)
    print(f"Grand API cost this run: ${grand_cost:.4f}")
    print(f"Wrote {OUT_DIR}/summary.json and per-method result JSONs.")


if __name__ == "__main__":
    main()
