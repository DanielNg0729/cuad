"""
RAG (retrieval-augmented) extraction for the CUAD highlighter.

The default pipeline (extract.extract) is *chunk-centric*: every chunk is sent to
the model once, asking for all 41 categories at the same time. This module is
*label-centric*:

    1. embed every chunk once            (OpenAI text-embedding-3-small)
    2. embed every category              ("<label>. <description>")
    3. for each category, rank the chunks and keep the top-k -- either plain
       cosine, or "hybrid" (BM25 lexical prefilter, then cosine-rerank)
    4. ask the model about that ONE category over just those k chunks

So instead of "N chunk calls, each covering 41 fields" we do "41 label calls,
each covering k retrieved chunks". The output shape is identical to
extract.extract(), so app.py and the front-end don't care which pipeline ran.

Embeddings need OPENAI_API_KEY (Groq has no embedding endpoint). Chunk
embeddings are cached on the parsed doc by the caller (app.py), so re-running RAG
with a different LLM doesn't re-embed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import re
import time
from typing import Optional

import numpy as np
from pydantic import Field, create_model

import extract  # reuse categories, chain wiring, span validation, offsets, pricing

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_1M = 0.02          # USD / 1M tokens for text-embedding-3-small
DEFAULT_TOP_K = 3
# Hybrid retrieval ("hybrid 5-5"): BM25 prefilter to the top 5 chunks (lexical
# recall), then cosine-rerank that shortlist and keep the top 5 (dense
# precision). Matches H_bm5_cos5 from RAG_Research/Result6, the best-scoring
# retriever found there (F1 0.430, beating plain cosine and plain BM25).
HYBRID_BM25_N = 5
HYBRID_TOP_K = 5
# RAG fires ONE call per category (~41), so it parallelises far more than the
# scan path (a handful of chunk calls). Each call is small and independent, so a
# higher ceiling than extract.CONCURRENCY (4) is safe and ~4x faster in practice
# (41 labels: ~21s @ 4 -> ~5.5s @ 12). Lower it if you hit 429 rate limits.
RAG_CONCURRENCY = 12
# text-embedding-3-small rejects any input over 8192 tokens. Cap by TOKENS, not
# chars: token-dense content (markdown tables, numbers) can be ~2 chars/token, so
# a 30k-char chunk was ~14k tokens -> a 400 error. Leave margin under 8192.
_MAX_TOKENS = 8_000
_CHAR_CAP = 16_000                 # fallback cap (~2 chars/token) only if tiktoken is unavailable
# Smaller batches (by chars AND item count) so a big document splits into several
# requests we can fire concurrently, instead of one giant serial request.
_BATCH_CHARS = 100_000
_BATCH_ITEMS = 128
_EMBED_WORKERS = 8                 # concurrent embedding requests for a large doc
# Fail FAST: a stalled request should surface as an error in the UI, not sit at
# "0%" for minutes. SDK default is 600s per attempt -- far too long for a call
# that normally finishes in <5s. Worst case here: ~30s x (1 retry) ~= 60s.
_EMBED_TIMEOUT = 30.0              # seconds per request
_EMBED_RETRIES = 1                 # SDK auto-retries transient errors / 429s this many times

# One generic schema reused for every label: the specific category to look for is
# passed in the prompt (not baked into the schema), so we build the chain once.
LabelExtraction = create_model(
    "LabelExtraction",
    spans=(Optional[list[str]], Field(
        default=None,
        description="Every EXACT verbatim substring from the provided sections that "
                    "matches the requested clause category. Empty list if none apply.",
    )),
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


# --- embeddings -------------------------------------------------------------

def _client():
    from openai import OpenAI
    # Bounded timeout so a stalled connection fails fast (and retries) instead of
    # hanging for the SDK's 600s default -- that was the "4 min and still going".
    return OpenAI(timeout=_EMBED_TIMEOUT, max_retries=_EMBED_RETRIES)


_ENCODER = None


def _encoder():
    """tiktoken encoder for the embedding model (cl100k_base), built once."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _truncate_for_embedding(texts: list[str]) -> list[str]:
    """Truncate each text to <= _MAX_TOKENS so no single input trips the embedding
    model's 8192-token limit. Uses tiktoken (exact token count); falls back to a
    conservative char cap only if tiktoken can't be loaded."""
    try:
        enc = _encoder()
    except Exception:  # noqa: BLE001
        return [(t or "")[:_CHAR_CAP] for t in texts]
    out, n_trunc = [], 0
    for t in texts:
        t = t or ""
        toks = enc.encode(t, disallowed_special=())
        if len(toks) > _MAX_TOKENS:
            t = enc.decode(toks[:_MAX_TOKENS])
            n_trunc += 1
        out.append(t)
    if n_trunc:
        print(f"[rag] truncated {n_trunc} oversized chunk(s) to {_MAX_TOKENS} tokens "
              f"for embedding", flush=True)
    return out


def _batches(texts: list[str]) -> list[list[str]]:
    """Split into sub-lists bounded by both total chars (_BATCH_CHARS) and item
    count (_BATCH_ITEMS), so each embedding request stays small and we get several
    batches to fire concurrently on a large document."""
    out, batch, size = [], [], 0
    for t in texts:
        if batch and (size + len(t) > _BATCH_CHARS or len(batch) >= _BATCH_ITEMS):
            out.append(batch)
            batch, size = [], 0
        batch.append(t)
        size += len(t)
    if batch:
        out.append(batch)
    return out


def embed_texts(texts: list[str], model: str = EMBED_MODEL,
                progress_cb=None) -> tuple[np.ndarray, int]:
    """Embed `texts` with the OpenAI embedding model. Returns an (n, d) float32
    matrix (L2-normalised, so a dot product IS cosine similarity) and the total
    tokens billed. Each input is capped at _CHAR_CAP chars; batches are embedded
    concurrently so a large document doesn't serialise into a long wait.

    progress_cb(done, total) fires as each batch lands, so the UI can show how
    many chunks have been embedded so far."""
    if not texts:
        return np.zeros((0, 0), dtype="float32"), 0
    client = _client()
    capped = _truncate_for_embedding(texts)   # keep every input under the token limit
    batches = _batches(capped)
    total = len(texts)

    def run(batch):
        resp = client.embeddings.create(model=model, input=batch)
        return ([d.embedding for d in resp.data],
                getattr(resp.usage, "total_tokens", 0) or 0)

    def report(done):
        if progress_cb:
            try:
                progress_cb(done, total)
            except Exception:  # noqa: BLE001
                pass

    t0 = time.time()
    print(f"[rag] embedding {total} chunk(s) via {model} in {len(batches)} batch(es) "
          f"(timeout {_EMBED_TIMEOUT:.0f}s, up to {_EMBED_RETRIES + 1} attempts each)…",
          flush=True)
    report(0)
    done = 0
    try:
        if len(batches) == 1:
            results = [run(batches[0])]
            report(total)
        else:
            results = [None] * len(batches)
            workers = min(_EMBED_WORKERS, len(batches))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run, b): i for i, b in enumerate(batches)}
                for fut in concurrent.futures.as_completed(futs):
                    i = futs[fut]
                    results[i] = fut.result()   # keep original order
                    done += len(batches[i])
                    report(done)
    except Exception as e:  # noqa: BLE001
        # Turn a silent network stall into a visible, actionable error.
        print(f"[rag] EMBEDDING FAILED after {time.time() - t0:.1f}s: "
              f"{type(e).__name__}: {str(e)[:200]}", flush=True)
        raise RuntimeError(
            f"Embedding call to OpenAI failed ({type(e).__name__}). RAG mode must reach "
            f"api.openai.com for '{model}' — check your network / proxy / firewall, or "
            f"use Full-scan mode instead. Details: {str(e)[:200]}"
        ) from e

    vecs: list[list[float]] = []
    total_tokens = 0
    for v, tok in results:
        vecs.extend(v)
        total_tokens += tok
    print(f"[rag] embedded {len(texts)} text(s) in {len(batches)} batch(es) "
          f"({time.time() - t0:.1f}s, {total_tokens:,} tokens)", flush=True)

    arr = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms, total_tokens


# --- BM25 (dependency-free) --------------------------------------------------

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


# --- retrieval + per-label extraction --------------------------------------

def _retrieve_idxs(search: str, top_k: int, n_chunks: int, ci: int, sims: np.ndarray,
                   bm25: Optional["BM25"], query_tokens: Optional[list[list[str]]],
                   hybrid_n: int) -> list[int]:
    """Indices of the chunks to send to the model for category `ci`.

    'cosine' = plain cosine top-k. 'hybrid' = BM25 prefilter to the top
    `hybrid_n` chunks (lexical recall), then cosine-rerank that shortlist and
    keep the top_k (dense precision)."""
    k = min(top_k, n_chunks)
    if k == 0:
        return []
    if search == "hybrid":
        bm_row = bm25.scores(query_tokens[ci])
        cand = [int(x) for x in np.argsort(-bm_row)[:min(hybrid_n, n_chunks)]]
        cos = sims[ci] if sims.size else np.zeros(n_chunks)
        cand.sort(key=lambda i: -float(cos[i]))       # rerank the BM25 shortlist by cosine
        return cand[:k]
    row = sims[ci] if sims.size else np.zeros(n_chunks)
    return [int(x) for x in np.argsort(-row)[:k]]


async def _rag_async(categories: list[dict], chunks: list[str], sims: np.ndarray,
                     top_k: int, model_id: str, concurrency: int,
                     progress_cb=None, search: str = "cosine",
                     bm25: Optional["BM25"] = None,
                     query_tokens: Optional[list[list[str]]] = None,
                     hybrid_n: int = HYBRID_BM25_N) -> tuple[dict[str, list[str]], dict, dict]:
    """For each category (capped by `concurrency`): retrieve its top-k chunks
    (cosine, or BM25-prefilter + cosine-rerank when search='hybrid'), ask the
    model for verbatim spans, keep the ones that really appear."""
    chain = extract.build_chain(model_id, LabelExtraction,
                                system_prompt=RAG_SYSTEM, user_prompt=RAG_USER)
    sem = asyncio.Semaphore(concurrency)
    n = len(categories)
    print(f"[rag] model={model_id} | {n} categor(ies) | search={search} | top_k={top_k}"
          + (f" | bm25_n={hybrid_n}" if search == "hybrid" else "")
          + f" | {len(chunks)} chunk(s) | concurrency={concurrency}", flush=True)

    prog = {"completed": 0, "ok": 0, "failed": 0}

    async def process(ci: int, cat: dict):
        async with sem:
            idxs = _retrieve_idxs(search, top_k, len(chunks), ci, sims, bm25, query_tokens, hybrid_n)
            context = "\n\n---\n\n".join(chunks[i] for i in idxs)
            try:
                res = await extract._call_with_retry(
                    chain, {"label": cat["label"], "description": cat["description"],
                            "chunk": context})
                ok = True
            except Exception as e:  # noqa: BLE001
                res, ok = e, False
            prog["completed"] += 1
            prog["ok" if ok else "failed"] += 1
            status = "OK" if ok else f"FAILED: {str(res)[:100]}"
            print(f"  [{prog['completed']}/{n}] {cat['label'][:40]:40} "
                  f"top{len(idxs)} -> {status}", flush=True)
            if progress_cb:
                try:
                    progress_cb(prog["completed"], n, prog["ok"], prog["failed"])
                except Exception:  # noqa: BLE001
                    pass
            return cat["label"], context, res

    results = await asyncio.gather(*(process(ci, c) for ci, c in enumerate(categories)))

    preds: dict[str, list[str]] = {c["label"]: [] for c in categories}
    usage = {"input": 0, "output": 0}
    n_ok, n_failed, errors = 0, 0, []
    for label, context, res in results:
        if isinstance(res, Exception) or res is None:
            n_failed += 1
            if isinstance(res, Exception):
                errors.append(str(res))
            continue
        n_ok += 1
        raw = res.get("raw") if isinstance(res, dict) else None
        parsed = res.get("parsed") if isinstance(res, dict) else None
        um = getattr(raw, "usage_metadata", None) if raw is not None else None
        if um:
            usage["input"] += um.get("input_tokens", 0)
            usage["output"] += um.get("output_tokens", 0)
        if parsed is None:
            continue
        for span in (parsed.spans or []):
            v = extract._validate_span(span, context)
            if v:
                preds[label].append(v)
    for label in preds:
        preds[label] = list(dict.fromkeys(preds[label]))   # dedupe, keep order

    stats = {"n_ok": n_ok, "n_failed": n_failed, "error": errors[0] if errors else None}
    return preds, usage, stats


def rag_extract(text: str, chunks: list[str], model_id: str,
                categories: Optional[list[dict]] = None, top_k: int = DEFAULT_TOP_K,
                chunk_embeddings: Optional[np.ndarray] = None,
                concurrency: int = RAG_CONCURRENCY, progress_cb=None,
                search: str = "cosine", hybrid_n: int = HYBRID_BM25_N) -> dict:
    """RAG extraction for one already-parsed document. Same return shape as
    extract.extract(), plus mode/top_k/embed metadata.

    `search`: "cosine" (default, plain cosine top-k) or "hybrid" (BM25
    prefilter to the top `hybrid_n` chunks, then cosine-rerank down to
    `top_k` -- the "hybrid 5-5" method when hybrid_n=top_k=5).
    `chunk_embeddings` lets the caller pass a cached (n_chunks, d) matrix so
    re-runs skip re-embedding; if None, chunks are embedded here.
    progress_cb(completed, total, ok, failed) fires as each category finishes."""
    t0 = time.time()
    categories = categories or extract.load_categories()

    embed_tokens = 0
    if chunk_embeddings is None:
        chunk_embeddings, ct = embed_texts(chunks)
        embed_tokens += ct
    label_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    label_emb, lt = embed_texts(label_texts)
    embed_tokens += lt

    # (n_labels, n_chunks) cosine-similarity matrix; both sides are L2-normalised.
    # Needed even in hybrid mode, since hybrid reranks the BM25 shortlist by cosine.
    if chunk_embeddings.size and label_emb.size:
        sims = label_emb @ chunk_embeddings.T
    else:
        sims = np.zeros((len(categories), len(chunks)), dtype="float32")

    bm25, query_tokens = None, None
    if search == "hybrid":
        bm25 = BM25([_tok(c) for c in chunks])
        query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]

    preds, usage, stats = asyncio.run(
        _rag_async(categories, chunks, sims, top_k, model_id, concurrency, progress_cb,
                  search=search, bm25=bm25, query_tokens=query_tokens, hybrid_n=hybrid_n)
    )

    entities = extract._build_entities(text, preds)
    spans_by_label = {label: spans for label, spans in preds.items() if spans}

    tin, tout = usage["input"], usage["output"]
    llm_cost = extract.estimate_cost(model_id, tin, tout)
    embed_cost = embed_tokens / 1e6 * EMBED_PRICE_PER_1M
    cost = llm_cost + embed_cost

    print(f"[rag] done in {time.time() - t0:.1f}s | model={model_id} | search={search} | "
          f"top_k={top_k}" + (f" | bm25_n={hybrid_n}" if search == "hybrid" else ""), flush=True)
    print(f"  label calls OK/failed : {stats['n_ok']}/{stats['n_failed']}", flush=True)
    print(f"  clause types hit      : {len(spans_by_label)}/{len(categories)}   "
          f"highlights: {len(entities)}", flush=True)
    print(f"  LLM tokens in/out     : {tin:,} / {tout:,}", flush=True)
    print(f"  embed tokens          : {embed_tokens:,}", flush=True)
    print(f"  est. cost (USD)       : ${cost:.4f}  (llm ${llm_cost:.4f} + embed ${embed_cost:.4f})",
          flush=True)
    if stats["error"]:
        print(f"  sample error          : {stats['error'][:200]}", flush=True)

    return {
        "entities": entities,
        "spans_by_label": spans_by_label,
        "usage": usage,
        "model": model_id,
        "mode": "hybrid" if search == "hybrid" else "rag",
        "search": search,
        "top_k": top_k,
        "hybrid_n": hybrid_n if search == "hybrid" else None,
        "n_chunks": len(chunks),
        "n_labels": len(categories),
        "n_ok": stats["n_ok"],
        "n_failed": stats["n_failed"],
        "error": stats["error"],
        "embed_tokens": embed_tokens,
        "estimated_cost_usd": round(cost, 6),
        "n_entities": len(entities),
    }


def retrieve_context_for_label(label: str, description: str, chunks: list[str],
                               chunk_embeddings: np.ndarray, search: str, top_k: int,
                               hybrid_n: int = HYBRID_BM25_N,
                               bm25: Optional["BM25"] = None) -> tuple[str, Optional["BM25"]]:
    """Re-run retrieval for ONE label with the same scorer/top_k the document was
    last extracted with, so a later AI-verify pass sees exactly the chunks the
    extractor saw instead of a generic +/- char window around each answer.

    `search` is "cosine" (RAG mode) or "hybrid" (BM25 top-`hybrid_n` prefilter,
    cosine-rerank to `top_k`). `bm25` lets the caller pass a cached index (built
    over `chunks`, which don't change for a given doc) so repeated verify calls
    don't rebuild it; if None and search=='hybrid', one is built here and handed
    back for the caller to cache.

    Returns (context, bm25) -- bm25 is None when search != 'hybrid'."""
    n_chunks = len(chunks)
    if n_chunks == 0:
        return "", bm25
    label_emb, _ = embed_texts([f"{label}. {description}"])
    if label_emb.size and chunk_embeddings.size:
        row = (label_emb @ chunk_embeddings.T)[0]
    else:
        row = np.zeros(n_chunks, dtype="float32")
    sims = row[None, :]

    if search == "hybrid":
        if bm25 is None:
            bm25 = BM25([_tok(c) for c in chunks])
        query_tokens = [_tok(f"{label} {description}")]
        idxs = _retrieve_idxs("hybrid", top_k, n_chunks, 0, sims, bm25, query_tokens, hybrid_n)
    else:
        idxs = _retrieve_idxs("cosine", top_k, n_chunks, 0, sims, None, None, hybrid_n)

    return "\n\n---\n\n".join(chunks[i] for i in idxs), bm25
