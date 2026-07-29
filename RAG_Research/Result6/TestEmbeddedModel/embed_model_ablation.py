"""
Embedding-model ablation: does swapping the embedder improve RETRIEVAL, holding
chunking and retrieval method fixed?

Same substrate as the rest of Result6 -- the 6-contract section chunking
(~220 chunks), the same 41 categories, the same gold answers -- so numbers are
directly comparable to H_bm5_cos5 / M7_section_hybrid_top5 / master_summary.csv.
Only the embedder varies; BM25 (lexical) is embedder-independent, so the two
hybrid configs below isolate exactly the cosine-rerank stage.

5 embedders:
    text-embedding-3-small   (OpenAI baseline; reuses RAG_Research/.cache, no new
                              API calls)
    qwen3-embedding:0.6b     (Ollama, local)
    bge-m3                   (Ollama, local)
    nomic-embed-text         (Ollama, local)
    mxbai-embed-large        (Ollama, local)

3 retrieval configs (all top_k=5, so the comparison is apples-to-apples):
    cosine_top5        search=cosine, top_k=5
    hybrid_bm5_cos5     search=hybrid, hybrid_n=5,  top_k=5  (= H_bm5_cos5)
    hybrid_bm10_cos5    search=hybrid, hybrid_n=10, top_k=5  (= M7_section_hybrid_top5)

Retrieval-only: NO LLM calls, NO OpenAI spend beyond the already-cached category/
chunk embeddings. Ollama calls are local and free. Coverage is identical across
embedders (it's about chunk TEXT, not vectors) -- only R@5 varies.

Run (Ollama must be running locally with the 4 models pulled):
    python RAG_Research/Result6/TestEmbeddedModel/embed_model_ablation.py
"""

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # .../Result6/TestEmbeddedModel
RESULT6 = HERE.parent                                    # .../Result6
RAG_RESEARCH = RESULT6.parent                             # .../RAG_Research
ROOT = RAG_RESEARCH.parent                                # repo root

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAG_RESEARCH))
os.chdir(ROOT)

from OpenAITest import load_categories, CATEGORY_CSV                          # noqa: E402
from evaluate import get_answers                                              # noqa: E402
from rag_research import embed_cached, BM25, _tok, retrieve_idxs              # noqa: E402

CACHE_DIR = HERE / ".cache"
SECTION_CHUNKS = RESULT6 / "section_chunking.json"
MODEL = "gpt-5.4"   # unused for LLM calls here, kept only for naming consistency

CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]

# embedder key -> ("openai" | ollama model tag)
EMBEDDERS = {
    "openai_text-embedding-3-small": None,     # None = reuse rag_research.embed_cached
    "ollama_qwen3-embedding-0.6b": "qwen3-embedding:0.6b",
    "ollama_bge-m3": "bge-m3",
    "ollama_nomic-embed-text": "nomic-embed-text",
    "ollama_mxbai-embed-large": "mxbai-embed-large",
}

# config name -> (search, top_k, hybrid_n)
CONFIGS = {
    "cosine_top5": ("cosine", 5, None),
    "hybrid_bm5_cos5": ("hybrid", 5, 5),
    "hybrid_bm10_cos5": ("hybrid", 5, 10),
}


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def chunk_contains(chunk: str, gold: str) -> bool:
    nc, ng = _norm(chunk), _norm(gold)
    if not ng:
        return False
    if ng in nc:
        return True
    gt_toks = set(ng.split())
    if not gt_toks:
        return False
    ch_toks = set(nc.split())
    return len(gt_toks & ch_toks) / len(gt_toks) >= 0.9


def _cache_key(texts: list[str]) -> str:
    return hashlib.sha1((" ".join(texts)).encode("utf-8")).hexdigest()[:16]


def ollama_embed(texts: list[str], model: str) -> np.ndarray:
    """Embed with a local Ollama model via langchain_ollama, L2-normalised so a
    dot product is cosine similarity (matches rag_research.embed_texts' contract)."""
    from langchain_ollama import OllamaEmbeddings
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    client = OllamaEmbeddings(model=model)
    vecs = np.asarray(client.embed_documents(list(texts)), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_cached_any(texts: list[str], tag: str, embedder_key: str, ollama_model) -> tuple[np.ndarray, float]:
    """Disk-cached embeddings for ANY embedder (OpenAI reuses the shared
    rag_research cache; Ollama models get their own cache here, keyed by model
    name so the 5 embedders never collide). Returns (matrix, wall_seconds)."""
    if ollama_model is None:
        mat, _ = embed_cached(texts, tag)   # OpenAI path: shared project-wide cache
        return mat, 0.0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ollama_model.replace(":", "_").replace("/", "_")
    path = CACHE_DIR / f"{safe}_{tag}_{_cache_key(texts)}.npy"
    if path.exists():
        return np.load(path), 0.0
    t0 = time.time()
    mat = ollama_embed(texts, ollama_model)
    dt = time.time() - t0
    np.save(path, mat)
    return mat, dt


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    cat_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]

    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)

    chunkdata = {c["contract_id"]: c["chunks"]
                 for c in json.loads(SECTION_CHUNKS.read_text(encoding="utf-8"))["data"]}
    chunkmap = {cid: chunkdata[cid] for cid in CONTRACTS}

    # --- BM25 index + gold-chunk locations: embedder-independent, computed ONCE ---
    print("Building BM25 index + gold-chunk locations (embedder-independent)...", flush=True)
    bm25s = {cid: BM25([_tok(c) for c in chunks]) for cid, chunks in chunkmap.items()}

    gold_total = 0
    gold: list[tuple[str, str, list[int]]] = []   # (cid, label, hit_chunk_idxs), only reachable ones
    for cid, chunks in chunkmap.items():
        for label in labels:
            for g in gt.get(f"{cid}__{label}", []):
                gold_total += 1
                hits = [k for k, ch in enumerate(chunks) if chunk_contains(ch, g)]
                if hits:
                    gold.append((cid, label, hits))
    coverage = len(gold) / gold_total if gold_total else 0.0
    print(f"  gold total={gold_total}  reachable={len(gold)}  coverage={coverage:.3f}\n")

    rows = []
    for embedder_key, ollama_model in EMBEDDERS.items():
        print(f"[{embedder_key}] embedding {len(cat_texts)} categories + "
              f"{sum(len(c) for c in chunkmap.values())} chunks...", flush=True)
        cat_emb, cat_dt = embed_cached_any(cat_texts, "cats", embedder_key, ollama_model)

        sims = {}
        embed_wall = cat_dt
        for cid, chunks in chunkmap.items():
            ch_emb, dt = embed_cached_any(chunks, "chunks_" + _cache_key([cid]), embedder_key, ollama_model)
            embed_wall += dt
            sims[cid] = cat_emb @ ch_emb.T if (cat_emb.size and ch_emb.size) else \
                np.zeros((len(categories), len(chunks)), dtype="float32")
        print(f"  done ({embed_wall:.1f}s embedding wall time this run; 0.0s = fully cached)\n", flush=True)

        for cfg_name, (search, top_k, hybrid_n) in CONFIGS.items():
            hit = 0
            for cid, label, hit_idxs in gold:
                ci = labels.index(label)
                idxs = set(retrieve_idxs(search, top_k, len(chunkmap[cid]), ci,
                                         sims[cid], bm25s[cid], query_tokens, hybrid_n or 10))
                if set(hit_idxs) & idxs:
                    hit += 1
            r_at_5 = hit / len(gold) if gold else 0.0
            rows.append({
                "embedder": embedder_key, "config": cfg_name,
                "search": search, "top_k": top_k, "hybrid_n": hybrid_n or "",
                "coverage": round(coverage, 4), "reachable": len(gold),
                "retrieved": hit, "R_at_5": round(r_at_5, 4),
                "embed_wall_seconds": round(embed_wall, 1),
            })

    # ---- print + write ----
    print("=" * 78)
    print(f"{'embedder':<32}{'config':<20}{'reach':>6}{'hit':>6}{'R@5':>7}")
    print("-" * 78)
    for r in rows:
        print(f"{r['embedder']:<32}{r['config']:<20}{r['reachable']:>6}{r['retrieved']:>6}"
              f"{r['R_at_5']:>7.3f}")
    print("=" * 78)

    with open(HERE / "embed_model_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "embed_model_ablation.json").write_text(
        json.dumps({"contracts": CONTRACTS, "gold_total": gold_total,
                    "coverage": round(coverage, 4), "rows": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {HERE / 'embed_model_ablation.csv'} and embed_model_ablation.json")


if __name__ == "__main__":
    main()
