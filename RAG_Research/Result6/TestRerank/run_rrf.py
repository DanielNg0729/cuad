"""
Reciprocal Rank Fusion (RRF) retrieval test -- kept in its OWN folder, separate from
../TestAblation/, per request. Same section chunking / 6 contracts / 41 categories /
gpt-5.4 as the rest of Result6, so numbers are directly comparable to the
cosine_top5 / hybrid_bm10_cos5 arms already in ../TestAblation/results/.

RRF is genuinely different from the existing "hybrid" method (BM25 prefilter ->
cosine RERANK, where BM25's score is discarded once used as a gatekeeper): RRF ranks
chunks independently under BOTH BM25 and cosine, takes each ranker's own top-N
shortlist, UNIONS them (a chunk strong in only one ranker still gets partial credit --
the current hybrid method excludes it outright if BM25 didn't shortlist it), and fuses
by rank position:

    RRF_score(chunk) = 1/(rrf_k + bm25_rank(chunk))  +  1/(rrf_k + cosine_rank(chunk))

(rank is 1-indexed within that ranker's own shortlist; a chunk absent from a given
list contributes 0 from that list). rrf_k=60 is the community-standard constant from
the original Cormack et al. RRF paper. Final top-5 are the highest RRF-scored chunks
in the union.

6 arms: shortlist N in {10, 15} x embedder in {openai, qwen3, bge-m3}, all at final
top-5 (matching most ../TestAblation arms for direct comparison).

Run:
    python RAG_Research/Result6/TestRerank/run_rrf.py
"""

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # .../Result6/TestRerank
RESULT6 = HERE.parent                                     # .../Result6
RAG_RESEARCH = RESULT6.parent                               # .../RAG_Research
ROOT = RAG_RESEARCH.parent                                   # repo root

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAG_RESEARCH))
os.chdir(ROOT)

import asyncio                                                                # noqa: E402
from dotenv import load_dotenv                                                # noqa: E402
from OpenAITest import load_categories, CATEGORY_CSV                          # noqa: E402
from evaluate import get_answers                                              # noqa: E402
from rag_research import (                                                    # noqa: E402
    eval_metrics, breakdown, prf, answer_categories, embed_cached,
    BM25, _tok, estimate_cost,
)

load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY missing -- add it to .env")

MODEL = "gpt-5.4"
CONCURRENCY = 12
RRF_K = 60   # standard RRF smoothing constant (Cormack et al.) -- NOT the top_k final count
SECTION_CHUNKS = RESULT6 / "section_chunking.json"
OUT_RESULTS = HERE / "results"
CACHE_DIR = HERE / ".cache"
ABLATION_RESULTS = RESULT6 / "TestAblation" / "results"   # for the comparison table only

CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]

OLLAMA_MODEL_NAME = {"qwen3": "qwen3-embedding:0.6b", "bge3": "bge-m3"}
EMBEDDERS = ["openai", "qwen3", "bge3"]
TOP_K = 5

# explicit (embedder, shortlist_n) pairs -- NOT a blanket cross product, since N=20/25
# were only requested for qwen3/bge-m3, not OpenAI.
ARM_CONFIGS = [(e, n) for e in EMBEDDERS for n in (10, 15)] + \
              [(e, n) for e in ("qwen3", "bge3") for n in (20, 25)]

FORCE_RERUN: set[str] = set()


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
    from langchain_ollama import OllamaEmbeddings
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    vecs = np.asarray(OllamaEmbeddings(model=model).embed_documents(list(texts)), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_cached_ollama(texts: list[str], tag: str, embedder_key: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = OLLAMA_MODEL_NAME[embedder_key].replace(":", "_").replace("/", "_")
    path = CACHE_DIR / f"{safe}_{tag}_{_cache_key(texts)}.npy"
    if path.exists():
        return np.load(path)
    mat = ollama_embed(texts, OLLAMA_MODEL_NAME[embedder_key])
    np.save(path, mat)
    return mat


def retrieve_rrf(shortlist_n: int, top_k: int, n_chunks: int, ci: int,
                 sims_row: np.ndarray, bm25: BM25, query_tokens: list[list[str]],
                 rrf_k: int = RRF_K) -> list[int]:
    """Union BM25's own top-`shortlist_n` and cosine's own top-`shortlist_n` (each
    ranked independently over ALL chunks), fuse by reciprocal rank, return the
    final top_k by fused score."""
    n = min(shortlist_n, n_chunks)
    bm_scores = bm25.scores(query_tokens[ci])
    bm_shortlist = [int(x) for x in np.argsort(-bm_scores)[:n]]
    cos_shortlist = [int(x) for x in np.argsort(-sims_row)[:n]] if sims_row.size else []

    bm_rank = {idx: r for r, idx in enumerate(bm_shortlist)}      # 0-indexed
    cos_rank = {idx: r for r, idx in enumerate(cos_shortlist)}

    candidates = set(bm_shortlist) | set(cos_shortlist)
    rrf_score = {}
    for idx in candidates:
        s = 0.0
        if idx in bm_rank:
            s += 1.0 / (rrf_k + bm_rank[idx] + 1)
        if idx in cos_rank:
            s += 1.0 / (rrf_k + cos_rank[idx] + 1)
        rrf_score[idx] = s

    ranked = sorted(candidates, key=lambda i: -rrf_score[i])
    return ranked[:min(top_k, len(ranked))]


def score_arm(preds: dict, gt: dict) -> dict:
    pred_nbest = {
        k: [{"text": s, "probability": 1.0} for s in preds.get(k, [])]
           + [{"text": "", "probability": 0.0}]
        for k in gt
    }
    metrics = eval_metrics(pred_nbest, gt, category=None)
    by_contract, agg = breakdown(preds, gt)
    totals = {k: sum(agg[c][k] for c in agg) for k in ("tp", "fp", "fn")}
    p, r, f1, f2 = prf(totals["tp"], totals["fp"], totals["fn"])
    return {"by_contract": by_contract, "agg": agg, "totals": totals,
            "metrics": metrics, "micro": {"precision": p, "recall": r, "f1": f1, "f2": f2}}


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    cat_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]

    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)
    kept = {c["label"] for c in categories}
    gt = {k: v for k, v in gt.items() if k.rsplit("__", 1)[1] in kept}

    chunkdata = {c["contract_id"]: c["chunks"]
                 for c in json.loads(SECTION_CHUNKS.read_text(encoding="utf-8"))["data"]}
    chunkmap = {cid: chunkdata[cid] for cid in CONTRACTS}
    n_chunks_total = sum(len(v) for v in chunkmap.values())

    print("Building BM25 index (shared across all embedders -- lexical, embedder-agnostic)...",
          flush=True)
    bm25s = {cid: BM25([_tok(c) for c in chunks]) for cid, chunks in chunkmap.items()}

    print(f"Embedding categories + chunks for embedders: {EMBEDDERS} "
          "(OpenAI reuses the shared project cache; Ollama models reuse/build a local "
          "cache here, free either way)...", flush=True)
    cat_emb = {}
    for e in EMBEDDERS:
        cat_emb[e] = embed_cached(cat_texts, "cats")[0] if e == "openai" else \
            embed_cached_ollama(cat_texts, "cats", e)

    sims = {e: {} for e in EMBEDDERS}
    for cid, chunks in chunkmap.items():
        for e in EMBEDDERS:
            ch_emb = embed_cached(chunks, "chunks_" + _cache_key([cid]))[0] if e == "openai" else \
                embed_cached_ollama(chunks, "chunks_" + _cache_key([cid]), e)
            sims[e][cid] = cat_emb[e] @ ch_emb.T if (cat_emb[e].size and ch_emb.size) else \
                np.zeros((len(categories), len(chunks)), dtype="float32")

    # --- gold-chunk locations for R@k / coverage: embedder-independent ---
    gold_total = 0
    gold: list[tuple[str, str, list[int]]] = []
    for cid, chunks in chunkmap.items():
        for label in labels:
            for g in gt.get(f"{cid}__{label}", []):
                gold_total += 1
                hits = [k for k, ch in enumerate(chunks) if chunk_contains(ch, g)]
                if hits:
                    gold.append((cid, label, hits))
    coverage = len(gold) / gold_total if gold_total else 0.0
    print(f"  gold total={gold_total}  reachable={len(gold)}  coverage={coverage:.4f}\n", flush=True)

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    grand_new_spend = 0.0
    summary_rows = []

    arms = [(f"{e}_rrf_n{n}_top{TOP_K}", e, n) for e, n in ARM_CONFIGS]

    for arm, embedder, shortlist_n in arms:
        print(f"=== ARM: {arm} (embedder={embedder}, shortlist_n={shortlist_n}, "
              f"top_k={TOP_K}) ===", flush=True)

        arm_json = OUT_RESULTS / arm / f"{MODEL}.json"
        if arm not in FORCE_RERUN and arm_json.exists():
            print(f"  already have {arm_json} -- loading from disk, no new LLM spend\n", flush=True)
            prev = json.loads(arm_json.read_text(encoding="utf-8"))
            jpc = prev["metrics"].get("jaccard_per_category", {}) or {}
            best_cat = max(jpc, key=jpc.get) if jpc else ""
            worst_cat = min(jpc, key=jpc.get) if jpc else ""
            jac_avg = (sum(jpc.values()) / len(jpc)) if jpc else 0.0
            summary_rows.append({
                "arm": arm, "embedder": embedder, "chunking": "section",
                "search": "rrf", "k": TOP_K, "shortlist_n": shortlist_n, "rrf_k": RRF_K,
                "n_chunks": n_chunks_total,
                "tp": prev["micro"]["tp"], "fp": prev["micro"]["fp"], "fn": prev["micro"]["fn"],
                "precision": prev["micro"]["precision"], "recall": prev["micro"]["recall"],
                "f1": prev["micro"]["f1"], "f2": prev["micro"]["f2"],
                "aupr": round(prev["metrics"]["aupr"], 4), "best_f1": round(prev["metrics"]["best_f1"], 4),
                "best_f2": round(prev["metrics"]["best_f2"], 4),
                "jaccard_avg": round(jac_avg, 4),
                "jaccard_best": round(jpc.get(best_cat, 0.0), 4), "jaccard_best_cat": best_cat,
                "jaccard_worst": round(jpc.get(worst_cat, 0.0), 4), "jaccard_worst_cat": worst_cat,
                "coverage": prev.get("coverage", round(coverage, 4)), "R_at_k": prev.get("r_at_k", 0.0),
                "cost_usd": prev["cost_usd"], "new_spend_usd": 0.0,
            })
            continue

        sim = sims[embedder]
        retrieved_idxs_by_key: dict[str, list[int]] = {}
        context_by_key: dict[str, str] = {}
        tasks = []
        for cid, chunks in chunkmap.items():
            for ci, c in enumerate(categories):
                idxs = retrieve_rrf(shortlist_n, TOP_K, len(chunks), ci, sim[cid][ci],
                                    bm25s[cid], query_tokens, RRF_K)
                key = f"{cid}__{c['label']}"
                retrieved_idxs_by_key[key] = idxs
                ctx = "\n\n---\n\n".join(chunks[i] for i in idxs)
                context_by_key[key] = ctx
                tasks.append({"key": key, "label": c["label"],
                              "description": c["description"], "context": ctx})

        t0 = time.time()
        preds, usage, stats = asyncio.run(answer_categories(MODEL, tasks, CONCURRENCY))
        n_ok, n_failed = stats["n_ok"], stats["n_failed"]
        cost = estimate_cost(MODEL, usage["input"], usage["output"])
        grand_new_spend += cost

        scored = score_arm(preds, gt)
        by_contract, metrics, micro = scored["by_contract"], scored["metrics"], scored["micro"]

        for cid in by_contract:
            for cat in by_contract[cid]:
                key = f"{cid}__{cat}"
                idxs = retrieved_idxs_by_key.get(key, [])
                by_contract[cid][cat]["context"] = context_by_key.get(key, "")
                by_contract[cid][cat]["retrieved_chunk_idxs"] = idxs
                by_contract[cid][cat]["retrieved_chunks"] = {str(i): chunkmap[cid][i] for i in idxs}

        hit = 0
        for cid, label, hit_idxs in gold:
            if set(hit_idxs) & set(retrieved_idxs_by_key[f"{cid}__{label}"]):
                hit += 1
        r_at_k = hit / len(gold) if gold else 0.0

        jpc = metrics.get("jaccard_per_category", {}) or {}
        if jpc:
            best_cat = max(jpc, key=jpc.get); worst_cat = min(jpc, key=jpc.get)
            jac_avg = sum(jpc.values()) / len(jpc)
        else:
            best_cat = worst_cat = ""; jac_avg = 0.0

        out = {
            "arm": arm, "model": MODEL,
            "retrieval": {"embedder": embedder, "search": "rrf", "shortlist_n": shortlist_n,
                         "rrf_k": RRF_K, "top_k": TOP_K, "chunking": "section"},
            "contracts": CONTRACTS,
            "n_llm_calls_ok": n_ok, "n_llm_calls_failed": n_failed,
            "tokens": usage, "cost_usd": round(cost, 6),
            "micro": {"tp": scored["totals"]["tp"], "fp": scored["totals"]["fp"],
                      "fn": scored["totals"]["fn"], **{k: round(v, 4) for k, v in micro.items()}},
            "metrics": metrics,
            "by_category_counts": scored["agg"],
            "by_contract": by_contract,
            "r_at_k": round(r_at_k, 4), "coverage": round(coverage, 4),
        }
        adir = OUT_RESULTS / arm
        adir.mkdir(parents=True, exist_ok=True)
        (adir / f"{MODEL}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

        summary_rows.append({
            "arm": arm, "embedder": embedder, "chunking": "section",
            "search": "rrf", "k": TOP_K, "shortlist_n": shortlist_n, "rrf_k": RRF_K,
            "n_chunks": n_chunks_total,
            "tp": scored["totals"]["tp"], "fp": scored["totals"]["fp"], "fn": scored["totals"]["fn"],
            "precision": round(micro["precision"], 4), "recall": round(micro["recall"], 4),
            "f1": round(micro["f1"], 4), "f2": round(micro["f2"], 4),
            "aupr": round(metrics["aupr"], 4), "best_f1": round(metrics["best_f1"], 4),
            "best_f2": round(metrics["best_f2"], 4),
            "jaccard_avg": round(jac_avg, 4),
            "jaccard_best": round(jpc.get(best_cat, 0.0), 4), "jaccard_best_cat": best_cat,
            "jaccard_worst": round(jpc.get(worst_cat, 0.0), 4), "jaccard_worst_cat": worst_cat,
            "coverage": round(coverage, 4), "R_at_k": round(r_at_k, 4),
            "cost_usd": round(cost, 6), "new_spend_usd": round(cost, 6),
        })
        print(f"    TP={scored['totals']['tp']} FP={scored['totals']['fp']} "
              f"FN={scored['totals']['fn']} | P={micro['precision']:.3f} R={micro['recall']:.3f} "
              f"F1={micro['f1']:.3f} | R@k={r_at_k:.3f} | cost=${cost:.4f} (NEW) | "
              f"{(time.time()-t0)/60:.1f} min\n", flush=True)

    fields = list(summary_rows[0].keys())
    with open(HERE / "rerank_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    (HERE / "rerank_summary.json").write_text(
        json.dumps({"model": MODEL, "contracts": CONTRACTS, "gold_total": gold_total,
                    "coverage": round(coverage, 4), "grand_new_spend_usd": round(grand_new_spend, 6),
                    "rows": summary_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print(f"{'arm':<28}{'embedder':<10}{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}{'AUPR':>7}"
          f"{'Jac.avg':>9}{'R@k':>7}{'cost':>9}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['arm']:<28}{r['embedder']:<10}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['f1']:>7.3f}{r['f2']:>7.3f}{r['aupr']:>7.3f}{r['jaccard_avg']:>9.3f}"
              f"{r['R_at_k']:>7.3f}{('$%.3f' % r['cost_usd']):>9}")
    print("=" * 100)
    print(f"NEW spend this run: ${grand_new_spend:.4f}")
    print(f"\nWrote {HERE / 'rerank_summary.csv'}, rerank_summary.json, and "
          f"results/<arm>/{MODEL}.json (each with retrieved_chunks index->text map)")


if __name__ == "__main__":
    main()
