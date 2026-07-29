"""
Full-extraction ablation: OpenAI text-embedding-3-small vs qwen3-embedding:0.6b
(the winning local embedder from ../TestEmbeddedModel's retrieval-only test),
across 3 retrieval configs, with REAL gpt-5.4 calls. Same section chunking / 6
contracts / 41 categories / gold as the rest of Result6, so numbers are directly
comparable to M7_section_hybrid_top5, master_summary.csv, etc.

5 arms (not 6 -- BM25-only retrieval doesn't use embeddings at all, so it is a
single embedder-agnostic arm, never duplicated per-embedder):

  A  bm25_top10                BM25-only (lexical), top-10.            [NEW LLM calls]
  B  openai_cosine_top5        OpenAI embeddings, cosine-only, top-5.  [NEW LLM calls]
  C  openai_hybrid_bm10_cos5   OpenAI embeddings, BM25(10)->cosine(5).
                                == M7_section_hybrid_top5 exactly -- REUSES its
                                already-paid-for predictions/usage/cost, no new
                                LLM calls; only reconstructs+saves the retrieved
                                chunk TEXT (deterministic from cached embeddings).
  D  qwen3_cosine_top5         qwen3-embedding:0.6b, cosine-only, top-5. [NEW LLM calls]
  E  qwen3_hybrid_bm10_cos5    qwen3-embedding:0.6b, BM25(10)->cosine(5). [NEW LLM calls]

For EVERY arm, the retrieved chunk TEXT (not just indices) is saved per
(contract, category) in results/<arm>/gpt-5.4.json under "by_contract"."<cat>"."context",
so you can manually inspect exactly what the model was shown.

Metrics kept identical to master_summary.py's schema: precision/recall/f1/f2/aupr/
best_f1, jaccard avg/best/worst(+category), coverage, R@k, cost.

Run:
    python RAG_Research/Result6/TestAblation/run_ablation.py
"""

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # .../Result6/TestAblation
RESULT6 = HERE.parent                                     # .../Result6
RAG_RESEARCH = RESULT6.parent                              # .../RAG_Research
ROOT = RAG_RESEARCH.parent                                  # repo root

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAG_RESEARCH))
os.chdir(ROOT)

import asyncio                                                                # noqa: E402
from dotenv import load_dotenv                                                # noqa: E402
from OpenAITest import load_categories, CATEGORY_CSV                          # noqa: E402
from evaluate import get_answers                                              # noqa: E402
from rag_research import (                                                    # noqa: E402
    eval_metrics, breakdown, prf, answer_categories, embed_cached,
    BM25, _tok, retrieve_idxs, estimate_cost,
)

load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY missing -- add it to .env")

MODEL = "gpt-5.4"
CONCURRENCY = 12
SECTION_CHUNKS = RESULT6 / "section_chunking.json"
M7_JSON = RESULT6 / "results" / "M7_section_hybrid_top5" / f"{MODEL}.json"
OUT_RESULTS = HERE / "results"
CACHE_DIR = HERE / ".cache"

CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]

# arm -> (embedder, search, top_k, hybrid_n)  embedder is None for bm25 (embedder-agnostic)
# NOTE: bm25_top5 / bm25_top10 each appear ONCE, not once per embedder -- BM25 is purely
# lexical and never touches an embedding vector, so it is identical regardless of embedder
# (see hybrid_bm5_cos5 in ../TestEmbeddedModel, which scored identically across all 5
# embedders for the same reason). Running it "per embedder" would just be 3x the cost for
# 3 identical result files.
ARMS = {
    "bm25_top5":               (None,      "bm25",    5, None),
    "bm25_top10":              (None,      "bm25",   10, None),
    "openai_cosine_top5":      ("openai",  "cosine",  5, None),
    "openai_hybrid_bm10_cos5": ("openai",  "hybrid",  5, 10),
    "qwen3_cosine_top5":       ("qwen3",   "cosine",  5, None),
    "qwen3_hybrid_bm10_cos5":  ("qwen3",   "hybrid",  5, 10),
    "bge3_cosine_top5":        ("bge3",    "cosine",  5, None),
    "bge3_hybrid_bm10_cos5":   ("bge3",    "hybrid",  5, 10),
    "openai_cosine_top8":      ("openai",  "cosine",  8, None),
    "qwen3_cosine_top8":       ("qwen3",   "cosine",  8, None),
    "bge3_cosine_top8":        ("bge3",    "cosine",  8, None),
    "openai_hybrid_bm10_cos8": ("openai",  "hybrid",  8, 10),
    "qwen3_hybrid_bm10_cos8":  ("qwen3",   "hybrid",  8, 10),
    "bge3_hybrid_bm10_cos8":   ("bge3",    "hybrid",  8, 10),
}
OLLAMA_MODEL_NAME = {"qwen3": "qwen3-embedding:0.6b", "bge3": "bge-m3"}
# Arms whose results/<arm>/gpt-5.4.json already exists are SKIPPED (loaded from disk,
# no new LLM spend) unless listed here to force a re-run.
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


def prf_from_counts(tp, fp, fn):
    return prf(tp, fp, fn)


def score_arm(preds: dict, gt: dict) -> dict:
    """Full project metric suite, identical to rag_research.run_method's scoring."""
    pred_nbest = {
        k: [{"text": s, "probability": 1.0} for s in preds.get(k, [])]
           + [{"text": "", "probability": 0.0}]
        for k in gt
    }
    metrics = eval_metrics(pred_nbest, gt, category=None)
    by_contract, agg = breakdown(preds, gt)
    totals = {k: sum(agg[c][k] for c in agg) for k in ("tp", "fp", "fn")}
    p, r, f1, f2 = prf_from_counts(totals["tp"], totals["fp"], totals["fn"])
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

    print("Building BM25 index (embedder-agnostic)...", flush=True)
    bm25s = {cid: BM25([_tok(c) for c in chunks]) for cid, chunks in chunkmap.items()}

    embedder_keys = sorted({e for e, *_ in ARMS.values() if e})   # e.g. {"openai","qwen3","bge3"}
    print(f"Embedding categories + chunks for embedders: {embedder_keys} "
          "(OpenAI reuses the shared project cache; Ollama models reuse/build a local "
          "cache here, free either way)...", flush=True)
    cat_emb = {}
    for e in embedder_keys:
        cat_emb[e] = embed_cached(cat_texts, "cats")[0] if e == "openai" else \
            embed_cached_ollama(cat_texts, "cats", e)

    sims = {e: {} for e in embedder_keys}
    for cid, chunks in chunkmap.items():
        for e in embedder_keys:
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
    grand_cost = 0.0
    summary_rows = []

    for arm, (embedder, search, top_k, hybrid_n) in ARMS.items():
        print(f"=== ARM: {arm} (search={search}, top_k={top_k}, hybrid_n={hybrid_n}, "
              f"embedder={embedder}) ===", flush=True)

        arm_json = OUT_RESULTS / arm / f"{MODEL}.json"
        if arm not in FORCE_RERUN and arm_json.exists():
            print(f"  already have {arm_json} -- loading from disk, no new LLM spend\n", flush=True)
            prev = json.loads(arm_json.read_text(encoding="utf-8"))

            # Backfill an explicit {index: chunk_text} map (deterministic from the
            # already-saved retrieved_chunk_idxs + chunkmap -- no LLM/embedding calls
            # needed) so each chunk's content is directly addressable by its index,
            # not just visible as one concatenated blob in "context".
            patched = False
            for cid, cats in prev["by_contract"].items():
                for cat, entry in cats.items():
                    if "retrieved_chunks" not in entry:
                        idxs = entry.get("retrieved_chunk_idxs", [])
                        entry["retrieved_chunks"] = {str(i): chunkmap[cid][i] for i in idxs}
                        patched = True
            if patched:
                arm_json.write_text(json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  patched retrieved_chunks (index -> text) into {arm_json}", flush=True)

            jpc = prev["metrics"].get("jaccard_per_category", {}) or {}
            best_cat = max(jpc, key=jpc.get) if jpc else ""
            worst_cat = min(jpc, key=jpc.get) if jpc else ""
            jac_avg = (sum(jpc.values()) / len(jpc)) if jpc else 0.0
            summary_rows.append({
                "arm": arm, "embedder": embedder or "n/a (lexical)", "chunking": "section",
                "search": search, "k": top_k, "bm25_prefilter": hybrid_n or "",
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

        sim = sims.get(embedder, {}) if embedder else {}

        # build retrieval + context for EVERY (contract, category) -- always, so the
        # retrieved chunk text is saved even for the reused M7 arm.
        retrieved_idxs_by_key: dict[str, list[int]] = {}
        context_by_key: dict[str, str] = {}
        tasks = []
        for cid, chunks in chunkmap.items():
            for ci, c in enumerate(categories):
                s = sim[cid] if sim else np.zeros((len(categories), len(chunks)), dtype="float32")
                idxs = retrieve_idxs(search, top_k, len(chunks), ci, s, bm25s[cid],
                                     query_tokens, hybrid_n or 10)
                key = f"{cid}__{c['label']}"
                retrieved_idxs_by_key[key] = idxs
                ctx = "\n\n---\n\n".join(chunks[i] for i in idxs)
                context_by_key[key] = ctx
                tasks.append({"key": key, "label": c["label"],
                              "description": c["description"], "context": ctx})

        t0 = time.time()
        if arm == "openai_hybrid_bm10_cos5" and M7_JSON.exists():
            print("  reusing M7_section_hybrid_top5's existing predictions (no new LLM calls)",
                  flush=True)
            m7 = json.loads(M7_JSON.read_text(encoding="utf-8"))
            preds = {}
            for cid, cats in m7["by_contract"].items():
                for cat, entry in cats.items():
                    preds[f"{cid}__{cat}"] = list(entry.get("predictions", []))
            usage = {"input": m7["tokens"]["input"], "output": m7["tokens"]["output"]}
            n_ok, n_failed = m7["n_llm_calls_ok"], m7["n_llm_calls_failed"]
            llm_cost = estimate_cost(MODEL, usage["input"], usage["output"])
            reused_cost = m7["cost_usd"]
        else:
            preds, usage, stats = asyncio.run(answer_categories(MODEL, tasks, CONCURRENCY))
            n_ok, n_failed = stats["n_ok"], stats["n_failed"]
            llm_cost = estimate_cost(MODEL, usage["input"], usage["output"])
            reused_cost = None

        scored = score_arm(preds, gt)
        by_contract, metrics, micro = scored["by_contract"], scored["metrics"], scored["micro"]

        # inject the retrieved chunk TEXT into by_contract so it's saved alongside
        # ground_truth/predictions/tp/fn/fp for manual inspection.
        for cid in by_contract:
            for cat in by_contract[cid]:
                key = f"{cid}__{cat}"
                idxs = retrieved_idxs_by_key.get(key, [])
                by_contract[cid][cat]["context"] = context_by_key.get(key, "")
                by_contract[cid][cat]["retrieved_chunk_idxs"] = idxs
                by_contract[cid][cat]["retrieved_chunks"] = {str(i): chunkmap[cid][i] for i in idxs}

        # R@k / coverage for this arm's exact retrieval config
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

        cost = reused_cost if reused_cost is not None else llm_cost
        grand_cost += (llm_cost if reused_cost is None else 0.0)   # reused arm: $0 NEW spend

        out = {
            "arm": arm, "model": MODEL,
            "retrieval": {"embedder": embedder, "search": search, "top_k": top_k,
                         "hybrid_n": hybrid_n, "chunking": "section"},
            "contracts": CONTRACTS,
            "n_llm_calls_ok": n_ok, "n_llm_calls_failed": n_failed,
            "reused_from": "M7_section_hybrid_top5" if reused_cost is not None else None,
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
            "arm": arm, "embedder": embedder or "n/a (lexical)", "chunking": "section",
            "search": search, "k": top_k, "bm25_prefilter": hybrid_n or "",
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
            "cost_usd": round(cost, 6),
            "new_spend_usd": 0.0 if reused_cost is not None else round(llm_cost, 6),
        })
        print(f"    TP={scored['totals']['tp']} FP={scored['totals']['fp']} "
              f"FN={scored['totals']['fn']} | P={micro['precision']:.3f} R={micro['recall']:.3f} "
              f"F1={micro['f1']:.3f} | R@k={r_at_k:.3f} | "
              f"cost=${cost:.4f} ({'reused' if reused_cost is not None else 'NEW'}) | "
              f"{(time.time()-t0)/60:.1f} min\n", flush=True)

    # ---- write master summary ----
    fields = list(summary_rows[0].keys())
    with open(HERE / "ablation_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    (HERE / "ablation_summary.json").write_text(
        json.dumps({"model": MODEL, "contracts": CONTRACTS, "gold_total": gold_total,
                    "coverage": round(coverage, 4), "grand_new_spend_usd": round(grand_cost, 6),
                    "rows": summary_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print(f"{'arm':<28}{'embedder':<12}{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}{'AUPR':>7}"
          f"{'Jac.avg':>9}{'R@k':>7}{'cost':>9}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['arm']:<28}{r['embedder']:<12}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['f1']:>7.3f}{r['f2']:>7.3f}{r['aupr']:>7.3f}{r['jaccard_avg']:>9.3f}"
              f"{r['R_at_k']:>7.3f}{('$%.3f' % r['cost_usd']):>9}")
    print("=" * 100)
    print(f"NEW spend this run: ${grand_cost:.4f}  (arm C reused M7's existing predictions -- $0 new)")
    print(f"\nWrote {HERE / 'ablation_summary.csv'}, ablation_summary.json, "
          f"and results/<arm>/{MODEL}.json (each with retrieved chunk text under by_contract.*.context)")


if __name__ == "__main__":
    main()
