"""
RRF + cross-encoder reranking: does adding a real reranker on top of RRF's own
candidate pool beat RRF's own fusion-score selection?

Takes the SAME retrieval substrate as run_rrf.py, but instead of picking the final
top-5 by RRF's reciprocal-rank-fusion score, it widens the candidate pool to the
FULL union of BM25's top-N and cosine's top-N shortlists (not narrowed to top-5),
scores every (category query, chunk text) pair with Qwen3-Reranker
(tomaarsen/Qwen3-Reranker-0.6B-seq-cls, a real cross-encoder -- full cross-attention
between query and document, not a cosine-similarity comparison of separately-computed
vectors), and takes the final top-5 by THAT score instead.

Only 2 arms, one per embedder's own best RRF width (established in run_rrf.py):
  qwen3_rrf_n10_rerank_top5   (qwen3-embedding:0.6b's best RRF width was N=10)
  bge3_rrf_n25_rerank_top5    (bge-m3's best RRF width was N=25)

The reranker itself is embedder-agnostic (it reads raw chunk text, not vectors) --
only the CANDIDATE POOL being reranked differs per embedder, since that pool comes
from each embedder's own cosine shortlist union'd with the (embedder-independent)
BM25 shortlist.

Requires: CUDA-enabled torch + sentence-transformers (see this session's environment
setup -- torch was swapped from CPU-only to +cu126 for this). Falls back to CPU if
no GPU, just slower.

Run:
    python RAG_Research/Result6/TestRerank/run_rrf_rerank.py
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

os.environ.setdefault("HF_HOME", "D:/huggingface")   # keep model downloads off C:

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
RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
TOP_K = 5
SECTION_CHUNKS = RESULT6 / "section_chunking.json"
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

OLLAMA_MODEL_NAME = {"qwen3": "qwen3-embedding:0.6b", "bge3": "bge-m3"}

# (embedder, RRF shortlist N) -- each embedder's own best RRF width from run_rrf.py
ARM_CONFIGS = [("qwen3", 10), ("bge3", 25)]

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


def rrf_candidate_union(shortlist_n: int, n_chunks: int, ci: int, sims_row: np.ndarray,
                        bm25: BM25, query_tokens: list[list[str]]) -> list[int]:
    """The FULL union of BM25's top-N and cosine's top-N (unrestricted to any final
    top_k) -- this is the wide candidate pool the reranker gets to choose from."""
    n = min(shortlist_n, n_chunks)
    bm_scores = bm25.scores(query_tokens[ci])
    bm_shortlist = set(int(x) for x in np.argsort(-bm_scores)[:n])
    cos_shortlist = set(int(x) for x in np.argsort(-sims_row)[:n]) if sims_row.size else set()
    return sorted(bm_shortlist | cos_shortlist)


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

    print("Building BM25 index...", flush=True)
    bm25s = {cid: BM25([_tok(c) for c in chunks]) for cid, chunks in chunkmap.items()}

    needed_embedders = sorted({e for e, _ in ARM_CONFIGS})
    print(f"Embedding categories + chunks for embedders: {needed_embedders}...", flush=True)
    cat_emb = {e: embed_cached_ollama(cat_texts, "cats", e) for e in needed_embedders}
    sims = {e: {} for e in needed_embedders}
    for cid, chunks in chunkmap.items():
        for e in needed_embedders:
            ch_emb = embed_cached_ollama(chunks, "chunks_" + _cache_key([cid]), e)
            sims[e][cid] = cat_emb[e] @ ch_emb.T if (cat_emb[e].size and ch_emb.size) else \
                np.zeros((len(categories), len(chunks)), dtype="float32")

    # gold-chunk locations for R@k / coverage (embedder-independent)
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

    arms = [(f"{e}_rrf_n{n}_rerank_top{TOP_K}", e, n) for e, n in ARM_CONFIGS]
    need_reranker = any(
        FORCE_RERUN or not (OUT_RESULTS / arm / f"{MODEL}.json").exists()
        for arm, _, _ in arms
    )

    reranker = None
    if need_reranker:
        print(f"Loading reranker {RERANKER_MODEL} (CUDA if available)...", flush=True)
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        reranker = CrossEncoder(RERANKER_MODEL, device=device, trust_remote_code=True)
        print(f"  loaded on {device} in {time.time()-t0:.1f}s\n", flush=True)

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    grand_new_spend = 0.0
    summary_rows = []

    for arm, embedder, shortlist_n in arms:
        print(f"=== ARM: {arm} (embedder={embedder}, RRF shortlist N={shortlist_n}, "
              f"reranked final top-{TOP_K}) ===", flush=True)

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
                "search": "rrf+rerank", "k": TOP_K, "shortlist_n": shortlist_n,
                "reranker": RERANKER_MODEL, "n_chunks": n_chunks_total,
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

        # Build EVERY (contract, category) candidate pool first, then score ALL
        # pairs across the whole arm in ONE batched reranker.predict() call.
        # CrossEncoder.predict() internally mini-batches efficiently; calling it
        # once per category (246x) instead paid ~246x fixed per-call overhead and
        # was the reason the first attempt at this stalled for 50+ minutes with
        # zero throughput -- this is the fix.
        t_rerank0 = time.time()
        per_key_cands: dict[str, list[int]] = {}
        all_pairs: list[tuple[str, str]] = []
        pair_owner: list[tuple[str, int]] = []   # (key, position within that key's candidates)
        for cid, chunks in chunkmap.items():
            for ci, c in enumerate(categories):
                cand_idxs = rrf_candidate_union(shortlist_n, len(chunks), ci, sim[cid][ci],
                                                bm25s[cid], query_tokens)
                key = f"{cid}__{c['label']}"
                per_key_cands[key] = cand_idxs
                query_text = cat_texts[ci]
                for pos, idx in enumerate(cand_idxs):
                    all_pairs.append((query_text, chunks[idx]))
                    pair_owner.append((key, pos))

        print(f"  scoring {len(all_pairs)} (query, chunk) pairs across "
              f"{len(per_key_cands)} categories in one batched reranker call...", flush=True)
        all_scores = reranker.predict(all_pairs, batch_size=64, show_progress_bar=False) \
            if all_pairs else []
        print(f"  reranked in {time.time()-t_rerank0:.1f}s", flush=True)

        scores_by_key: dict[str, list[float]] = {k: [0.0] * len(v) for k, v in per_key_cands.items()}
        for (key, pos), score in zip(pair_owner, all_scores):
            scores_by_key[key][pos] = float(score)

        retrieved_idxs_by_key: dict[str, list[int]] = {}
        context_by_key: dict[str, str] = {}
        tasks = []
        for cid, chunks in chunkmap.items():
            for ci, c in enumerate(categories):
                key = f"{cid}__{c['label']}"
                cand_idxs = per_key_cands[key]
                scores = scores_by_key[key]
                order = sorted(range(len(cand_idxs)), key=lambda k: -scores[k])
                idxs = [cand_idxs[k] for k in order[:TOP_K]]

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
            "retrieval": {"embedder": embedder, "search": "rrf+rerank", "shortlist_n": shortlist_n,
                         "reranker": RERANKER_MODEL, "top_k": TOP_K, "chunking": "section"},
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
            "search": "rrf+rerank", "k": TOP_K, "shortlist_n": shortlist_n,
            "reranker": RERANKER_MODEL, "n_chunks": n_chunks_total,
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
    with open(HERE / "rerank_reranker_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    (HERE / "rerank_reranker_summary.json").write_text(
        json.dumps({"model": MODEL, "contracts": CONTRACTS, "gold_total": gold_total,
                    "coverage": round(coverage, 4), "grand_new_spend_usd": round(grand_new_spend, 6),
                    "rows": summary_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    for r in summary_rows:
        print(f"{r['arm']:<32}{r['embedder']:<8}P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} F2={r['f2']:.3f} AUPR={r['aupr']:.3f} "
              f"Jac.avg={r['jaccard_avg']:.3f} R@k={r['R_at_k']:.3f} cost=${r['cost_usd']:.3f}")
    print("=" * 100)
    print(f"NEW spend this run: ${grand_new_spend:.4f}")
    print(f"\nWrote {HERE / 'rerank_reranker_summary.csv'}, .json, and "
          f"results/<arm>/{MODEL}.json")


if __name__ == "__main__":
    main()
