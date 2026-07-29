"""
CompareContractEval -- the best RAG method from Result6, run over ALL 102 CUAD
test contracts instead of the 6-contract sample.

Method (fixed): the single best-performing arm found across TestAblation +
TestRerank -- `qwen3_rrf_n10_top5`:

    embedder    qwen3-embedding:0.6b  (local, Ollama, free)
    chunking    section (1.1/1.2/ARTICLE split, packed to 1500 chars)
    retrieval   Reciprocal Rank Fusion: BM25 top-10 shortlist UNION cosine top-10
                shortlist, fused by 1/(60 + rank), final top-5
    extraction  one structured-output LLM call per (contract, category)

On the 6-contract set with gpt-5.4 this scored F1=0.4566 / P=0.4107 / R=0.5140 /
AUPR=0.362 -- the best of every arm tested. This script holds that method fixed
and changes exactly two things: the model (default gpt-4.1) and the contract set
(all 102). Everything else -- chunk params, RRF constants, span validation,
scorer -- is identical, so the numbers stay comparable to Result6.

Output schema is identical to ../RAG_Research/Result6/TestAblation/, i.e. per
(contract, category): ground_truth / predictions / tp / fn / fp, the retrieved
chunk text both as a concatenated `context` blob and as an explicit
`retrieved_chunks` {index -> text} map, plus `retrieved_chunk_idxs`.

Run:
    python CompareContractEval/run_compare.py --limit-contracts 2   # smoke test
    python CompareContractEval/run_compare.py                       # full 102
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAG_RESEARCH = ROOT / "RAG_Research"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAG_RESEARCH))
os.chdir(ROOT)

import asyncio                                                                # noqa: E402
from dotenv import load_dotenv                                                # noqa: E402
from OpenAITest import load_categories, CATEGORY_CSV                          # noqa: E402
from evaluate import get_answers                                              # noqa: E402
from rag_research import (                                                    # noqa: E402
    eval_metrics, breakdown, prf, answer_categories, BM25, _tok, estimate_cost,
)

load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY missing -- add it to .env")

# --- method defaults (overridable from the CLI) ------------------------------
EMBEDDER_KEY = "qwen3"
OLLAMA_MODEL_NAME = {"qwen3": "qwen3-embedding:0.6b", "bge3": "bge-m3"}
SEARCH = "rrf"
SHORTLIST_N = 10
TOP_K = 5
RRF_K = 60
CHUNKING = "section"

CHUNK_FILE = HERE / "section_chunking_102.json"
OUT_RESULTS = HERE / "results"
CACHE_DIR = HERE / ".cache"


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def chunk_contains(chunk: str, gold: str) -> bool:
    """Same fuzzy containment used by Result6's diagnose.py / TestAblation: exact
    normalised substring, falling back to >=90% gold-token overlap."""
    nc, ng = _norm(chunk), _norm(gold)
    if not ng:
        return False
    if ng in nc:
        return True
    gt_toks = set(ng.split())
    if not gt_toks:
        return False
    return len(gt_toks & set(nc.split())) / len(gt_toks) >= 0.9


def _cache_key(texts: list[str]) -> str:
    return hashlib.sha1((" ".join(texts)).encode("utf-8")).hexdigest()[:16]


def ollama_embed(texts: list[str]) -> np.ndarray:
    from langchain_ollama import OllamaEmbeddings
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    vecs = np.asarray(OllamaEmbeddings(model=OLLAMA_MODEL).embed_documents(list(texts)),
                      dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_cached_ollama(texts: list[str], tag: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
    path = CACHE_DIR / f"{safe}_{tag}_{_cache_key(texts)}.npy"
    if path.exists():
        return np.load(path)
    mat = ollama_embed(texts)
    np.save(path, mat)
    return mat


def retrieve_rrf(shortlist_n: int, top_k: int, n_chunks: int, ci: int,
                 sims_row: np.ndarray, bm25: BM25, query_tokens: list[list[str]],
                 rrf_k: int = RRF_K) -> list[int]:
    """Union BM25's own top-N and cosine's own top-N (each ranked independently over
    ALL chunks of this contract), fuse by reciprocal rank, return the final top_k."""
    n = min(shortlist_n, n_chunks)
    bm_scores = bm25.scores(query_tokens[ci])
    bm_shortlist = [int(x) for x in np.argsort(-bm_scores)[:n]]
    cos_shortlist = [int(x) for x in np.argsort(-sims_row)[:n]] if sims_row.size else []

    bm_rank = {idx: r for r, idx in enumerate(bm_shortlist)}
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
    return sorted(candidates, key=lambda i: -rrf_score[i])[:min(top_k, len(candidates))]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit-contracts", type=int, default=0,
                    help="Only run the first N contracts (smoke test). 0 = all.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if the result JSON already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build retrieval + print the projected token/cost bill, make no LLM calls.")
    ap.add_argument("--shortlist-n", type=int, default=SHORTLIST_N,
                    help=f"BM25/cosine shortlist width N for RRF (default {SHORTLIST_N}).")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help=f"Final number of chunks sent to the LLM (default {TOP_K}).")
    args = ap.parse_args()

    MODEL = args.model
    shortlist_n, top_k = args.shortlist_n, args.top_k
    arm = f"{EMBEDDER_KEY}_rrf_n{shortlist_n}_top{top_k}"
    tag = "smoke" if args.limit_contracts else "all102"

    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    cat_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]

    chunkfile = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    chunkdata = {c["contract_id"]: c["chunks"] for c in chunkfile["data"]}
    contracts = [c["contract_id"] for c in chunkfile["data"] if c["chunks"]]
    if args.limit_contracts:
        contracts = contracts[:args.limit_contracts]
    chunkmap = {cid: chunkdata[cid] for cid in contracts}
    n_chunks_total = sum(len(v) for v in chunkmap.values())

    print(f"CompareContractEval | arm={arm} | model={MODEL} | "
          f"{len(contracts)} contracts | {n_chunks_total} section chunks", flush=True)

    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=contracts)
    kept = set(labels)
    gt = {k: v for k, v in gt.items() if k.rsplit("__", 1)[1] in kept}
    gold_answers = sum(len(v) for v in gt.values())
    print(f"  ground truth: {len(gt)} (contract, category) pairs, "
          f"{gold_answers} gold answers", flush=True)

    print("Building per-contract BM25 indexes...", flush=True)
    bm25s = {cid: BM25([_tok(c) for c in chunks]) for cid, chunks in chunkmap.items()}

    print(f"Embedding 41 categories + {n_chunks_total} chunks with {OLLAMA_MODEL} "
          "(local, free; cached on disk)...", flush=True)
    t_emb = time.time()
    cat_emb = embed_cached_ollama(cat_texts, "cats")
    sims = {}
    for i, (cid, chunks) in enumerate(chunkmap.items(), 1):
        ch_emb = embed_cached_ollama(chunks, "chunks_" + _cache_key([cid]))
        sims[cid] = cat_emb @ ch_emb.T if (cat_emb.size and ch_emb.size) else \
            np.zeros((len(categories), len(chunks)), dtype="float32")
        if i % 20 == 0 or i == len(chunkmap):
            print(f"    embedded {i}/{len(chunkmap)} contracts "
                  f"({(time.time()-t_emb)/60:.1f} min)", flush=True)
    print(f"  embedding done in {(time.time()-t_emb)/60:.1f} min (no API cost)\n", flush=True)

    # --- gold-chunk locations for coverage / R@k (embedder-independent) ---
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
    print(f"  gold total={gold_total}  reachable={len(gold)}  coverage={coverage:.4f}\n",
          flush=True)

    # --- retrieval (no API cost) ---
    retrieved_idxs_by_key: dict[str, list[int]] = {}
    context_by_key: dict[str, str] = {}
    tasks = []
    for cid, chunks in chunkmap.items():
        for ci, c in enumerate(categories):
            idxs = retrieve_rrf(shortlist_n, top_k, len(chunks), ci, sims[cid][ci],
                                bm25s[cid], query_tokens, RRF_K)
            key = f"{cid}__{c['label']}"
            retrieved_idxs_by_key[key] = idxs
            ctx = "\n\n---\n\n".join(chunks[i] for i in idxs)
            context_by_key[key] = ctx
            tasks.append({"key": key, "label": c["label"],
                          "description": c["description"], "context": ctx})

    ctx_chars = sum(len(t["context"]) for t in tasks)
    est_in = ctx_chars / 4 + len(tasks) * 120          # +prompt/schema overhead per call
    est_out = len(tasks) * 60
    est_cost = estimate_cost(MODEL, est_in, est_out)
    print(f"  {len(tasks)} LLM calls queued | {ctx_chars/1e6:.2f}M context chars "
          f"| projected ~{est_in/1e6:.2f}M in / ~{est_out/1e3:.0f}k out "
          f"| projected cost ~${est_cost:.2f}", flush=True)

    if args.dry_run:
        print("\n--dry-run: stopping before any LLM call.")
        return

    arm_json = OUT_RESULTS / f"{arm}__{tag}" / f"{MODEL}.json"
    if arm_json.exists() and not args.force:
        print(f"\n{arm_json} already exists -- nothing to do (use --force to re-run).")
        return

    print(f"\nCalling {MODEL} at concurrency {args.concurrency}...", flush=True)
    t0 = time.time()
    preds, usage, stats = asyncio.run(answer_categories(MODEL, tasks, args.concurrency))
    elapsed = time.time() - t0
    cost = estimate_cost(MODEL, usage["input"], usage["output"])
    print(f"  {stats['n_ok']} ok / {stats['n_failed']} failed | "
          f"{usage['input']:,} in + {usage['output']:,} out tokens | "
          f"${cost:.4f} | {elapsed/60:.1f} min", flush=True)

    scored = score_arm(preds, gt)
    by_contract, metrics, micro = scored["by_contract"], scored["metrics"], scored["micro"]

    for cid in by_contract:
        for cat in by_contract[cid]:
            key = f"{cid}__{cat}"
            idxs = retrieved_idxs_by_key.get(key, [])
            by_contract[cid][cat]["context"] = context_by_key.get(key, "")
            by_contract[cid][cat]["retrieved_chunk_idxs"] = idxs
            by_contract[cid][cat]["retrieved_chunks"] = {str(i): chunkmap[cid][i] for i in idxs}

    hit = sum(1 for cid, label, hit_idxs in gold
              if set(hit_idxs) & set(retrieved_idxs_by_key.get(f"{cid}__{label}", [])))
    r_at_k = hit / len(gold) if gold else 0.0

    jpc = metrics.get("jaccard_per_category", {}) or {}
    if jpc:
        best_cat = max(jpc, key=jpc.get); worst_cat = min(jpc, key=jpc.get)
        jac_avg = sum(jpc.values()) / len(jpc)
    else:
        best_cat = worst_cat = ""; jac_avg = 0.0

    out = {
        "arm": arm, "model": MODEL, "scope": tag,
        "retrieval": {"embedder": EMBEDDER_KEY, "embedder_model": OLLAMA_MODEL,
                      "search": SEARCH, "shortlist_n": shortlist_n, "rrf_k": RRF_K,
                      "top_k": top_k, "chunking": CHUNKING,
                      "chunk_chars": chunkfile["metadata"]["chunk_chars"]},
        "n_contracts": len(contracts), "n_chunks": n_chunks_total,
        "contracts": contracts,
        "n_llm_calls_ok": stats["n_ok"], "n_llm_calls_failed": stats["n_failed"],
        "tokens": usage, "cost_usd": round(cost, 6),
        "wall_seconds": round(elapsed, 1),
        "micro": {"tp": scored["totals"]["tp"], "fp": scored["totals"]["fp"],
                  "fn": scored["totals"]["fn"], **{k: round(v, 4) for k, v in micro.items()}},
        "metrics": metrics,
        "by_category_counts": scored["agg"],
        "by_contract": by_contract,
        "r_at_k": round(r_at_k, 4), "coverage": round(coverage, 4),
        "gold_total": gold_total,
    }
    arm_json.parent.mkdir(parents=True, exist_ok=True)
    arm_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    row = {
        "arm": arm, "model": MODEL, "scope": tag, "embedder": OLLAMA_MODEL,
        "chunking": CHUNKING, "search": SEARCH, "shortlist_n": shortlist_n,
        "rrf_k": RRF_K, "k": top_k,
        "n_contracts": len(contracts), "n_chunks": n_chunks_total,
        "gold_total": gold_total,
        "tp": scored["totals"]["tp"], "fp": scored["totals"]["fp"], "fn": scored["totals"]["fn"],
        "precision": round(micro["precision"], 4), "recall": round(micro["recall"], 4),
        "f1": round(micro["f1"], 4), "f2": round(micro["f2"], 4),
        "aupr": round(metrics["aupr"], 4), "best_f1": round(metrics["best_f1"], 4),
        "best_f2": round(metrics["best_f2"], 4),
        "jaccard_avg": round(jac_avg, 4),
        "jaccard_best": round(jpc.get(best_cat, 0.0), 4), "jaccard_best_cat": best_cat,
        "jaccard_worst": round(jpc.get(worst_cat, 0.0), 4), "jaccard_worst_cat": worst_cat,
        "coverage": round(coverage, 4), "R_at_k": round(r_at_k, 4),
        "input_tokens": usage["input"], "output_tokens": usage["output"],
        "cost_usd": round(cost, 6), "wall_minutes": round(elapsed / 60, 2),
    }
    csv_path = HERE / f"compare_summary__{arm}__{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    (HERE / f"compare_summary__{arm}__{tag}.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"{'model':<10}{'contracts':>10}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}"
          f"{'F1':>8}{'F2':>8}{'AUPR':>8}{'Jac':>8}{'R@k':>7}")
    print("-" * 92)
    print(f"{MODEL:<10}{len(contracts):>10}{row['tp']:>6}{row['fp']:>6}{row['fn']:>6}"
          f"{row['precision']:>8.3f}{row['recall']:>8.3f}{row['f1']:>8.3f}{row['f2']:>8.3f}"
          f"{row['aupr']:>8.3f}{row['jaccard_avg']:>8.3f}{row['R_at_k']:>7.3f}")
    print("=" * 92)
    print(f"cost ${cost:.4f} | {elapsed/60:.1f} min | "
          f"{arm_json} ({arm_json.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
