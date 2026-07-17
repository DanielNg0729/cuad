"""
Master summary across EVERY method in Result6/results (old + the hybrid grid).

For each method it collects, from the result JSON plus a re-derived retrieval pass:
  F1 / Precision / Recall / F2 / AUPR / best-F1   (end-to-end, micro)
  Jaccard: macro-average, best category (+name), worst category (+name)
  coverage : gold answers reachable in that method's chunk substrate / all gold
  R@k      : of reachable gold, fraction whose chunk is in the top-k retrieved
  chunks / search / k / bm25_prefilter / cost

Writes master_summary.csv + master_summary.json. No API calls (embeddings cached).

Run (after all methods have been run into results/):
    python RAG_Research/Result6/master_summary.py
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_Research"))
os.chdir(ROOT)

from OpenAITest import load_categories, CATEGORY_CSV                        # noqa: E402
from evaluate import get_answers                                            # noqa: E402
from rag_research import embed_cached, _cache_key, BM25, _tok, retrieve_idxs  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "gpt-5.4"
CHUNK_FILES = {"markdown": ROOT / "test_chunking.json",
               "section":  HERE / "section_chunking.json"}
CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]


def _norm(s):
    return " ".join((s or "").split()).lower()


def contains(chunk, gold):
    ng = _norm(gold)
    return bool(ng) and ng in _norm(chunk)


def build_substrate(path, cat_emb, qtok, labels, gt):
    cd = {c["contract_id"]: c["chunks"]
          for c in json.loads(path.read_text(encoding="utf-8"))["data"]}
    chunkmap = {cid: cd[cid] for cid in CONTRACTS}
    sims, bm = {}, {}
    for cid, chunks in chunkmap.items():
        e, _ = embed_cached(chunks, "chunks_" + _cache_key([cid]))
        sims[cid] = cat_emb @ e.T
        bm[cid] = BM25([_tok(c) for c in chunks])
    gold = []
    for cid, chunks in chunkmap.items():
        for label in labels:
            for g in gt.get(f"{cid}__{label}", []):
                hits = [k for k, ch in enumerate(chunks) if contains(ch, g)]
                if hits:
                    gold.append((cid, label, hits))
    n_chunks = sum(len(v) for v in chunkmap.values())
    return {"chunkmap": chunkmap, "sims": sims, "bm": bm, "gold": gold, "n_chunks": n_chunks}


def r_at_k(sub, labels, qtok, search, k, hn):
    gold = sub["gold"]
    ok = 0
    for cid, lab, hits in gold:
        ci = labels.index(lab)
        idxs = set(retrieve_idxs(search, k, len(sub["chunkmap"][cid]), ci,
                                 sub["sims"][cid], sub["bm"][cid], qtok, hn))
        ok += bool(set(hits) & idxs)
    return ok / len(gold) if gold else 0.0


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    cat_emb, _ = embed_cached([f'{c["label"]}. {c["description"]}' for c in categories], "cats")
    qtok = [_tok(f'{c["label"]} {c["description"]}') for c in categories]
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)
    total_gold = sum(len(v) for v in gt.values())

    subs = {name: build_substrate(path, cat_emb, qtok, labels, gt)
            for name, path in CHUNK_FILES.items()}

    paths = sorted(RESULTS.glob(f"*/{MODEL}.json"))
    rows = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        mk = d["method"]
        r = d["retrieval"]
        chunking = r.get("chunking", "markdown")
        sub = subs[chunking]
        mi, me = d["micro"], d["metrics"]
        jpc = me.get("jaccard_per_category", {}) or {}
        if jpc:
            best_cat = max(jpc, key=jpc.get); worst_cat = min(jpc, key=jpc.get)
            jac_avg = sum(jpc.values()) / len(jpc)
        else:
            best_cat = worst_cat = ""; jac_avg = 0.0

        # retrieval reach at this method's config (check methods retrieve like their base)
        search = r["search"]; k = r["top_k"]; hn = r.get("hybrid_n") or 10
        rk = r_at_k(sub, labels, qtok, search, k, hn)
        coverage = len(sub["gold"]) / total_gold

        rows.append({
            "method": mk, "chunking": chunking, "search": search, "k": k,
            "bm25_prefilter": (r.get("hybrid_n") if search == "hybrid" else ""),
            "n_chunks": sub["n_chunks"],
            "tp": mi["tp"], "fp": mi["fp"], "fn": mi["fn"],
            "precision": mi["precision"], "recall": mi["recall"],
            "f1": mi["f1"], "f2": mi["f2"],
            "aupr": round(me["aupr"], 4), "best_f1": round(me["best_f1"], 4),
            "jaccard_avg": round(jac_avg, 4),
            "jaccard_best": round(jpc.get(best_cat, 0.0), 4), "jaccard_best_cat": best_cat,
            "jaccard_worst": round(jpc.get(worst_cat, 0.0), 4), "jaccard_worst_cat": worst_cat,
            "coverage": round(coverage, 4), "R_at_k": round(rk, 4),
            "cost_usd": d["cost_usd"],
        })

    # order: markdown M-methods first (by name), then hybrid grid by (prefilter, k)
    def sortkey(r):
        if r["search"] == "hybrid" and isinstance(r["bm25_prefilter"], int):
            return (1, r["bm25_prefilter"], r["k"])
        return (0, 0, r["method"])
    rows.sort(key=sortkey)

    fields = list(rows[0].keys())
    with open(HERE / "master_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    (HERE / "master_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{len(rows)} methods | total gold {total_gold} | "
          f"markdown {subs['markdown']['n_chunks']} chunks / section {subs['section']['n_chunks']} chunks\n")
    print(f"{'method':<26}{'F1':>6}{'P':>6}{'R':>6}{'Jac.avg':>8}{'R@k':>6}{'cov':>6}{'cost':>7}")
    for r in rows:
        print(f"{r['method']:<26}{r['f1']:>6.3f}{r['precision']:>6.3f}{r['recall']:>6.3f}"
              f"{r['jaccard_avg']:>8.3f}{r['R_at_k']:>6.2f}{r['coverage']:>6.2f}{('$%.2f'%r['cost_usd']):>7}")
    print(f"\nWrote {HERE / 'master_summary.csv'} and master_summary.json")


if __name__ == "__main__":
    main()
