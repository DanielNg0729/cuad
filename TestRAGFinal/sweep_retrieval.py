"""
FREE retrieval-only sweep over the RRF parameter space. No LLM calls, no cost.

For every (shortlist_n, top_k, rrf_k) it measures R@k -- the share of gold answers whose
containing chunk made it into the retrieved set. That is the ceiling on recall: the
extractor can never return an answer it was never shown.

It also projects end-to-end F1 using the conversion rates measured from the two real
102-contract runs, so a config can be judged before paying for it:

    from n10_top5 -> n20_top8 :  R@k +0.076  ->  TP +88, FP +271
    i.e. d(TP)/d(R@k) ~ 1158 ,  d(FP)/d(R@k) ~ 3566   (per unit R@k, over 4182 pairs)

That linear model is crude and only trustworthy near the measured range, but it is enough
to answer "is any RRF config likely to beat what we already have?"

Run:
    python TestRAGFinal/sweep_retrieval.py
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys, os                                                              # noqa: E402
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "RAG_Research"))
os.chdir(ROOT)

from OpenAITest import load_categories, CATEGORY_CSV                        # noqa: E402
from evaluate import get_answers                                            # noqa: E402
from rag_research import BM25, _tok                                         # noqa: E402
import run_compare as rc                                                    # noqa: E402

# anchors measured on the real runs (102 contracts, gpt-4.1, qwen3)
ANCHORS = {
    (10, 5): {"r_at_k": 0.693, "tp": 1269, "fp": 1449, "fn": 1374, "f1": 0.4734},
    (20, 8): {"r_at_k": 0.769, "tp": 1357, "fp": 1720, "fn": 1286, "f1": 0.4744},
}
GOLD_SPANS = 2643


def project(r_at_k: float):
    """Linear extrapolation of TP/FP from R@k, anchored on the two measured runs."""
    (n1, k1), a1 = list(ANCHORS.items())[0]
    (n2, k2), a2 = list(ANCHORS.items())[1]
    dr = a2["r_at_k"] - a1["r_at_k"]
    dtp = (a2["tp"] - a1["tp"]) / dr
    dfp = (a2["fp"] - a1["fp"]) / dr
    tp = a1["tp"] + dtp * (r_at_k - a1["r_at_k"])
    fp = a1["fp"] + dfp * (r_at_k - a1["r_at_k"])
    fn = (a1["tp"] + a1["fn"]) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    cat_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]

    chunkfile = json.loads((HERE / "section_chunking_102.json").read_text(encoding="utf-8"))
    chunkmap = {c["contract_id"]: c["chunks"] for c in chunkfile["data"] if c["chunks"]}
    contracts = list(chunkmap)

    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=contracts)
    kept = set(labels)
    gt = {k: v for k, v in gt.items() if k.rsplit("__", 1)[1] in kept}

    print(f"{len(contracts)} contracts, {sum(len(v) for v in chunkmap.values())} chunks")
    bm25s = {cid: BM25([_tok(c) for c in ch]) for cid, ch in chunkmap.items()}

    rc.OLLAMA_MODEL = "qwen3-embedding:0.6b"          # cached from the real runs -> instant
    cat_emb = rc.embed_cached_ollama(cat_texts, "cats")
    sims = {}
    for cid, chunks in chunkmap.items():
        ch = rc.embed_cached_ollama(chunks, "chunks_" + rc._cache_key([cid]))
        sims[cid] = cat_emb @ ch.T if (cat_emb.size and ch.size) else \
            np.zeros((len(categories), len(chunks)), dtype="float32")

    gold = []
    for cid, chunks in chunkmap.items():
        for label in labels:
            for g in gt.get(f"{cid}__{label}", []):
                hits = [k for k, c in enumerate(chunks) if rc.chunk_contains(c, g)]
                if hits:
                    gold.append((cid, label, set(hits)))
    print(f"{len(gold)} reachable gold answers\n")

    label_idx = {l: i for i, l in enumerate(labels)}
    rows = []
    grid_n = [10, 15, 20, 30, 50]
    grid_k = [3, 5, 8, 12, 16]
    grid_rrf = [10, 60, 200]

    for rrf_k in grid_rrf:
        for n in grid_n:
            for k in grid_k:
                if k > n:
                    continue
                hit = 0
                for cid, label, gold_idx in gold:
                    ci = label_idx[label]
                    idxs = rc.retrieve_rrf(n, k, len(chunkmap[cid]), ci, sims[cid][ci],
                                           bm25s[cid], query_tokens, rrf_k)
                    if gold_idx & set(idxs):
                        hit += 1
                r_at_k = hit / len(gold)
                p, r, f1 = project(r_at_k)
                rows.append({"rrf_k": rrf_k, "shortlist_n": n, "top_k": k,
                             "r_at_k": round(r_at_k, 4),
                             "proj_precision": round(p, 4), "proj_recall": round(r, 4),
                             "proj_f1": round(f1, 4),
                             "measured": "YES" if (n, k) in ANCHORS and rrf_k == 60 else ""})
                print(f"  rrf_k={rrf_k:>3} N={n:>2} k={k:>2}  R@k={r_at_k:.4f}  "
                      f"proj F1={f1:.4f}{'   <-- MEASURED' if rows[-1]['measured'] else ''}",
                      flush=True)

    rows.sort(key=lambda x: -x["r_at_k"])
    with open(HERE / "rrf_retrieval_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 78)
    print("TOP 12 BY R@k (retrieval ceiling)")
    print("=" * 78)
    print(f"{'rrf_k':>6}{'N':>5}{'k':>4}{'R@k':>9}{'projP':>9}{'projR':>9}{'projF1':>9}")
    for r in rows[:12]:
        print(f"{r['rrf_k']:>6}{r['shortlist_n']:>5}{r['top_k']:>4}{r['r_at_k']:>9.4f}"
              f"{r['proj_precision']:>9.4f}{r['proj_recall']:>9.4f}{r['proj_f1']:>9.4f}"
              f"{'  <-- MEASURED' if r['measured'] else ''}")

    best_f1 = max(rows, key=lambda x: x["proj_f1"])
    print("\n" + "=" * 78)
    print(f"BEST PROJECTED F1: rrf_k={best_f1['rrf_k']} N={best_f1['shortlist_n']} "
          f"k={best_f1['top_k']} -> {best_f1['proj_f1']:.4f} (R@k={best_f1['r_at_k']:.4f})")
    print(f"Already measured  : n20 top8 -> 0.4744 ; n10 top5 -> 0.4734")
    print("=" * 78)
    print(f"\nWrote {HERE / 'rrf_retrieval_sweep.csv'}")


if __name__ == "__main__":
    main()
