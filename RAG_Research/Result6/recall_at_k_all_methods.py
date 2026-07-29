"""
Recall@k for EVERY method, at the exact (chunking, search, k) each one actually uses.

For each method it reports the full retrieval funnel, so you can see where its recall
goes -- not just one number:

  coverage  = gold answers reachable in that method's chunk substrate / all gold
  R@k       = of the reachable gold, fraction whose chunk is in the top-k retrieved
  reach     = coverage x R@k  = fraction of ALL gold whose chunk was retrieved
  X|R       = of the retrieved gold, fraction the LLM then actually extracted
  end2end R = the method's real recall (from its result JSON) -- sanity check that
              coverage x R@k x X|R ~= recall

Method configs are imported from rag_research.METHODS, so this can never drift from
what was run. No API calls (embeddings come from the on-disk cache).

  M3 caveat: M3 retrieves top-1 like M2, but its answer stage is a check pass that reads
  the WHOLE contract. So R@k is NOT M3's binding constraint and its funnel will not
  reconcile -- reported for completeness and flagged.

Run:
    python RAG_Research/Result6/recall_at_k_all_methods.py
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
from evaluate import get_answers, _is_match                                 # noqa: E402
from rag_research import (METHODS, embed_cached, _cache_key, BM25, _tok,    # noqa: E402
                          retrieve_idxs)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "gpt-5.4"
ORDER = ["M1_top2_cosine", "M2_top1_cosine", "M3_top1_cosine_llmcheck",
         "M4_top1_bm25", "M5_section_top2_cosine", "M6_section_hybrid_bm25_cosine",
         "M7_section_hybrid_top5",
         # Ablation: hybrid vs single-scorer retrieval at equal top-3 budget.
         "H_bm5_cos3", "M8_section_bm25_top3", "M9_section_cosine_top3"]
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


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    cat_emb, _ = embed_cached([f'{c["label"]}. {c["description"]}' for c in categories], "cats")
    qtok = [_tok(f'{c["label"]} {c["description"]}') for c in categories]
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)
    total_gold = sum(len(v) for v in gt.values())

    # build (and cache) each substrate's chunkmap + indexes + reachable-gold list
    subs = {}
    for name, path in CHUNK_FILES.items():
        cd = {c["contract_id"]: c["chunks"]
              for c in json.loads(path.read_text(encoding="utf-8"))["data"]}
        chunkmap = {cid: cd[cid] for cid in CONTRACTS}
        sims, bm = {}, {}
        for cid, chunks in chunkmap.items():
            e, _ = embed_cached(chunks, "chunks_" + _cache_key([cid]))
            sims[cid] = cat_emb @ e.T
            bm[cid] = BM25([_tok(c) for c in chunks])
        gold = []   # (cid, label, gold_text, hit_chunk_idxs)
        for cid, chunks in chunkmap.items():
            for label in labels:
                for g in gt.get(f"{cid}__{label}", []):
                    hits = [k for k, ch in enumerate(chunks) if contains(ch, g)]
                    if hits:
                        gold.append((cid, label, g, hits))
        subs[name] = {"chunkmap": chunkmap, "sims": sims, "bm": bm, "gold": gold,
                      "n_chunks": sum(len(v) for v in chunkmap.values())}

    rows = []
    for mk in ORDER:
        pred_path = RESULTS / mk / f"{MODEL}.json"
        if not pred_path.exists():
            print(f"(skipping {mk}: {pred_path} not found yet)")
            continue
        cfg = METHODS[mk]
        sub = subs["section" if cfg.get("chunking") == "section" else "markdown"]
        search, k, hn = cfg["search"], cfg["top_k"], cfg.get("hybrid_n", 10)
        chunkmap, sims, bm, gold = sub["chunkmap"], sub["sims"], sub["bm"], sub["gold"]

        preds = json.loads(pred_path.read_text(encoding="utf-8"))["by_contract"]

        retrieved = extracted = 0
        for cid, label, g, hits in gold:
            ci = labels.index(label)
            idxs = set(retrieve_idxs(search, k, len(chunkmap[cid]), ci, sims[cid], bm[cid], qtok, hn))
            if set(hits) & idxs:
                retrieved += 1
                plist = preds.get(cid, {}).get(label, {}).get("predictions", [])
                if any(_is_match(g, p, "Parties" in label) for p in plist):
                    extracted += 1

        reachable = len(gold)
        coverage = reachable / total_gold
        r_at_k = retrieved / reachable if reachable else 0.0
        reach = retrieved / total_gold
        x_given_r = extracted / retrieved if retrieved else 0.0
        end2end = json.loads(pred_path.read_text(encoding="utf-8"))["micro"]["recall"]

        rows.append({
            "method": mk, "chunking": cfg.get("chunking", "markdown"),
            "search": search, "k": k, "hybrid_n": hn if search == "hybrid" else "",
            "n_chunks": sub["n_chunks"], "coverage": round(coverage, 3),
            "R_at_k": round(r_at_k, 3), "reach_covxRk": round(reach, 3),
            "X_given_R": round(x_given_r, 3), "end2end_recall": round(end2end, 3),
            "note": "retrieval bypassed by full-contract check" if cfg["check"] else "",
        })

    # ---- print ----
    print(f"Total gold answers: {total_gold}   "
          f"(markdown chunks {subs['markdown']['n_chunks']}, section chunks {subs['section']['n_chunks']})\n")
    hdr = (f"{'method':<30}{'chunk':>9}{'search':>8}{'k':>3}{'chunks':>8}"
           f"{'cover':>7}{'R@k':>7}{'reach':>7}{'X|R':>7}{'recall':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<30}{r['chunking']:>9}{r['search']:>8}{r['k']:>3}{r['n_chunks']:>8}"
              f"{r['coverage']:>7.2f}{r['R_at_k']:>7.2f}{r['reach_covxRk']:>7.2f}"
              f"{r['X_given_R']:>7.2f}{r['end2end_recall']:>8.2f}"
              + (f"   <- {r['note']}" if r["note"] else ""))
    print("\n  cover = gold reachable in this substrate / all gold")
    print("  R@k   = of reachable gold, fraction whose chunk is in the top-k  (THE retrieval metric)")
    print("  reach = cover x R@k = fraction of ALL gold whose chunk was retrieved")
    print("  X|R   = of retrieved gold, fraction the LLM then extracted (prompt/model stage)")
    print("  recall= method's real end-to-end recall (~ cover x R@k x X|R, except M3)")

    with open(HERE / "recall_at_k_all_methods.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "recall_at_k_all_methods.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {HERE / 'recall_at_k_all_methods.csv'}")


if __name__ == "__main__":
    main()
