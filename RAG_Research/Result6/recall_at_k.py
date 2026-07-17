"""
Retrieval recall@k -- the numbers behind the R@k table in ANALYSIS.md.

                # gold answers whose containing chunk is in the top-k retrieved
    R@k  =  ------------------------------------------------------------------
                # gold answers that live in SOME chunk   ("reachable")

Gold answers that appear in NO chunk are excluded from the denominator: those are a
docling/chunking loss (Stage 1 coverage), and the retriever should not be blamed for
them. Retrieval itself is done by calling the PRODUCTION retrieve_idxs() from
rag_research.py, so this measures what the real run did, not a re-implementation.

Two containment definitions are reported, because that choice is load-bearing:
  lenient : normalised substring, else >=90% of the gold's tokens present in the chunk
  strict  : normalised verbatim substring only

And two populations, because a short gold span (party names, dates) can occur in many
chunks, which lets ANY retriever get lucky and flatters the lexical retriever:
  all        : every gold answer located in >=1 chunk
  unambiguous: gold answers located in EXACTLY 1 chunk -- a real retrieval test

No API calls: chunk/category embeddings come from the on-disk cache.

Run:
    python RAG_Research/Result6/recall_at_k.py
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

from OpenAITest import load_categories, CATEGORY_CSV                       # noqa: E402
from evaluate import get_answers                                            # noqa: E402
from rag_research import embed_cached, _cache_key, BM25, _tok, retrieve_idxs  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = [1, 2, 3, 4, 5, 8]
CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]


def _norm(s: str) -> str:
    """Collapse every whitespace run (incl. docling's non-breaking spaces) + lowercase."""
    return " ".join((s or "").split()).lower()


def contains_strict(chunk: str, gold: str) -> bool:
    ng = _norm(gold)
    return bool(ng) and ng in _norm(chunk)


def contains_lenient(chunk: str, gold: str) -> bool:
    if contains_strict(chunk, gold):
        return True
    g = set(_norm(gold).split())
    if not g:
        return False
    return len(g & set(_norm(chunk).split())) / len(g) >= 0.9


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]

    chunkdata = {c["contract_id"]: c["chunks"]
                 for c in json.loads((ROOT / "test_chunking.json").read_text(encoding="utf-8"))["data"]}
    chunkmap = {cid: chunkdata[cid] for cid in CONTRACTS}
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)

    # retrieval indexes -- identical to the ones the run used (embeddings from cache)
    cat_emb, _ = embed_cached([f'{c["label"]}. {c["description"]}' for c in categories], "cats")
    qtok = [_tok(f'{c["label"]} {c["description"]}') for c in categories]
    sims, bm25s = {}, {}
    for cid, chunks in chunkmap.items():
        ch_emb, _ = embed_cached(chunks, "chunks_" + _cache_key([cid]))
        sims[cid] = cat_emb @ ch_emb.T
        bm25s[cid] = BM25([_tok(c) for c in chunks])

    rows = []
    for defn, contains in (("lenient", contains_lenient), ("strict", contains_strict)):
        # locate every gold answer: which chunks hold it?
        gold = []          # (cid, label, hits)
        n_total = 0
        for cid, chunks in chunkmap.items():
            for label in labels:
                for g in gt.get(f"{cid}__{label}", []):
                    n_total += 1
                    hits = [k for k, ch in enumerate(chunks) if contains(ch, g)]
                    if hits:
                        gold.append((cid, label, hits))
        unamb = [g for g in gold if len(g[2]) == 1]

        print(f"\n=== containment = {defn} ===")
        print(f"  gold answers                : {n_total}")
        print(f"  located in >=1 chunk        : {len(gold)} ({len(gold)/n_total:.1%})  <- R@k denominator")
        print(f"  located in EXACTLY 1 chunk  : {len(unamb)}  <- unambiguous subset")

        for pop_name, pop in (("all", gold), ("unambiguous", unamb)):
            print(f"\n  population = {pop_name}  (n={len(pop)})")
            print(f"    {'k':>3}{'cosine':>9}{'bm25':>8}")
            for k in KS:
                res = {}
                for search in ("cosine", "bm25"):
                    ok = 0
                    for cid, label, hits in pop:
                        ci = labels.index(label)
                        idxs = set(retrieve_idxs(search, k, len(chunkmap[cid]), ci,
                                                 sims[cid], bm25s[cid], qtok))
                        ok += bool(set(hits) & idxs)
                    res[search] = ok / len(pop) if pop else 0.0
                print(f"    {k:>3}{res['cosine']:>9.2f}{res['bm25']:>8.2f}")
                rows.append({"containment": defn, "population": pop_name, "n": len(pop),
                             "k": k, "cosine_R_at_k": round(res["cosine"], 4),
                             "bm25_R_at_k": round(res["bm25"], 4)})

    with open(HERE / "recall_at_k.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["containment", "population", "n", "k",
                                          "cosine_R_at_k", "bm25_R_at_k"])
        w.writeheader()
        w.writerows(rows)
    (HERE / "recall_at_k.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {HERE / 'recall_at_k.csv'} and recall_at_k.json")
    print("\nNote: R@k measures ONLY whether the right chunk was retrieved. It does not")
    print("say the LLM will extract from it (see X|R in diagnose.py), and it says nothing")
    print("about false positives on categories with no gold -- which is why BM25 wins R@k")
    print("but still loses on end-to-end F1.")


if __name__ == "__main__":
    main()
