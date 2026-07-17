"""
Does the finer SECTION chunking improve retrieval? Compares R@k on the two chunk
substrates, so we can say whether "more chunks" actually helped the retriever --
independently of what the LLM then did with the chunks.

  markdown : test_chunking.json          (~60 chunks over the 6 contracts, docling md)
  section  : Result6/section_chunking.json (~220 chunks, 1.1/1.2 split, ~1500 chars)

R@k definition and denominator handling are identical to recall_at_k.py: a gold
answer counts if the top-k retrieved chunks contain any chunk holding it, and the
denominator excludes gold answers absent from every chunk. Because the two substrates
have different coverage, coverage is reported alongside so the comparison is honest.

No API calls: embeddings come from the on-disk cache.

Run:
    python RAG_Research/Result6/recall_at_k_section.py
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
KS = [1, 2, 3, 5, 8]
CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]
SUBSTRATES = {
    "markdown": ROOT / "test_chunking.json",
    "section":  HERE / "section_chunking.json",
}


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

    rows = []
    summary = {}
    for sub, path in SUBSTRATES.items():
        cd = {c["contract_id"]: c["chunks"]
              for c in json.loads(path.read_text(encoding="utf-8"))["data"]}
        chunkmap = {cid: cd[cid] for cid in CONTRACTS}
        sims, bm = {}, {}
        for cid, chunks in chunkmap.items():
            # Same tag the run used; embed_cached hashes the chunk TEXT into the
            # filename, so markdown vs section never collide despite the shared tag.
            e, _ = embed_cached(chunks, "chunks_" + _cache_key([cid]))
            sims[cid] = cat_emb @ e.T
            bm[cid] = BM25([_tok(c) for c in chunks])

        gold, n_total = [], 0
        for cid, chunks in chunkmap.items():
            for label in labels:
                for g in gt.get(f"{cid}__{label}", []):
                    n_total += 1
                    hits = [k for k, ch in enumerate(chunks) if contains(ch, g)]
                    if hits:
                        gold.append((cid, label, hits))
        n_chunks = sum(len(v) for v in chunkmap.values())
        cov = len(gold) / n_total

        print(f"\n=== {sub}  ({n_chunks} chunks, coverage {cov:.1%}, reachable gold={len(gold)}) ===")
        print(f"    {'k':>3}{'cosine':>9}{'bm25':>8}{'hybrid':>9}")
        rk = {}
        for k in KS:
            res = {}
            for search in ("cosine", "bm25", "hybrid"):
                ok = 0
                for cid, label, hits in gold:
                    ci = labels.index(label)
                    idxs = set(retrieve_idxs(search, k, len(chunkmap[cid]), ci,
                                             sims[cid], bm[cid], qtok, hybrid_n=10))
                    ok += bool(set(hits) & idxs)
                res[search] = ok / len(gold) if gold else 0.0
            rk[k] = res
            print(f"    {k:>3}{res['cosine']:>9.2f}{res['bm25']:>8.2f}{res['hybrid']:>9.2f}")
            rows.append({"substrate": sub, "n_chunks": n_chunks, "coverage": round(cov, 4),
                         "k": k, "cosine": round(res["cosine"], 4),
                         "bm25": round(res["bm25"], 4), "hybrid": round(res["hybrid"], 4)})
        summary[sub] = {"n_chunks": n_chunks, "coverage": round(cov, 4), "rk": rk}

    # headline deltas at the k the methods actually use
    print("\n" + "=" * 60)
    print("DID FINER CHUNKING HELP RETRIEVAL?  (R@k: section - markdown)")
    print("=" * 60)
    for k in KS:
        for s in ("cosine", "bm25", "hybrid"):
            d = summary["section"]["rk"][k][s] - summary["markdown"]["rk"][k][s]
            print(f"  k={k} {s:<7} {summary['markdown']['rk'][k][s]:.2f} -> "
                  f"{summary['section']['rk'][k][s]:.2f}  ({d:+.2f})")
        print()

    with open(HERE / "recall_at_k_section.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["substrate", "n_chunks", "coverage", "k",
                                          "cosine", "bm25", "hybrid"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {HERE / 'recall_at_k_section.csv'}")

    # ---- the honest comparison: R@k at fixed k is unfair because a section chunk is
    # ~6x smaller, so "top-2 of 220" shows the LLM far less text than "top-2 of 60".
    # Compare at equal CONTEXT BUDGET (tokens actually handed to the model) instead. ----
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s):
        return len(enc.encode(s, disallowed_special=()))

    def rebuild(path):
        cd = {c["contract_id"]: c["chunks"]
              for c in json.loads(path.read_text(encoding="utf-8"))["data"]}
        cm = {cid: cd[cid] for cid in CONTRACTS}
        sims, bm = {}, {}
        for cid, chunks in cm.items():
            e, _ = embed_cached(chunks, "chunks_" + _cache_key([cid]))
            sims[cid] = cat_emb @ e.T
            bm[cid] = BM25([_tok(c) for c in chunks])
        gold = []
        for cid, chunks in cm.items():
            for label in labels:
                for g in gt.get(f"{cid}__{label}", []):
                    h = [j for j, ch in enumerate(chunks) if contains(ch, g)]
                    if h:
                        gold.append((cid, label, h))
        return cm, sims, bm, gold

    def avg_tokens(cm, sims, bm, search, k):
        tt = []
        for cid, chunks in cm.items():
            for ci in range(len(labels)):
                idx = retrieve_idxs(search, k, len(chunks), ci, sims[cid], bm[cid], qtok, 10)
                tt.append(sum(ntok(chunks[i]) for i in idx))
        return sum(tt) / len(tt)

    def rk(cm, sims, bm, gold, search, k):
        ok = sum(bool(set(h) & set(retrieve_idxs(search, k, len(cm[cid]), labels.index(lab),
                                                  sims[cid], bm[cid], qtok, 10)))
                 for cid, lab, h in gold)
        return ok / len(gold)

    mdc = rebuild(SUBSTRATES["markdown"])
    secc = rebuild(SUBSTRATES["section"])
    print("\n" + "=" * 64)
    print("EQUAL CONTEXT BUDGET (tokens the LLM actually sees, not k)")
    print("=" * 64)
    print(f"  markdown cosine top-2 : {avg_tokens(*mdc[:3], 'cosine', 2):>6.0f} tok  "
          f"R@2={rk(*mdc, 'cosine', 2):.2f}   (this is M1, F1 0.426)")
    print(f"  section  cosine top-2 : {avg_tokens(*secc[:3], 'cosine', 2):>6.0f} tok  "
          f"R@2={rk(*secc, 'cosine', 2):.2f}   (this is M5, F1 0.356)")
    print(f"  section  hybrid top-3 : {avg_tokens(*secc[:3], 'hybrid', 3):>6.0f} tok  "
          f"R@3={rk(*secc, 'hybrid', 3):.2f}   (this is M6, F1 0.395)")
    print("  -> section methods were STARVED: same k, ~4-6x less text than markdown.")
    print("  matching markdown's token budget needs a higher k on section chunks:")
    for k in (5, 6, 7, 8):
        print(f"    section cosine top-{k}: {avg_tokens(*secc[:3], 'cosine', k):>6.0f} tok  "
              f"R@{k}={rk(*secc, 'cosine', k):.2f}")


if __name__ == "__main__":
    main()
