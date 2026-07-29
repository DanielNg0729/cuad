"""
Root-cause decomposition: WHERE does the RAG pipeline actually lose points?

A missed gold answer can die at exactly one of three stages. This script attributes
every gold answer to the stage that killed it, which is the only honest way to say
whether "RAG sucks", the embeddings are bad, the retriever is bad, or the prompt is bad.

  Stage 1  COVERAGE   -- is the gold answer even present in the chunked markdown?
                         If docling dropped/garbled it, no retriever or prompt can
                         ever recover it. This is the hard ceiling.
  Stage 2  RETRIEVAL  -- given the answer IS in some chunk, did top-k retrieval pick
                         that chunk? If not, that is the embedding/BM25's fault.
  Stage 3  EXTRACTION -- given the right chunk WAS retrieved, did the LLM actually
                         return the span? If not, that is the prompt/model's fault.

  recall = coverage x retrieval x extraction

Also attributes FALSE POSITIVES, separating the structural problem: top-k retrieval
always returns k chunks even for a category that does not occur in the contract at
all, so the model is handed text and asked to find something that isn't there.

Run (after rag_research.py has written Result6/results):
    python RAG_Research/Result6/diagnose.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_Research"))
os.chdir(ROOT)

from OpenAITest import load_categories, CATEGORY_CSV      # noqa: E402
from evaluate import get_answers, _is_match, get_jaccard  # noqa: E402
from rag_research import (                                # noqa: E402
    embed_cached, _cache_key, BM25, _tok, retrieve_idxs,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "gpt-5.4"

CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]

# retrieval configs to diagnose, mapped to the method whose predictions they produced.
# (search, top_k, hybrid_n, chunking) -- chunking picks which chunk file/substrate
# (and therefore which coverage/gold_chunkidx) this method's retrieval is checked against.
CONFIGS = {
    "M1_top2_cosine": ("cosine", 2, None, "markdown"),
    "M2_top1_cosine": ("cosine", 1, None, "markdown"),
    "M4_top1_bm25":   ("bm25", 1, None, "markdown"),
    # Ablation: hybrid vs single-scorer retrieval at equal top-3 budget, section chunking.
    "H_bm5_cos3":            ("hybrid", 3, 5, "section"),
    "M8_section_bm25_top3":  ("bm25", 3, None, "section"),
    "M9_section_cosine_top3": ("cosine", 3, None, "section"),
}
CHUNK_FILES = {"markdown": "test_chunking.json", "section": "section_chunking.json"}


def _norm(s: str) -> str:
    """Collapse every whitespace run (incl. the non-breaking spaces docling emits)
    and lowercase, so 'in\xa0the' and 'in the' compare equal."""
    return " ".join((s or "").split()).lower()


def chunk_contains(chunk: str, gold: str) -> bool:
    """Is the gold answer present in this chunk? Exact (normalised) containment first;
    fall back to token-recall >= 0.9 to survive small PDF-extraction differences."""
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


def build_substrate(chunking: str, categories, labels, gt):
    """Stage-1 coverage + retrieval indexes for one chunk substrate (markdown or
    section), reused by every CONFIGS entry that shares that chunking."""
    chunk_path = HERE / CHUNK_FILES[chunking] if chunking == "section" else ROOT / CHUNK_FILES[chunking]
    chunkdata = {c["contract_id"]: c["chunks"]
                 for c in json.loads(chunk_path.read_text(encoding="utf-8"))["data"]}
    chunkmap = {cid: chunkdata[cid] for cid in CONTRACTS}

    cat_texts = [f'{c["label"]}. {c["description"]}' for c in categories]
    cat_emb, _ = embed_cached(cat_texts, "cats")
    query_tokens = [_tok(f'{c["label"]} {c["description"]}') for c in categories]
    sims, bm25s = {}, {}
    for cid, chunks in chunkmap.items():
        ch_emb, _ = embed_cached(chunks, "chunks_" + _cache_key([cid]))
        sims[cid] = cat_emb @ ch_emb.T
        bm25s[cid] = BM25([_tok(c) for c in chunks])

    gold_total = 0
    gold_in_some_chunk = 0
    gold_chunkidx: dict[tuple[str, str, int], list[int]] = {}
    uncovered = []
    for cid, chunks in chunkmap.items():
        for label in labels:
            answers = gt.get(f"{cid}__{label}", [])
            for ai, gold in enumerate(answers):
                gold_total += 1
                hits = [k for k, ch in enumerate(chunks) if chunk_contains(ch, gold)]
                gold_chunkidx[(cid, label, ai)] = hits
                if hits:
                    gold_in_some_chunk += 1
                else:
                    uncovered.append((cid, label, gold[:70]))

    return {"chunking": chunking, "chunkmap": chunkmap, "sims": sims, "bm25s": bm25s,
           "query_tokens": query_tokens, "gold_chunkidx": gold_chunkidx,
           "gold_total": gold_total, "gold_in_some_chunk": gold_in_some_chunk,
           "uncovered": uncovered}


def main():
    categories = [{"label": l.title(), "description": d}
                  for l, d in load_categories(CATEGORY_CSV).items()]
    labels = [c["label"] for c in categories]
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)

    # Only build the substrate(s) actually needed by CONFIGS (markdown and/or section).
    needed = {cfg[3] for cfg in CONFIGS.values()}
    subs = {name: build_substrate(name, categories, labels, gt) for name in needed}

    for name, sub in subs.items():
        print("=" * 78)
        print(f"STAGE 1 -- COVERAGE ({name} chunking): can the answer be found at all?")
        print("=" * 78)
        gt_total, in_chunk, unc = sub["gold_total"], sub["gold_in_some_chunk"], sub["uncovered"]
        print(f"  gold answers total                    : {gt_total}")
        print(f"  present in at least one chunk         : {in_chunk} ({in_chunk / gt_total:.1%})")
        print(f"  NOT present in any chunk (unreachable): {len(unc)} ({len(unc) / gt_total:.1%})"
              f"  <- docling/chunking loss, hard ceiling")
        if unc:
            print("\n  examples of unreachable gold answers:")
            for cid, label, g in unc[:8]:
                print(f"    [{label}] {cid[:28]:<28} {g!r}")
        print()

    # --- Stages 2 & 3, per retrieval config ---
    rows = []
    for method, (search, top_k, hybrid_n, chunking) in CONFIGS.items():
        pred_path = RESULTS / method / f"{MODEL}.json"
        if not pred_path.exists():
            print(f"(missing {pred_path}, skipping {method})")
            continue
        doc = json.loads(pred_path.read_text(encoding="utf-8"))
        by_contract = doc["by_contract"]
        sub = subs[chunking]
        chunkmap, sims, bm25s = sub["chunkmap"], sub["sims"], sub["bm25s"]
        query_tokens, gold_chunkidx = sub["query_tokens"], sub["gold_chunkidx"]

        retrieved_ok = 0        # gold was in a chunk AND retrieval picked that chunk
        retrieval_miss = 0      # gold was in a chunk but retrieval picked other chunks
        extracted_ok = 0        # retrieval hit AND the model returned a matching span
        extraction_miss = 0     # retrieval hit but the model failed to return it
        by_cat: dict[str, dict] = {}   # per-category tallies -- THIS answers "which category?"
        instances = []                 # per-gold-answer record -- THIS answers "show me the example"

        def cat_row(label):
            return by_cat.setdefault(label, {
                "category": label, "n_gold": 0,
                "retr_hit": 0, "retr_miss": 0, "extr_ok": 0, "extr_miss": 0,
                "fp_zero_gt": 0, "fp_with_gt": 0,
            })

        for cid, chunks in chunkmap.items():
            for ci, label in enumerate(labels):
                answers = gt.get(f"{cid}__{label}", [])
                if not answers:
                    continue
                idxs = set(retrieve_idxs(search, top_k, len(chunks), ci,
                                         sims[cid], bm25s[cid], query_tokens, hybrid_n or 10))
                preds = by_contract.get(cid, {}).get(label, {}).get("predictions", [])
                substr_ok = "Parties" in label
                for ai, gold in enumerate(answers):
                    hits = gold_chunkidx[(cid, label, ai)]
                    if not hits:
                        continue                      # already counted as uncovered
                    row = cat_row(label)
                    row["n_gold"] += 1
                    if not (set(hits) & idxs):
                        retrieval_miss += 1
                        row["retr_miss"] += 1
                        instances.append({"contract": cid, "category": label, "gold": gold,
                                          "retrieved": False, "extracted": None})
                        continue
                    retrieved_ok += 1
                    row["retr_hit"] += 1
                    hit_ok = any(_is_match(gold, p, substr_ok) for p in preds)
                    if hit_ok:
                        extracted_ok += 1
                        row["extr_ok"] += 1
                    else:
                        extraction_miss += 1
                        row["extr_miss"] += 1
                    instances.append({"contract": cid, "category": label, "gold": gold,
                                      "retrieved": True, "extracted": hit_ok})

        reachable = retrieved_ok + retrieval_miss
        r_recall = retrieved_ok / reachable if reachable else 0.0
        x_recall = extracted_ok / retrieved_ok if retrieved_ok else 0.0

        # false-positive attribution (also per-category)
        fp_zero_gt = fp_with_gt = 0
        for cid in chunkmap:
            for label in labels:
                e = by_contract.get(cid, {}).get(label)
                if not e:
                    continue
                n_fp = len(e.get("fp", []))
                if not n_fp:
                    continue
                row = cat_row(label)
                if not e.get("ground_truth"):
                    fp_zero_gt += n_fp        # category doesn't exist in this contract at all
                    row["fp_zero_gt"] += n_fp
                else:
                    fp_with_gt += n_fp        # category exists, but the model grabbed the wrong span
                    row["fp_with_gt"] += n_fp
        total_fp = fp_zero_gt + fp_with_gt

        rows.append({
            "method": method, "chunking": chunking, "reachable": reachable,
            "retrieved_ok": retrieved_ok, "retrieval_miss": retrieval_miss,
            "r_recall": r_recall,
            "extracted_ok": extracted_ok, "extraction_miss": extraction_miss,
            "x_recall": x_recall,
            "fp_zero_gt": fp_zero_gt, "fp_with_gt": fp_with_gt, "total_fp": total_fp,
            "by_category": by_cat, "instances": instances,
        })

        # per-category CSV: exactly "which category, and was it retrieval's or the LLM's fault"
        cat_rows = sorted(by_cat.values(),
                          key=lambda r: (r["retr_miss"] + r["extr_miss"] + r["fp_with_gt"]),
                          reverse=True)
        import csv as _csv
        with open(HERE / f"diagnose_by_category_{method}.csv", "w", newline="",
                 encoding="utf-8-sig") as f:
            w = _csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()) if cat_rows else
                                ["category", "n_gold", "retr_hit", "retr_miss",
                                 "extr_ok", "extr_miss", "fp_zero_gt", "fp_with_gt"])
            w.writeheader()
            w.writerows(cat_rows)
        (HERE / f"diagnose_instances_{method}.json").write_text(
            json.dumps(instances, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("STAGE 2 -- RETRIEVAL  (did top-k pick the chunk that holds the answer?)")
    print("STAGE 3 -- EXTRACTION (given the right chunk, did the LLM return the span?)")
    print("=" * 78)
    print(f"{'method':<26}{'reach':>6}{'retr-hit':>9}{'retr-miss':>10}{'R@k':>7}"
          f"{'extr-ok':>9}{'extr-miss':>10}{'X|R':>7}")
    for r in rows:
        print(f"{r['method']:<26}{r['reachable']:>6}{r['retrieved_ok']:>9}{r['retrieval_miss']:>10}"
              f"{r['r_recall']:>7.2f}{r['extracted_ok']:>9}{r['extraction_miss']:>10}{r['x_recall']:>7.2f}")

    print("\n  Reading it: R@k is the retriever's fault. X|R is the prompt/model's fault.")
    print("  end-to-end recall = coverage x R@k x X|R")
    for r in rows:
        cov = subs[r["chunking"]]["gold_in_some_chunk"] / subs[r["chunking"]]["gold_total"]
        print(f"    {r['method']:<26} {cov:.2f} x {r['r_recall']:.2f} x {r['x_recall']:.2f} "
              f"= {cov * r['r_recall'] * r['x_recall']:.2f}")

    print("\n" + "=" * 78)
    print("FALSE POSITIVES -- where does the noise come from?")
    print("=" * 78)
    print(f"{'method':<26}{'FP total':>10}{'FP on 0-GT cats':>17}{'FP on real cats':>17}")
    for r in rows:
        print(f"{r['method']:<26}{r['total_fp']:>10}{r['fp_zero_gt']:>17}{r['fp_with_gt']:>17}")
    print("\n  FP on 0-GT cats = the category does not appear in that contract at all, but")
    print("  top-k retrieval still handed the model a chunk and asked it to find one.")

    print("\n" + "=" * 78)
    print("WORST CATEGORIES per method (top 8 by retr-miss + extr-miss + wrong-span FP)")
    print("=" * 78)
    for r in rows:
        print(f"\n  {r['method']}  ({r['chunking']} chunking)")
        cat_rows = sorted(r["by_category"].values(),
                          key=lambda c: (c["retr_miss"] + c["extr_miss"] + c["fp_with_gt"]),
                          reverse=True)[:8]
        print(f"    {'category':<32}{'n_gold':>7}{'retr-miss':>10}{'extr-miss':>10}{'fp(real)':>9}")
        for c in cat_rows:
            print(f"    {c['category'][:32]:<32}{c['n_gold']:>7}{c['retr_miss']:>10}"
                  f"{c['extr_miss']:>10}{c['fp_with_gt']:>9}")
    print(f"\n  Full per-category CSV: {HERE}/diagnose_by_category_<method>.csv")
    print(f"  Full per-instance JSON (exact gold text + retrieved?/extracted? per answer): "
          f"{HERE}/diagnose_instances_<method>.json")

    # Keep the summary JSON lean: per-category/per-instance detail already lives in the
    # dedicated diagnose_by_category_*.csv / diagnose_instances_*.json files per method.
    lean_rows = [{k: v for k, v in r.items() if k not in ("by_category", "instances")}
                for r in rows]
    (HERE / "diagnosis.json").write_text(json.dumps({
        "model": MODEL, "contracts": CONTRACTS,
        "coverage": {name: {"gold_total": sub["gold_total"],
                            "in_some_chunk": sub["gold_in_some_chunk"],
                            "unreachable": len(sub["uncovered"]),
                            "coverage_rate": round(sub["gold_in_some_chunk"] / sub["gold_total"], 4),
                            "unreachable_examples": [
                                {"contract": c, "category": l, "gold": g}
                                for c, l, g in sub["uncovered"]]}
                    for name, sub in subs.items()},
        "stages": lean_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {HERE / 'diagnosis.json'}")


if __name__ == "__main__":
    main()
