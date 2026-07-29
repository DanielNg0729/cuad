"""
Per-category breakdown of the 102-contract run, under BOTH scoring protocols,
plus the retrieval-vs-extraction attribution for every false negative.

Writes:
  per_category.csv          one row per CUAD category: native TP/FP/FN + F1/Jaccard,
                            ContractEval-protocol TP/TN/FP/FN + F1, and R@k
  failure_attribution.csv   per category: how many FNs are retrieval's fault
                            (gold text never appeared in the retrieved chunks) vs
                            extraction's fault (gold text was there, model missed it)

Run:
    python TestRAGFinal/per_category.py
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = HERE / "results" / "qwen3_rrf_n10_top5__all102" / "gpt-4.1.json"


def norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def toks(s: str) -> set:
    return set(norm(s).split())


def covers(preds, gold) -> bool:
    g = norm(gold)
    return bool(g) and any(g in norm(p) for p in preds)


def in_context(ctx: str, gold: str) -> bool:
    """Fuzzy containment, same rule as run_compare.chunk_contains."""
    nc, ng = norm(ctx), norm(gold)
    if not ng:
        return False
    if ng in nc:
        return True
    gt = set(ng.split())
    return bool(gt) and len(gt & set(nc.split())) / len(gt) >= 0.9


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(DEFAULT_RESULT))
    ap.add_argument("--suffix", default="", help="Suffix for the output CSV filenames.")
    args = ap.parse_args()
    sfx = args.suffix
    d = json.loads(Path(args.result).read_text(encoding="utf-8"))
    by_contract = d["by_contract"]
    native_counts = d["by_category_counts"]
    jpc = d["metrics"].get("jaccard_per_category", {}) or {}

    ce = {}          # category -> ContractEval-protocol counts
    attrib = {}      # category -> retrieval/extraction FN attribution
    for cid, cats in by_contract.items():
        for cat, e in cats.items():
            gold = [g for g in (e.get("ground_truth") or []) if norm(g)]
            preds = [p for p in (e.get("predictions") or []) if norm(p)]
            ctx = e.get("context", "")
            c = ce.setdefault(cat, {"tp": 0, "tn": 0, "fp": 0, "fn": 0,
                                    "n_pos": 0, "n_neg": 0, "jac": []})
            a = attrib.setdefault(cat, {"retrieval_miss": 0, "extraction_miss": 0,
                                        "abstained": 0})
            if gold:
                c["n_pos"] += 1
                if any(covers(preds, g) for g in gold):
                    c["tp"] += 1
                else:
                    c["fn"] += 1
                    # attribute: was the gold text even in the retrieved context?
                    reachable = any(in_context(ctx, g) for g in gold)
                    if not reachable:
                        a["retrieval_miss"] += 1
                    else:
                        a["extraction_miss"] += 1
                    if not preds:
                        a["abstained"] += 1
                A, B = set(), set()
                for p in preds:
                    A |= toks(p)
                for g in gold:
                    B |= toks(g)
                c["jac"].append(len(A & B) / len(A | B) if (A | B) else 0.0)
            else:
                c["n_neg"] += 1
                if preds:
                    c["fp"] += 1
                else:
                    c["tn"] += 1

    rows = []
    for cat in sorted(ce):
        n = native_counts.get(cat, {"tp": 0, "fp": 0, "fn": 0})
        np_, nr, nf1 = prf(n["tp"], n["fp"], n["fn"])
        c = ce[cat]
        cp, cr, cf1 = prf(c["tp"], c["fp"], c["fn"])
        rows.append({
            "category": cat,
            "n_positive_pairs": c["n_pos"], "n_negative_pairs": c["n_neg"],
            "native_tp": n["tp"], "native_fp": n["fp"], "native_fn": n["fn"],
            "native_precision": round(np_, 4), "native_recall": round(nr, 4),
            "native_f1": round(nf1, 4),
            "native_jaccard": round(jpc.get(cat, 0.0), 4),
            "ce_tp": c["tp"], "ce_tn": c["tn"], "ce_fp": c["fp"], "ce_fn": c["fn"],
            "ce_precision": round(cp, 4), "ce_recall": round(cr, 4), "ce_f1": round(cf1, 4),
            "ce_jaccard_positive": round(sum(c["jac"]) / len(c["jac"]), 4) if c["jac"] else 0.0,
        })

    rows.sort(key=lambda r: -r["ce_f1"])
    with open(HERE / f"per_category{sfx}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    arows = []
    for cat in sorted(attrib):
        a = attrib[cat]
        tot = a["retrieval_miss"] + a["extraction_miss"]
        arows.append({
            "category": cat, "total_fn": tot,
            "retrieval_miss": a["retrieval_miss"], "extraction_miss": a["extraction_miss"],
            "retrieval_share": round(a["retrieval_miss"] / tot, 4) if tot else 0.0,
            "abstained_returned_nothing": a["abstained"],
            "dominant": ("retrieval" if a["retrieval_miss"] > a["extraction_miss"]
                         else "extraction" if a["extraction_miss"] > a["retrieval_miss"]
                         else "tied"),
        })
    arows.sort(key=lambda r: -r["total_fn"])
    with open(HERE / f"failure_attribution{sfx}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0].keys()))
        w.writeheader(); w.writerows(arows)

    tr = sum(r["retrieval_miss"] for r in arows)
    te = sum(r["extraction_miss"] for r in arows)
    print(f"Wrote per_category{sfx}.csv ({len(rows)} categories) and failure_attribution{sfx}.csv")
    print(f"\nFalse negatives (ContractEval protocol, lenient): {tr + te}")
    print(f"  retrieval  (gold never in retrieved chunks): {tr:>5}  ({tr/(tr+te):.1%})")
    print(f"  extraction (gold present, model missed it) : {te:>5}  ({te/(tr+te):.1%})")

    print(f"\nTop 10 categories by ContractEval F1:")
    for r in rows[:10]:
        print(f"  {r['category']:<36}{r['ce_f1']:>7.3f}  (n_pos={r['n_positive_pairs']})")
    print(f"\nBottom 10 categories by ContractEval F1:")
    for r in rows[-10:]:
        print(f"  {r['category']:<36}{r['ce_f1']:>7.3f}  (n_pos={r['n_positive_pairs']})")


if __name__ == "__main__":
    main()
