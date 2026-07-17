"""
For each CUAD category, which method wins?

Scans every method in Result6/results and, for each category that has ground truth,
computes that method's per-category F1 (from tp/fp/fn) and Jaccard (from the stored
jaccard_per_category). Reports the best method by F1 (primary) and by Jaccard, plus
the full per-method F1 vector so ties are visible.

Tie-break for the F1 winner: higher F1, then higher Jaccard, then fewer false positives.

Writes best_per_category.json (+ a flat best_per_category.csv). Fed into the Excel by
make_excel.py.

Run:
    python RAG_Research/Result6/best_per_category.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from OpenAITest import load_categories, CATEGORY_CSV   # noqa: E402
from evaluate import get_answers                        # noqa: E402

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


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0)


def main():
    categories = sorted(l.title() for l in load_categories(CATEGORY_CSV))
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)
    gold_answers = {c: sum(len(gt.get(f"{cid}__{c}", [])) for cid in CONTRACTS) for c in categories}

    methods = {}
    for p in sorted(RESULTS.glob(f"*/{MODEL}.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        methods[d["method"]] = {
            "counts": d["by_category_counts"],
            "jac": d["metrics"].get("jaccard_per_category", {}) or {},
        }

    out = {}
    for cat in categories:
        if gold_answers[cat] == 0:
            continue   # no gold -> F1 undefined, skip
        f1_by, jac_by = {}, {}
        for mk, m in methods.items():
            c = m["counts"].get(cat, {"tp": 0, "fp": 0, "fn": 0})
            f1_by[mk] = round(prf(c["tp"], c["fp"], c["fn"]), 4)
            if cat in m["jac"]:
                jac_by[mk] = round(m["jac"][cat], 4)
        # F1 winner: F1, then Jaccard, then fewer FP
        def f1_key(mk):
            c = methods[mk]["counts"].get(cat, {"tp": 0, "fp": 0, "fn": 0})
            return (f1_by[mk], jac_by.get(mk, 0.0), -c["fp"])
        best_f1_m = max(f1_by, key=f1_key)
        best_jac_m = max(jac_by, key=jac_by.get) if jac_by else ""
        out[cat] = {
            "gold_answers": gold_answers[cat],
            "best_f1_method": best_f1_m, "best_f1": f1_by[best_f1_m],
            "best_jaccard_method": best_jac_m,
            "best_jaccard": jac_by.get(best_jac_m, 0.0) if best_jac_m else 0.0,
            "f1_by_method": f1_by, "jaccard_by_method": jac_by,
        }

    (HERE / "best_per_category.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(HERE / "best_per_category.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "gold_answers", "best_F1_method", "best_F1",
                    "best_Jaccard_method", "best_Jaccard"])
        for cat, r in out.items():
            w.writerow([cat, r["gold_answers"], r["best_f1_method"], r["best_f1"],
                        r["best_jaccard_method"], r["best_jaccard"]])

    # how often does each method win?
    from collections import Counter
    wins = Counter(r["best_f1_method"] for r in out.values())
    print(f"Best-F1 method per category over {len(out)} categories with gold:\n")
    print(f"  {'category':<34}{'best method (F1)':<26}{'F1':>6}{'best (Jac)':>16}{'Jac':>7}")
    for cat, r in out.items():
        print(f"  {cat:<34}{r['best_f1_method']:<26}{r['best_f1']:>6.2f}"
              f"{r['best_jaccard_method']:>16}{r['best_jaccard']:>7.2f}")
    print(f"\n  win counts (times a method had the best F1):")
    for m, n in wins.most_common():
        print(f"    {m:<30}{n}")
    print(f"\nWrote {HERE / 'best_per_category.json'} and best_per_category.csv")


if __name__ == "__main__":
    main()
