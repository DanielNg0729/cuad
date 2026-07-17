"""
What KIND of errors is each method making? The headline F1 hides the answer.

Two lenses, both reproducible from the result JSONs (no API calls):

  1. IOU-threshold sweep. The scorer calls a prediction correct only at token-Jaccard
     >= 0.5 vs the gold. Re-score at 0.5 -> 0.1. If F1 rockets as the bar drops, most
     "misses" are the model finding the RIGHT clause with the WRONG span extent -- a
     scoring/boundary problem, not a retrieval or reasoning problem.

  2. FP / FN composition. Bucket every false positive and false negative:
       FP: zero-gold category (no abstention) | near-miss of a real gold (boundary) | genuine
       FN: a prediction overlaps it (boundary) | nothing near it (genuine miss)

Writes error_breakdown.csv + prints a table for all methods.

Run:
    python RAG_Research/Result6/error_breakdown.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import get_jaccard, _is_match, _norm_ws   # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "gpt-5.4"
ORDER = ["M1_top2_cosine", "M2_top1_cosine", "M3_top1_cosine_llmcheck",
         "M4_top1_bm25", "M5_section_top2_cosine", "M6_section_hybrid_bm25_cosine",
         "M7_section_hybrid_top5"]


def load(m):
    return json.loads((RESULTS / m / f"{MODEL}.json").read_text(encoding="utf-8"))["by_contract"]


def counts_at_iou(bc, iou):
    tp = fp = fn = 0
    for cats in bc.values():
        for cat, e in cats.items():
            ans, preds, sub = e["ground_truth"], list(dict.fromkeys(e["predictions"])), "Parties" in cat
            if not ans:
                fp += len(preds); continue
            def hit(a, p): return (sub and _norm_ws(a) in _norm_ws(p)) or get_jaccard(a, p) >= iou
            for a in ans:
                tp += 1 if any(hit(a, p) for p in preds) else 0
                fn += 0 if any(hit(a, p) for p in preds) else 1
            for p in preds:
                fp += 0 if any(hit(a, p) for a in ans) else 1
    return tp, fp, fn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r


def composition(bc):
    fp_zero = fp_near = fp_gen = fn_bound = fn_miss = 0
    gold_n, pred_n = [], []
    for cats in bc.values():
        for cat, e in cats.items():
            ans, preds, sub = e["ground_truth"], list(dict.fromkeys(e["predictions"])), "Parties" in cat
            if ans:
                gold_n.append(len(ans)); pred_n.append(len(preds))
            for p in preds:
                if not ans:
                    fp_zero += 1
                elif any(_is_match(a, p, sub) for a in ans):
                    pass
                elif any(get_jaccard(a, p) > 0 or (sub and _norm_ws(a) in _norm_ws(p)) for a in ans):
                    fp_near += 1
                else:
                    fp_gen += 1
            for a in ans:
                if any(_is_match(a, p, sub) for p in preds):
                    continue
                fn_bound += 1 if any(get_jaccard(a, p) > 0 for p in preds) else 0
                fn_miss += 0 if any(get_jaccard(a, p) > 0 for p in preds) else 1
    return dict(fp_zero=fp_zero, fp_near=fp_near, fp_gen=fp_gen, fn_bound=fn_bound,
                fn_miss=fn_miss, gold_per_cat=float(np.mean(gold_n)),
                pred_per_cat=float(np.mean(pred_n)))


def main():
    rows = []
    print("IOU sweep (F1) -- if F1 climbs as the match bar drops, it's a BOUNDARY problem:\n")
    print(f"{'method':<30}{'F1@.5':>7}{'F1@.3':>7}{'F1@.1':>7}{'lift':>7}")
    for m in ORDER:
        bc = load(m)
        f5 = f1(*counts_at_iou(bc, 0.5))[0]
        f3 = f1(*counts_at_iou(bc, 0.3))[0]
        f1_ = f1(*counts_at_iou(bc, 0.1))[0]
        print(f"{m:<30}{f5:>7.3f}{f3:>7.3f}{f1_:>7.3f}{(f1_ - f5):>+7.3f}")
        c = composition(bc)
        rows.append({"method": m, "f1_iou50": round(f5, 3), "f1_iou30": round(f3, 3),
                     "f1_iou10": round(f1_, 3), **{k: round(v, 3) if isinstance(v, float) else v
                                                    for k, v in c.items()}})

    print("\nFalse positives -- where the noise comes from:")
    print(f"{'method':<30}{'FP':>5}{'zero-gt':>9}{'boundary':>10}{'genuine':>9}")
    for r in rows:
        fp = r["fp_zero"] + r["fp_near"] + r["fp_gen"]
        print(f"{r['method']:<30}{fp:>5}{r['fp_zero']:>9}{r['fp_near']:>10}{r['fp_gen']:>9}")

    print("\nFalse negatives -- misses that are really boundary vs really gone:")
    print(f"{'method':<30}{'FN':>5}{'boundary':>10}{'genuine':>9}{'pred/cat':>10}{'gold/cat':>10}")
    for r in rows:
        fn = r["fn_bound"] + r["fn_miss"]
        print(f"{r['method']:<30}{fn:>5}{r['fn_bound']:>10}{r['fn_miss']:>9}"
              f"{r['pred_per_cat']:>10.2f}{r['gold_per_cat']:>10.2f}")

    with open(HERE / "error_breakdown.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {HERE / 'error_breakdown.csv'}")


if __name__ == "__main__":
    main()
