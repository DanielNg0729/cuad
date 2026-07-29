"""
Summarize the all-41 prompt ablation across models.

Reads results/experiment_all41/<model>.json (written by experiment_prompt.py) and
produces:
  1. an aggregate TP/FP/FN/precision/recall/F1 table per (model, arm),
  2. a per-category view: F1 per arm, averaged across the available models, to
     show which categories are consistently weak and where ArmB helps vs hurts,
  3. total new API cost.

Run:
    python analyze_experiment.py
"""

import json
from pathlib import Path

MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.5"]
ARMS = ["Arm0_baseline_batched", "ArmA_baseline_grouped", "ArmB_example_grouped"]
SHORT = {"Arm0_baseline_batched": "Arm0", "ArmA_baseline_grouped": "ArmA",
         "ArmB_example_grouped": "ArmB"}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    docs = {}
    for m in MODELS:
        path = Path(f"results/experiment_all41/{m}.json")
        if path.exists():
            docs[m] = json.loads(path.read_text(encoding="utf-8"))
    if not docs:
        print("No experiment JSONs found yet."); return

    # 1. Aggregate table
    print("=" * 74)
    print("AGGREGATE (all 41 categories, 3 contracts)")
    print("=" * 74)
    print(f"{'model':<14}{'arm':<6}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}{'cost':>9}")
    cost_total = 0.0
    for m in MODELS:
        if m not in docs:
            continue
        for arm in ARMS:
            t = docs[m]["arms"][arm]["totals"]
            p, r, f = prf(t["tp"], t["fp"], t["fn"])
            c = docs[m]["arms"][arm]["cost_usd"]
            cost_total += c
            print(f"{m:<14}{SHORT[arm]:<6}{t['tp']:>5}{t['fp']:>5}{t['fn']:>5}"
                  f"{p:>8.2f}{r:>8.2f}{f:>8.2f}{('$%.3f'%c):>9}")
        print()

    # 2. Per-category F1, averaged across models that exist, per arm
    cats = sorted(next(iter(docs.values()))["arms"][ARMS[0]]["by_category_counts"])
    rows = []
    for cat in cats:
        # skip categories with no ground truth anywhere
        has_gt = any(
            docs[m]["arms"][arm]["by_category_counts"].get(cat, {}).get("tp", 0)
            + docs[m]["arms"][arm]["by_category_counts"].get(cat, {}).get("fn", 0) > 0
            for m in docs for arm in ARMS)
        if not has_gt:
            continue
        f_by_arm = {}
        for arm in ARMS:
            fs = []
            for m in docs:
                c = docs[m]["arms"][arm]["by_category_counts"].get(cat, {"tp": 0, "fp": 0, "fn": 0})
                fs.append(prf(c["tp"], c["fp"], c["fn"])[2])
            f_by_arm[arm] = sum(fs) / len(fs)
        rows.append((f_by_arm[ARMS[0]], cat, f_by_arm))

    print("=" * 74)
    print("PER-CATEGORY F1 (avg across models), sorted worst Arm0 first")
    print(f"  delta = ArmB - Arm0  (>0 prompt helps, <0 prompt hurts)")
    print("=" * 74)
    print(f"{'category':<34}{'Arm0':>7}{'ArmA':>7}{'ArmB':>7}{'B-0':>8}")
    for _, cat, fb in sorted(rows):
        d = fb[ARMS[2]] - fb[ARMS[0]]
        mark = "  <<" if d <= -0.10 else ("  ++" if d >= 0.10 else "")
        print(f"{cat:<34}{fb[ARMS[0]]:>7.2f}{fb[ARMS[1]]:>7.2f}{fb[ARMS[2]]:>7.2f}{d:>+8.2f}{mark}")

    # 3. consistently-weak: low F1 in ALL arms
    print("\n" + "=" * 74)
    print("CONSISTENTLY WEAK (avg F1 < 0.34 in every arm) -- prompt-independent")
    print("=" * 74)
    for _, cat, fb in sorted(rows):
        if all(fb[a] < 0.34 for a in ARMS):
            print(f"  {cat}")

    print(f"\nTotal new API cost across models: ${cost_total:.4f}")


if __name__ == "__main__":
    main()
