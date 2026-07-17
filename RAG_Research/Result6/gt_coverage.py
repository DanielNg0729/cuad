"""
Ground-truth category coverage across the 6 Result6 contracts.

Answers "how many categories actually have an answer in the ground truth?" -- which
is the ceiling any extractor can hit, and the denominator that makes a per-category
F1 meaningful (a category with 0 gold answers can only ever produce false positives).

Writes Result6/ground_truth_category_coverage.csv with one row per category:
    Category, ContractsWithAnswer (0-6), TotalGoldAnswers, then one column per
    contract holding that contract's gold-answer count.

Run:
    python RAG_Research/Result6/gt_coverage.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from OpenAITest import load_categories, CATEGORY_CSV   # noqa: E402
from evaluate import get_answers                        # noqa: E402

OUT = Path(__file__).resolve().parent
CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]
# short, spreadsheet-friendly column names
SHORT = ["BIOPURE", "Bizzingo", "AIRSPAN", "Ambassador", "ColumbiaLabs", "DrivenDeliveries"]


def main():
    import os
    os.chdir(ROOT)

    categories = [l.title() for l in load_categories(CATEGORY_CSV)]
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)

    rows = []
    for cat in sorted(categories):
        per_contract = []
        for cid in CONTRACTS:
            per_contract.append(len(gt.get(f"{cid}__{cat}", [])))
        rows.append({
            "Category": cat,
            "ContractsWithAnswer": sum(1 for n in per_contract if n > 0),
            "TotalGoldAnswers": sum(per_contract),
            **{SHORT[i]: per_contract[i] for i in range(len(CONTRACTS))},
        })

    # sort: the most-covered categories first, so the empty ones sink to the bottom
    rows.sort(key=lambda r: (-r["ContractsWithAnswer"], -r["TotalGoldAnswers"], r["Category"]))

    path = OUT / "ground_truth_category_coverage.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["Category", "ContractsWithAnswer",
                                          "TotalGoldAnswers"] + SHORT)
        w.writeheader()
        w.writerows(rows)

    n_any = sum(1 for r in rows if r["ContractsWithAnswer"] > 0)
    n_all6 = sum(1 for r in rows if r["ContractsWithAnswer"] == 6)
    n_none = sum(1 for r in rows if r["ContractsWithAnswer"] == 0)
    total_answers = sum(r["TotalGoldAnswers"] for r in rows)

    print(f"Contracts            : {len(CONTRACTS)}")
    print(f"Categories total     : {len(categories)}")
    print(f"  with >=1 gold answer in >=1 contract : {n_any}")
    print(f"  with a gold answer in ALL 6 contracts: {n_all6}")
    print(f"  with NO gold answer anywhere         : {n_none}  <- can only produce false positives")
    print(f"Total gold answers   : {total_answers}")
    print(f"\nWrote {path}")

    print(f"\n{'category':<36}{'#contracts':>11}{'#answers':>10}")
    for r in rows:
        print(f"{r['Category']:<36}{r['ContractsWithAnswer']:>11}{r['TotalGoldAnswers']:>10}")


if __name__ == "__main__":
    main()
