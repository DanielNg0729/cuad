"""
Consolidate every arm in results/ into one comparison table, under BOTH scoring
protocols, and place them against the ContractEval paper's published numbers.

Reads   results/<arm>/<model>.json
Writes  all_arms_summary.csv   one row per arm: native metrics + ContractEval-protocol
                               metrics + cost + failure attribution
        all_arms_summary.md    the same table, formatted for the report

Run:
    python TestRAGFinal/summarize_arms.py
"""

import csv
import json
from pathlib import Path

import contracteval_score as ce  # same folder

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def attribute_fns(by_contract):
    """Split ContractEval-protocol false negatives into retrieval vs extraction."""
    retr = extr = 0
    for cats in by_contract.values():
        for e in cats.values():
            gold = [g for g in (e.get("ground_truth") or []) if ce.norm(g)]
            preds = [p for p in (e.get("predictions") or []) if ce.norm(p)]
            if not gold or any(ce.covers(preds, g) for g in gold):
                continue
            ctx = ce.norm(e.get("context", ""))
            reachable = False
            for g in gold:
                ng = ce.norm(g)
                if ng and (ng in ctx or
                           len(set(ng.split()) & set(ctx.split())) / len(set(ng.split())) >= 0.9):
                    reachable = True
                    break
            if reachable:
                extr += 1
            else:
                retr += 1
    return retr, extr


def main():
    rows = []
    for arm_dir in sorted(RESULTS.iterdir()):
        if not arm_dir.is_dir() or "smoke" in arm_dir.name:
            continue
        for f in sorted(arm_dir.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            bc = d["by_contract"]
            len_ = ce.score(bc, "lenient")
            str_ = ce.score(bc, "strict")
            r = d.get("retrieval", {})
            m = d["micro"]
            retr, extr = attribute_fns(bc)
            tot_fn = retr + extr
            rows.append({
                "arm": d.get("arm"), "model": d.get("model"),
                "shortlist_n": r.get("shortlist_n"), "top_k": r.get("top_k"),
                "n_contracts": d.get("n_contracts"),
                # native scorer
                "nat_tp": m["tp"], "nat_fp": m["fp"], "nat_fn": m["fn"],
                "nat_precision": m["precision"], "nat_recall": m["recall"],
                "nat_f1": m["f1"], "nat_f2": m["f2"],
                "nat_aupr": round(d["metrics"]["aupr"], 4),
                "nat_jaccard": round(d["metrics"].get("jaccard_similarity", 0.0), 4),
                "r_at_k": d.get("r_at_k"), "coverage": d.get("coverage"),
                # ContractEval protocol
                "ce_tp": len_["tp"], "ce_tn": len_["tn"], "ce_fp": len_["fp"], "ce_fn": len_["fn"],
                "ce_precision": len_["precision"], "ce_recall": len_["recall"],
                "ce_f1": len_["f1"], "ce_f2": len_["f2"],
                "ce_jaccard": len_["jaccard_positive"],
                "ce_false_rate": len_["false_no_related_clause_rate_all"],
                "ce_f1_strict": str_["f1"], "ce_f2_strict": str_["f2"],
                # failure attribution
                "fn_retrieval": retr, "fn_extraction": extr,
                "fn_retrieval_share": round(retr / tot_fn, 4) if tot_fn else 0.0,
                # economics
                "input_tokens": d["tokens"]["input"], "output_tokens": d["tokens"]["output"],
                "cost_usd": d.get("cost_usd"), "wall_minutes": round(d.get("wall_seconds", 0) / 60, 2),
            })

    rows.sort(key=lambda x: -x["ce_f1"])
    with open(HERE / "all_arms_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    paper = ce.PAPER["GPT 4.1"]
    lines = []
    lines.append("# All arms — 102 CUAD contracts, gpt-4.1\n")
    lines.append("## Under ContractEval's protocol (comparable to the paper)\n")
    lines.append("| System | P | R | F1 | F2 | Jaccard | False rate | Cost |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    pp, pr = ce.pr_from_f1f2(paper[0], paper[1])
    lines.append(f"| **ContractEval GPT-4.1 (full document, published)** | {pp:.3f}* | {pr:.3f}* | "
                 f"**{paper[0]:.3f}** | {paper[1]:.3f} | {paper[2]:.3f} | {paper[3]:.3f} | ~$50 |")
    for r in rows:
        lines.append(f"| RAG RRF n{r['shortlist_n']} top-{r['top_k']} | {r['ce_precision']:.3f} | "
                     f"{r['ce_recall']:.3f} | {r['ce_f1']:.3f} | {r['ce_f2']:.3f} | "
                     f"{r['ce_jaccard']:.3f} | {r['ce_false_rate']:.3f} | ${r['cost_usd']:.2f} |")
    lines.append("\n\\* derived from the paper's published F1/F2.\n")

    lines.append("## Under this project's native scorer (per gold span, Jaccard >= 0.5)\n")
    lines.append("| Arm | TP | FP | FN | P | R | F1 | F2 | AUPR | Jaccard | R@k | Cost |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| RRF n{r['shortlist_n']} top-{r['top_k']} | {r['nat_tp']} | {r['nat_fp']} | "
                     f"{r['nat_fn']} | {r['nat_precision']:.3f} | {r['nat_recall']:.3f} | "
                     f"{r['nat_f1']:.3f} | {r['nat_f2']:.3f} | {r['nat_aupr']:.3f} | "
                     f"{r['nat_jaccard']:.3f} | {r['r_at_k']:.3f} | ${r['cost_usd']:.2f} |")

    lines.append("\n## Where the false negatives come from (ContractEval protocol)\n")
    lines.append("| Arm | Total FN | Retrieval | Extraction | Retrieval share |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| RRF n{r['shortlist_n']} top-{r['top_k']} | "
                     f"{r['fn_retrieval'] + r['fn_extraction']} | {r['fn_retrieval']} | "
                     f"{r['fn_extraction']} | {r['fn_retrieval_share']:.1%} |")
    (HERE / "all_arms_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote all_arms_summary.csv and all_arms_summary.md")


if __name__ == "__main__":
    main()
