"""
Re-score this project's predictions under the ContractEval paper's protocol, so
our RAG numbers can be compared to their Table 3 like-for-like.

Why this file exists
--------------------
Our harness and ContractEval's measure DIFFERENT things, so our native F1 and
their F1 are not comparable as printed:

  | | this project (evaluate.py / breakdown) | ContractEval (arXiv 2508.03080) |
  |-|---------------------------------------|---------------------------------|
  | unit counted   | one per GOLD SPAN            | one per (contract, question) PAIR |
  | match rule     | token-Jaccard >= 0.5         | prediction FULLY COVERS the gold span |
  | empty label    | can only produce FP          | scored as TN when the model abstains |
  | abstention     | none (pipeline always answers)| model told to say "no related clause" |
  | context        | top-5 retrieved chunks       | the ENTIRE contract |

ContractEval definitions (their Section 3.4), reproduced exactly:
  TP  label non-empty AND the model's prediction fully covers the labeled span
  TN  label empty     AND the model correctly predicts "no related clause"
  FP  label empty     BUT the model predicts a non-empty clause
  FN  label non-empty BUT the model outputs "no related clause" OR fails to
      fully cover the label span
  F1 = 2PR/(P+R),  F2 = 5PR/(4P+R)
  Jaccard = mean token-set Jaccard on POSITIVE cases only
  False "no related clause" rate = share of cases where the model returned
      nothing although a non-empty gold clause exists

Our pipeline has no literal "no related clause" string -- returning an empty span
list is the same signal, and is treated as such here.

Two readings of "fully covers the labeled span" are reported, because CUAD
questions can have several gold spans:
  lenient  TP if ANY gold span is fully covered by some prediction  (primary)
  strict   TP only if EVERY gold span is fully covered

Run:
    python CompareContractEval/contracteval_score.py \
        --result CompareContractEval/results/qwen3_rrf_n10_top5__all102/gpt-4.1.json
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ContractEval Table 3, proprietary models (F1, F2, Jaccard, false-rate)
PAPER = {
    "GPT 4.1":                (0.641, 0.672, 0.472, 0.071),
    "GPT 4.1 mini":           (0.644, 0.678, 0.435, 0.072),
    "Claude Sonnet 4":        (0.523, 0.578, 0.458, 0.025),
    "Gemini 2.5 Pro Preview": (0.497, 0.604, 0.506, 0.011),
}


def pr_from_f1f2(f1: float, f2: float) -> tuple[float, float]:
    """The paper prints only F1 and F2. Two equations, two unknowns:
        F1 = 2PR/(P+R)        F2 = 5PR/(4P+R)
    Substituting PR = F1(P+R)/2 into the F2 identity gives R = mP with
    m = (4*F2 - 2.5*F1) / (2.5*F1 - F2), hence P = F1(1+m)/(2m).
    Verified to reproduce the published F1/F2 to 4 decimals for all four
    proprietary models."""
    m = (4 * f2 - 2.5 * f1) / (2.5 * f1 - f2)
    p = f1 * (1 + m) / (2 * m)
    return p, m * p


def norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def toks(s: str) -> set:
    return set(norm(s).split())


def covers(preds: list[str], gold: str) -> bool:
    """ContractEval's 'the prediction fully covers the labeled span': the gold
    text appears in full inside one of the predicted spans."""
    g = norm(gold)
    if not g:
        return False
    return any(g in norm(p) for p in preds)


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return p, r, f1, f2


def score(by_contract: dict, mode: str = "lenient") -> dict:
    tp = tn = fp = fn = 0
    n_pos = n_neg = 0
    jacs = []
    false_nrc = 0                 # gold non-empty but model returned nothing
    covered_but_not_returned = 0  # gold non-empty, model answered, but no full cover

    for cid, cats in by_contract.items():
        for cat, e in cats.items():
            gold = [g for g in (e.get("ground_truth") or []) if norm(g)]
            preds = [p for p in (e.get("predictions") or []) if norm(p)]

            if gold:
                n_pos += 1
                if mode == "strict":
                    ok = all(covers(preds, g) for g in gold)
                else:
                    ok = any(covers(preds, g) for g in gold)
                if ok:
                    tp += 1
                else:
                    fn += 1
                    if not preds:
                        false_nrc += 1
                    else:
                        covered_but_not_returned += 1
                # Jaccard on positive cases: token sets of all predicted vs all gold
                A, B = set(), set()
                for p in preds:
                    A |= toks(p)
                for g in gold:
                    B |= toks(g)
                jacs.append(len(A & B) / len(A | B) if (A | B) else 0.0)
            else:
                n_neg += 1
                if preds:
                    fp += 1
                else:
                    tn += 1

    p, r, f1, f2 = prf(tp, fp, fn)
    total = tp + tn + fp + fn
    return {
        "mode": mode,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pairs": total, "n_positive": n_pos, "n_negative": n_neg,
        "positive_share": round(n_pos / total, 4) if total else 0.0,
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(f1, 4), "f2": round(f2, 4),
        "jaccard_positive": round(sum(jacs) / len(jacs), 4) if jacs else 0.0,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "false_no_related_clause_rate_all": round(false_nrc / total, 4) if total else 0.0,
        "false_no_related_clause_rate_positive": round(false_nrc / n_pos, 4) if n_pos else 0.0,
        "n_false_no_related_clause": false_nrc,
        "n_answered_but_not_covering": covered_but_not_returned,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="Path to the run's result JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = json.loads(Path(args.result).read_text(encoding="utf-8"))
    by_contract = d["by_contract"]

    lenient = score(by_contract, "lenient")
    strict = score(by_contract, "strict")

    out = {
        "source_result": str(args.result),
        "model": d.get("model"), "arm": d.get("arm"),
        "n_contracts": d.get("n_contracts"),
        "native_metrics": {
            "note": "this project's own scorer: per-GOLD-SPAN counting, token-Jaccard >= 0.5",
            **d["micro"],
            "aupr": round(d["metrics"]["aupr"], 4),
            "jaccard_similarity": round(d["metrics"].get("jaccard_similarity", 0.0), 4),
            "r_at_k": d.get("r_at_k"), "coverage": d.get("coverage"),
        },
        "contracteval_protocol": {
            "note": "ContractEval Sec 3.4: per-(contract,question) counting, "
                    "prediction must FULLY COVER the gold span, empty label + "
                    "abstention = TN",
            "lenient_any_gold_covered": lenient,
            "strict_all_golds_covered": strict,
        },
        "paper_reference_table3": PAPER,
        "cost_usd": d.get("cost_usd"),
        "tokens": d.get("tokens"),
    }
    dest = Path(args.out) if args.out else HERE / "contracteval_protocol_scores.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    nm = out["native_metrics"]
    print("=" * 96)
    print("A) THIS PROJECT'S NATIVE SCORER (per gold span, Jaccard >= 0.5) -- not comparable to the paper")
    print("=" * 96)
    print(f"  TP={nm['tp']}  FP={nm['fp']}  FN={nm['fn']}   "
          f"P={nm['precision']:.4f}  R={nm['recall']:.4f}  F1={nm['f1']:.4f}  F2={nm['f2']:.4f}")
    print(f"  AUPR={nm['aupr']:.4f}  R@k={nm['r_at_k']}  coverage={nm['coverage']}")

    print()
    print("=" * 96)
    print("B) RE-SCORED UNDER ContractEval's PROTOCOL -- directly comparable to their Table 3")
    print("=" * 96)
    hdr = (f"{'reading':<10}{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}"
           f"{'F1':>8}{'F2':>8}{'Jac':>8}{'FalseRate':>11}")
    print(hdr); print("-" * 96)
    for s in (lenient, strict):
        print(f"{s['mode']:<10}{s['tp']:>6}{s['tn']:>6}{s['fp']:>6}{s['fn']:>6}"
              f"{s['precision']:>8.4f}{s['recall']:>8.4f}{s['f1']:>8.4f}{s['f2']:>8.4f}"
              f"{s['jaccard_positive']:>8.4f}{s['false_no_related_clause_rate_all']:>11.4f}")
    print("-" * 96)
    print(f"  pairs={lenient['n_pairs']}  positive={lenient['n_positive']} "
          f"({lenient['positive_share']:.1%})  negative={lenient['n_negative']}")
    print(f"  (paper reports 4,128 data points, 30% positive / 70% negative)")

    print()
    print("=" * 96)
    print("C) HEAD-TO-HEAD vs ContractEval Table 3")
    print("=" * 96)
    print(f"{'system':<40}{'P*':>8}{'R*':>8}{'F1':>8}{'F2':>8}{'Jaccard':>9}{'FalseRate':>11}")
    print("-" * 96)
    derived = {}
    for name, (f1, f2, jac, fr) in PAPER.items():
        dp, dr = pr_from_f1f2(f1, f2)
        derived[name] = {"precision_derived": round(dp, 4), "recall_derived": round(dr, 4),
                         "f1": f1, "f2": f2, "jaccard": jac, "false_rate": fr}
        print(f"{'  ' + name + ' (full doc, paper)':<40}{dp:>8.3f}{dr:>8.3f}"
              f"{f1:>8.3f}{f2:>8.3f}{jac:>9.3f}{fr:>11.3f}")
    print("-" * 96)
    mdl = out["model"]
    rcfg = d.get("retrieval", {})
    desc = f"RRF n{rcfg.get('shortlist_n','?')} top-{rcfg.get('top_k','?')}"
    for s in (lenient, strict):
        label = f"  OURS {mdl} {desc} ({s['mode']})"
        print(f"{label:<40}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['f2']:>8.3f}"
              f"{s['jaccard_positive']:>9.3f}{s['false_no_related_clause_rate_all']:>11.3f}")
    print("=" * 96)
    print("  P*/R* for the paper rows are DERIVED from its published F1/F2 (see pr_from_f1f2);")
    print("  the paper itself prints only F1, F2, Jaccard and false rate.")
    out["paper_reference_table3_derived"] = derived

    base = PAPER["GPT 4.1"]
    print(f"\nvs paper GPT 4.1 (same model, same dataset, full-document, their scorer):")
    for s in (lenient, strict):
        d_f1, d_f2, d_j = (s["f1"] - base[0], s["f2"] - base[1],
                           s["jaccard_positive"] - base[2])
        verdict = "BETTER" if d_f1 > 0 else "WORSE"
        print(f"  {s['mode']:<8} F1 {s['f1']:.3f} vs {base[0]:.3f} = {d_f1:+.3f} ({verdict})   "
              f"F2 {d_f2:+.3f}   Jaccard {d_j:+.3f}")

    out["headline"] = {
        "paper_gpt41_full_document": {"f1": base[0], "f2": base[1], "jaccard": base[2],
                                      "false_rate": base[3], "approx_cost_usd": 50},
        "ours_gpt41_rag_lenient": {"f1": lenient["f1"], "f2": lenient["f2"],
                                   "jaccard": lenient["jaccard_positive"],
                                   "false_rate": lenient["false_no_related_clause_rate_all"],
                                   "cost_usd": d.get("cost_usd")},
        "ours_gpt41_rag_strict": {"f1": strict["f1"], "f2": strict["f2"],
                                  "jaccard": strict["jaccard_positive"],
                                  "false_rate": strict["false_no_related_clause_rate_all"],
                                  "cost_usd": d.get("cost_usd")},
        "delta_f1_lenient": round(lenient["f1"] - base[0], 4),
        "delta_f1_strict": round(strict["f1"] - base[0], 4),
    }
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
