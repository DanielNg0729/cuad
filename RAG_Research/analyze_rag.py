"""
Summarize the RAG-method comparison written by rag_research.py.

Reads RAG_Research/results/<method>/<model>.json and produces:
  1. a per-(method, model) table: TP/FP/FN, micro P/R/F1/F2, AUPR, best-F1,
     Jaccard(License Grant) and macro-Jaccard, cost;
  2. a per-category F1 view: for one model, F1 per method, so you can see which
     categories each retrieval strategy wins/loses on;
  3. writes RAG_Research/SUMMARY.md with the same tables (+ the full-scan baseline
     from results/*__3contracts.json for context).

Run:
    python RAG_Research/analyze_rag.py
    python RAG_Research/analyze_rag.py --per-category-model gpt-5.5
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "RAG_Research" / "results"
METHOD_ORDER = ["M1_top2_cosine", "M2_top1_cosine", "M3_top1_cosine_llmcheck", "M4_top1_bm25"]
MODELS = ["gpt-5.4", "gpt-5.5"]

# Existing full-scan baselines (evaluate.py on OpenAITest nbest, same 3 contracts).
BASELINES = {
    "gpt-5.4": ROOT / "results" / "gpt-5.4__openai__all__3contracts.json",
    "gpt-5.5": ROOT / "results" / "gpt-5.5__openai__all__3contracts.json",
}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return p, r, f1, f2


def macro_jaccard(metrics: dict) -> float:
    per = metrics.get("jaccard_per_category") or {}
    return sum(per.values()) / len(per) if per else 0.0


def load(method, model):
    p = RES / method / f"{model}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def baseline_row(model):
    p = BASELINES.get(model)
    if not p or not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    tp, fp, fn = m["true_positives"], m["false_positives"], m["false_negatives"]
    pr, rc, f1, f2 = prf(tp, fp, fn)
    return {"method": "baseline_fullscan", "model": model, "tp": tp, "fp": fp, "fn": fn,
            "precision": pr, "recall": rc, "f1": f1, "f2": f2,
            "aupr": m["aupr"], "best_f1": m["best_f1"],
            "jac_lg": m["jaccard_similarity"], "jac_macro": macro_jaccard(m),
            "cost_usd": None}


def rows():
    out = []
    for model in MODELS:
        b = baseline_row(model)
        if b:
            out.append(b)
        for method in METHOD_ORDER:
            d = load(method, model)
            if not d:
                continue
            mi, me = d["micro"], d["metrics"]
            out.append({"method": method, "model": model,
                        "tp": mi["tp"], "fp": mi["fp"], "fn": mi["fn"],
                        "precision": mi["precision"], "recall": mi["recall"],
                        "f1": mi["f1"], "f2": mi["f2"],
                        "aupr": me["aupr"], "best_f1": me["best_f1"],
                        "jac_lg": me["jaccard_similarity"], "jac_macro": macro_jaccard(me),
                        "cost_usd": d["cost_usd"]})
    return out


def fmt_table(rs) -> list[str]:
    head = (f"| {'method':<24} | {'model':<8} | TP | FP | FN | Prec | Rec |  F1  |  F2  "
            f"| AUPR | bestF1 | Jac-LG | Jac-macro |  cost  |")
    sep = "|" + "|".join(["-" * w for w in (26, 10, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 11, 8)]) + "|"
    lines = [head, sep]
    for r in rs:
        cost = " n/a " if r["cost_usd"] is None else f"${r['cost_usd']:.3f}"
        lines.append(
            f"| {r['method']:<24} | {r['model']:<8} | {r['tp']:>2} | {r['fp']:>2} | {r['fn']:>2} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['f2']:.3f} "
            f"| {r['aupr']:.3f} | {r['best_f1']:.3f} | {r['jac_lg']:.3f} | {r['jac_macro']:.3f}   "
            f"| {cost:>6} |")
    return lines


def per_category_f1(model):
    """Per category: micro F1 per method (using by_category_counts), for one model."""
    data = {m: load(m, model) for m in METHOD_ORDER}
    data = {m: d for m, d in data.items() if d}
    if not data:
        return []
    cats = sorted({c for d in data.values() for c in d["by_category_counts"]})
    lines = [f"| {'category':<34} | " + " | ".join(f"{m.split('_',1)[1][:12]:>12}" for m in data) + " |"]
    lines.append("|" + "-" * 36 + "|" + "|".join(["-" * 14] * len(data)) + "|")
    for cat in cats:
        # only show categories that have ground truth somewhere
        has_gt = any(d["by_category_counts"].get(cat, {}).get("tp", 0)
                     + d["by_category_counts"].get(cat, {}).get("fn", 0) > 0 for d in data.values())
        if not has_gt:
            continue
        cells = []
        for m, d in data.items():
            c = d["by_category_counts"].get(cat, {"tp": 0, "fp": 0, "fn": 0})
            cells.append(f"{prf(c['tp'], c['fp'], c['fn'])[2]:>12.2f}")
        lines.append(f"| {cat:<34} | " + " | ".join(cells) + " |")
    return lines


def findings(rs) -> list[str]:
    """Data-derived bullet points comparing methods per model."""
    by = {}
    for r in rs:
        by.setdefault(r["model"], {})[r["method"]] = r
    out = []
    for model, mm in by.items():
        base = mm.get("baseline_fullscan")
        ranked = sorted((r for k, r in mm.items() if k != "baseline_fullscan"),
                        key=lambda r: -r["f1"])
        if not ranked:
            continue
        best = ranked[0]
        vs_base = ""
        if base:
            delta = best["f1"] - base["f1"]
            vs_base = (f" — {'beats' if delta > 0 else 'below'} full-scan baseline "
                       f"(F1 {base['f1']:.3f}) by {delta:+.3f}")
        out.append(f"- **{model}**: best RAG method is **{best['method']}** "
                   f"(F1 {best['f1']:.3f}, P {best['precision']:.3f}, R {best['recall']:.3f}){vs_base}.")
        m1, m2 = mm.get("M1_top2_cosine"), mm.get("M2_top1_cosine")
        m3, m4 = mm.get("M3_top1_cosine_llmcheck"), mm.get("M4_top1_bm25")
        if m1 and m2:
            out.append(f"    - top-2 vs top-1 cosine: F1 {m1['f1']:.3f} vs {m2['f1']:.3f} "
                       f"(more chunks → recall {m1['recall']:.3f} vs {m2['recall']:.3f}).")
        if m2 and m4:
            out.append(f"    - cosine vs BM25 (both top-1): F1 {m2['f1']:.3f} vs {m4['f1']:.3f}.")
        if m2 and m3:
            effects = []
            if m3["fp"] < m2["fp"]:
                effects.append("prunes false positives")
            if m3["tp"] > m2["tp"]:
                effects.append("recovers misses from the full contract")
            effect = "; ".join(effects) or "little net change"
            out.append(f"    - the LLM check pass (M3 vs M2): F1 {m2['f1']:.3f} → {m3['f1']:.3f}, "
                       f"FP {m2['fp']} → {m3['fp']}, TP {m2['tp']} → {m3['tp']} ({effect}).")
    return out


def rebuild_summary_json():
    """Rewrite results/summary.json from the per-method result JSONs on disk (the
    source of truth), so it stays complete even after a partial re-run."""
    rows, descs = [], {}
    for model in MODELS:
        for method in METHOD_ORDER:
            d = load(method, model)
            if not d:
                continue
            descs[method] = d.get("method_description", "")
            mi, me = d["micro"], d["metrics"]
            rows.append({"method": method, "model": model,
                         "tp": mi["tp"], "fp": mi["fp"], "fn": mi["fn"],
                         "precision": mi["precision"], "recall": mi["recall"],
                         "f1": mi["f1"], "f2": mi["f2"],
                         "aupr": round(me["aupr"], 4), "best_f1": round(me["best_f1"], 4),
                         "best_f2": round(me["best_f2"], 4),
                         "jaccard_license_grant": round(me["jaccard_similarity"], 4),
                         "cost_usd": d["cost_usd"]})
    summary = {"models": MODELS, "methods": descs,
               "grand_cost_usd": round(sum(r["cost_usd"] for r in rows), 6), "rows": rows}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category-model", default="gpt-5.5")
    ap.add_argument("--no-md", action="store_true", help="Print only; don't write SUMMARY.md")
    args = ap.parse_args()

    rs = rows()
    if not rs:
        print("No result JSONs found under RAG_Research/results/. Run rag_research.py first.")
        return

    table = fmt_table(rs)
    print("\n".join(table))
    pc = per_category_f1(args.per_category_model)

    if not args.no_md:
        md = []
        md.append("# RAG pipeline research — 4 retrieval methods on 3 CUAD contracts\n")
        md.append("Same 3 contracts, chunks, span-validation and scorer as every other experiment; "
                  "only the retrieval strategy (and one optional verification pass) changes. "
                  "Metrics via the project's `evaluate.evaluate`, so numbers are directly comparable "
                  "to the full-scan `results/*__3contracts.json` baselines.\n")
        md.append("## Methods\n")
        for m in METHOD_ORDER:
            d = next((load(m, mo) for mo in MODELS if load(m, mo)), None)
            if d:
                md.append(f"- **{m}** — {d['method_description']}")
        md.append("\n## Results (micro over all 41 categories × 3 contracts)\n")
        md.append("`Prec/Rec/F1/F2` are micro over tp/fp/fn. `Jac-LG` = Jaccard on License Grant "
                  "(the reference complex category); `Jac-macro` = mean Jaccard across categories "
                  "with ground truth. `baseline_fullscan` = existing all-41-per-chunk run.\n")
        md.extend(table)
        md.append("\n## Findings\n")
        md.extend(findings(rs))
        md.append(f"\n## Per-category F1 by method — {args.per_category_model}\n")
        md.extend(pc)
        (ROOT / "RAG_Research" / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        rebuild_summary_json()
        print(f"\nWrote {ROOT / 'RAG_Research' / 'SUMMARY.md'} and results/summary.json")


if __name__ == "__main__":
    main()
