"""
Bad-case analysis: for the single best method in each of TestAblation and TestRerank,
list EVERY category where a missed gold answer (false negative) was missed because
the retriever handed the model the WRONG chunks -- not because the LLM failed to
extract it from the right chunk.

No API calls, no recomputation of retrieval -- every result JSON already saved the
retrieved chunk text (context / retrieved_chunks) alongside ground_truth/fn, so this
is a pure post-hoc text-containment check:

    for every (contract, category, missed gold answer):
        if gold text NOT found in that category's retrieved `context`:
            -> RETRIEVAL MISS (wrong chunks retrieved; the LLM never had a chance)
        else:
            -> EXTRACTION MISS (right chunks were there; the LLM just didn't pull it out)

Writes BadCaseAnalysisForRetrival.md into EACH target's own folder (TestAblation/ and
TestRerank/), not a single combined file, so each stays self-contained.

Run:
    python RAG_Research/Result6/bad_case_retrieval_analysis.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent   # .../Result6
MODEL = "gpt-5.4"

TARGETS = [
    {
        "folder": HERE / "TestAblation",
        "arm": "bge3_cosine_top5",
        "label": "Best TestAblation method (bge-m3, cosine-only, top-5) -- F1=0.449, the "
                 "single best arm across all 14 TestAblation arms.",
    },
    {
        "folder": HERE / "TestRerank",
        "arm": "qwen3_rrf_n10_top5",
        "label": "Best TestRerank method (qwen3-embedding:0.6b, RRF fusion, BM25/cosine "
                 "shortlist N=10, top-5) -- F1=0.457, the best arm across TestAblation + "
                 "TestRerank combined.",
    },
]


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def context_contains(context: str, gold: str) -> bool:
    """Same fuzzy containment check used throughout Result6 (../diagnose.py etc.):
    exact normalized substring, falling back to >=90% token-overlap so minor
    PDF-extraction noise doesn't cause a false 'retrieval miss'."""
    nc, ng = _norm(context), _norm(gold)
    if not ng:
        return False
    if ng in nc:
        return True
    gt_toks = set(ng.split())
    if not gt_toks:
        return False
    ctx_toks = set(nc.split())
    return len(gt_toks & ctx_toks) / len(gt_toks) >= 0.9


def analyze(arm_json: dict) -> dict:
    """Returns {category: {"retrieval_miss": [...], "extraction_miss": [...]}} plus
    overall totals. Each miss entry is (contract, gold_text)."""
    by_cat: dict[str, dict] = {}
    total_fn = total_retrieval_miss = total_extraction_miss = 0

    for cid, cats in arm_json["by_contract"].items():
        for cat, entry in cats.items():
            fn_list = entry.get("fn", [])
            if not fn_list:
                continue
            context = entry.get("context", "")
            row = by_cat.setdefault(cat, {"retrieval_miss": [], "extraction_miss": []})
            for gold in fn_list:
                total_fn += 1
                if context_contains(context, gold):
                    row["extraction_miss"].append((cid, gold))
                    total_extraction_miss += 1
                else:
                    row["retrieval_miss"].append((cid, gold))
                    total_retrieval_miss += 1

    return {"by_cat": by_cat, "total_fn": total_fn,
            "total_retrieval_miss": total_retrieval_miss,
            "total_extraction_miss": total_extraction_miss}


def render_md(target: dict, arm_json: dict, analysis: dict) -> str:
    by_cat = analysis["by_cat"]
    # only categories with at least one retrieval miss -- the user's actual ask
    retrieval_cats = {cat: row for cat, row in by_cat.items() if row["retrieval_miss"]}
    retrieval_cats_sorted = sorted(retrieval_cats.items(),
                                   key=lambda kv: -len(kv[1]["retrieval_miss"]))

    lines = []
    lines.append(f"# Bad-Case Analysis: Retrieval Failures -- `{target['arm']}`\n")
    lines.append(f"{target['label']}\n")
    lines.append(
        "Every false negative (missed gold answer) is attributed to exactly one of two "
        "stages, using the retrieved chunk text already saved in "
        f"`results/{target['arm']}/{MODEL}.json` (`by_contract.<contract>.<category>.context`) "
        "-- no re-running of retrieval or the LLM:\n"
    )
    lines.append(
        "- **Retrieval miss**: the gold answer text does not appear anywhere in the chunks "
        "that were retrieved for that category -- the retriever picked the WRONG chunks; "
        "the LLM was never given a chance.\n"
        "- **Extraction miss**: the gold answer text IS present in the retrieved chunks, "
        "but the LLM still didn't return it -- a model/prompt problem, not retrieval.\n"
    )

    r = analysis
    pct_retr = (r["total_retrieval_miss"] / r["total_fn"] * 100) if r["total_fn"] else 0.0
    pct_extr = (r["total_extraction_miss"] / r["total_fn"] * 100) if r["total_fn"] else 0.0
    lines.append("## Summary\n")
    lines.append(f"- Total false negatives (missed gold answers): **{r['total_fn']}**")
    lines.append(f"- Caused by retrieval (wrong chunks retrieved): **{r['total_retrieval_miss']}** "
                 f"({pct_retr:.1f}%)")
    lines.append(f"- Caused by extraction (right chunks, LLM still missed it): "
                 f"**{r['total_extraction_miss']}** ({pct_extr:.1f}%)")
    lines.append(f"- Categories with at least one retrieval-caused miss: "
                 f"**{len(retrieval_cats_sorted)}** of {len(by_cat)} categories that had any FN at all\n")

    lines.append("## Every category with a retrieval-caused miss (sorted by count, most first)\n")
    lines.append(f"| Category | Retrieval misses | Extraction misses (same category) |")
    lines.append(f"|---|---|---|")
    for cat, row in retrieval_cats_sorted:
        lines.append(f"| {cat} | {len(row['retrieval_miss'])} | {len(row['extraction_miss'])} |")
    lines.append("")

    lines.append("## Full detail: every retrieval-missed gold answer, by category\n")
    for cat, row in retrieval_cats_sorted:
        lines.append(f"### {cat} ({len(row['retrieval_miss'])} retrieval miss"
                     f"{'es' if len(row['retrieval_miss']) != 1 else ''})\n")
        for cid, gold in row["retrieval_miss"]:
            short_cid = cid[:55] + ("..." if len(cid) > 55 else "")
            gold_short = gold if len(gold) <= 160 else gold[:160] + "..."
            lines.append(f"- **{short_cid}**: {gold_short!r}")
        lines.append("")

    lines.append(
        "\n---\n*Generated by `bad_case_retrieval_analysis.py`. Re-run any time after "
        "re-running the underlying arm -- this script makes no API calls.*\n"
    )
    return "\n".join(lines)


def main():
    for target in TARGETS:
        arm_path = target["folder"] / "results" / target["arm"] / f"{MODEL}.json"
        arm_json = json.loads(arm_path.read_text(encoding="utf-8"))
        analysis = analyze(arm_json)
        md = render_md(target, arm_json, analysis)
        out_path = target["folder"] / "BadCaseAnalysisForRetrival.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"[{target['arm']}] total_fn={analysis['total_fn']} "
              f"retrieval_miss={analysis['total_retrieval_miss']} "
              f"extraction_miss={analysis['total_extraction_miss']} "
              f"categories_with_retrieval_miss={sum(1 for v in analysis['by_cat'].values() if v['retrieval_miss'])}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
