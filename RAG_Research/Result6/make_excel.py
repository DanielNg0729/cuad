"""
Excel workbook: per-category evaluation for the 6-contract / gpt-5.4 run, with
charts sorted worst-first so the weak categories are obvious at a glance.

Sheets
  Summary      - the method-level result table (TP/FP/FN, P/R/F1/F2, AUPR, Jaccard, cost)
  PerCategory  - one row per category: gold coverage, then TP/FP/FN/P/R/F1/Jaccard per method
  Chart_F1     - categories sorted ASCENDING by best-method F1 (worst first) + bar chart
  Chart_Jaccard- categories sorted ASCENDING by best-method Jaccard + bar chart

Run (after rag_research.py has written Result6/results):
    python RAG_Research/Result6/make_excel.py
"""

import json
import os
import sys
from pathlib import Path

import xlsxwriter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from OpenAITest import load_categories, CATEGORY_CSV   # noqa: E402
from evaluate import get_answers                        # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "gpt-5.4"
METHODS = ["M1_top2_cosine", "M2_top1_cosine", "M3_top1_cosine_llmcheck", "M4_top1_bm25",
           "M5_section_top2_cosine", "M6_section_hybrid_bm25_cosine", "M7_section_hybrid_top5"]
SHORT = {"M1_top2_cosine": "M1 md top2-cos", "M2_top1_cosine": "M2 md top1-cos",
         "M3_top1_cosine_llmcheck": "M3 md top1+chk", "M4_top1_bm25": "M4 md top1-bm25",
         "M5_section_top2_cosine": "M5 sec top2-cos",
         "M6_section_hybrid_bm25_cosine": "M6 sec hybrid-3",
         "M7_section_hybrid_top5": "M7 sec hybrid-5"}
# Sort the charts by the best-performing method on this 6-contract run (M1, F1 0.426).
# Sorting by a method that zeroes out whole categories (M3 on gpt-5.4) would push
# categories to the "worst" end that the other methods actually handle fine.
BEST = "M1_top2_cosine"

CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
    "AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEMENT AGREEMENT",
    "Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT",
    "DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGREEMENT",
]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    docs = {}
    for m in METHODS:
        path = RESULTS / m / f"{MODEL}.json"
        if path.exists():
            docs[m] = json.loads(path.read_text(encoding="utf-8"))
    if not docs:
        raise SystemExit(f"No result JSONs under {RESULTS}. Run rag_research.py first.")

    categories = sorted(l.title() for l in load_categories(CATEGORY_CSV))
    gt = get_answers(json.loads((ROOT / "test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)

    # gold coverage per category
    gold_answers, gold_contracts = {}, {}
    for cat in categories:
        counts = [len(gt.get(f"{cid}__{cat}", [])) for cid in CONTRACTS]
        gold_answers[cat] = sum(counts)
        gold_contracts[cat] = sum(1 for n in counts if n > 0)

    # per-category stats per method
    stats = {}   # stats[method][cat] = dict
    for m, d in docs.items():
        agg = d["by_category_counts"]
        jac = d["metrics"].get("jaccard_per_category", {})
        stats[m] = {}
        for cat in categories:
            c = agg.get(cat, {"tp": 0, "fp": 0, "fn": 0})
            p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
            stats[m][cat] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                             "precision": p, "recall": r, "f1": f1,
                             "jaccard": jac.get(cat)}   # None when the category has no gold

    out = HERE / "evaluation_by_category.xlsx"
    wb = xlsxwriter.Workbook(str(out))
    hdr = wb.add_format({"bold": True, "bg_color": "#DDDDDD", "border": 1, "text_wrap": True,
                         "valign": "vcenter", "align": "center"})
    txt = wb.add_format({"border": 1})
    num = wb.add_format({"border": 1, "num_format": "0.000"})
    integer = wb.add_format({"border": 1, "num_format": "0"})
    bad = wb.add_format({"border": 1, "num_format": "0.000", "bg_color": "#F8C9C4"})
    nogold = wb.add_format({"border": 1, "italic": True, "font_color": "#999999"})

    # ---------------- Summary ----------------
    ws = wb.add_worksheet("Summary")
    ws.set_column(0, 0, 24)
    ws.set_column(1, 12, 10)
    cols = ["method", "TP", "FP", "FN", "Precision", "Recall", "F1", "F2",
            "AUPR", "best_F1", "Jaccard(LicGrant)", "cost_usd"]
    for j, c in enumerate(cols):
        ws.write(0, j, c, hdr)
    for i, m in enumerate(METHODS):
        if m not in docs:
            continue
        d = docs[m]
        mi, me = d["micro"], d["metrics"]
        row = [SHORT[m], mi["tp"], mi["fp"], mi["fn"], mi["precision"], mi["recall"],
               mi["f1"], mi["f2"], me["aupr"], me["best_f1"], me["jaccard_similarity"],
               d["cost_usd"]]
        for j, v in enumerate(row):
            ws.write(i + 1, j, v, txt if j == 0 else (integer if j <= 3 else num))
    ws.write(len(METHODS) + 2, 0, f"Model: {MODEL} | {len(CONTRACTS)} contracts | 41 categories", txt)

    # ---------------- PerCategory ----------------
    ws = wb.add_worksheet("PerCategory")
    ws.freeze_panes(2, 1)
    ws.set_column(0, 0, 34)
    ws.set_column(1, 2, 11)
    head1 = ["", "Gold", "Gold"]
    for m in METHODS:
        head1 += [SHORT.get(m, m)] * 7
    head2 = ["Category", "Answers", "Contracts"]
    for _ in METHODS:
        head2 += ["TP", "FP", "FN", "Prec", "Rec", "F1", "Jaccard"]
    for j, v in enumerate(head1):
        ws.write(0, j, v, hdr)
    for j, v in enumerate(head2):
        ws.write(1, j, v, hdr)
    ws.set_column(3, 3 + 7 * len(METHODS), 8)

    for i, cat in enumerate(categories):
        r = i + 2
        ws.write(r, 0, cat, txt)
        ws.write(r, 1, gold_answers[cat], integer)
        ws.write(r, 2, gold_contracts[cat], integer)
        for k, m in enumerate(METHODS):
            base = 3 + k * 7
            s = stats.get(m, {}).get(cat)
            if not s:
                continue
            ws.write(r, base + 0, s["tp"], integer)
            ws.write(r, base + 1, s["fp"], integer)
            ws.write(r, base + 2, s["fn"], integer)
            ws.write(r, base + 3, s["precision"], num)
            ws.write(r, base + 4, s["recall"], num)
            # highlight a weak F1 on a category that DOES have gold
            f1_fmt = bad if (gold_answers[cat] > 0 and s["f1"] < 0.34) else num
            ws.write(r, base + 5, s["f1"], f1_fmt)
            if s["jaccard"] is None:
                ws.write(r, base + 6, "no gold", nogold)
            else:
                ws.write(r, base + 6, s["jaccard"], num)

    note = ("Rows highlighted red = F1 < 0.34 on a category that HAS gold answers (a real miss). "
            "'no gold' = the category never appears in these 6 contracts, so it can only "
            "generate false positives and has no recall/Jaccard to speak of.")
    ws.write(len(categories) + 3, 0, note, txt)

    # ---------------- charts ----------------
    def chart_sheet(name, metric, title, only_with_gold=True):
        """Write a sheet whose rows are sorted ASCENDING by the best method's `metric`
        (worst categories first) and attach a grouped bar chart over all methods."""
        cats = [c for c in categories if (gold_answers[c] > 0 or not only_with_gold)]

        def key(c):
            v = stats.get(BEST, {}).get(c, {}).get(metric)
            return 1e9 if v is None else v
        cats.sort(key=key)   # worst first

        wsx = wb.add_worksheet(name)
        wsx.set_column(0, 0, 34)
        wsx.set_column(1, 1 + len(METHODS), 13)
        wsx.write(0, 0, "Category", hdr)
        for j, m in enumerate(METHODS):
            wsx.write(0, j + 1, SHORT[m], hdr)
        for i, c in enumerate(cats):
            wsx.write(i + 1, 0, c, txt)
            for j, m in enumerate(METHODS):
                v = stats.get(m, {}).get(c, {}).get(metric)
                wsx.write(i + 1, j + 1, 0.0 if v is None else v, num)

        ch = wb.add_chart({"type": "column"})
        n = len(cats)
        for j, m in enumerate(METHODS):
            ch.add_series({
                "name":       [name, 0, j + 1],
                "categories": [name, 1, 0, n, 0],
                "values":     [name, 1, j + 1, n, j + 1],
                "gap": 40,
            })
        ch.set_title({"name": title})
        ch.set_x_axis({"name": "Category (sorted worst -> best by " + SHORT[BEST] + ")",
                       "num_font": {"rotation": -45, "size": 8}})
        ch.set_y_axis({"name": metric.title(), "min": 0, "max": 1})
        ch.set_size({"width": 1500, "height": 620})
        ch.set_legend({"position": "top"})
        wsx.insert_chart(1, len(METHODS) + 3, ch)
        return n

    n1 = chart_sheet("Chart_F1", "f1",
                     f"F1 per category ({MODEL}, 6 contracts) - worst first")
    n2 = chart_sheet("Chart_Jaccard", "jaccard",
                     f"Jaccard per category ({MODEL}, 6 contracts) - worst first")

    # ---------------- Recall_at_k (per-method retrieval funnel) ----------------
    funnel_path = HERE / "recall_at_k_all_methods.json"
    have_funnel = funnel_path.exists()
    if have_funnel:
        funnel = json.loads(funnel_path.read_text(encoding="utf-8"))
        wsr = wb.add_worksheet("Recall_at_k")
        wsr.set_column(0, 0, 30)
        wsr.set_column(1, 11, 11)
        cols = ["method", "chunking", "search", "k", "n_chunks", "coverage",
                "R_at_k", "reach_covxRk", "X_given_R", "end2end_recall"]
        nice = ["method", "chunking", "search", "k", "chunks", "coverage",
                "R@k", "reach=cov*R@k", "X|R", "end2end recall"]
        for j, c in enumerate(nice):
            wsr.write(0, j, c, hdr)
        for i, r in enumerate(funnel):
            for j, c in enumerate(cols):
                v = r.get(c, "")
                fmt = txt if j <= 2 else (integer if c in ("k", "n_chunks") else num)
                wsr.write(i + 1, j, SHORT.get(v, v) if c == "method" else v, fmt)
        wsr.write(len(funnel) + 2, 0,
                  "R@k = of gold reachable in that method's chunks, fraction whose chunk is in "
                  "the top-k retrieved. reach = coverage*R@k. X|R = of retrieved gold, fraction "
                  "the LLM then extracted. M3 bypasses retrieval (full-contract check).", txt)

        # grouped bar chart: coverage / R@k / reach / X|R / recall per method
        chr_ = wb.add_chart({"type": "column"})
        nrows = len(funnel)
        for j, (col, name) in enumerate([("coverage", "coverage"), ("R_at_k", "R@k"),
                                         ("reach_covxRk", "reach"), ("X_given_R", "X|R"),
                                         ("end2end_recall", "end2end recall")]):
            cidx = cols.index(col)
            chr_.add_series({
                "name": name,
                "categories": ["Recall_at_k", 1, 0, nrows, 0],
                "values": ["Recall_at_k", 1, cidx, nrows, cidx],
                "gap": 60,
            })
        chr_.set_title({"name": f"Retrieval funnel per method ({MODEL}, 6 contracts)"})
        chr_.set_x_axis({"num_font": {"rotation": -30, "size": 9}})
        chr_.set_y_axis({"name": "fraction", "min": 0, "max": 1})
        chr_.set_size({"width": 1150, "height": 560})
        chr_.set_legend({"position": "top"})
        wsr.insert_chart(len(funnel) + 5, 0, chr_)

    # ---------------- Hybrid grid: 3 tables (BM25 prefilter 5/10/15) ----------------
    master_path = HERE / "master_summary.json"
    have_master = master_path.exists()
    if have_master:
        master = json.loads(master_path.read_text(encoding="utf-8"))
        # index hybrid rows by (prefilter, k)
        by_bk = {(r["bm25_prefilter"], r["k"]): r for r in master
                 if r["search"] == "hybrid" and isinstance(r["bm25_prefilter"], int)}
        wsg = wb.add_worksheet("Hybrid_grid")
        wsg.set_column(0, 0, 16)
        wsg.set_column(1, 7, 11)
        metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
                   ("jaccard_avg", "Jaccard avg"), ("R_at_k", "R@k"), ("cost_usd", "cost$")]
        KS = [1, 2, 3, 5, 8]
        row0 = 0
        for bm in (5, 10, 15):
            wsg.write(row0, 0, f"BM25 prefilter = {bm}  (section chunks, cosine rerank)", hdr)
            wsg.write(row0 + 1, 0, "cosine k", hdr)
            for j, (_, name) in enumerate(metrics):
                wsg.write(row0 + 1, j + 1, name, hdr)
            for i, k in enumerate(KS):
                cell = by_bk.get((bm, k))
                note = ""
                if cell is None and k > bm:            # degenerate: capped at prefilter
                    cell = by_bk.get((bm, bm)); note = f" (=k{bm})"
                wsg.write(row0 + 2 + i, 0, f"{k}{note}", txt)
                for j, (key, _) in enumerate(metrics):
                    if cell is None:
                        wsg.write(row0 + 2 + i, j + 1, "", txt)
                    else:
                        wsg.write(row0 + 2 + i, j + 1, cell[key],
                                  num if key != "cost_usd" else num)
            row0 += 2 + len(KS) + 2
        wsg.write(row0, 0, "R@k = retrieval recall (right chunk in top-k). k>prefilter is "
                           "capped at the prefilter size, so it repeats that row.", txt)

    # ---------------- Best method per category ----------------
    bpc_path = HERE / "best_per_category.json"
    have_bpc = bpc_path.exists()
    if have_bpc:
        bpc = json.loads(bpc_path.read_text(encoding="utf-8"))
        wsb = wb.add_worksheet("Best_per_category")
        wsb.set_column(0, 0, 34)
        wsb.set_column(1, 5, 24)
        for j, c in enumerate(["Category", "gold", "best method (F1)", "F1",
                               "best method (Jaccard)", "Jaccard"]):
            wsb.write(0, j, c, hdr)
        # sort by best F1 ascending (worst categories first)
        for i, (cat, r) in enumerate(sorted(bpc.items(), key=lambda kv: kv[1]["best_f1"])):
            wsb.write(i + 1, 0, cat, txt)
            wsb.write(i + 1, 1, r["gold_answers"], integer)
            wsb.write(i + 1, 2, r["best_f1_method"], txt)
            wsb.write(i + 1, 3, r["best_f1"], num)
            wsb.write(i + 1, 4, r["best_jaccard_method"], txt)
            wsb.write(i + 1, 5, r["best_jaccard"], num)
        # win tally
        from collections import Counter
        wins = Counter(r["best_f1_method"] for r in bpc.values())
        wsb.write(len(bpc) + 2, 0, "How many categories each method wins (best F1):", hdr)
        for i, (m, n) in enumerate(wins.most_common()):
            wsb.write(len(bpc) + 3 + i, 0, m, txt)
            wsb.write(len(bpc) + 3 + i, 1, n, integer)

    # ---------------- Master summary (all methods) ----------------
    if have_master:
        wsm = wb.add_worksheet("Master_summary")
        wsm.set_column(0, 0, 26)
        wsm.set_column(1, 20, 11)
        cols = [("method", "method"), ("chunking", "chunk"), ("search", "search"),
                ("k", "k"), ("bm25_prefilter", "bm25_n"), ("n_chunks", "chunks"),
                ("f1", "F1"), ("precision", "Prec"), ("recall", "Rec"), ("f2", "F2"),
                ("aupr", "AUPR"), ("best_f1", "bestF1"), ("jaccard_avg", "Jac avg"),
                ("jaccard_best", "Jac best"), ("jaccard_best_cat", "best cat"),
                ("jaccard_worst", "Jac worst"), ("jaccard_worst_cat", "worst cat"),
                ("coverage", "coverage"), ("R_at_k", "R@k"), ("cost_usd", "cost$")]
        for j, (_, name) in enumerate(cols):
            wsm.write(0, j, name, hdr)
        floatcols = {"f1", "precision", "recall", "f2", "aupr", "best_f1", "jaccard_avg",
                     "jaccard_best", "jaccard_worst", "coverage", "R_at_k", "cost_usd"}
        for i, r in enumerate(master):
            for j, (key, _) in enumerate(cols):
                v = r.get(key, "")
                wsm.write(i + 1, j, v, num if key in floatcols else (integer if key in ("k", "n_chunks") else txt))

    wb.close()
    print(f"Wrote {out}")
    print(f"  Summary        : {len(docs)} methods")
    print(f"  PerCategory    : {len(categories)} categories")
    print(f"  Chart_F1       : {n1} categories with gold, sorted worst-first")
    print(f"  Chart_Jaccard  : {n2} categories with gold, sorted worst-first")
    print(f"  Recall_at_k    : {'per-method funnel + chart' if have_funnel else 'SKIPPED'}")
    print(f"  Hybrid_grid    : {'3 tables (bm25 5/10/15)' if have_master else 'SKIPPED (run master_summary.py)'}")
    print(f"  Best_per_category: {'yes' if have_bpc else 'SKIPPED (run best_per_category.py)'}")
    print(f"  Master_summary : {'all methods' if have_master else 'SKIPPED (run master_summary.py)'}")


if __name__ == "__main__":
    main()
