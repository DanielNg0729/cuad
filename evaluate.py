"""
Evaluate CUAD nbest predictions with a richer metric set.

Inputs (unchanged):
  - test.json   : SQuAD-style CUAD ground truth
  - nbest_predictions_.json
                : {qa_id: [{"text": str, "probability": float}, ...]}

Metrics:
  - AUPR                       Area under the precision-recall curve
  - Precision @ 80% recall     Highest precision we can hit while keeping recall >= 0.8
  - Precision @ 90% recall     Same idea, stricter recall bar
  - Best F1                    Max F1 across all confidence thresholds
  - Best F2                    Max F2 (recall-weighted) across all thresholds
  - Jaccard similarity         Mean token-overlap between gt and best-matching pred
                               for a single chosen category (defaults to "License Grant")

A prediction matches a ground-truth answer when their token Jaccard is at least
IOU_THRESH=0.5 (the original CUAD paper's setting). The "Parties" category also
accepts a substring containment match because gt spans there are very short.

CLI:
    python evaluate.py --model_path ./trained_models/qwen3-1.7b
    python evaluate.py --model_path ./trained_models/qwen3-1.7b --category "License Grant"
    python evaluate.py --model_path ./trained_models/qwen3-1.7b --category all
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

# Token-overlap threshold for calling a prediction a match. From the CUAD paper.
IOU_THRESH = 0.5

# "Complex" categories whose spans tend to be long, multi-clause, and worth
# scoring by Jaccard overlap rather than just precision/recall.
DEFAULT_JACCARD_CATEGORY = "License Grant"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_questions_from_csv(csv_path="category_descriptions.csv"):
    """Map category name -> description (kept for parity with the old script)."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    out = {}
    for i in range(df.shape[0]):
        category = df.iloc[i, 0].split("Category: ")[1]
        description = df.iloc[i, 1].split("Description: ")[1]
        out[category.title()] = description
    return out


def get_answers(test_json_dict):
    """Flatten SQuAD-style ground truth into {qa_id: [answer_text, ...]}."""
    results = {}
    for contract in test_json_dict["data"]:
        for para in contract["paragraphs"]:
            for qa in para["qas"]:
                results[qa["id"]] = [a["text"] for a in qa["answers"]]
    return results


def get_preds(nbest_preds_dict, conf):
    """Filter the nbest dict to predictions whose probability > conf."""
    results = {}
    for qid, preds in nbest_preds_dict.items():
        kept = {}
        for p in preds:
            if p["text"]:  # the empty-string entry stands for "no answer"
                kept[p["text"]] = p["probability"]
        results[qid] = [t for t, prob in kept.items() if prob > conf]
    return results


# ---------------------------------------------------------------------------
# Core matching primitives
# ---------------------------------------------------------------------------

def get_jaccard(gt: str, pred: str) -> float:
    """Token-level Jaccard. Lowercases, strips light punctuation, splits on
    whitespace. Identical to the original CUAD implementation."""
    for token in [".", ",", ";", ":"]:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower().replace("/", " ")
    pred = pred.lower().replace("/", " ")
    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))
    if not gt_words and not pred_words:
        return 0.0
    return len(gt_words & pred_words) / len(gt_words | pred_words)


def _is_match(ans: str, pred: str, substr_ok: bool) -> bool:
    if substr_ok and ans in pred:
        return True
    return get_jaccard(ans, pred) >= IOU_THRESH


def compute_precision_recall(gt_dict, preds_dict, category=None):
    """Count tp/fp/fn across all (or one) categories and return (P, R)."""
    tp = fp = fn = 0
    for key, answers in gt_dict.items():
        if category and not key.endswith(f"__{category}"):
            continue
        substr_ok = "Parties" in key
        preds = preds_dict.get(key, [])

        if not answers:
            # No ground truth -> every prediction is a false positive.
            fp += len(preds)
            continue

        # True positives: each gt that has any matching prediction.
        for ans in answers:
            if any(_is_match(ans, p, substr_ok) for p in preds):
                tp += 1
            else:
                fn += 1

        # False positives: each prediction that matches no gt answer.
        for p in preds:
            if not any(_is_match(a, p, substr_ok) for a in answers):
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall    = tp / (tp + fn) if (tp + fn) else np.nan
    return precision, recall


# ---------------------------------------------------------------------------
# PR curve and derived metrics
# ---------------------------------------------------------------------------

def process_precisions(precisions):
    """Monotone-decreasing precision envelope (standard PR-curve smoothing)."""
    rev = list(precisions[::-1])
    for i in range(1, len(rev)):
        rev[i] = max(rev[i - 1], rev[i])
    return rev[::-1]


def get_precisions_recalls(pred_dict, gt_dict, category=None):
    """Sweep confidence thresholds high -> low to trace out a PR curve."""
    precisions, recalls, confs = [1.0], [0.0], []
    for conf in list(np.arange(0.99, 0, -0.01)) + [0.001, 0.0]:
        thresholded = get_preds(pred_dict, conf)
        p, r = compute_precision_recall(gt_dict, thresholded, category=category)
        # NaN occurs when there are no predictions at this threshold; treat as
        # precision=1, recall=0 so the curve still extends cleanly.
        precisions.append(0.0 if np.isnan(p) else p)
        recalls.append(0.0 if np.isnan(r) else r)
        confs.append(conf)
    return precisions, recalls, confs


def get_aupr(precisions, recalls):
    aupr = metrics.auc(recalls, process_precisions(precisions))
    return 0.0 if np.isnan(aupr) else aupr


def get_prec_at_recall(precisions, recalls, recall_thresh):
    """Highest precision among points whose recall >= the threshold."""
    smoothed = process_precisions(precisions)
    for prec, recall in zip(smoothed, recalls):
        if recall >= recall_thresh:
            return prec
    return 0.0


def get_best_fbeta(precisions, recalls, confs, beta):
    """Walk the PR curve and return the best F-beta score plus its threshold."""
    best_f, best_conf = 0.0, 0.0
    smoothed = process_precisions(precisions)
    # confs is one element shorter (no conf for the (P=1, R=0) anchor).
    for p, r, c in zip(smoothed[1:], recalls[1:], confs):
        denom = (beta ** 2) * p + r
        if denom == 0:
            continue
        f = (1 + beta ** 2) * p * r / denom
        if f > best_f:
            best_f, best_conf = f, c
    return best_f, best_conf


# ---------------------------------------------------------------------------
# Jaccard for a single (complex) category
# ---------------------------------------------------------------------------

def compute_jaccard_for_category(pred_dict, gt_dict, category, conf=0.0):
    """Mean best-match Jaccard for one category.

    For each QA in the category that has at least one ground-truth answer,
    take max_p Jaccard(answer, p) for every answer, then average. A QA with
    answers but no predictions contributes 0 (worst overlap).
    """
    thresholded = get_preds(pred_dict, conf)
    scores = []
    for key, answers in gt_dict.items():
        if not key.endswith(f"__{category}") or not answers:
            continue
        preds = thresholded.get(key, [])
        for ans in answers:
            scores.append(max((get_jaccard(ans, p) for p in preds), default=0.0))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def evaluate(pred_dict, gt_dict, category=None, jaccard_category=None):
    """Compute every metric for one (category | all) selection.

    `category` = None evaluates over all categories; a string evaluates only QAs
    whose id ends with "__<category>". `jaccard_category` controls which single
    category the Jaccard metric is computed for (defaults to `category` if set,
    otherwise to DEFAULT_JACCARD_CATEGORY).
    """
    precisions, recalls, confs = get_precisions_recalls(pred_dict, gt_dict, category)

    jaccard_cat = jaccard_category or category or DEFAULT_JACCARD_CATEGORY

    return {
        "category":           category or "all",
        "aupr":               get_aupr(precisions, recalls),
        "prec_at_80_recall":  get_prec_at_recall(precisions, recalls, 0.80),
        "prec_at_90_recall":  get_prec_at_recall(precisions, recalls, 0.90),
        "best_f1":            get_best_fbeta(precisions, recalls, confs, beta=1)[0],
        "best_f2":            get_best_fbeta(precisions, recalls, confs, beta=2)[0],
        "jaccard_category":   jaccard_cat,
        "jaccard_similarity": compute_jaccard_for_category(pred_dict, gt_dict, jaccard_cat),
    }


def get_results(model_path, gt_dict, category=None, jaccard_category=None, verbose=False):
    """Load a model's nbest file and run `evaluate` against the gt."""
    pred_path = os.path.join(model_path, "nbest_predictions_.json")
    pred_dict = load_json(pred_path)

    # Align prediction keys to the gt keyspace: missing -> empty, extra -> drop.
    for qid in set(gt_dict) - set(pred_dict):
        pred_dict[qid] = []
    pred_dict = {k: v for k, v in pred_dict.items() if k in gt_dict}

    results = evaluate(pred_dict, gt_dict, category, jaccard_category)
    results["name"] = os.path.basename(os.path.normpath(model_path))

    if verbose:
        print_results(results)
    return results


def print_results(results: dict):
    print(f"\n== {results['name']}  [category: {results['category']}] ==")
    print(f"  AUPR                        {results['aupr']:.4f}")
    print(f"  Precision @ 80% recall      {results['prec_at_80_recall']:.4f}")
    print(f"  Precision @ 90% recall      {results['prec_at_90_recall']:.4f}")
    print(f"  Best F1                     {results['best_f1']:.4f}")
    print(f"  Best F2                     {results['best_f2']:.4f}")
    print(f"  Jaccard ({results['jaccard_category']}){' ' * max(1, 20 - len(results['jaccard_category']))}"
          f"{results['jaccard_similarity']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./trained_models/qwen3-1.7b",
                    help="Folder containing nbest_predictions_.json")
    ap.add_argument("--data", default="./data/test.json",
                    help="CUAD test set in SQuAD format")
    ap.add_argument("--save_dir", default="./results",
                    help="Where to write the result JSON")
    ap.add_argument("--category", default="all",
                    help='Category name (e.g. "License Grant") or "all"')
    ap.add_argument("--jaccard_category", default=None,
                    help="Category used for the Jaccard metric. Defaults to "
                         f"--category when specific, else '{DEFAULT_JACCARD_CATEGORY}'.")
    args = ap.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    gt_dict = get_answers(load_json(args.data))

    category = None if args.category.lower() == "all" else args.category
    results = get_results(args.model_path, gt_dict,
                          category=category,
                          jaccard_category=args.jaccard_category,
                          verbose=True)

    suffix = "all" if category is None else category.replace(" ", "_")
    save_path = os.path.join(args.save_dir, f"{results['name']}__{suffix}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {save_path}")
