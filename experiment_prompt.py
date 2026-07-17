"""
Root-cause ablation across ALL 41 CUAD categories, on the same 3 contracts.

Three arms, holding model / chunks / validate_span / scorer constant and varying
only how the categories are prompted:

  Arm0  "baseline-batched"  : all 41 categories in ONE schema, original yes/no
                              descriptions  (read from the existing nbest file --
                              no new API cost).
  ArmA  "baseline-grouped"  : same yes/no descriptions, but categories sent in
                              small CUAD-group schemas (isolates the focus/
                              attention effect of NOT batching all 41 at once).
  ArmB  "example-grouped"   : same grouping, but each field carries a definition
                              + a positive EXAMPLE span + an explicit "return the
                              verbatim substring" instruction (isolates prompt
                              quality on top of focus).

Arm0 vs ArmA  -> does focus alone help?
ArmA vs ArmB  -> does prompt quality help on top of focus?

Output per model -> results/experiment_all41/<model>.json with, for every arm,
every contract, every category: ground_truth, predictions, tp, fn, fp.

Run:
    python experiment_prompt.py --model gpt-5.4
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, create_model

from OpenAITest import (
    SYSTEM_PROMPT, USER_PROMPT, validate_span, call_with_retry,
    load_categories, estimate_cost, CATEGORY_CSV,
)
from evaluate import get_answers, get_jaccard, _is_match

CONTRACTS = [
    "BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT",
    "BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endorsement Agreement",
    "AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement",
]

# Focused-arm grouping: CUAD groups 1-6 kept intact, plus the ungrouped ("-")
# categories split into small thematic buckets. Labels are the .title()-cased
# forms used in the CUAD qa-ids. The union must be exactly the 41 categories.
GROUPS = [
    ["Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
     "Notice Period To Terminate Renewal"],
    ["Non-Compete", "Exclusivity", "No-Solicit Of Customers",
     "Competitive Restriction Exception"],
    ["Change Of Control", "Anti-Assignment"],
    ["License Grant", "Non-Transferable License", "Affiliate License-Licensor",
     "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
     "Irrevocable Or Perpetual License"],
    ["Post-Termination Services", "Audit Rights"],
    ["Uncapped Liability", "Cap On Liability"],
    ["Document Name", "Parties", "Governing Law"],
    ["Most Favored Nation", "Revenue/Profit Sharing", "Price Restrictions",
     "Minimum Commitment", "Volume Restriction"],
    ["Ip Ownership Assignment", "Joint Ip Ownership", "Source Code Escrow"],
    ["No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
     "Rofr/Rofo/Rofn"],
    ["Liquidated Damages", "Warranty Duration", "Insurance",
     "Covenant Not To Sue", "Third Party Beneficiary"],
]

# ArmB: definition + a SYNTHETIC example (never drawn from the test contracts) +
# explicit extraction instruction, for each of the 41 categories.
_EX = "Example of a triggering clause (illustrative, not from this contract): "
_RET = " Return the EXACT verbatim substring from THIS section that does this, or null."
EXAMPLE_DRIVEN = {
    "Document Name":
        "Find the name/title of this contract (often a heading)." + _EX
        + "'DISTRIBUTION AND LICENSE AGREEMENT'." + _RET,
    "Parties":
        "Find the names of the contracting parties (the entities or individuals who "
        "signed)." + _EX + "'between Acme Corp., a Delaware corporation, and Beta LLC'." + _RET,
    "Agreement Date":
        "Find the date the contract was signed / entered into." + _EX
        + "'This Agreement is made as of January 15, 2020'." + _RET,
    "Effective Date":
        "Find the date the contract BECOMES effective (may differ from the signing "
        "date; look for 'effective as of ___')." + _EX
        + "'This Agreement shall take effect on March 1, 2020'." + _RET,
    "Expiration Date":
        "Find when the contract's INITIAL term ends. May be a calendar date, a "
        "DURATION from the start ('for N months/years from the Effective Date'), or "
        "'perpetual' -- recognise duration phrasing, not just dates." + _EX
        + "'This Agreement shall remain in effect for three (3) years from the Effective Date'." + _RET,
    "Renewal Term":
        "Find the renewal / extension term after the initial term (automatic or by "
        "notice)." + _EX
        + "'Thereafter this Agreement shall automatically renew for successive one (1) year periods'." + _RET,
    "Notice Period To Terminate Renewal":
        "Find the notice period required to stop or terminate a renewal." + _EX
        + "'unless either party gives written notice of non-renewal at least ninety (90) days prior to the end of the term'." + _RET,
    "Governing Law":
        "Find which state's or country's law governs the contract." + _EX
        + "'This Agreement shall be governed by the laws of the State of New York'." + _RET,
    "Most Favored Nation":
        "Find a clause entitling a party to better terms if a third party later gets "
        "better terms on the same goods/technology/services." + _EX
        + "'If Seller offers more favorable pricing to any other customer, such pricing shall be extended to Buyer'." + _RET,
    "Non-Compete":
        "Find a restriction on a party competing with the counterparty or operating "
        "in a certain geography/business/technology sector (may not say 'non-compete')." + _EX
        + "'During the Term, Distributor shall not sell or distribute any products that compete with the Products'." + _RET,
    "Exclusivity":
        "Find an exclusive-dealing commitment: dealing only with the counterparty, "
        "buying all requirements from one party, or not selling/licensing to third parties." + _EX
        + "'Supplier shall purchase all of its requirements for Widgets exclusively from Company'." + _RET,
    "No-Solicit Of Customers":
        "Find a restriction on soliciting or contracting the counterparty's customers "
        "or partners (during or after the contract)." + _EX
        + "'For two years after termination, Consultant shall not solicit any customer of the Company'." + _RET,
    "Competitive Restriction Exception":
        "Find an EXCEPTION or carve-out to a non-compete, exclusivity, or no-solicit "
        "restriction (look for 'except', 'shall not apply to')." + _EX
        + "'except that Distributor may continue to sell products it distributed prior to the Effective Date'." + _RET,
    "No-Solicit Of Employees":
        "Find a restriction on soliciting or hiring the counterparty's employees or "
        "contractors." + _EX
        + "'Neither party shall solicit for employment any employee of the other party during the Term'." + _RET,
    "Non-Disparagement":
        "Find a requirement not to disparage the counterparty." + _EX
        + "'Each party agrees not to make any disparaging statements about the other party'." + _RET,
    "Termination For Convenience":
        "Find a clause letting a party terminate WITHOUT cause, just by giving notice." + _EX
        + "'Either party may terminate this Agreement for any reason upon thirty (30) days written notice'." + _RET,
    "Rofr/Rofo/Rofn":
        "Find a right of first refusal / first offer / first negotiation to purchase, "
        "license, market, or distribute equity, technology, assets, or services." + _EX
        + "'Company shall have a right of first refusal to purchase such assets on the same terms'." + _RET,
    "Change Of Control":
        "Find a clause giving a party rights (terminate, or require consent/notice) if "
        "the other undergoes a merger, acquisition, stock sale, or transfer of all assets." + _EX
        + "'In the event of a change of control of Licensee, Licensor may terminate this Agreement'." + _RET,
    "Anti-Assignment":
        "Find a clause requiring consent or notice to assign the contract to a third party." + _EX
        + "'Neither party may assign this Agreement without the prior written consent of the other'." + _RET,
    "Revenue/Profit Sharing":
        "Find a clause requiring one party to share revenue or profit with the other." + _EX
        + "'Licensee shall pay Licensor 20% of net revenues derived from the Product'." + _RET,
    "Price Restrictions":
        "Find a restriction on a party's ability to raise or reduce prices." + _EX
        + "'Distributor shall not increase the resale price of the Products by more than 5% per year'." + _RET,
    "Minimum Commitment":
        "Find a minimum order size or minimum amount/units per period a party must buy." + _EX
        + "'Buyer shall purchase a minimum of 10,000 units per calendar quarter'." + _RET,
    "Volume Restriction":
        "Find a clause imposing a consequence -- a fee increase, surcharge, or consent "
        "requirement -- when a party's usage, order volume, or consumption EXCEEDS a "
        "stated threshold." + _EX
        + "'If Licensee's monthly active users exceed 1,000, Licensee shall pay an additional fee per user above that threshold'." + _RET,
    "Ip Ownership Assignment":
        "Find a clause assigning IP created by one party to the counterparty." + _EX
        + "'All inventions developed under this Agreement shall be the sole property of Company'." + _RET,
    "Joint Ip Ownership":
        "Find a clause providing joint or shared ownership of IP between the parties." + _EX
        + "'Any jointly developed intellectual property shall be owned jointly by the parties'." + _RET,
    "License Grant":
        "Find a clause where one party grants a license to the other." + _EX
        + "'Licensor hereby grants to Licensee a non-exclusive license to use the Software'." + _RET,
    "Non-Transferable License":
        "Find language limiting a party's ability to TRANSFER the granted license." + _EX
        + "'The license granted herein is personal and non-transferable'." + _RET,
    "Affiliate License-Licensor":
        "Find a license grant that comes from or includes IP of AFFILIATES of the "
        "LICENSOR (the grantor side)." + _EX
        + "'Licensor and its Affiliates hereby grant to Licensee a license under the Patents'." + _RET,
    "Affiliate License-Licensee":
        "Find a license grant that extends to AFFILIATES of the LICENSEE (the receiving "
        "side, including sublicensees)." + _EX
        + "'The license is granted to Licensee and its Affiliates'." + _RET,
    "Unlimited/All-You-Can-Eat-License":
        "Find a clause granting an 'enterprise', 'all-you-can-eat', or unlimited usage license." + _EX
        + "'Company grants an enterprise-wide, unlimited license to use the Software at all of Customer's locations'." + _RET,
    "Irrevocable Or Perpetual License":
        "Find a license grant stated to be irrevocable or perpetual." + _EX
        + "'Licensor grants a perpetual, irrevocable license to the Technology'." + _RET,
    "Source Code Escrow":
        "Find a requirement to deposit source code into escrow, releasable on certain "
        "events (bankruptcy, insolvency)." + _EX
        + "'Licensor shall deposit the source code with an escrow agent, releasable upon Licensor's insolvency'." + _RET,
    "Post-Termination Services":
        "Find obligations that continue AFTER termination/expiration (transition, "
        "wind-down, last-buy, payment, IP transfer)." + _EX
        + "'Upon termination, Supplier shall continue to provide spare parts for two (2) years'." + _RET,
    "Audit Rights":
        "Find a right to audit the books, records, or facilities of the counterparty "
        "for compliance." + _EX
        + "'Company may, upon reasonable notice, audit Licensee's records to verify royalty payments'." + _RET,
    "Uncapped Liability":
        "Find a clause stating a party's liability is UNCAPPED for a breach (or a "
        "specific breach type like IP infringement or confidentiality)." + _EX
        + "'Nothing herein shall limit a party's liability for breach of its confidentiality obligations'." + _RET,
    "Cap On Liability":
        "Find a clause LIMITING/capping liability for breach (a maximum recovery "
        "amount, or a time limit to bring claims)." + _EX
        + "'In no event shall either party's total liability exceed the fees paid in the prior twelve months'." + _RET,
    "Liquidated Damages":
        "Find a clause awarding liquidated damages for breach, or a termination fee." + _EX
        + "'Upon early termination, Customer shall pay a termination fee of $50,000'." + _RET,
    "Warranty Duration":
        "Find the DURATION of any warranty against defects or errors (a length of time)." + _EX
        + "'Supplier warrants the Products against defects for twelve (12) months from delivery'." + _RET,
    "Insurance":
        "Find a requirement to maintain insurance for the benefit of the counterparty." + _EX
        + "'Contractor shall maintain commercial general liability insurance of at least $1,000,000 per occurrence'." + _RET,
    "Covenant Not To Sue":
        "Find a clause where a party agrees NOT to challenge the counterparty's IP "
        "ownership/validity, or not to bring certain claims." + _EX
        + "'Licensee agrees not to challenge the validity of Licensor's patents'." + _RET,
    "Third Party Beneficiary":
        "Find a clause naming a non-party who can enforce rights under the contract." + _EX
        + "'The Guarantor shall be a third-party beneficiary of Section 5'." + _RET,
}


def fname_of(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def build_schema(descriptions: dict[str, str]):
    fields, field_to_label = {}, {}
    for label, desc in descriptions.items():
        fn = fname_of(label)
        field_to_label[fn] = label
        fields[fn] = (Optional[str], Field(default=None, description=desc))
    return create_model("Cats", **fields), field_to_label


def build_chain(model, schema):
    # timeout: cap a single request so one stalled call can't hang asyncio.gather
    # forever; max_retries lets the client transparently re-issue a timed-out call.
    llm = ChatOpenAI(model=model, temperature=0, timeout=120, max_retries=2)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    return prompt | llm.with_structured_output(schema, include_raw=True)


async def run_arm(model, chunkmap, desc_for_label, groups, concurrency):
    """Run one grouped arm; return {qa_key: [spans]} and token usage."""
    sem = asyncio.Semaphore(concurrency)
    usage = {"input": 0, "output": 0}
    preds = {f"{cid}__{label}": [] for cid in chunkmap for label in desc_for_label}

    for group in groups:
        schema, f2l = build_schema({l: desc_for_label[l] for l in group})
        chain = build_chain(model, schema)
        items = [(cid, ch) for cid, chunks in chunkmap.items() for ch in chunks]

        async def one(cid, ch):
            async with sem:
                return cid, ch, await call_with_retry(chain, {"chunk": ch})

        results = await asyncio.gather(*(one(cid, ch) for cid, ch in items),
                                       return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                continue
            cid, ch, res = r
            if res is None:
                continue
            raw, parsed = res.get("raw"), res.get("parsed")
            um = getattr(raw, "usage_metadata", None)
            if um:
                usage["input"] += um.get("input_tokens", 0)
                usage["output"] += um.get("output_tokens", 0)
            if parsed is None:
                continue
            for fn, label in f2l.items():
                span = validate_span(getattr(parsed, fn, None), ch)
                if span:
                    preds[f"{cid}__{label}"].append(span)
    return preds, usage


def arm0_from_nbest(model, gt):
    """Original 41-field predictions for our 3 contracts, from the nbest file."""
    nbest = json.load(open(
        f"trained_models/{model}__openai/nbest_predictions_.json", encoding="utf-8"))
    return {key: [p["text"] for p in nbest.get(key, []) if p["text"]] for key in gt}


def breakdown(preds, gt):
    """Per (contract, category): ground_truth / predictions / tp / fn / fp, plus
    aggregate tp/fp/fn counts per category."""
    by_contract, agg = {}, {}
    for key, answers in gt.items():
        cid, cat = key.rsplit("__", 1)
        substr_ok = "Parties" in key
        plist = list(dict.fromkeys(preds.get(key, [])))  # dedupe, keep order
        if answers:
            tp = [a for a in answers if any(_is_match(a, p, substr_ok) for p in plist)]
            fn = [a for a in answers if a not in tp]
            fp = [p for p in plist if not any(_is_match(a, p, substr_ok) for a in answers)]
        else:
            tp, fn, fp = [], [], list(plist)
        by_contract.setdefault(cid, {})[cat] = {
            "ground_truth": answers, "predictions": plist,
            "tp": tp, "fn": fn, "fp": fp,
        }
        a = agg.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
        a["tp"] += len(tp); a["fp"] += len(fp); a["fn"] += len(fn)
    return by_contract, agg


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    # sanity: the grouping must cover exactly the 41 categories
    flat = [c for g in GROUPS for c in g]
    cats_csv = {l.title() for l in load_categories(CATEGORY_CSV)}
    assert set(flat) == cats_csv and len(flat) == 41, \
        f"group mismatch: missing={cats_csv - set(flat)} extra={set(flat) - cats_csv}"
    assert set(EXAMPLE_DRIVEN) == cats_csv, \
        f"example-desc mismatch: missing={cats_csv - set(EXAMPLE_DRIVEN)}"

    gt = get_answers(json.loads(Path("test.json").read_text(encoding="utf-8")),
                     contract_ids=CONTRACTS)
    chunkdata = {c["contract_id"]: c["chunks"]
                 for c in json.loads(Path("test_chunking.json").read_text(encoding="utf-8"))["data"]}
    chunkmap = {cid: chunkdata[cid] for cid in CONTRACTS}

    baseline_desc = {l.title(): d for l, d in load_categories(CATEGORY_CSV).items()}

    print(f"Model={args.model} | 41 categories | {len(GROUPS)} groups | "
          f"{len(CONTRACTS)} contracts | concurrency={args.concurrency}")

    t0 = time.time()
    arms_raw = {}
    arms_raw["Arm0_baseline_batched"] = (arm0_from_nbest(args.model, gt), None)
    print("  ArmA (baseline-grouped) running...")
    arms_raw["ArmA_baseline_grouped"] = asyncio.run(
        run_arm(args.model, chunkmap, baseline_desc, GROUPS, args.concurrency))
    print(f"    done {(time.time()-t0)/60:.1f} min")
    print("  ArmB (example-grouped) running...")
    arms_raw["ArmB_example_grouped"] = asyncio.run(
        run_arm(args.model, chunkmap, EXAMPLE_DRIVEN, GROUPS, args.concurrency))
    print(f"    done {(time.time()-t0)/60:.1f} min")

    out = {"model": args.model, "contracts": CONTRACTS, "arms": {}}
    totals = {}
    cost_total = 0.0
    for name, (preds, use) in arms_raw.items():
        by_contract, agg = breakdown(preds, gt)
        cost = estimate_cost(args.model, use["input"], use["output"]) if use else 0.0
        cost_total += cost
        out["arms"][name] = {
            "cost_usd": round(cost, 6),
            "tokens": use,
            "totals": {k: sum(agg[c][k] for c in agg) for k in ("tp", "fp", "fn")},
            "by_category_counts": agg,
            "by_contract": by_contract,
        }
        totals[name] = out["arms"][name]["totals"]

    outdir = Path("results/experiment_all41")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{args.model}.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  {'arm':<26}{'TP':>5}{'FP':>5}{'FN':>5}{'cost':>9}")
    for name, t in totals.items():
        c = out["arms"][name]["cost_usd"]
        print(f"  {name:<26}{t['tp']:>5}{t['fp']:>5}{t['fn']:>5}{('$%.3f'%c):>9}")
    print(f"\n  Wrote {outpath}")
    print(f"  New API cost (this model): ${cost_total:.4f} | {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
