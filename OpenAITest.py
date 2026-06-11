"""
CUAD extraction with an OpenAI model (via LangChain), over pre-chunked input.

Chunking happens elsewhere and lands in test_chunking.json (one entry per
contract: a contract_id and a ready-made list of chunks). This script just reads
those chunks, sends each one to the model, and collects the verbatim spans it
finds -- no chunking here.

Kept chatty on purpose: it prints the chunk count, the char size of each chunk,
and whether each call succeeded, so you can eyeball what's going on. Calls run
concurrently and progress is checkpointed, so an interrupted run resumes.

Run the whole set:
    python OpenAITest.py
Process just one contract (by its contract ID):
    python OpenAITest.py --doc "LohaCompanyltd_..._Supply Agreement"

Output is the nbest format that evaluate.py reads:
    python evaluate.py --model_path <out_dir>
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, create_model

CATEGORY_CSV = "category_descriptions.csv"
MAX_TOKENS = 1200
CONCURRENCY = 4            # parallel in-flight calls; lower it if you hit rate limits

SYSTEM_PROMPT = (
    "You are a legal contract analyst. For each schema field, return the EXACT "
    "verbatim substring from the section that matches that field's description. "
    "Copy character-by-character; never paraphrase or invent. Set a field to "
    "null if its category does not appear in the section."
)
USER_PROMPT = 'Contract section:\n"""\n{chunk}\n"""'


# --- Schema -----------------------------------------------------------------

def load_categories(csv_path: str) -> dict[str, str]:
    """{label: description} from the CUAD category CSV (col 0 = label, 1 = desc)."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cats = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).split("Category:", 1)[-1].strip()
        desc = str(row.iloc[1]).split("Description:", 1)[-1].strip()
        cats[label] = desc
    return cats


def build_schema(categories: dict[str, str]):
    """One Pydantic model with an Optional[str] field per category.
    Returns (Schema, {field_name: qa_id_label})."""
    fields, field_to_label = {}, {}
    for label, desc in categories.items():
        fname = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        field_to_label[fname] = label.title()
        fields[fname] = (Optional[str], Field(default=None, description=desc))
    return create_model("ContractExtraction", **fields), field_to_label


def build_chain(model: str, schema, max_tokens: int = MAX_TOKENS):
    # ChatOpenAI picks up OPENAI_API_KEY from the environment on its own.
    llm = ChatOpenAI(model=model, temperature=0, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    return prompt | llm.with_structured_output(schema)


# --- Extraction -------------------------------------------------------------

def validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """Keep only spans that literally appear in the chunk -- drops anything the
    model paraphrased or hallucinated."""
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if span and span in chunk:
        return span
    if span and span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx:idx + len(span)]
    return None


# OpenAI's 429 errors carry a "Please try again in 1.2s" hint -- honour it.
_RETRY_RE = re.compile(r"try again in\s+(?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


async def call_with_retry(chain, inputs: dict, retries: int = 5):
    """ainvoke with reactive backoff: on a 429, sleep as long as OpenAI's
    'try again in ...' hint asks (or a small default), then retry."""
    for attempt in range(retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:
            if attempt == retries or "429" not in str(e) and "rate" not in str(e).lower():
                raise
            m = _RETRY_RE.search(str(e))
            wait = (float(m.group(2)) + (int(m.group(1)) * 60 if m.group(1) else 0)) if m else 2.0
            print(f"    rate-limited, sleeping {wait:.1f}s (attempt {attempt + 1})")
            await asyncio.sleep(wait + 0.5)


async def extract_contract(chain, field_to_label, chunks: list[str],
                           labels: list[str], concurrency: int) -> dict[str, list[str]]:
    """One LLM call per chunk (capped by `concurrency`); collect the validated
    spans per category. Prints each chunk's size and outcome."""
    sem = asyncio.Semaphore(concurrency)
    print(f"  {len(chunks)} chunks to process")

    async def process(i: int, chunk: str):
        async with sem:
            try:
                res = await call_with_retry(chain, {"chunk": chunk})
                print(f"    chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) -> OK")
                return res
            except Exception as e:
                print(f"    chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) -> FAILED: {str(e)[:120]}")
                return e

    results = await asyncio.gather(*(process(i, c) for i, c in enumerate(chunks)))

    preds = {label: [] for label in labels}
    for chunk, res in zip(chunks, results):
        if isinstance(res, Exception) or res is None:
            continue
        for fname, label in field_to_label.items():
            span = validate_span(getattr(res, fname, None), chunk)
            if span:
                preds[label].append(span)
    return preds


# --- Runner -----------------------------------------------------------------

def pick_contract(contracts: list[dict], contract_id: str) -> dict:
    """Find the one contract whose ID matches `contract_id`, or bail with a clear
    error if no contract in the loaded file has that ID."""
    for c in contracts:
        if c["contract_id"] == contract_id:
            return c
    sample = "\n  ".join(c["contract_id"] for c in contracts[:5])
    raise SystemExit(
        f"Contract ID {contract_id!r} not found in this data file "
        f"({len(contracts)} contracts). First few valid IDs:\n  {sample}"
    )


def load_checkpoint(path: Path) -> dict:
    """Resume an interrupted run: the checkpoint IS the nbest dict so far."""
    if path.exists():
        print(f"Resuming from checkpoint at {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(nbest: dict, path: Path):
    path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")


async def run(args):
    categories = load_categories(CATEGORY_CSV)
    schema, field_to_label = build_schema(categories)
    label_set = sorted(set(field_to_label.values()))
    chain = build_chain(args.model, schema, args.max_tokens)

    contracts = json.loads(Path(args.data).read_text(encoding="utf-8"))["data"]

    # --doc lets you process a single contract (by ID) instead of the whole set.
    if args.doc is not None:
        contracts = [pick_contract(contracts, args.doc)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "_checkpoint.json"
    nbest = load_checkpoint(checkpoint_path)   # {} on a fresh run

    print(f"OpenAI eval | model={args.model} | {len(contracts)} contract(s) "
          f"| concurrency={args.concurrency} | already done: {len(nbest)} QAs")

    t0, done = time.time(), 0
    for ci, contract in enumerate(contracts):
        cid = contract["contract_id"]
        chunks = contract["chunks"]
        # Each contract answers the same set of categories; the QA id CUAD's
        # evaluator expects is just "<contract_id>__<label>".
        target_ids = [f"{cid}__{label}" for label in label_set]

        # Resume: each contract is written atomically, so if all its target QAs
        # are already in the checkpoint we skip it without any LLM calls.
        if all(qid in nbest for qid in target_ids):
            done += len(target_ids)
            continue

        print(f"\n[{ci + 1}/{len(contracts)}] {cid[:60]} "
              f"({sum(len(c) for c in chunks):,} chars)")
        preds = await extract_contract(
            chain, field_to_label, chunks, label_set, args.concurrency,
        )

        for label in label_set:
            spans = list(dict.fromkeys(preds.get(label, [])))  # dedupe, keep order
            nbest[f"{cid}__{label}"] = (
                [{"text": s, "probability": 1.0} for s in spans]
                + [{"text": "", "probability": 0.0}]           # no-answer sentinel
            )
            done += 1

        save_checkpoint(nbest, checkpoint_path)   # persist after every contract

        hits = sum(1 for p in preds.values() if p)
        print(f"  {hits}/{len(label_set)} labels hit, {done} QAs total")

    out_path = out_dir / "nbest_predictions_.json"
    out_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(nbest)} QA predictions to {out_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    if checkpoint_path.exists():       # finished cleanly -> drop the checkpoint
        checkpoint_path.unlink()
    print(f"Next: python evaluate.py --model_path {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing -- add it to .env or your shell.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test_chunking.json",
                    help="Pre-chunked input (contract_id + chunks per contract).")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--doc", default=None,
                    help="Process only the contract with this ID.")
    ap.add_argument("--out", default=None, help="Output dir (default trained_models/<model>).")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    if args.out is None:
        args.out = "trained_models/" + args.model.replace(":", "-").replace("/", "_") + "__openai"

    asyncio.run(run(args))
