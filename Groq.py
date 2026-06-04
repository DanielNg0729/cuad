"""
This file run groq hosted LLM using langchain to test against the CUAD dataset using zero-shot prompting technique and write predictions in the nbest format that
can be use in the evaluate.py
This file can use to test model for each category or for all 41 categories 

"""

import argparse
import asyncio
import json
import os
import re
import time

from pathlib import Path
from typing import Optional # This support can be None or that specific type

import pandas as pd # Process excel file using pandas
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate # For parsing in the template wanted
from langchain_groq import ChatGroq
from pydantic import Field, create_model # Langchain use pydantic object

# All chunking logic lives in chunking.py — see that module to add/tune strategies.
from chunking import CHUNK_CHARS, OVERLAP, CHUNK_STRATEGY, make_chunks

MAX_CONCURRENCY = 1
DEFAULT_MAX_TOKENS = 1200
DESC_MAX_CHARS = 110
CATEGORY_CSV = "category_descriptions.csv"

# "Copy verbatim" / "null if absent" instructions live in the system prompt
# once instead of being duplicated into all 41 field descriptions — that
# alone shaved ~1000 tokens off every call.

SYSTEM_PROMPT = (
    "You are a legal contract analyst. For each schema field, return the "
    "EXACT verbatim substring from the section that matches that field's "
    "description. Copy character-by-character; never paraphrase or invent. "
    "Set a field to null if its category does not appear in the section."
)

USER_PROMPT = "Contract section:\n\"\"\"\n{chunk}\n\"\"\""


# Schema: All 41 fields in 1 pydantic class

def load_categories(csv_path: str) -> dict[str,list[str]]:
    """
    Parse CUAD dataset category_description.csv into {label: [description, answer_format, group]}.
    """

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    out = {}
    for _,row in df.iterrows():
        label = str(row.iloc[0]).split("Category:",1)[-1].strip()
        desc  = str(row.iloc[1]).split("Description:", 1)[-1].strip()
        fmt   = str(row.iloc[2]).split("Answer Format:", 1)[-1].strip()
        grp   = str(row.iloc[3]).split("Group:", 1)[-1].strip()
        out[label] = [desc, fmt, grp]
    
    return out

def label_to_field(label:str) -> str:
    # Human label -> snake case pydantic safe field name 
    """
    Good Pydantic field name:
        - lowercase letters / uppercase letters / digits / underscore are okay
        - cannot start with a digit
        - should not start with underscore
        - should not contain spaces, hyphens, brackets, punctuation, etc.
        - should not be a Python keyword like class, for, if, etc.
    So have to remove number, underscore at first char and lower case everything
    """
    return re.sub(r"[^a-z0-9]+","_",label.lower()).strip("_")


def filter_categories(categories:dict,category:str) -> dict:
    """
    Narrow categories into a single one, or return all if category == "all"
    Also matching case insensitive so user can pass either the csv form "Cap on Liability" or the id form "Cap On Liability"
    
    """
    if category.lower() == "all":
        return categories
    
    match = next((c for c in categories if c.lower() == category.lower()),None) # Take the first match category

    if match is None:
        raise SystemExit(
            f"Unknown Category {category!r}. Use 'all' or one of the category in the original dataset"
        )
    
    return {match: categories[match]}


def build_schema(categories: dict[str,list[str]]):
    """
    A single pydantic class covering all 41 fields. Return (Schema, field_to_label).

    The
    "copy verbatim / null if absent" boilerplate is in the system prompt
    instead of being duplicated 41× into every schema."
    
    """
    fields, field_to_label = {} , {}
    for label, (desc,_fmt,_grp) in categories.items():
        fname = label_to_field(label)
        field_to_label[fname] = label.title()
        fields[fname] = (
            Optional[str],
            Field(default=None,description=desc)
        )
    
    return create_model("ContractExtraction", **fields), field_to_label


def build_chain(model: str, schema, max_tokens:int = DEFAULT_MAX_TOKENS):
    """
    prompt | ChatGroq.with_structured_output(schema)
    """

    llm = ChatGroq(model = model, temperature=0, max_tokens= max_tokens).with_structured_output(schema)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),("human",USER_PROMPT)])

    return prompt | llm

def validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """Drop predictions that don't literally appear in the chunk. Even strong
    models occasionally paraphrase — we want raw extractions only."""
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if not span:
        return None
    if span in chunk:
        return span
    if span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx:idx + len(span)]
    return None

"""
API-error detection:
- Treat rate-limit, quota, and auth errors as "Stop the whole run and save" rather than per-contract failure
- Anything else (timeouts, parse errors,...) is just a skipped chunk

"""

class FatalAPIError(RuntimeError):
    """
    Signal that Groq API is unusable. Caught by run() to checkpoint and exit cleanly
    """

_FATAL_TYPES = {"RateLimitError", "AuthenticationError",
                "PermissionDeniedError", "InsufficientQuotaError"}
_FATAL_KEYWORDS = ("rate limit", "rate_limit", "quota", "insufficient_quota",
                   "invalid_api_key", "unauthorized", "429", "401", "403")

def is_fatal_api_error(exc: BaseException) -> bool:
    # True if the errors mean we should stop the run, not just skip a chunk
    if type(exc).__name__ in _FATAL_TYPES:
        return True
    msg = str(exc).lower()
    return any(k in msg for k in _FATAL_KEYWORDS)

# Instead of pre-throttling based on estimates (which over-counts because Groq
# only bills actual_output, not max_tokens), let Groq tell us when it's full.
# Every 429 from Groq includes a "Please try again in <duration>" hint — we
# parse it and sleep exactly that long. Durations come in two flavors:
#   - TPM (tokens/minute):  "try again in 8.594999999s"
#   - TPD (tokens/day):     "try again in 34m12.864s"   (and rarely with hours)

_RETRY_AFTER_RE = re.compile(
    r"try again in\s+(?:(\d+)h\s*)?(?:(\d+)m\s*)?([\d.]+)\s*s",
    re.IGNORECASE,
)
MAX_RETRIES = 5                # per-call attempts before giving up
MAX_RETRY_WAIT_SECONDS = 3600 * 2   # 2h cap — anything longer = bail and resume later

def _retry_seconds(exc: BaseException) -> Optional[float]:
    """Pull the wait duration out of a 429 message and convert to seconds.
    Handles 'Xs', 'Ym Zs', 'Wh Ym Zs' formats. Returns None if not a parseable
    rate-limit hint."""
    m = _RETRY_AFTER_RE.search(str(exc))
    if not m:
        return None
    h, mn, s = m.groups()
    total = float(s)
    if mn: total += int(mn) * 60
    if h:  total += int(h) * 3600
    return total

async def _sleep_with_eta(seconds: float, reason: str):
    """Sleep, but tell the user upfront when we'll wake. For long waits print
    a midway update so the run doesn't look hung."""
    wake_at = time.time() + seconds
    wake_str = time.strftime("%H:%M:%S", time.localtime(wake_at))
    print(f"  {reason}: sleeping {seconds/60:.1f} min, resume at ~{wake_str}")
    if seconds > 600:
        # Midway heartbeat for waits > 10 min so user can see the script is alive.
        await asyncio.sleep(seconds / 2)
        remaining = wake_at - time.time()
        print(f"  ... still waiting, ~{remaining/60:.1f} min remaining")
        await asyncio.sleep(max(0, remaining))
    else:
        await asyncio.sleep(seconds)

_NON_RETRIABLE = {"AuthenticationError", "PermissionDeniedError",
                  "InsufficientQuotaError"}


async def call_with_retry(chain, inputs: dict, max_retries: int = MAX_RETRIES):
    """ainvoke + reactive backoff. Sleeps exactly as long as Groq asks (TPM or
    TPD), then retries. Bails on auth errors or waits > MAX_RETRY_WAIT_SECONDS."""
    for attempt in range(max_retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:
            msg_low = str(e).lower()
            # Auth/quota/permission — no amount of waiting fixes these.
            if type(e).__name__ in _NON_RETRIABLE:
                raise FatalAPIError(str(e)) from e
            # 413 "Request too large": a single call exceeds the TPM cap
            # entirely. Retrying with the same payload always fails — bail
            # immediately so the user can lower --chunk_chars / --max_tokens.
            if "413" in msg_low or "request too large" in msg_low:
                raise FatalAPIError(
                    f"Single request exceeds model's TPM cap (no retry can "
                    f"help). Lower --chunk_chars and/or --max_tokens, then "
                    f"resume.\n  {e}"
                ) from e
            wait_s = _retry_seconds(e)
            # Fallback for 429s without a parseable hint: wait 60s.
            if wait_s is None:
                if "429" not in msg_low and "rate_limit" not in msg_low:
                    raise   # not a rate-limit error at all
                wait_s = 60.0
            # Cap: anything longer than the cap is better handled by exiting
            # and letting the user resume tomorrow.
            if wait_s > MAX_RETRY_WAIT_SECONDS:
                raise FatalAPIError(
                    f"Groq asked to wait {wait_s/60:.0f}min "
                    f"(> {MAX_RETRY_WAIT_SECONDS/60:.0f}min cap). "
                    f"Likely TPD exhausted — resume tomorrow."
                ) from e
            if attempt >= max_retries:
                raise FatalAPIError(
                    f"Rate-limited {max_retries}+ times in a row: {e}"
                ) from e
            sleep_s = wait_s + 0.5     # tiny cushion so we don't race the window
            await _sleep_with_eta(
                sleep_s,
                f"rate-limited (attempt {attempt+1}/{max_retries})",
            )


async def extract_contract(chain, field_to_label, context: str,
                           all_labels: list[str],
                           concurrency: int,
                           chunk_chars: int = CHUNK_CHARS,
                           overlap: int = OVERLAP,
                           strategy: str = CHUNK_STRATEGY) -> dict[str, list[str]]:
    """Fire one LLM call per chunk in parallel (capped by `concurrency`).
    Each call retries reactively on 429 using Groq's own retry-after hint."""
    chunk_list = make_chunks(context, strategy=strategy,
                             size=chunk_chars, overlap=overlap)
    predictions = {label: [] for label in all_labels}
    sem = asyncio.Semaphore(concurrency)

    async def process(chunk: str):
        async with sem:
            try:
                return await call_with_retry(chain, {"chunk": chunk})
            except FatalAPIError:
                raise
            except Exception as e:
                return e   # transient — caller skips this chunk

    results = await asyncio.gather(
        *(process(c) for c in chunk_list),
        return_exceptions=True,
    )

    for chunk_text, result in zip(chunk_list, results):
        if isinstance(result, Exception):
            if is_fatal_api_error(result):
                raise FatalAPIError(str(result)) from result
            continue  # transient — skip this chunk
        if result is None:
            continue
        for fname, label in field_to_label.items():
            valid = validate_span(getattr(result, fname, None), chunk_text)
            if valid:
                predictions[label].append(valid)
    return predictions

def save_checkpoint(nbest, path: Path):
    path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        print(f"Resuming from checkpoint at {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


async def run(args):
    print(f"Groq CUAD eval | model={args.model} | data={args.data} | "
          f"limit={args.limit or 'full'} | out={args.out}")

    categories = filter_categories(load_categories(CATEGORY_CSV), args.category)
    schema, field_to_label = build_schema(categories)
    label_set = set(field_to_label.values())   # qa_id-form labels we're testing
    all_labels = list(label_set)
    chain = build_chain(args.model, schema, max_tokens=args.max_tokens)

    # Rough per-call reservation (Groq counts input + max_tokens against TPM).
    # ~56 tokens of JSON scaffolding per schema field, measured.
    chunk_tok_est = args.chunk_chars // 4
    schema_tok_est = 56 * len(all_labels) + 100
    est_per_call = chunk_tok_est + schema_tok_est + args.max_tokens + 100
    scope = "all 41 categories" if args.category.lower() == "all" else f"category {args.category!r}"
    print(f"  testing {scope} ({len(all_labels)} field schema), "
          f"concurrency={args.concurrency}, chunk_strategy={args.chunk_strategy}, "
          f"chunk_chars={args.chunk_chars}, max_tokens={args.max_tokens}")
    print(f"  est per-call reservation ~{est_per_call} tok "
          f"(must be < model's TPM cap or you'll get 413s)")

    contracts = json.loads(Path(args.data).read_text(encoding="utf-8"))["data"]
    print(f"  {len(contracts)} contracts loaded")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "nbest_predictions_.json"
    checkpoint_path = out_dir / "_checkpoint.json"
    nbest = load_checkpoint(checkpoint_path)

    t0, n_qas_done = time.time(), 0

    for ci, contract in enumerate(contracts):
        title, ctx = contract["title"], contract["paragraphs"][0]["context"]
        qas = contract["paragraphs"][0]["qas"]

        # Only the QAs whose category we're testing (all 41, or just one).
        target_qas = [qa for qa in qas
                      if qa["id"].split("__")[-1] in label_set]

        # Resume: contracts are processed atomically, so if every target QA
        # is already in nbest we can skip the whole contract.
        if target_qas and all(qa["id"] in nbest for qa in target_qas):
            n_qas_done += len(target_qas)
            if args.limit and n_qas_done >= args.limit:
                break
            continue

        n_chunks = len(make_chunks(ctx, strategy=args.chunk_strategy,
                                   size=args.chunk_chars, overlap=args.overlap))
        print(f"\n[{ci+1}/{len(contracts)}] {title[:60]}  "
              f"({len(ctx):,} chars, {n_chunks} chunks)")

        contract_start = time.time()
        try:
            predictions = await extract_contract(
                chain, field_to_label, ctx, all_labels, args.concurrency,
                chunk_chars=args.chunk_chars, overlap=args.overlap,
                strategy=args.chunk_strategy,
            )
        except FatalAPIError as e:
            # Out of API budget or auth broke — save what we have and bail.
            # The in-flight contract isn't in `nbest` yet, so resume re-tries it.
            print(f"\n!! Groq API unusable: {e}")
            print(f"   Saved {len(nbest)} QAs to checkpoint. Re-run the same "
                  f"command to resume.")
            save_checkpoint(nbest, checkpoint_path)
            raise SystemExit(1)
        except Exception as e:
            # Anything else: one weird contract can't kill the whole run.
            print(f"  contract failed: {e}")
            predictions = {label: [] for label in all_labels}

        # Flatten to nbest: {qa_id: [{"text": ..., "probability": ...}, ...]}
        # Only the categories under test get written.
        for qa in target_qas:
            label = qa["id"].split("__")[-1]
            spans = list(dict.fromkeys(predictions.get(label, [])))  # dedupe
            nbest[qa["id"]] = (
                [{"text": s, "probability": 1.0} for s in spans]
                + [{"text": "", "probability": 0.0}]                 # no-answer sentinel
            )
            n_qas_done += 1

        save_checkpoint(nbest, checkpoint_path)

        elapsed = time.time() - contract_start
        eta = (time.time() - t0) / (ci + 1) * (len(contracts) - ci - 1)
        n_hit = sum(1 for p in predictions.values() if p)
        print(f"  done in {elapsed:.1f}s, {n_hit}/{len(all_labels)} labels hit, "
              f"eta {eta/60:.1f}min")

        if args.limit and n_qas_done >= args.limit:
            print(f"\nReached --limit {args.limit}, stopping")
            break

    final_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(nbest)} QA predictions to {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nNext: python evaluate.py --model_path {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing — add it to .env or your shell.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="test.json")
    ap.add_argument("--model", default="llama-3.3-70b-versatile",
                    help="Any Groq-hosted model id (llama-3.3-70b-versatile, "
                         "openai/gpt-oss-120b, deepseek-r1-distill-llama-70b, ...).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on QAs to process. Useful for smoke tests.")
    ap.add_argument("--out",   default=None,
                    help="Output directory. Defaults to trained_models/<model>.")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                    help=f"Max parallel in-flight calls (default {MAX_CONCURRENCY}). "
                         "Lower if you keep hitting rate limits even with retry.")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"Per-call output token cap (default {DEFAULT_MAX_TOKENS}). "
                         "Groq reserves max_tokens against TPM, so lowering this "
                         "shrinks per-call reservation. 41-field JSON is mostly "
                         "nulls — 1500 is usually plenty.")
    ap.add_argument("--chunk_chars", type=int, default=CHUNK_CHARS,
                    help=f"Characters per chunk (default {CHUNK_CHARS}). Smaller "
                         "chunks shrink per-call input but produce more chunks. "
                         "For 6K-TPM models (llama-3.1-8b-instant, gpt-oss-20b) "
                         "use 4000-5000.")
    ap.add_argument("--overlap", type=int, default=OVERLAP,
                    help=f"Chunk overlap in chars (default {OVERLAP}). Only the "
                         "'fixed' strategy (and oversized-section fallback) use it.")
    ap.add_argument("--chunk_strategy", default=CHUNK_STRATEGY,
                    choices=["fixed", "recursive", "section"],
                    help=f"How to cut a contract into LLM-sized pieces "
                         f"(default {CHUNK_STRATEGY!r}). 'fixed'=blind windows "
                         "(old behaviour); 'recursive'=split on paragraph/line/"
                         "sentence boundaries; 'section'=split on contract "
                         "headers (ARTICLE/Section/1.1/(a)/ALL-CAPS) and pack "
                         "whole sections.")
    ap.add_argument("--category", default="all",
                    help='Which CUAD category to test: "all" (default, the full '
                         '41-field schema) or one category name, e.g. '
                         '"License Grant". A single category yields a 1-field '
                         'schema — far cheaper per call.')
    args = ap.parse_args()

    if args.out is None:
        model_slug = args.model.replace(":", "-").replace("/", "_")
        args.out = f"trained_models/{model_slug}"
        # Isolate single-category runs so they don't overwrite an "all" run.
        if args.category.lower() != "all":
            cat_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.category).strip("_")
            args.out += f"__{cat_slug}"

    asyncio.run(run(args))