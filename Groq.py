"""
Step 2 of the Groq pipeline.

Takes the pre-chunked file from chunking.py (test_chunking.json by default),
runs a Groq-hosted LLM over each contract's chunks with a zero-shot
structured-extraction prompt, and writes the predictions in the nbest format
that evaluate.py reads.

All the chunking happens in chunking.py now -- this file doesn't cut anything,
it just sends the chunks it's handed. If you want a different chunk strategy or
size, re-run chunking.py and point --data at its output.

You can run it on a single category or all 41 at once.

Pipeline:
    python chunking.py --strategy section          # -> test_chunking.json
    python Groq.py --data test_chunking.json --model <groq-model-id>
    python evaluate.py --model_path <out_dir>
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

MAX_CONCURRENCY = 1
DEFAULT_MAX_TOKENS = 1200
DESC_MAX_CHARS = 110
CATEGORY_CSV = "category_descriptions.csv"
DEFAULT_DATA = "test_chunking.json"   # produced by chunking.py (chunks only)
DEFAULT_GOLD = "test.json"            # original CUAD file: questions + gold answers

# The "copy verbatim" / "null if absent" rules sit in the system prompt once
# rather than getting repeated into all 41 field descriptions. That alone cut
# about 1000 tokens off every call.

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
    """Throw away any prediction that isn't actually in the chunk word for word.
    Even good models paraphrase now and then, and we only want raw extractions."""
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
How we sort API errors:
- rate-limit, quota and auth errors mean "stop everything and save", not just a
  failed contract
- everything else (timeouts, parse errors, ...) is just a chunk we skip

"""

class FatalAPIError(RuntimeError):
    """
    Means the Groq API is unusable. run() catches this, checkpoints, and exits
    cleanly so you can resume later.
    """

_FATAL_TYPES = {"RateLimitError", "AuthenticationError",
                "PermissionDeniedError", "InsufficientQuotaError"}
_FATAL_KEYWORDS = ("rate limit", "rate_limit", "quota", "insufficient_quota",
                   "invalid_api_key", "unauthorized", "429", "401", "403")

def is_fatal_api_error(exc: BaseException) -> bool:
    # True when the error means stop the run, not just skip a chunk
    if type(exc).__name__ in _FATAL_TYPES:
        return True
    msg = str(exc).lower()
    return any(k in msg for k in _FATAL_KEYWORDS)

# Rather than guess ahead of time how much we can send (which overshoots, since
# Groq bills actual output, not max_tokens), we let Groq tell us when it's full.
# Every 429 carries a "Please try again in <duration>" hint, so we parse that and
# sleep exactly that long. The duration shows up in two forms:
#   - TPM (tokens/minute):  "try again in 8.594999999s"
#   - TPD (tokens/day):     "try again in 34m12.864s"   (rarely with hours too)

_RETRY_AFTER_RE = re.compile(
    r"try again in\s+(?:(\d+)h\s*)?(?:(\d+)m\s*)?([\d.]+)\s*s",
    re.IGNORECASE,
)
MAX_RETRIES = 5                # how many times we retry a call before giving up
MAX_RETRY_WAIT_SECONDS = 3600 * 2   # 2h cap; longer than that, bail and resume later

def _retry_seconds(exc: BaseException) -> Optional[float]:
    """Dig the wait time out of a 429 message and turn it into seconds. Handles
    the 'Xs', 'Ym Zs' and 'Wh Ym Zs' shapes. Returns None if there's no
    rate-limit hint to parse."""
    m = _RETRY_AFTER_RE.search(str(exc))
    if not m:
        return None
    h, mn, s = m.groups()
    total = float(s)
    if mn: total += int(mn) * 60
    if h:  total += int(h) * 3600
    return total

async def _sleep_with_eta(seconds: float, reason: str):
    """Sleep, but say up front when we'll wake back up. On long waits print a
    halfway update so it doesn't look like the run hung."""
    wake_at = time.time() + seconds
    wake_str = time.strftime("%H:%M:%S", time.localtime(wake_at))
    print(f"  {reason}: sleeping {seconds/60:.1f} min, resume at ~{wake_str}")
    if seconds > 600:
        # Halfway heartbeat for waits over 10 min so you can tell it's alive.
        await asyncio.sleep(seconds / 2)
        remaining = wake_at - time.time()
        print(f"  ... still waiting, ~{remaining/60:.1f} min remaining")
        await asyncio.sleep(max(0, remaining))
    else:
        await asyncio.sleep(seconds)

_NON_RETRIABLE = {"AuthenticationError", "PermissionDeniedError",
                  "InsufficientQuotaError"}


async def call_with_retry(chain, inputs: dict, max_retries: int = MAX_RETRIES):
    """ainvoke with reactive backoff: sleep for exactly as long as Groq asks
    (TPM or TPD), then try again. Gives up on auth errors or any wait longer
    than MAX_RETRY_WAIT_SECONDS."""
    for attempt in range(max_retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:
            msg_low = str(e).lower()
            # Auth / quota / permission: waiting won't fix any of these.
            if type(e).__name__ in _NON_RETRIABLE:
                raise FatalAPIError(str(e)) from e
            # 413 "Request too large": one call is over the TPM cap by itself.
            # Resending the same payload will always fail, so bail right away and
            # let the user lower --chunk_chars / --max_tokens.
            if "413" in msg_low or "request too large" in msg_low:
                raise FatalAPIError(
                    f"Single request exceeds model's TPM cap (no retry can "
                    f"help). Lower --chunk_chars and/or --max_tokens, then "
                    f"resume.\n  {e}"
                ) from e
            wait_s = _retry_seconds(e)
            # If a 429 didn't come with a parseable hint, just wait 60s.
            if wait_s is None:
                if "429" not in msg_low and "rate_limit" not in msg_low:
                    raise   # not a rate-limit error at all
                wait_s = 60.0
            # If the wait is longer than our cap, it's better to quit and pick
            # it back up tomorrow than to sit here.
            if wait_s > MAX_RETRY_WAIT_SECONDS:
                raise FatalAPIError(
                    f"Groq asked to wait {wait_s/60:.0f}min "
                    f"(> {MAX_RETRY_WAIT_SECONDS/60:.0f}min cap). "
                    f"Probably out of daily tokens -- resume tomorrow."
                ) from e
            if attempt >= max_retries:
                raise FatalAPIError(
                    f"Rate-limited {max_retries}+ times in a row: {e}"
                ) from e
            sleep_s = wait_s + 0.5     # small cushion so we don't race the window
            await _sleep_with_eta(
                sleep_s,
                f"rate-limited (attempt {attempt+1}/{max_retries})",
            )


async def extract_contract(chain, field_to_label, chunk_list: list[str],
                           all_labels: list[str],
                           concurrency: int) -> dict[str, list[str]]:
    """One LLM call per chunk, run in parallel up to `concurrency`. Each call
    backs off on a 429 using Groq's own retry-after hint. The chunks come from
    chunking.py."""
    predictions = {label: [] for label in all_labels}
    sem = asyncio.Semaphore(concurrency)

    async def process(chunk: str):
        async with sem:
            try:
                return await call_with_retry(chain, {"chunk": chunk})
            except FatalAPIError:
                raise
            except Exception as e:
                return e   # transient -- the caller skips this chunk

    results = await asyncio.gather(
        *(process(c) for c in chunk_list),
        return_exceptions=True,
    )

    for chunk_text, result in zip(chunk_list, results):
        if isinstance(result, Exception):
            if is_fatal_api_error(result):
                raise FatalAPIError(str(result)) from result
            continue  # transient -- skip this chunk
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

    # The chunks come from chunking.py -- text only, keyed by contract_id. The
    # questions and gold answers live in the original test.json, and we join the
    # two back together here by contract_id, so the ground truth never sits next
    # to the chunks we feed the model.
    chunk_file = json.loads(Path(args.data).read_text(encoding="utf-8"))
    meta = chunk_file.get("metadata", {})
    chunk_entries = chunk_file["data"]
    if "chunks" not in (chunk_entries[0] if chunk_entries else {}):
        raise SystemExit(
            f"{args.data!r} is not a chunk file. Run chunking.py first:\n"
            f"  python chunking.py --strategy section"
        )
    chunks_by_id = {c["contract_id"]: c["chunks"] for c in chunk_entries}

    # The original test file gives us the qas (ids / questions / gold answers).
    # The link between the two files is contract_id == title == the part of a
    # qa id before "__".
    contracts = json.loads(Path(args.gold).read_text(encoding="utf-8"))["data"]

    # Rough guess at what each call reserves (Groq counts input + max_tokens
    # against the TPM limit). Measured at ~56 tokens of JSON scaffolding per
    # field. We size the estimate off the largest chunk chunking.py actually
    # produced, which is what matters for the 'section' strategy (no size cap).
    max_chunk_chars = meta.get("max_chunk_chars") or meta.get("chunk_chars") or 5000
    chunk_tok_est = max_chunk_chars // 4
    schema_tok_est = 56 * len(all_labels) + 100
    est_per_call = chunk_tok_est + schema_tok_est + args.max_tokens + 100
    scope = "all 41 categories" if args.category.lower() == "all" else f"category {args.category!r}"
    print(f"  testing {scope} ({len(all_labels)} field schema), "
          f"concurrency={args.concurrency}, max_tokens={args.max_tokens}")
    print(f"  chunks from {args.data}: strategy={meta.get('strategy', '?')!r}, "
          f"largest chunk {max_chunk_chars:,} chars, "
          f"{meta.get('total_chunks', '?')} chunks total")
    print(f"  est per-call reservation ~{est_per_call} tok "
          f"(must be < model's TPM cap or you'll get 413s)")
    print(f"  {len(contracts)} contracts loaded")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "nbest_predictions_.json"
    checkpoint_path = out_dir / "_checkpoint.json"
    nbest = load_checkpoint(checkpoint_path)

    t0, n_qas_done = time.time(), 0

    for ci, contract in enumerate(contracts):
        title = contract["title"]
        qas = contract["paragraphs"][0]["qas"]
        chunk_list = chunks_by_id.get(title)
        if chunk_list is None:
            print(f"\n[{ci+1}/{len(contracts)}] {title[:60]}  "
                  f"-- no chunks in {args.data} (re-run chunking.py); skipping")
            continue

        # Just the QAs for the category/categories we're testing (all 41, or one).
        target_qas = [qa for qa in qas
                      if qa["id"].split("__")[-1] in label_set]

        # Resume support: we write a contract all at once, so if every target QA
        # is already in nbest we can skip the whole thing.
        if target_qas and all(qa["id"] in nbest for qa in target_qas):
            n_qas_done += len(target_qas)
            if args.limit and n_qas_done >= args.limit:
                break
            continue

        ctx_chars = sum(len(c) for c in chunk_list)
        print(f"\n[{ci+1}/{len(contracts)}] {title[:60]}  "
              f"({ctx_chars:,} chars, {len(chunk_list)} chunks)")

        contract_start = time.time()
        try:
            predictions = await extract_contract(
                chain, field_to_label, chunk_list, all_labels, args.concurrency,
            )
        except FatalAPIError as e:
            # Out of budget or auth broke: save what we've got and stop. The
            # current contract isn't in `nbest` yet, so a resume will retry it.
            print(f"\n!! Groq API unusable: {e}")
            print(f"   Saved {len(nbest)} QAs to checkpoint. Re-run the same "
                  f"command to resume.")
            save_checkpoint(nbest, checkpoint_path)
            raise SystemExit(1)
        except Exception as e:
            # Anything else: don't let one odd contract take down the whole run.
            print(f"  contract failed: {e}")
            predictions = {label: [] for label in all_labels}

        # Flatten into nbest: {qa_id: [{"text": ..., "probability": ...}, ...]}.
        # Only the categories we're testing get written.
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

    # Finished cleanly, so we don't need the checkpoint anymore.
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nNext: python evaluate.py --model_path {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing -- add it to .env or your shell.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default=DEFAULT_DATA,
                    help=f"Pre-chunked file from chunking.py (default "
                         f"{DEFAULT_DATA!r}). Holds chunk text only, keyed by "
                         "contract_id. Re-run chunking.py to change the "
                         "chunking strategy / size.")
    ap.add_argument("--gold",  default=DEFAULT_GOLD,
                    help=f"Original CUAD file with questions + gold answers "
                         f"(default {DEFAULT_GOLD!r}), joined to the chunks by "
                         "contract_id. Gold answers are never sent to the LLM.")
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
                         "shrinks the per-call reservation. The 41-field JSON is "
                         "mostly nulls, so 1500 is usually plenty.")
    ap.add_argument("--category", default="all",
                    help='Which CUAD category to test: "all" (default, the full '
                         '41-field schema) or a single category name, e.g. '
                         '"License Grant". One category means a 1-field schema, '
                         'which is much cheaper per call.')
    args = ap.parse_args()

    if args.out is None:
        model_slug = args.model.replace(":", "-").replace("/", "_")
        args.out = f"trained_models/{model_slug}"
        # Keep single-category runs separate so they don't clobber an "all" run.
        if args.category.lower() != "all":
            cat_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.category).strip("_")
            args.out += f"__{cat_slug}"

    asyncio.run(run(args))