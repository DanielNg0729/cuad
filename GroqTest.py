"""
Section-aware CUAD extraction with a Groq-hosted LLM (via LangChain).

Instead of cutting a contract into blind fixed-width windows, we split it on its
own structure -- ARTICLE / Section / 1.1 / (a) / ALL-CAPS headings -- keep each
header glued to its body, and pack whole sections into chunks up to a size cap.
This keeps clauses intact and headers attached to the text they label, which is
exactly what the per-field extraction needs.

Output is the nbest format consumed by evaluate.py:
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
from langchain_groq import ChatGroq
from pydantic import Field, create_model

CATEGORY_CSV = "category_descriptions.csv"
MAX_TOKENS = 1200
CONCURRENCY = 1             # parallel in-flight calls; raise only if not rate-limited
# Safety net only: a section larger than this can't be sent (Groq rejects it
# with 413 "request too large"), so we split just those rare giants. ~97% of
# real sections are far smaller and pass through whole. ~16k chars ≈ 4k tokens.
MAX_CHARS = 16000

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
    llm = ChatGroq(model=model, temperature=0, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    return prompt | llm.with_structured_output(schema)


# --- Section-aware chunking -------------------------------------------------

# A line is a section header if it looks like: ARTICLE IV / Section 12.3 /
# "1." "1.1" "12.3.4" / "(a)" "(iv)" / an ALL-CAPS heading line.
# Only TOP-LEVEL section starts -- not sub-clauses. We deliberately do NOT match
# "1.1", "1.2.3", "(a)", "(iv)" etc., so a whole section (with all its
# sub-paragraphs) stays together in one chunk.
_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLCDM]+\b"  # ARTICLE IV / Section 12
    r"|[0-9]{1,2}\.(?!\d)\s"                                  # top-level "1. " / "12. " (not 1.1)
    r"|[A-Z][A-Z0-9 ,;:'&/().\-]{3,}\s*$"                     # ALL-CAPS heading line
    r")"
)


def _split_oversized(text: str, limit: int) -> list[str]:
    """Split ONE section that's too big to send, losslessly, on the best
    available boundary (paragraph > line > word). Only the rare giant section
    ever reaches here."""
    if len(text) <= limit:
        return [text]
    for sep in ("\n\n", "\n", " "):
        if sep not in text:
            continue
        parts = text.split(sep)
        out, buf = [], ""
        for j, part in enumerate(parts):
            piece = part + (sep if j < len(parts) - 1 else "")  # keep sep between, not after last
            if buf and len(buf) + len(piece) > limit:
                out.append(buf)
                buf = piece
            else:
                buf += piece
        if buf:
            out.append(buf)
        # A single piece may still exceed the limit -> recurse with a finer sep.
        result = []
        for c in out:
            result += [c] if len(c) <= limit else _split_oversized(c, limit)
        return result
    return [text[i:i + limit] for i in range(0, len(text), limit)]  # no separators at all


def chunk_by_section(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split the contract on its top-level headers (ARTICLE / Section / "1." /
    ALL-CAPS), keeping each header glued to the body beneath it. One section =
    one chunk; only a section bigger than `max_chars` is split further so it can
    actually be sent. Lossless: ''.join(result) == text exactly."""
    sections, cur = [], []
    for line in text.splitlines(keepends=True):
        if cur and _HEADER_RE.match(line.strip()):
            sections.append("".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        sections.append("".join(cur))

    chunks = []
    for sec in (sections or [text]):
        chunks += _split_oversized(sec, max_chars)
    return chunks


# --- Extraction -------------------------------------------------------------

def validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """Keep only spans that literally appear in the chunk (raw extractions)."""
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if span and span in chunk:
        return span
    if span and span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx:idx + len(span)]
    return None


_RETRY_RE = re.compile(r"try again in\s+(?:(\d+)m\s*)?([\d.]+)\s*s", re.IGNORECASE)


async def call_with_retry(chain, inputs: dict, retries: int = 5):
    """ainvoke with reactive backoff: on a 429, sleep exactly as long as Groq's
    'try again in ...' hint asks, then retry."""
    for attempt in range(retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:
            m = _RETRY_RE.search(str(e))
            if m is None or attempt == retries:
                raise
            wait = float(m.group(2)) + (int(m.group(1)) * 60 if m.group(1) else 0)
            print(f"  rate-limited, sleeping {wait:.1f}s (attempt {attempt + 1})")
            await asyncio.sleep(wait + 0.5)


async def extract_contract(chain, field_to_label, context: str,
                           labels: list[str], concurrency: int) -> dict[str, list[str]]:
    """One LLM call per section-chunk (capped by `concurrency`); collect the
    validated spans per category."""
    sem = asyncio.Semaphore(concurrency)
    chunks = chunk_by_section(context)

    async def process(chunk: str):
        async with sem:
            try:
                return await call_with_retry(chain, {"chunk": chunk})
            except Exception as e:
                return e  # transient -> skip this chunk

    results = await asyncio.gather(*(process(c) for c in chunks))

    preds = {label: [] for label in labels}
    errors, first_err = 0, None
    for chunk, res in zip(chunks, results):
        if isinstance(res, Exception):
            errors += 1
            first_err = first_err or res
            continue
        if res is None:
            continue
        for fname, label in field_to_label.items():
            span = validate_span(getattr(res, fname, None), chunk)
            if span:
                preds[label].append(span)
    if errors:
        # Surface the real cause instead of silently returning zeros.
        print(f"  !! {errors}/{len(chunks)} chunks failed -> e.g. {str(first_err)[:180]}")
    return preds


# --- Runner -----------------------------------------------------------------

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
    label_set = set(field_to_label.values())
    chain = build_chain(args.model, schema, args.max_tokens)

    contracts = json.loads(Path(args.data).read_text(encoding="utf-8"))["data"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "_checkpoint.json"
    nbest = load_checkpoint(checkpoint_path)   # {} on a fresh run

    print(f"Groq section-aware eval | model={args.model} | {len(contracts)} contracts "
          f"| concurrency={args.concurrency} | already done: {len(nbest)} QAs")

    t0, done = time.time(), 0
    for ci, contract in enumerate(contracts):
        title = contract["title"]
        ctx = contract["paragraphs"][0]["context"]
        qas = contract["paragraphs"][0]["qas"]
        target_ids = [qa["id"] for qa in qas if qa["id"].split("__")[-1] in label_set]

        # Resume: contracts are written atomically, so if every target QA is
        # already in the checkpoint we skip the whole contract (no LLM calls).
        if target_ids and all(qid in nbest for qid in target_ids):
            done += len(target_ids)
            if args.limit and done >= args.limit:
                break
            continue

        n_chunks = len(chunk_by_section(ctx))
        print(f"\n[{ci + 1}/{len(contracts)}] {title[:60]} "
              f"({len(ctx):,} chars, {n_chunks} chunks)")

        preds = await extract_contract(
            chain, field_to_label, ctx, list(label_set), args.concurrency,
        )

        for qa in qas:
            label = qa["id"].split("__")[-1]
            if label not in label_set:
                continue
            spans = list(dict.fromkeys(preds.get(label, [])))  # dedupe, keep order
            nbest[qa["id"]] = (
                [{"text": s, "probability": 1.0} for s in spans]
                + [{"text": "", "probability": 0.0}]           # no-answer sentinel
            )
            done += 1

        save_checkpoint(nbest, checkpoint_path)   # persist after every contract

        hits = sum(1 for p in preds.values() if p)
        print(f"  {hits}/{len(label_set)} labels hit, {done} QAs total")
        if args.limit and done >= args.limit:
            print(f"\nReached --limit {args.limit}, stopping")
            break

    out_path = out_dir / "nbest_predictions_.json"
    out_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(nbest)} QA predictions to {out_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    if checkpoint_path.exists():       # finished cleanly -> drop the checkpoint
        checkpoint_path.unlink()
    print(f"Next: python evaluate.py --model_path {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing -- add it to .env or your shell.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test.json")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--limit", type=int, default=None, help="Cap QAs (smoke test).")
    ap.add_argument("--out", default=None, help="Output dir (default trained_models/<model>).")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    if args.out is None:
        args.out = "trained_models/" + args.model.replace(":", "-").replace("/", "_") + "__section"

    asyncio.run(run(args))
