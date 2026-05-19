"""
Run a Groq-served LLM (Llama 3.x, Mixtral, Gemma, DeepSeek, Qwen, etc.)
against the CUAD test set using zero-shot prompting, and write predictions
in the nbest format expected by evaluate.py.

Mirrors Qwen34bModelTesting.py:
  - Same chunking + label-batching schema strategy
  - Same per-contract checkpointing
  - Same verbatim-substring validation

What's different:
  - LLM backend is langchain-groq's ChatGroq (cloud API) instead of Ollama.
  - Concurrency is bumped up (cloud calls are I/O-bound) but capped to stay
    inside Groq's per-minute rate limits.
  - API key is read from .env via python-dotenv.

Examples:
    python GroqModelTesting.py --model llama-3.3-70b-versatile
    python GroqModelTesting.py --model openai/gpt-oss-120b --limit 50
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


# Sliding window over each contract. Groq-hosted models all handle long
# contexts, but 6K char chunks keep latency and token cost predictable.
CHUNK_CHARS = 6000
OVERLAP = 400

# Concurrent in-flight requests. Cloud calls are I/O bound, but Groq enforces
# per-minute rate limits — 5 stays comfortably under the free tier.
MAX_CONCURRENCY = 5

# Output token cap. Generous because Groq models are fast and we want full JSON.
NUM_PREDICT = 800

# Labels per structured-output schema. Large hosted models (70B+) can handle
# more fields at once than local 1.7B-4B ones, so we batch a bit larger.
LABELS_PER_BATCH = 10

CATEGORY_CSV = "category_descriptions.csv"


# ---------------------------------------------------------------------------
# Category loading + schema construction (same shape as the Ollama script)
# ---------------------------------------------------------------------------

def load_categories(csv_path: str) -> dict[str, list[str]]:
    """Parse CUAD's category_descriptions.csv into
    {label: [description, answer_format, group]}."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    out = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).split("Category:", 1)[-1].strip()
        desc  = str(row.iloc[1]).split("Description:", 1)[-1].strip()
        fmt   = str(row.iloc[2]).split("Answer Format:", 1)[-1].strip()
        grp   = str(row.iloc[3]).split("Group:", 1)[-1].strip()
        out[label] = [desc, fmt, grp]
    return out


def label_to_field(label: str) -> str:
    """Human label -> snake_case identifier safe for Pydantic field names."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def build_extraction_schemas(categories, batch_size=LABELS_PER_BATCH):
    """One Pydantic class per batch of labels.

    Even though hosted models are stronger than local ones, splitting into
    smaller schemas still helps: fewer fields per call means more reliable
    JSON, and a single bad batch only loses ~10 labels' worth of work.
    """
    items = list(categories.items())
    schemas = []
    for batch_idx, start in enumerate(range(0, len(items), batch_size)):
        fields, field_to_label = {}, {}
        for label, (desc, fmt, _) in items[start:start + batch_size]:
            fname = label_to_field(label)
            field_to_label[fname] = label
            fields[fname] = (
                Optional[str],
                Field(
                    default=None,
                    description=(
                        f"{desc}. Expected format: {fmt}. "
                        "Return the exact verbatim substring from the section "
                        "if present, or null if not in this section."
                    ),
                ),
            )
        SchemaCls = create_model(f"ContractExtractionBatch{batch_idx}", **fields)
        schemas.append((SchemaCls, field_to_label))
    return schemas


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def build_chains(model: str, schemas: list) -> list:
    """Build one LCEL chain (prompt | ChatGroq.with_structured_output) per
    schema batch. Each chain returns a Pydantic instance directly."""
    chains = []
    for schema, _ in schemas:
        llm = ChatGroq(
            model=model,
            temperature=0,
            max_tokens=NUM_PREDICT,
        ).with_structured_output(schema)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a legal contract analyst. Given ONE SECTION of a contract, "
             "for each field in the schema return the EXACT verbatim substring "
             "from the section that matches the field's description. Copy "
             "character-by-character. If a field's category doesn't appear in "
             "the section, set it to null. Do NOT paraphrase. Do NOT invent text."),
            ("human",
             "Contract section:\n"
             "\"\"\"\n{chunk}\n\"\"\"\n\n"
             "Extract the requested categories. Set fields to null where the "
             "category is not present in the section."),
        ])
        chains.append(prompt | llm)
    return chains


# ---------------------------------------------------------------------------
# Chunking + validation
# ---------------------------------------------------------------------------

def chunks(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP):
    """Yield overlapping windows. Overlap exists so a span that straddles a
    chunk boundary still appears whole in at least one chunk."""
    if len(text) <= size:
        yield text
        return
    i = 0
    while i < len(text):
        yield text[i:i + size]
        if i + size >= len(text):
            break
        i += size - overlap


def validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """Drop any predicted span that doesn't literally appear in the chunk.
    Even strong models occasionally paraphrase; we only want raw extractions."""
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if not span:
        return None
    if span in chunk:
        return span
    # Case-insensitive fallback: re-fetch the original casing.
    if span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx:idx + len(span)]
    return None


# ---------------------------------------------------------------------------
# Per-contract extraction
# ---------------------------------------------------------------------------

async def extract_contract(chains, schemas, context, all_labels):
    """Run every schema batch across every chunk and collect surviving spans
    per label."""
    chunk_list = list(chunks(context))
    predictions = {label: [] for label in all_labels}

    for batch_idx, (chain, (_, field_to_label)) in enumerate(zip(chains, schemas)):
        inputs = [{"chunk": c} for c in chunk_list]
        try:
            results = await chain.abatch(
                inputs,
                config={"max_concurrency": MAX_CONCURRENCY},
                return_exceptions=True,
            )
        except Exception as e:
            print(f"      batch {batch_idx} died entirely: {e}")
            continue

        for chunk_text, result in zip(chunk_list, results):
            if isinstance(result, Exception) or result is None:
                continue
            for fname, label in field_to_label.items():
                valid = validate_span(getattr(result, fname, None), chunk_text)
                if valid:
                    predictions[label].append(valid)
    return predictions


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(nbest, path: Path):
    path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        print(f"Resuming from checkpoint at {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def run(args):
    print(f"Running CUAD eval with Groq model: {args.model}")
    print(f"  data:             {args.data}")
    print(f"  limit:            {args.limit or 'full set'}")
    print(f"  labels per batch: {LABELS_PER_BATCH}")
    print(f"  concurrency:      {MAX_CONCURRENCY}")
    print(f"  output:           {args.out}")

    categories = load_categories(CATEGORY_CSV)
    all_labels = list(categories.keys())
    schemas = build_extraction_schemas(categories)
    chains = build_chains(args.model, schemas)
    print(f"  {len(all_labels)} labels -> {len(schemas)} schema batches")

    contracts = json.loads(Path(args.data).read_text(encoding="utf-8"))["data"]
    print(f"  {len(contracts)} contracts")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "nbest_predictions_.json"
    checkpoint_path = out_dir / "_checkpoint.json"
    nbest = load_checkpoint(checkpoint_path)

    t0 = time.time()
    n_qas_done = 0

    for ci, contract in enumerate(contracts):
        title = contract["title"]
        ctx = contract["paragraphs"][0]["context"]
        qas = contract["paragraphs"][0]["qas"]

        # Resume: contracts are processed atomically, so the first qa_id is a
        # safe sentinel for "already done".
        if qas[0]["id"] in nbest:
            n_qas_done += len(qas)
            if args.limit and n_qas_done >= args.limit:
                break
            continue

        n_chunks = len(list(chunks(ctx)))
        n_calls = n_chunks * len(schemas)
        print(f"\n[{ci+1}/{len(contracts)}] {title[:60]}")
        print(f"  {len(ctx):,} chars, {n_chunks} chunks * {len(schemas)} batches "
              f"= {n_calls} calls")

        contract_start = time.time()
        try:
            predictions = await extract_contract(chains, schemas, ctx, all_labels)
        except Exception as e:
            # Catch-all so one weird contract can't kill the whole run.
            print(f"  contract failed: {e}")
            predictions = {label: [] for label in all_labels}

        # Flatten into nbest format: {qa_id: [{"text": ..., "probability": ...}]}
        for qa in qas:
            label = qa["id"].split("__")[-1]
            spans = list(dict.fromkeys(predictions.get(label, [])))  # dedupe
            entries = [{"text": s, "probability": 1.0} for s in spans]
            entries.append({"text": "", "probability": 0.0})  # "no answer" sentinel
            nbest[qa["id"]] = entries
            n_qas_done += 1

        save_checkpoint(nbest, checkpoint_path)

        elapsed = time.time() - contract_start
        total_elapsed = time.time() - t0
        remaining = (total_elapsed / (ci + 1)) * (len(contracts) - ci - 1)
        n_with_spans = sum(1 for p in predictions.values() if p)
        print(f"  done in {elapsed:.1f}s ({elapsed/max(n_calls,1):.1f}s/call), "
              f"{n_with_spans}/{len(all_labels)} labels hit, "
              f"eta {remaining/60:.1f}min")

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
    ap.add_argument("--data", default="test.json")
    ap.add_argument("--model", default="llama-3.3-70b-versatile",
                    help="Any model id served by Groq (e.g. llama-3.3-70b-versatile, "
                         "openai/gpt-oss-120b, deepseek-r1-distill-llama-70b).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on QAs to process. Useful for smoke tests.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to trained_models/<model>.")
    args = ap.parse_args()

    if args.out is None:
        model_clean = args.model.replace(":", "-").replace("/", "_")
        args.out = f"trained_models/{model_clean}"

    asyncio.run(run(args))
