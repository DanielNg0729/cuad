"""
Run a HuggingFace-loaded model (e.g. Qwen3-4B) against the CUAD test set
and write predictions in the same nbest format the paper's evaluate.py
expects.

The interesting bits are:
  - We pose all 41 label questions per chunk instead of one-at-a-time,
    so a single contract takes ~N_chunks LLM calls instead of ~41 * N_chunks.
  - The 41 fields are split into smaller Pydantic schemas (LABELS_PER_BATCH
    at a time) because small local models can't reliably emit a single
    41-field JSON object — they start truncating or hallucinating keys.
  - Every predicted span is checked against the chunk text. If the model
    paraphrased instead of copying verbatim, the prediction is dropped.
  - Progress is checkpointed per contract so a crash mid-run doesn't lose
    hours of work — just rerun the same command and it picks up.

Difference from the LangChain version: we load Qwen3 directly into GPU
memory via HuggingFace Transformers, build the prompt by hand, and parse
the model's text output into Pydantic ourselves. The public functions and
the data flow are identical — only build_chains() has different innards.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from pydantic import Field, ValidationError, create_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# Sliding window over each contract. 6K chars is ~1.5K tokens, which leaves
# headroom for the system prompt + JSON schema + response on a 4GB GPU.
CHUNK_CHARS = 6000
OVERLAP = 400

# One concurrent generation. PyTorch generate() is single-call anyway;
# we keep the variable for parity with the LangChain version.
MAX_CONCURRENCY = 1

# Output token cap. Pydantic structured output emits JSON, and 8 short
# fields fits comfortably in 600 tokens.
NUM_PREDICT = 600

# How many labels we ask the model about in a single call. Small local
# models (1.7B-class) start producing malformed JSON when this gets too
# high. 8 worked for me with qwen3:1.7b and qwen3:4b. Drop to 5 if you
# see "Invalid json output" errors regularly.
LABELS_PER_BATCH = 8

CATEGORY_CSV = "category_descriptions.csv"


def load_categories(csv_path: str) -> dict[str, list[str]]:
    """Parse the CUAD category_descriptions.csv.

    The CSV has four columns ("Category: X", "Description: Y", "Answer
    Format: Z", "Group: W") with the prefix repeated on every row, so
    we strip the prefix off each cell.

    Returns {label_name: [description, answer_format, group]}.
    """
    # utf-8-sig handles the BOM Excel likes to add on Windows
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    result = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).split("Category:", 1)[-1].strip()
        desc  = str(row.iloc[1]).split("Description:", 1)[-1].strip()
        fmt   = str(row.iloc[2]).split("Answer Format:", 1)[-1].strip()
        grp   = str(row.iloc[3]).split("Group:", 1)[-1].strip()
        result[label] = [desc, fmt, grp]
    return result


def label_to_field(label: str) -> str:
    """Turn a human label like "No-Solicit Of Customers" into a Python
    identifier ("no_solicit_of_customers") so Pydantic accepts it as a
    field name."""
    s = label.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def build_extraction_schemas(
    categories: dict[str, list[str]],
    batch_size: int = LABELS_PER_BATCH,
) -> list[tuple[type, dict[str, str]]]:
    """Build one Pydantic class per batch of labels.

    Asking a small model for 41 JSON fields in one shot is unreliable —
    the model will truncate, drop keys, or emit invalid JSON. Splitting
    into ~5-6 smaller schemas trades a few extra LLM calls for parser
    sanity.

    Each label becomes an Optional[str] field whose description carries
    the original label description and expected answer format. With
    LangChain + Ollama this gets fed to the model via JSON schema
    enforcement; here we paste the schema into the system prompt and
    rely on the model to follow it.

    Returns a list of (schema_class, field_name_to_label) pairs.
    """
    items = list(categories.items())
    schemas = []

    for batch_idx, start in enumerate(range(0, len(items), batch_size)):
        batch_items = items[start : start + batch_size]
        fields = {}
        field_to_label = {}

        for label, info in batch_items:
            desc, fmt, _ = info
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

        # Distinct class name per batch so any traceback points at the right one
        SchemaCls = create_model(f"ContractExtractionBatch{batch_idx}", **fields)
        schemas.append((SchemaCls, field_to_label))

    return schemas


class GenerationChain:
    """Drop-in stand-in for an LCEL chain.

    Exposes the same .invoke() and .abatch() shape that LangChain chains
    have, so extract_contract() doesn't need to know whether it's talking
    to Ollama or to a local HF model. Internally it does the four steps
    LangChain hides: format the chat with the model's template, tokenize,
    generate, decode, then parse the JSON output back into the Pydantic
    schema.
    """

    SYSTEM_TEMPLATE = (
        "You are a legal contract analyst. Given ONE SECTION of a contract, "
        "for each field in the schema return the EXACT verbatim substring "
        "from the section that matches the field's description. Copy "
        "character-by-character. If a field's category doesn't appear in "
        "the section, set it to null. Do NOT paraphrase. Do NOT invent text.\n\n"
        "Return JSON matching this exact schema:\n{schema}\n"
        "Output ONLY valid JSON. No markdown fences, no commentary, no preamble."
    )

    USER_TEMPLATE = (
        "Contract section:\n"
        '"""\n{chunk}\n"""\n\n'
        "Extract the requested categories. Set fields to null where "
        "the category is not present in the section."
    )

    def __init__(self, schema: type, tokenizer, model):
        self.schema = schema
        self.tokenizer = tokenizer
        self.model = model
        # Pre-render the schema string once so we don't pay it on every call
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        self.system_prompt = self.SYSTEM_TEMPLATE.format(schema=schema_str)

    @torch.inference_mode()
    def invoke(self, inputs: dict):
        """Run one generation. Returns a Pydantic instance or raises."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.USER_TEMPLATE.format(chunk=inputs["chunk"])},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = ids.input_ids.shape[1]

        out_ids = self.model.generate(
            **ids,
            max_new_tokens=NUM_PREDICT,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # generate() returns prompt + completion, so slice off the prompt
        text = self.tokenizer.decode(
            out_ids[0][prompt_len:], skip_special_tokens=True,
        )

        return self._parse(text)

    def _parse(self, text: str):
        """Pull a JSON object out of the model's free-text output and
        validate it against the schema. The model usually obeys but
        sometimes wraps output in markdown fences or adds commentary."""
        # Strip common wrappers
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"\s*```\s*$", "", text)

        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Invalid json output: no object found in {text[:100]!r}")

        try:
            data = json.loads(text[start : end + 1])
            return self.schema(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid json output: {e}") from e

    async def abatch(self, inputs_list, config=None, return_exceptions=False):
        """Run a list of inputs sequentially. config is accepted for
        signature parity with LangChain but ignored — PyTorch generate()
        is one-call-at-a-time on a single GPU."""
        results = []
        for inputs in inputs_list:
            try:
                results.append(self.invoke(inputs))
            except Exception as e:
                results.append(e if return_exceptions else None)
        return results


def build_chains(model: str, schemas: list) -> list:
    """Spin up one chain per schema batch.

    With LangChain we got a fresh ChatOllama per batch and let the server
    share the model. Here we load Qwen3 into PyTorch once and share the
    same tokenizer + model object across all batch chains (the schema
    differs but the underlying weights don't).

    4-bit quantization via bitsandbytes brings a 4B FP16 model from ~8GB
    down to ~2.5GB so it fits on a 4GB GPU. If you have more headroom,
    swap quantization_config for torch_dtype=torch.float16.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    hf_model.eval()

    chains = []
    for schema, _ in schemas:
        chains.append(GenerationChain(schema, tokenizer, hf_model))

    return chains


def chunks(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP):
    """Yield overlapping chunks. Overlap exists so a span that happens
    to straddle a chunk boundary still appears whole in at least one
    chunk."""
    if len(text) <= size:
        yield text
        return
    i = 0
    while i < len(text):
        yield text[i : i + size]
        if i + size >= len(text):
            break
        i += size - overlap


def validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """Reject any predicted span that doesn't literally appear in the
    chunk. Small models love to "helpfully" rewrite quotes — we want
    raw extractions only.

    Returns the cleaned span (with surrounding quote chars stripped) or
    None if the prediction doesn't survive the check.
    """
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if not span:
        return None
    if span in chunk:
        return span
    # Case-insensitive fallback: re-fetch the original casing from the chunk
    if span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx : idx + len(span)]
    return None


async def extract_contract(
    chains: list,
    schemas: list,
    context: str,
    all_labels: list[str],
) -> dict[str, list[str]]:
    """Run every schema batch across every chunk of one contract and
    collect surviving spans per label.

    Total LLM calls = num_chunks * num_batches. For an average contract
    that's ~5 * 6 = ~30 calls. Big contracts that produce 50+ chunks can
    blow this up, which is why we checkpoint after each contract.
    """
    chunk_list = list(chunks(context))
    predictions = {label: [] for label in all_labels}

    for batch_idx, (chain, (_, field_to_label)) in enumerate(zip(chains, schemas)):
        inputs = [{"chunk": c} for c in chunk_list]

        # return_exceptions=True so a single chunk's JSON parse failure
        # doesn't kill the rest of the batch
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
                # Bad JSON, timeout, or model refused — skip and move on
                continue

            for fname, label in field_to_label.items():
                raw_span = getattr(result, fname, None)
                valid_span = validate_span(raw_span, chunk_text)
                if valid_span:
                    predictions[label].append(valid_span)

    return predictions


def save_checkpoint(nbest: dict, checkpoint_path: Path):
    checkpoint_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint at {checkpoint_path}")
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {}


async def run(args):
    print(f"Running CUAD eval with {args.model} (PyTorch backend)")
    print(f"  data:             {args.data}")
    print(f"  limit:            {args.limit or 'full set'}")
    print(f"  labels per batch: {LABELS_PER_BATCH}")
    print(f"  output:           {args.out}")

    print(f"\nLoading categories from {CATEGORY_CSV}")
    categories = load_categories(CATEGORY_CSV)
    print(f"  loaded {len(categories)} labels")

    schemas = build_extraction_schemas(categories, batch_size=LABELS_PER_BATCH)
    all_labels = list(categories.keys())
    print(f"  split into {len(schemas)} schema batches "
          f"(~{LABELS_PER_BATCH} labels each)")

    chains = build_chains(args.model, schemas)

    print(f"\nLoading test data")
    ds = json.loads(Path(args.data).read_text(encoding="utf-8"))
    contracts = ds["data"]
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

        # Resume: if we've already processed any QA from this contract,
        # skip the whole thing (we always do contracts atomically)
        first_qa_id = contract["paragraphs"][0]["qas"][0]["id"]
        if first_qa_id in nbest:
            n_qas_done += len(contract["paragraphs"][0]["qas"])
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
            # Catch-all so one weird contract can't kill the whole run
            print(f"  contract failed: {e}")
            predictions = {label: [] for label in all_labels}

        # Flatten the per-label predictions into the nbest format that
        # evaluate.py expects: {qa_id: [{"text": str, "probability": float}, ...]}
        for qa in contract["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            label = qa_id.split("__")[-1]
            spans = predictions.get(label, [])
            spans = list(dict.fromkeys(spans))  # dedupe, preserve order
            entries = [{"text": s, "probability": 1.0} for s in spans]
            # The empty option is what evaluate.py uses as the "no answer" prediction
            entries.append({"text": "", "probability": 0.0})
            nbest[qa_id] = entries
            n_qas_done += 1

        save_checkpoint(nbest, checkpoint_path)

        elapsed = time.time() - contract_start
        total_elapsed = time.time() - t0
        avg_per_contract = total_elapsed / (ci + 1)
        remaining = avg_per_contract * (len(contracts) - ci - 1)
        n_with_spans = sum(1 for p in predictions.values() if p)
        print(f"  done in {elapsed:.1f}s ({elapsed/n_calls:.1f}s/call), "
              f"{n_with_spans}/{len(all_labels)} labels hit, "
              f"eta {remaining/60:.1f}min")

        if args.limit and n_qas_done >= args.limit:
            print(f"\nReached --limit {args.limit}, stopping")
            break

    final_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(nbest)} QA predictions to {final_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

    # Checkpoint and final file are redundant once we've written the final
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nNext: point model_path in evaluate.py at '{out_dir}' and run it")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B",
                    help="HuggingFace model ID (not an Ollama tag)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on QAs to process. Useful for smoke tests.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to trained_models/<model>.")
    args = ap.parse_args()

    # Derive a clean folder name from the model if the user didn't say
    if args.out is None:
        model_clean = args.model.replace("/", "_").replace(":", "-")
        args.out = f"trained_models/{model_clean}"

    asyncio.run(run(args))
