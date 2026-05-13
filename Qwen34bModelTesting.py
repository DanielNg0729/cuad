"""
CUAD eval using Ollama + LangChain.

V3 changes vs V2:
  • Schema được CHIA thành nhiều batches nhỏ (mặc định 8 labels/batch)
    → 1.7b model giờ output JSON valid được
    → 4b model cũng reliable hơn
  • Per-batch fault tolerance: nếu 1 batch parse fail, các batch khác vẫn save data
  • Output dir auto-derived từ model name nếu không có --out

Output: <out>/nbest_predictions_.json (paper format → dùng evaluate.py)
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import Field, create_model


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CHUNK_CHARS = 6000          # Size of each chunk
OVERLAP = 400               # Overlap between chunks
MAX_CONCURRENCY = 1         # Concurrent calls (fit GPU 4GB)
NUM_PREDICT = 600           # Max output tokens per call
LABELS_PER_BATCH = 8        # ★ KEY KNOB ★ Labels per LLM call.
                            #    Lower = more reliable, slower (more calls).
                            #    Higher = faster, but small models may fail JSON.
                            #    8 works for both qwen3:1.7b and qwen3:4b.
                            #    Try 5 if 1.7b still fails JSON often.
CATEGORY_CSV = "category_descriptions.csv"


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD 41 LABELS FROM CSV
# ═══════════════════════════════════════════════════════════════════════════
def load_categories(csv_path: str) -> dict[str, list[str]]:
    """Read paper CSV → {label: [desc, fmt, group]}"""
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
    """Convert label name to valid Python identifier for Pydantic."""
    s = label.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD MULTIPLE SCHEMAS (★ NEW IN V3 ★)
# ═══════════════════════════════════════════════════════════════════════════
def build_extraction_schemas(
    categories: dict[str, list[str]],
    batch_size: int = LABELS_PER_BATCH,
) -> list[tuple[type, dict[str, str]]]:
    """
    Chia 41 labels thành các batches nhỏ, mỗi batch = 1 Pydantic schema.
    
    Tại sao? Small LLMs (1.7b) không reliably output JSON 41-fields.
    Chia thành ~5 schemas có ~8 fields → mỗi call output JSON đơn giản
    → ít fail parse hơn nhiều.
    
    Returns: list of (schema_class, field_to_label_dict)
    """
    items = list(categories.items())
    schemas = []
    
    for batch_idx, start in enumerate(range(0, len(items), batch_size)):
        batch_items = items[start : start + batch_size]
        fields = {}
        field_to_label = {}
        
        for label, info in batch_items:
            desc, fmt, grp = info
            fname = label_to_field(label)
            field_to_label[fname] = label
            fields[fname] = (
                Optional[str],
                Field(
                    default=None,
                    description=(
                        f"{desc}. Expected format: {fmt}. "
                        f"Return the EXACT verbatim substring from the section "
                        f"if present, or null if not in this section."
                    )
                )
            )
        
        SchemaCls = create_model(f"ContractExtractionBatch{batch_idx}", **fields)
        schemas.append((SchemaCls, field_to_label))
    
    return schemas


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: BUILD CHAINS — one per batch (★ MODIFIED IN V3 ★)
# ═══════════════════════════════════════════════════════════════════════════
def build_chains(model: str, schemas: list) -> list:
    """Build separate LCEL chain for each schema batch."""
    chains = []
    for schema, _ in schemas:
        llm = ChatOllama(
            model=model,
            temperature=0,
            num_predict=NUM_PREDICT,
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
             "Extract the requested categories. Set fields to null when "
             "the category is not present in the section."),
        ])
        chains.append(prompt | llm)
    
    return chains


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: HELPERS (unchanged from V2)
# ═══════════════════════════════════════════════════════════════════════════
def chunks(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP):
    """Slide window through long text."""
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
    """Anti-hallucination: span must literally appear in chunk."""
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if not span:
        return None
    if span in chunk:
        return span
    if span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx : idx + len(span)]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: EXTRACT CONTRACT (★ MODIFIED IN V3 — loops over batches ★)
# ═══════════════════════════════════════════════════════════════════════════
async def extract_contract(
    chains: list,
    schemas: list,
    context: str,
    all_labels: list[str],
) -> dict[str, list[str]]:
    """
    Run all schema batches on all chunks.
    
    Total LLM calls = N_chunks × N_batches
    Ví dụ 5 chunks × 6 batches = 30 calls (vs V2: 5, vs V1: 205)
    
    Returns: {label: [valid spans found]}
    """
    chunk_list = list(chunks(context))
    predictions = {label: [] for label in all_labels}
    
    # Loop từng batch (mỗi batch xử lý ~8 labels)
    for batch_idx, (chain, (_, field_to_label)) in enumerate(zip(chains, schemas)):
        inputs = [{"chunk": c} for c in chunk_list]
        
        # return_exceptions=True: 1 chunk fail không kill cả batch
        try:
            results = await chain.abatch(
                inputs,
                config={"max_concurrency": MAX_CONCURRENCY},
                return_exceptions=True,
            )
        except Exception as e:
            print(f"      [batch {batch_idx}] entire batch failed: {e}")
            continue
        
        # Aggregate spans cho từng label trong batch
        for chunk_text, result in zip(chunk_list, results):
            if isinstance(result, Exception) or result is None:
                continue  # chunk này LLM lỗi parse JSON, skip
            
            for fname, label in field_to_label.items():
                raw_span = getattr(result, fname, None)
                valid_span = validate_span(raw_span, chunk_text)
                if valid_span:
                    predictions[label].append(valid_span)
    
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════
def save_checkpoint(nbest: dict, checkpoint_path: Path):
    checkpoint_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        print(f"📂 Resuming from checkpoint: {checkpoint_path}")
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: MAIN
# ═══════════════════════════════════════════════════════════════════════════
async def run(args):
    print("🚀 Starting CUAD evaluation V3 (batched schemas)")
    print(f"   Model:            {args.model}")
    print(f"   Data:             {args.data}")
    print(f"   Limit:            {args.limit or 'full set'}")
    print(f"   Labels per batch: {LABELS_PER_BATCH}")
    print(f"   Output:           {args.out}")
    
    print(f"\n📋 Loading categories from {CATEGORY_CSV}...")
    categories = load_categories(CATEGORY_CSV)
    print(f"   Loaded {len(categories)} labels")
    
    schemas = build_extraction_schemas(categories, batch_size=LABELS_PER_BATCH)
    all_labels = list(categories.keys())
    print(f"   Built {len(schemas)} schema batches "
          f"(~{LABELS_PER_BATCH} labels each)")
    
    chains = build_chains(args.model, schemas)
    
    print(f"\n📖 Loading test data...")
    ds = json.loads(Path(args.data).read_text(encoding="utf-8"))
    contracts = ds["data"]
    print(f"   {len(contracts)} contracts in test set")
    
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
        
        # Resume check
        first_qa_id = contract["paragraphs"][0]["qas"][0]["id"]
        if first_qa_id in nbest:
            n_qas_done += len(contract["paragraphs"][0]["qas"])
            if args.limit and n_qas_done >= args.limit:
                break
            continue
        
        n_chunks = len(list(chunks(ctx)))
        n_calls = n_chunks * len(schemas)
        print(f"\n[{ci+1}/{len(contracts)}] {title[:60]}")
        print(f"   {len(ctx):,} chars → {n_chunks} chunks × {len(schemas)} batches "
              f"= {n_calls} LLM calls")
        
        contract_start = time.time()
        try:
            predictions = await extract_contract(chains, schemas, ctx, all_labels)
        except Exception as e:
            print(f"   ❌ Contract failed: {e}")
            predictions = {label: [] for label in all_labels}
        
        # Map predictions → nbest format
        for qa in contract["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            label = qa_id.split("__")[-1]
            spans = predictions.get(label, [])
            spans = list(dict.fromkeys(spans))
            entries = [{"text": s, "probability": 1.0} for s in spans]
            entries.append({"text": "", "probability": 0.0})
            nbest[qa_id] = entries
            n_qas_done += 1
        
        save_checkpoint(nbest, checkpoint_path)
        
        elapsed = time.time() - contract_start
        total_elapsed = time.time() - t0
        avg_per_contract = total_elapsed / (ci + 1)
        remaining = avg_per_contract * (len(contracts) - ci - 1)
        n_with_spans = sum(1 for p in predictions.values() if p)
        print(f"   ✓ Done in {elapsed:.1f}s ({elapsed/n_calls:.1f}s/call) | "
              f"{n_with_spans}/{len(all_labels)} labels found | "
              f"ETA: {remaining/60:.1f}min")
        
        if args.limit and n_qas_done >= args.limit:
            print(f"\n⏸  Reached --limit {args.limit}")
            break
    
    final_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {len(nbest)} QA predictions → {final_path}")
    print(f"   Total time: {(time.time()-t0)/60:.1f} minutes")
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    print(f"\n📊 Next: edit evaluate.py → model_path = '{out_dir}' → python evaluate.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test.json")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="Output dir (default: trained_models/<model-clean>)")
    args = ap.parse_args()
    
    # Auto-derive --out từ --model nếu không có
    if args.out is None:
        model_clean = args.model.replace(":", "-").replace("/", "_")
        args.out = f"trained_models/{model_clean}"
    
    asyncio.run(run(args))