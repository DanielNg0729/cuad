"""
CUAD eval in qwen3:4b model using Ollama and Langchain for easier switching syntax

This code ask all 41 labels 1 in call instead of 41 calls in the original file
(due to the fact that qwen3:4b is a Large language model and can handle 41 question in 1 call instead of 1 like all model in the research paper)

Also add anti-hallucination: Oftenly, these model will provide the summary of the contract instead of the contract itself 0> verify that span needed must in the chunk

Output: trained_models/<model>/nbest_predictions_.json (Format the same with paper so can use on evaluate.py)

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
from pydantic import BaseModel,Field,create_model
from tqdm.asyncio import tqdm as atqdm

CHUNK_CHARS = 6000 # Size of each chunk (6k ~ 1.5k tokens) (i use this to fit with my GPU)
OVERLAP = 400 # Overlap between chunks to not cut out span
MAX_CONCURRENCY = 1 # Number of chunks running concurrent = 1 (fit better with my GPU)

NUM_PREDICT = 800
CATEGORY_CSV = "category_descriptions.csv"

# Load 41 Labels from CSV

def load_categories(csv_path = str) -> dict[str,list[str]]:
    """
    Read file category csv from the paper
    switch to more friendly format in python: each cell in the format:
    Category X: Description Y

    Returns: {label_name: [description, answer_format, group]}
    
    
    """
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).split("Category:", 1)[-1].strip()
        desc  = str(row.iloc[1]).split("Description:", 1)[-1].strip()
        fmt   = str(row.iloc[2]).split("Answer Format:", 1)[-1].strip()
        grp   = str(row.iloc[3]).split("Group:", 1)[-1].strip()
        result[label] = [desc, fmt, grp]
    return result

def label_to_field(label: str) -> str:
    """
    Convert label name to python indetifier that suitable for pydantic field
    """
    s = label.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)  # Replace non-alphanumeric with _
    s = s.strip('_')                     # remove _ at start end
    return s

# Now Build Pydantic Schema with 41 fields

def build_extraction_schema(categories: dict[str, list[str]]) -> tuple[type, dict[str, str]]:
    """
    Create dynamic pydantic class with 41 fields, each fields is with 1 label
    Each field is a Optional[str] with description from:
    - desc: describe the label
    - fmt: Expected answer format (For LLM to output in that format)

    Pydantic will use this description to produce JSON schema for Ollama -> LLM know what to fill in

    Return (Pydantic class (for with_structure_output function), dict mapping field_name -> original label to map back) 
    """
    fields = {}
    field_to_label = {}

    for label,info in categories.items():
        desc,fmt,grp = info
        fname = label_to_field(label)
        field_to_label[fname] = label
        fields[fname] = (
            Optional[str],
            Field(
                default= None,
                description= (
                    f"{desc}. Expected format: {fmt}. "
                    f"Return the EXACT verbatim substring from the section "
                    f"if present, or null if not in this section."
                )
            )
        )

    # Create dynamic class name ContractExtraction
    ExtractionSchema = create_model("ContractExtraction", **fields)
    return ExtractionSchema, field_to_label

# Build CLEL Chain

def build_chain(model : str, schema: type):
    """
    Build Langchain pipeline: prompt -> ChatOllama -> Pydantic Schema

    Using with_structured_output(schema) -> force LLM to return JSON in the right schema
    - Send JSON schema of pydantic class to Ollama
    - Ollama use format= json to force output is valid json
    - Parse JSON -> validate through pydantic -> return object
    
    """
    llm = ChatOllama(
        model=model,
        temperature=0,           # Deterministic — không random
        num_predict=NUM_PREDICT, # Giới hạn output để tránh lan man
    ).with_structured_output(schema)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a legal contract analyst. You will be given ONE SECTION of a "
         "contract. Your job: for EACH category listed in the schema, return the "
         "EXACT verbatim substring from the section that matches it, copied "
         "character-by-character. If a category doesn't appear in this section, "
         "set that field to null. Do NOT paraphrase. Do NOT make up text. Only "
         "return text that literally appears in the section."),
        ("human",
         "Contract section:\n"
         "\"\"\"\n{chunk}\n\"\"\"\n\n"
         "Extract all 41 categories from the section. Set fields to null "
         "where the category is not present."),
    ])

    return prompt | llm

def chunks(text: str, size: int = CHUNK_CHARS, overlap : int = OVERLAP):
    """
    Divide long text into chunks with overlaps
    
    """
    if len(text) <= size:
        yield text
        return
    i = 0
    while i < len(text):
        yield text[i:i+size]
        if i + size >= len(text):
            break
        i += size - overlap


def validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """
    Anti-hallucination check
    LLM usually paraphrase instead of copy verbatim. This function check whether the span actually in the chunk. If not -> eliminate

    Return: cleaned span if valid, else None
    
    """

    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if not span: return None
    if span in chunk:
        return span
    if span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx : idx + len(span)]
    
    return None

# Extract contract

async def extract_contract(
        chain,
        context: str,
        field_to_label: dict[str,str],
) -> dict[str,list[str]]:
    """
    Run LLM in all contract, return prediction for all 41 labels

    - Divide contract into chunks
    - Each chunk -> 1 LLM call -> return all 41 fields in 1 call
    - Aggregate: For each label, collect spans from chunks
    
    return dict {label_name: [List of valid spans]}
    """

    chunk_list = list(chunks(context))
    predictions = {label: [] for label in field_to_label.values()}

    inputs = [{"chunk": c} for c in chunk_list]
    results = await chain.abatch(
        inputs, 
        config={"max_concurrency": MAX_CONCURRENCY}
    )


    for chunk_text,result in zip(chunk_list,results):
        if result is None:
            continue # LLM parse error -> skip this chunk

        # Result is an instance of ContractExtraction (in Pydantic)
        # Access each field: result.governing_lax,...
        for fname,label in field_to_label.items():
            raw_span = getattr(result, fname, None)
            valid_span = validate_span(raw_span, chunk_text)
            if valid_span:
                predictions[label].append(valid_span)
        
    return predictions

# Checkpoint -Save/load result if still working
def save_checkpoint(nbest: dict, checkpoint_path: Path):
    """Save present result to a file to resume if crash"""
    checkpoint_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")

def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load result from working file if checkpoint exist"""
    if checkpoint_path.exists():
        print(f"📂 Resuming from checkpoint: {checkpoint_path}")
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {}



async def run(args):
    print("🚀 Starting CUAD evaluation V2")
    print(f"   Model: {args.model}")
    print(f"   Data:  {args.data}")
    print(f"   Limit: {args.limit or 'full set'}")
    
    # --- Load 41 labels ---
    print(f"\n📋 Loading categories from {CATEGORY_CSV}...")
    categories = load_categories(CATEGORY_CSV)
    print(f"   Loaded {len(categories)} labels")
    
    # --- Build Pydantic schema ---
    schema, field_to_label = build_extraction_schema(categories)
    print(f"   Schema: {schema.__name__} with {len(field_to_label)} fields")
    
    # --- Build chain ---
    chain = build_chain(args.model, schema)
    
    # --- Load test data ---
    print(f"\n📖 Loading test data...")
    ds = json.loads(Path(args.data).read_text(encoding="utf-8"))
    contracts = ds["data"]
    print(f"   {len(contracts)} contracts in test set")
    
    # --- Setup output ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "nbest_predictions_.json"
    checkpoint_path = out_dir / "_checkpoint.json"
    
    # Load checkpoint
    nbest = load_checkpoint(checkpoint_path)
    
    # --- Process contracts ---
    t0 = time.time()
    n_qas_done = 0
    
    for ci, contract in enumerate(contracts):
        title = contract["title"]
        ctx = contract["paragraphs"][0]["context"]
        
        # Check if contract has alr processed (resume)
        first_qa_id = contract["paragraphs"][0]["qas"][0]["id"]
        if first_qa_id in nbest:
            n_qas_done += len(contract["paragraphs"][0]["qas"])
            if args.limit and n_qas_done >= args.limit:
                break
            continue
        
        # --- Process contract này ---
        print(f"\n[{ci+1}/{len(contracts)}] {title[:60]}")
        print(f"   {len(ctx):,} chars → {len(list(chunks(ctx)))} chunks")
        
        contract_start = time.time()
        try:
            predictions = await extract_contract(chain, ctx, field_to_label)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            predictions = {label: [] for label in field_to_label.values()}
        
        # --- Map predictions → nbest format ---
        # Format paper: {qa_id: [{"text": "...", "probability": ...}, ...]}
        for qa in contract["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            label = qa_id.split("__")[-1]
            
            spans = predictions.get(label, [])
            # Dedupe (1 label có thể tìm thấy ở nhiều chunks)
            spans = list(dict.fromkeys(spans))
            
            # Format cho evaluate.py
            entries = [{"text": s, "probability": 1.0} for s in spans]
            entries.append({"text": "", "probability": 0.0})  # empty option
            nbest[qa_id] = entries
            n_qas_done += 1
        
        # --- Save checkpoint for each contract ---
        save_checkpoint(nbest, checkpoint_path)
        
        elapsed = time.time() - contract_start
        total_elapsed = time.time() - t0
        avg_per_contract = total_elapsed / (ci + 1)
        remaining = avg_per_contract * (len(contracts) - ci - 1)
        print(f"   ✓ Done in {elapsed:.1f}s | "
              f"Total: {total_elapsed/60:.1f}min | "
              f"ETA: {remaining/60:.1f}min")
        
        # --- Stop nếu đạt limit ---
        if args.limit and n_qas_done >= args.limit:
            print(f"\n⏸  Reached --limit {args.limit}")
            break
    
    # --- Save final output ---
    final_path.write_text(json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {len(nbest)} QA predictions → {final_path}")
    print(f"   Total time: {(time.time()-t0)/60:.1f} minutes")
    
    # --- Cleanup checkpoint ---
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    print(f"\n📊 Next step: chạy paper's scorer:")
    print(f"   1. Edit evaluate.py: đổi model_path = '{out_dir}'")
    print(f"   2. python evaluate.py")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test.json",
                    help="Path to CUAD test.json")
    ap.add_argument("--model", default="qwen3:4b",
                    help="Ollama model name")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max QAs to process (for smoke testing)")
    ap.add_argument("--out", default="trained_models/qwen3-4b",
                    help="Output directory")
    args = ap.parse_args()
    
    asyncio.run(run(args))