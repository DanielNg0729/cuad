# Test on Qwen3:4b model on CUAD dataset
import argparse
import asyncio
import json
import string
from pathlib import Path
from typing import Optional
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class Span(BaseModel):
    text: Optional[str] = Field(None, description="Exact verbatim substring from the section, or null if no match.",)
    confidence: float = Field(0.0,description="Your confidence in this extraction, 0.0 (low) to 1.0 (high).", ge=0.0, le=1.0,)


def build_chain(model):
    llm = ChatOllama(model = model, temperature=0).with_structured_output(Span)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You extract clauses from legal contracts. Return the EXACT verbatim "
         "substring from the given section that matches the requested category, "
         "and your confidence (0-1). If nothing matches, return null text and "
         "confidence 0."),
        ("human",
         "Category: {label}\nDescription: {desc}\n\n"
         "Section:\n\"\"\"\n{chunk}\n\"\"\""),
    ])

    return prompt | llm

CHUNK_CHARS = 8000
OVERLAP = 500

def parse_question(q):
    label = q.split('"')[1]
    desc = q.split("Details:", 1)[1].strip() if "Details:" in q else ""
    return label, desc

def chunks(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP):
    i = 0
    while i < len(text):
        yield text[i: i+ size]
        if i + size >= len(text):
            break
        i += size - overlap
async def predict_one(chain, context: str, label: str, desc: str) -> list[dict]:
    """Return list of {text, probability} dicts compatible with evaluate.py format."""
    chunk_list = list(chunks(context))
    inputs = [{"label": label, "desc": desc, "chunk": c} for c in chunk_list]
    results = await chain.abatch(inputs, config={"max_concurrency": 4})
 
    preds = []
    for chunk, r in zip(chunk_list, results):
        if r is None or not r.text:
            continue
        span = r.text.strip().strip('"').strip("'")
        if span and span in chunk:  # anti-hallucination
            preds.append({"text": span, "probability": float(r.confidence)})
 
    # Always include the empty prediction (evaluate.py expects this option)
    preds.append({"text": "", "probability": 0.0})
    return preds
 
 
async def run(args):
    chain = build_chain(args.model)
    ds = json.loads(Path(args.data).read_text(encoding="utf-8"))
 
    nbest = {}
    n_done = 0
    for doc in ds["data"]:
        ctx = doc["paragraphs"][0]["context"]
        for qa in doc["paragraphs"][0]["qas"]:
            if args.limit and n_done >= args.limit:
                break
            label, desc = parse_question(qa["question"])
            try:
                preds = await predict_one(chain, ctx, label, desc)
            except Exception as e:
                print(f"[!] {qa['id']}: {e}")
                preds = [{"text": "", "probability": 0.0}]
            nbest[qa["id"]] = preds
            n_done += 1
            if n_done % 10 == 0:
                print(f"[{n_done}] done")
        if args.limit and n_done >= args.limit:
            break
 
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nbest_predictions_.json"
    out_path.write_text(json.dumps(nbest, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Now run the paper's scorer (edit model_path in evaluate.py to point here):")
    print(f"    python evaluate.py")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to test.json")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="trained_models/qwen3-4b")
    args = ap.parse_args()
    asyncio.run(run(args))