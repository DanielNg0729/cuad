"""
Batch-mode CUAD extraction with OpenAI's Batch API -- 50% cheaper than live calls.

The Batch API is asynchronous: you upload every request as one JSONL file,
OpenAI churns through them within a 24h window, and you download the results
later. Half price, but not instant -- so this runs in two steps:

    python OpenAIBatch.py submit      # build + upload + kick off the batch
    python OpenAIBatch.py collect     # once it's done, pull results -> nbest

`submit` prints a batch id and writes a manifest (_batch.json) so `collect` knows
what to fetch and how to map each answer back to its chunk. Pass --wait to submit
to poll until the batch finishes and collect in one shot.

It reuses the schema / prompt / span-validation from OpenAITest.py, and the same
pre-chunked input (test_chunking.json), so results line up with the live script.

Output is the nbest format that evaluate.py reads:
    python evaluate.py --model_path <out_dir>
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.lib._parsing._completions import type_to_response_format_param

from OpenAITest import (
    CATEGORY_CSV, MAX_TOKENS, SYSTEM_PROMPT, USER_PROMPT,
    build_schema, load_categories, pick_contract, validate_span,
)

MANIFEST_NAME = "_batch.json"
# Batch is done when it lands in one of these; the rest are still in flight.
_TERMINAL = {"completed", "failed", "expired", "cancelled"}

def load_contracts(data_path: str, doc: str | None, index: int | None) -> list[dict]:
    """The pre-chunked contracts, optionally narrowed to a single contract --
    by id (--doc) or by position (--index). With neither, you get them all.
    Kept identical between submit and collect so custom_ids map back cleanly."""
    contracts = json.loads(Path(data_path).read_text(encoding="utf-8"))["data"]
    if doc is not None:
        return [pick_contract(contracts, doc)]
    if index is not None:
        if not (0 <= index < len(contracts)):
            raise SystemExit(f"--index {index} out of range (0..{len(contracts) - 1})")
        return [contracts[index]]
    return contracts

def submit(args):
    """Build one request per chunk, upload them as a JSONL, and start the batch."""
    categories = load_categories(CATEGORY_CSV)
    schema, _ = build_schema(categories)
    response_format = type_to_response_format_param(schema)  # strict JSON schema

    contracts = load_contracts(args.data, args.doc, args.index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # custom_id = "<contract index>-<chunk index>"; both are rebuilt the same way
    # at collect time, so we never have to ship the chunk text back and forth.
    input_path = out_dir / "_batch_input.jsonl"
    n_requests = 0
    with input_path.open("w", encoding="utf-8") as f:
        for ci, contract in enumerate(contracts):
            for k, chunk in enumerate(contract["chunks"]):
                body = {
                    "model": args.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT.format(chunk=chunk)},
                    ],
                    "response_format": response_format,
                }
                if args.max_tokens:           # omit when no cap (default)
                    body["max_tokens"] = args.max_tokens
                f.write(json.dumps({
                    "custom_id": f"{ci}-{k}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }) + "\n")
                n_requests += 1

    print(f"Built {n_requests} requests across {len(contracts)} contract(s) -> {input_path}")

    client = OpenAI()
    with input_path.open("rb") as fh:
        upload = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    manifest = {
        "batch_id": batch.id,
        "input_file_id": upload.id,
        "model": args.model,
        "data": args.data,
        "doc": args.doc,
        "index": args.index,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Submitted batch {batch.id} (status: {batch.status})")
    print(f"Manifest -> {out_dir / MANIFEST_NAME}")

    if args.wait:
        final = wait_for(client, batch.id, args.poll, args.max_wait)
        if final.status == "completed":
            collect(args)
    else:
        print("Check back later with:  python OpenAIBatch.py collect "
              f"--out {out_dir}")

def wait_for(client: OpenAI, batch_id: str, poll: int, max_wait_min: int):
    """Poll the batch until it's terminal, or until max_wait_min elapses (0 = no
    limit). Prints only when status/progress actually changes -- no spam -- and
    returns the last batch object so the caller can decide what to do."""
    deadline = time.time() + max_wait_min * 60 if max_wait_min else None
    last_line = None
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        line = f"  status={batch.status}  {counts.completed}/{counts.total} done, {counts.failed} failed"
        if line != last_line:                 # only print on a real change
            print(line)
            last_line = line
        if batch.status in _TERMINAL:
            return batch
        if deadline and time.time() >= deadline:
            print(f"  stopped watching after {max_wait_min} min -- batch is still "
                  f"'{batch.status}' in OpenAI's queue. This is normal, not an error;")
            print("  it keeps running server-side. Collect later: python OpenAIBatch.py collect")
            return batch
        time.sleep(poll)

def collect(args):
    """Download the finished batch, validate spans, and write the nbest file."""
    out_dir = Path(args.out)
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    client = OpenAI()
    batch = client.batches.retrieve(manifest["batch_id"])
    print(f"Batch {batch.id}: status={batch.status}")
    if batch.status != "completed":
        if batch.status not in _TERMINAL:
            print("Not finished yet -- try again later, or submit with --wait.")
        else:
            print(f"Batch ended as {batch.status}; nothing to collect.")
        return

    # Rebuild the exact same chunk layout so custom_ids map back to (contract, chunk).
    categories = load_categories(CATEGORY_CSV)
    schema, field_to_label = build_schema(categories)
    label_set = sorted(set(field_to_label.values()))
    contracts = load_contracts(manifest["data"], manifest["doc"], manifest.get("index"))

    # preds[ci] = {label: [spans...]}
    preds = {ci: {label: [] for label in label_set} for ci in range(len(contracts))}

    # A completed batch with no output file means every request errored.
    if not batch.output_file_id:
        print("Batch completed but produced NO output file -- all requests errored.")
        if batch.error_file_id:
            print(f"  Inspect the error file: {batch.error_file_id}")
        return

    raw = client.files.content(batch.output_file_id).read().decode("utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    ok, failed = 0, 0
    for line in lines:
        row = json.loads(line)
        cid = row["custom_id"]
        ci, k = (int(x) for x in cid.split("-"))
        chunk = contracts[ci]["chunks"][k]

        resp = row.get("response")
        if not resp or resp.get("status_code") != 200 or row.get("error"):
            failed += 1
            continue
        # content is None on a refusal, or truncated/missing -> json.loads(None)
        # raises TypeError, so guard both that and a bad JSON parse.
        content = resp["body"]["choices"][0]["message"].get("content")
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            failed += 1
            continue
        ok += 1
        for fname, label in field_to_label.items():
            span = validate_span(data.get(fname), chunk)
            if span:
                preds[ci][label].append(span)

    print(f"Parsed {ok} responses OK, {failed} failed/empty")

    # Requests that errored outright go to the error file, not the output file --
    # without this they'd silently vanish and look like 0 failures.
    if batch.error_file_id:
        err = client.files.content(batch.error_file_id).read().decode("utf-8")
        n_err = sum(1 for ln in err.splitlines() if ln.strip())
        print(f"  !! {n_err} request(s) errored entirely (error file {batch.error_file_id})")

    nbest = {}
    for ci, contract in enumerate(contracts):
        cid = contract["contract_id"]
        for label in label_set:
            spans = list(dict.fromkeys(preds[ci][label]))  # dedupe, keep order
            nbest[f"{cid}__{label}"] = (
                [{"text": s, "probability": 1.0} for s in spans]
                + [{"text": "", "probability": 0.0}]        # no-answer sentinel
            )

    out_path = out_dir / "nbest_predictions_.json"
    # ensure_ascii=False keeps spans readable (real chars, not \uXXXX escapes).
    out_path.write_text(json.dumps(nbest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(nbest)} QA predictions to {out_path}")
    print(f"Next: python evaluate.py --model_path {out_dir}")

def list_contracts(args):
    """Print each contract's index and id so you can pick one for --index/--doc."""
    contracts = json.loads(Path(args.data).read_text(encoding="utf-8"))["data"]
    print(f"{len(contracts)} contracts in {args.data}:")
    for i, c in enumerate(contracts):
        print(f"  [{i:>3}] {c['contract_id']}  ({c['num_chunks']} chunks)")

if __name__ == "__main__":
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "collect", "list"],
                    help="submit = start a batch; collect = fetch a finished one; "
                         "list = show contract indices/ids.")
    ap.add_argument("--data", default="test_chunking.json",
                    help="Pre-chunked input (contract_id + chunks per contract).")
    ap.add_argument("--model", default="gpt-4o-mini")
    # Pick ONE contract by id (--doc) or by position (--index); give neither for all.
    ap.add_argument("--doc", default=None, help="Only this contract id.")
    ap.add_argument("--index", type=int, default=None, help="Only this contract position (0-based).")
    ap.add_argument("--out", default=None, help="Output dir (default trained_models/<model>).")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--wait", action="store_true",
                    help="(submit) poll until done, then collect automatically.")
    ap.add_argument("--poll", type=int, default=60, help="(--wait) seconds between polls.")
    ap.add_argument("--max-wait", dest="max_wait", type=int, default=20,
                    help="(--wait) stop watching after this many minutes (0 = wait forever). "
                         "The batch keeps running server-side; collect it later.")
    args = ap.parse_args()

    if args.doc is not None and args.index is not None:
        raise SystemExit("Use --doc OR --index, not both.")

    if args.action == "list":      # read-only, no API key needed
        list_contracts(args)
        raise SystemExit(0)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing -- add it to .env or your shell.")

    if args.out is None:
        args.out = "trained_models/" + args.model.replace(":", "-").replace("/", "_") + "__batch"

    if args.action == "submit":
        submit(args)
    else:
        collect(args)
