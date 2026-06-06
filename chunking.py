"""
Step 1 of the Groq pipeline: chop each contract into LLM-sized pieces and dump
them to test_chunking.json, which Groq.py reads next.

I split this out from the actual LLM call for two reasons. Chunking is cheap and
deterministic, so doing it once up front means I can open the file and actually
look at the chunks before spending any API budget. And it lets me generate all
three strategies and compare them without making a single request.

The way you cut a contract really does matter here. A naive fixed-width cut
tends to slice a clause down the middle (so no chunk has it whole and Groq.py
throws it away), and it strands headings like "12. GOVERNING LAW" from the text
underneath them. The 'section' and 'recursive' strategies exist to avoid that.

Shape of the output file (what Groq.py expects):
    {
      "metadata": {"strategy": ..., "chunk_chars": ..., "overlap": ...,
                   "source": ..., "num_contracts": ...},
      "data": [
        {"contract_id": <title>, "chunks": ["...", ...]},
        ...
      ]
    }

Note we only keep the chunk text here -- no questions, no gold answers. That
keeps the ground truth out of the file we feed to the model. The contract_id is
just the contract's title, which also happens to be the prefix of every qa id
in test.json (qa["id"].split("__")[0]), so you can always join back to the
original to get the questions and answers when you need them.

Usage:
    python chunking.py --strategy section            # default
    python chunking.py --strategy recursive --chunk_chars 4000
    python chunking.py --data test.json --strategy fixed --overlap 400
"""

import argparse
import json
import re
from pathlib import Path

CHUNK_CHARS = 5000
OVERLAP = 400
CHUNK_STRATEGY = "section"   # one of: fixed | recursive | section
OUTPUT_FILE = "test_chunking.json"


# ---------------------------------------------------------------------------
# Chunking strategies
#
# Every strategy honors `size` (the char cap per chunk) so we never hand the
# model a chunk it'll reject with a 413 "request too large".
# ---------------------------------------------------------------------------

def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Dumb fixed-width windows with a bit of overlap. Only used as a fallback
    when there are no separators to work with (think a giant ASCII table)."""
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def _greedy_pack(pieces: list[str], size: int) -> list[str]:
    """Glue consecutive pieces together into chunks up to `size`, but never cut
    a piece in half -- so whole sections / paragraphs stay in one piece."""
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) > size:
            out.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        out.append(buf)
    return out


# Separators in order of preference: paragraph, then line, then sentence, then
# word. We only drop to a finer one for pieces that are still too big.
_SEPARATORS = ("\n\n", "\n", ". ", " ")

def _recursive_split(text: str, size: int, overlap: int,
                     seps: tuple[str, ...] = _SEPARATORS) -> list[str]:
    """Split on the best separator that's actually present, keep the separator
    attached to each piece, recurse into anything still too big, then pack the
    results back up so we don't end up with a pile of tiny chunks."""
    if len(text) <= size:
        return [text]
    for k, sep in enumerate(seps):
        if sep not in text:
            continue
        raw = text.split(sep)
        pieces = [p + sep for p in raw[:-1]] + [raw[-1]]
        out: list[str] = []
        for p in pieces:
            if len(p) <= size:
                out.append(p)
            else:
                out.extend(_recursive_split(p, size, overlap, seps[k + 1:]))
        return _greedy_pack(out, size)
    return _hard_split(text, size, overlap)


# What a section header looks like, checked against each stripped line:
#   "ARTICLE IV" / "Section 12.3"  |  "1."  "1.1"  "12.3.4"  |  "(a)" "(iv)"
#   or just an ALL-CAPS heading line like "GOVERNING LAW"
_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLCDM]+\b"
    r"|[0-9]+(?:\.[0-9]+)*\.?(?:\s|$)"
    r"|\([a-zA-Z0-9]{1,4}\)\s"
    r"|[A-Z][A-Z0-9 ,;:'&/().\-]{3,}\s*$"
    r")"
)

def _split_sections(text: str) -> list[str]:
    """Cut the text wherever a section header shows up, keeping each header
    stuck to the body underneath it."""
    sections, cur = [], []
    for ln in text.splitlines(keepends=True):
        if cur and _HEADER_RE.match(ln.strip()):
            sections.append("".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append("".join(cur))
    return sections


def _section_split(text: str, size: int, overlap: int) -> list[str]:
    """Keep sections whole. Only split a section if it's bigger than `size` on
    its own (then fall back to recursive), and pack the rest together up to the
    cap."""
    out: list[str] = []
    for sec in _split_sections(text):
        if len(sec) > size:
            out.extend(_recursive_split(sec, size, overlap))
        else:
            out.append(sec)
    return _greedy_pack(out, size)


def make_chunks(text: str, strategy: str = CHUNK_STRATEGY,
                size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Hand off to whichever strategy was picked. They all give back a list of
    chunks, each no bigger than `size` chars."""
    if len(text) <= size:
        return [text]
    if strategy == "fixed":
        return _hard_split(text, size, overlap)
    if strategy == "recursive":
        return _recursive_split(text, size, overlap)
    if strategy == "section":
        return _section_split(text, size, overlap)
    raise SystemExit(
        f"Unknown --strategy {strategy!r}. Use fixed | recursive | section."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_chunk_file(data_path: str, strategy: str, chunk_chars: int,
                     overlap: int) -> dict:
    """Load the CUAD test JSON and build the test_chunking.json structure. Each
    contract keeps only its `contract_id` (the title) and its chunks. We leave
    the questions and gold answers out on purpose -- Groq.py joins back to the
    original test.json by contract_id when it needs them."""
    contracts = json.loads(Path(data_path).read_text(encoding="utf-8"))["data"]

    out_contracts = []
    total_chunks = 0
    for contract in contracts:
        para = contract["paragraphs"][0]
        chunks = make_chunks(para["context"], strategy=strategy,
                             size=chunk_chars, overlap=overlap)
        total_chunks += len(chunks)
        out_contracts.append({
            "contract_id": contract["title"],   # same as qa["id"].split("__")[0]
            "chunks": chunks,
        })

    return {
        "metadata": {
            "strategy": strategy,
            "chunk_chars": chunk_chars,
            "overlap": overlap,
            "source": data_path,
            "num_contracts": len(out_contracts),
            "total_chunks": total_chunks,
        },
        "data": out_contracts,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Chunk every CUAD contract and write test_chunking.json "
                    "for Groq.py to consume."
    )
    ap.add_argument("--data", default="test.json",
                    help="CUAD test file to chunk (default test.json).")
    ap.add_argument("--strategy", default=CHUNK_STRATEGY,
                    choices=["fixed", "recursive", "section"],
                    help=f"How to cut a contract into LLM-sized pieces "
                         f"(default {CHUNK_STRATEGY!r}). 'fixed'=blind "
                         "overlapping windows; 'recursive'=split on paragraph/"
                         "line/sentence boundaries; 'section'=split on contract "
                         "headers (ARTICLE/Section/1.1/(a)/ALL-CAPS) and pack "
                         "whole sections.")
    ap.add_argument("--chunk_chars", type=int, default=CHUNK_CHARS,
                    help=f"Characters per chunk (default {CHUNK_CHARS}). Smaller "
                         "chunks shrink per-call input but produce more chunks.")
    ap.add_argument("--overlap", type=int, default=OVERLAP,
                    help=f"Chunk overlap in chars (default {OVERLAP}). Only the "
                         "'fixed' strategy (and oversized-section fallback) use it.")
    args = ap.parse_args()

    result = build_chunk_file(args.data, args.strategy, args.chunk_chars,
                              args.overlap)

    Path(OUTPUT_FILE).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = result["metadata"]
    print(f"Chunked {meta['num_contracts']} contracts from {args.data} "
          f"using strategy={meta['strategy']!r} "
          f"(chunk_chars={meta['chunk_chars']}, overlap={meta['overlap']})")
    print(f"  {meta['total_chunks']} chunks total "
          f"(~{meta['total_chunks'] / max(1, meta['num_contracts']):.1f} per contract)")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Next: python Groq.py --data {OUTPUT_FILE} --model <groq-model-id>")


if __name__ == "__main__":
    main()
