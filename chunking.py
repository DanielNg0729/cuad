"""
Stage 1 of the Groq CUAD pipeline: cut every contract into LLM-sized chunks and
write them to a single JSON file (test_chunking.json) that Groq.py then consumes.

Why split chunking off from the LLM call:
  - chunking is deterministic and free; the LLM call is slow and rate-limited.
    Running it once up front lets you eyeball / diff the chunks, and lets
    Groq.py focus purely on "send chunk -> collect spans".
  - the three strategies (fixed | recursive | section) can be generated and
    compared without burning any API budget.

How a contract is cut matters a lot for span extraction: a fixed-width cut
routinely splits a clause in half (so neither chunk contains it whole and
Groq.py's validate_span drops it) and orphans section headers like
"12. GOVERNING LAW" from the paragraph they label. The 'section' and 'recursive'
strategies exist to avoid exactly that.

Output schema (consumed by Groq.py):
    {
      "metadata": {"strategy": ..., "chunk_chars": ..., "overlap": ...,
                   "source": ..., "num_contracts": ...},
      "data": [
        {"contract_id": <title>, "chunks": ["...", ...]},
        ...
      ]
    }

Only the chunk text is stored here -- no questions and no gold answers, so the
ground truth is never duplicated alongside what gets fed to the LLM. Each entry
keeps `contract_id` (the contract's title, which is also the prefix of every
qa id in test.json: qa["id"].split("__")[0]), so this file links back to the
original test.json by contract id whenever the questions / answers are needed.

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
# All strategies respect `size` (the per-call char cap) so we never produce a
# chunk too large for the model to accept (which would later cause a 413
# "request too large" from Groq).
# ---------------------------------------------------------------------------

def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Blind overlapping fixed-width windows. Last resort for a blob with no
    usable separators (e.g. a giant ASCII table)."""
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
    """Concatenate consecutive pieces into chunks <= size WITHOUT ever splitting
    an individual piece (so whole sections / paragraphs stay intact)."""
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


# Separator priority: paragraph -> line -> sentence -> word. We descend to the
# next separator only for pieces that are still over `size`.
_SEPARATORS = ("\n\n", "\n", ". ", " ")

def _recursive_split(text: str, size: int, overlap: int,
                     seps: tuple[str, ...] = _SEPARATORS) -> list[str]:
    """Split on the highest-priority separator that exists, keeping the
    separator attached to each piece, recursing into any piece still too big,
    then greedily repacking so we don't emit a flood of tiny chunks."""
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


# Contract section headers we split on. Matched against each stripped line:
#   "ARTICLE IV" / "Section 12.3"  |  "1."  "1.1"  "12.3.4"  |  "(a)" "(iv)"
#   "GOVERNING LAW" (an ALL-CAPS heading line)
_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLCDM]+\b"
    r"|[0-9]+(?:\.[0-9]+)*\.?(?:\s|$)"
    r"|\([a-zA-Z0-9]{1,4}\)\s"
    r"|[A-Z][A-Z0-9 ,;:'&/().\-]{3,}\s*$"
    r")"
)

def _split_sections(text: str) -> list[str]:
    """Break text at detected section headers, keeping each header glued to the
    body beneath it."""
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
    """Section-aware: never split a section unless it alone exceeds `size`
    (then recurse), and pack whole sections together up to the cap."""
    out: list[str] = []
    for sec in _split_sections(text):
        if len(sec) > size:
            out.extend(_recursive_split(sec, size, overlap))
        else:
            out.append(sec)
    return _greedy_pack(out, size)


def make_chunks(text: str, strategy: str = CHUNK_STRATEGY,
                size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Dispatch to the chosen chunking strategy. All return a list of chunks,
    each <= `size` chars."""
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
    """Read the CUAD test JSON and return the test_chunking.json structure:
    every contract carries only its `contract_id` (the title) and its chunk
    list. Questions and gold answers are deliberately NOT copied here -- Groq.py
    re-joins to the original test.json by contract_id when it needs them."""
    contracts = json.loads(Path(data_path).read_text(encoding="utf-8"))["data"]

    out_contracts = []
    total_chunks = 0
    for contract in contracts:
        para = contract["paragraphs"][0]
        chunks = make_chunks(para["context"], strategy=strategy,
                             size=chunk_chars, overlap=overlap)
        total_chunks += len(chunks)
        out_contracts.append({
            "contract_id": contract["title"],   # == qa["id"].split("__")[0]
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
