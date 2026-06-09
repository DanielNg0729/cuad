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
        {"contract_id": <title>, "num_chunks": <int>, "chunks": ["...", ...]},
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
    python chunking.py --strategy markdown           # chunk docling .md (## / **)
"""

import argparse
import json
import re
from pathlib import Path

CHUNK_CHARS = 5000
OVERLAP = 400
CHUNK_STRATEGY = "section"   # one of: fixed | recursive | section | markdown
OUTPUT_FILE = "test_chunking.json"
MARKDOWN_DIR = "markdown_docling"   # docling .md files from PdfToMarkdownBatch.py


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


# A Markdown divider line in a docling-converted contract: an ATX heading
# ("## SUPPLY CONTRACT"), a bold run at line start ("**1.1** ..."), or a list
# item docling uses for clauses ("1. ..." / "- ..."). Each starts a new chunk.
_MD_DIVIDER_RE = re.compile(r"^(?:#{1,6}\s|\*\*|\d+(?:\.\d+)*\.\s|[-*]\s)")

def _split_markdown(md: str) -> list[str]:
    """Cut a docling Markdown contract at every divider line, keeping the divider
    stuck to the text underneath it. The leading "# <title>" line is dropped (the
    title already lives in contract_id). Heading lines (#, ##, ...) are never left
    on their own -- a "## PACKING" rides along with the first clause beneath it
    instead of becoming a useless heading-only chunk."""
    lines = md.splitlines(keepends=True)
    if lines and lines[0].lstrip().startswith("# "):
        lines = lines[1:]                       # drop the document-title heading
    sections, cur, cur_has_body = [], [], False
    for ln in lines:
        stripped = ln.lstrip()
        is_divider = bool(_MD_DIVIDER_RE.match(stripped))
        is_heading = stripped.startswith("#")
        if is_divider and cur and cur_has_body:
            sections.append("".join(cur))
            cur, cur_has_body = [], False
        cur.append(ln)
        if stripped.strip() and not is_heading:
            cur_has_body = True
    if cur:
        sections.append("".join(cur))
    return [s.strip() for s in sections if s.strip()]


def safe_filename(title: str) -> str:
    """Mirror of PdfToMarkdownBatch.safe_filename, so we can find the .md file
    that belongs to a given contract title."""
    name = re.sub(r'[<>:"/\\|?*]', "_", title).strip()
    return (name[:180] if len(name) > 180 else name) + ".md"


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
        f"Unknown --strategy {strategy!r}. Use fixed | recursive | section | markdown."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_chunk_file(data_path: str, strategy: str, chunk_chars: int,
                     overlap: int, markdown_dir: str = MARKDOWN_DIR) -> dict:
    """Load the CUAD test JSON and build the test_chunking.json structure. Each
    contract keeps its `contract_id` (the title), a `num_chunks` count, and its
    chunks. We leave the questions and gold answers out on purpose -- Groq.py
    joins back to the original test.json by contract_id when it needs them.

    The 'markdown' strategy ignores the raw context and instead reads the
    docling-converted .md file for each contract (built by PdfToMarkdownBatch.py),
    splitting on Markdown dividers (headings, bold runs, list items)."""
    contracts = json.loads(Path(data_path).read_text(encoding="utf-8"))["data"]
    md_dir = Path(markdown_dir)

    out_contracts = []
    total_chunks = 0
    for contract in contracts:
        title = contract["title"]
        if strategy == "markdown":
            md_path = md_dir / safe_filename(title)
            if not md_path.exists():
                raise SystemExit(
                    f"Missing Markdown file for {title!r}: {md_path}\n"
                    f"Run `python PdfToMarkdownBatch.py` first to build the "
                    f"{markdown_dir}/ folder."
                )
            chunks = _split_markdown(md_path.read_text(encoding="utf-8"))
        else:
            chunks = make_chunks(contract["paragraphs"][0]["context"],
                                 strategy=strategy, size=chunk_chars,
                                 overlap=overlap)
        total_chunks += len(chunks)
        out_contracts.append({
            "contract_id": title,               # same as qa["id"].split("__")[0]
            "num_chunks": len(chunks),
            "chunks": chunks,
        })

    # The 'markdown' strategy is structural -- chunk_chars/overlap don't apply.
    markdown = strategy == "markdown"
    return {
        "metadata": {
            "strategy": strategy,
            "chunk_chars": None if markdown else chunk_chars,
            "overlap": None if markdown else overlap,
            "source": markdown_dir if markdown else data_path,
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
                    choices=["fixed", "recursive", "section", "markdown"],
                    help=f"How to cut a contract into LLM-sized pieces "
                         f"(default {CHUNK_STRATEGY!r}). 'fixed'=blind "
                         "overlapping windows; 'recursive'=split on paragraph/"
                         "line/sentence boundaries; 'section'=split on contract "
                         "headers (ARTICLE/Section/1.1/(a)/ALL-CAPS) and pack "
                         "whole sections; 'markdown'=read the docling .md files "
                         "and split on Markdown dividers (##/**/list items), "
                         "ignoring --chunk_chars/--overlap.")
    ap.add_argument("--chunk_chars", type=int, default=CHUNK_CHARS,
                    help=f"Characters per chunk (default {CHUNK_CHARS}). Smaller "
                         "chunks shrink per-call input but produce more chunks.")
    ap.add_argument("--overlap", type=int, default=OVERLAP,
                    help=f"Chunk overlap in chars (default {OVERLAP}). Only the "
                         "'fixed' strategy (and oversized-section fallback) use it.")
    ap.add_argument("--markdown_dir", default=MARKDOWN_DIR,
                    help=f"Folder of docling .md files for the 'markdown' "
                         f"strategy (default {MARKDOWN_DIR!r}); built by "
                         "PdfToMarkdownBatch.py.")
    args = ap.parse_args()

    result = build_chunk_file(args.data, args.strategy, args.chunk_chars,
                              args.overlap, args.markdown_dir)

    Path(OUTPUT_FILE).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = result["metadata"]
    if meta["strategy"] == "markdown":
        print(f"Chunked {meta['num_contracts']} contracts from "
              f"{args.markdown_dir}/ using strategy='markdown' "
              f"(split on Markdown dividers, no size cap)")
    else:
        print(f"Chunked {meta['num_contracts']} contracts from {args.data} "
              f"using strategy={meta['strategy']!r} "
              f"(chunk_chars={meta['chunk_chars']}, overlap={meta['overlap']})")
    print(f"  {meta['total_chunks']} chunks total "
          f"(~{meta['total_chunks'] / max(1, meta['num_contracts']):.1f} per contract)")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Next: python Groq.py --data {OUTPUT_FILE} --model <groq-model-id>")


if __name__ == "__main__":
    main()
