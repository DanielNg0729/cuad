"""
Chunking strategies for splitting a contract's text into LLM-sized pieces.

A contract's `context` is one big string. How we cut it into pieces matters a lot
for span extraction: a fixed-width cut routinely splits a clause in half (so
neither chunk contains it whole and validate_span drops it) and orphans section
headers like "12. GOVERNING LAW" from the paragraph they label — losing the
single strongest signal for that category.

Public entry point: `make_chunks(text, strategy, size, overlap)`.
All strategies respect `size` (the per-call char cap) so we never regress into
413 "request too large" errors.
"""

import re

CHUNK_CHARS = 5000           # per-chunk char cap (keep under the model's TPM cap)
OVERLAP = 400                # fixed-strategy window overlap, in chars
CHUNK_STRATEGY = "section"   # one of: fixed | recursive | section


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Original behaviour: blind overlapping fixed-width windows. Last resort
    for a blob with no usable separators (e.g. a giant ASCII table)."""
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
        f"Unknown --chunk_strategy {strategy!r}. "
        "Use fixed | recursive | section."
    )
