"""Convert the contract `context` text stored in test.json into structured Markdown.

Unlike ConvertPDFtoMarkdown.py (which runs docling on the source PDFs), this works
directly off the already-extracted `context` field in the CUAD test.json. The context
is a flat run-on blob, so we apply heuristics to recover structure:

  * numbered headings        ->  "3. PACKING"            => "## 3. PACKING"
  * numbered sub-clauses     ->  "1.1 This is a ..."     => "**1.1** This is a ..."
  * ARTICLE / SECTION lines  ->  "ARTICLE IV"            => "## ARTICLE IV"
  * ALL-CAPS title lines      ->  "SUPPLY CONTRACT"       => "# SUPPLY CONTRACT"
  * page-footer noise (lone page numbers, "Source: ..." lines) is stripped.

Output: one .md file per contract under ./markdown_contracts/, named after the title.
"""

import json
import re
from pathlib import Path

INPUT_JSON = Path("test.json")
OUTPUT_DIR = Path("markdown_contracts")

# A section number like 1, 1.1, 12.3.4 (1-3 dot-separated groups, each 1-2 digits).
SECTION_NUM = r"\d{1,2}(?:\.\d{1,2}){0,3}"

# Candidate section start: a section number sitting at a clause boundary, i.e. preceded
# by start-of-string or whitespace, and not part of a larger number/decimal.
SECTION_START = re.compile(rf"(?<!\S)({SECTION_NUM})\.?\s+")

# Heading-ish leading keywords that introduce a top-level block.
KEYWORD_HEADING = re.compile(
    r"^(ARTICLE|SECTION|EXHIBIT|SCHEDULE|APPENDIX|ANNEX|RECITALS?|WITNESSETH)\b.*",
    re.IGNORECASE,
)

# Footer noise produced by the original PDF->text extraction.
SOURCE_LINE = re.compile(r"^Source:\s", re.IGNORECASE)
PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def looks_like_caps_title(text: str) -> bool:
    """True if the line is a short, mostly-uppercase title (no trailing sentence)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3 or len(text) > 80:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.85


def split_into_blocks(context: str):
    """Break the flat context into logical lines/blocks at section boundaries."""
    # Normalise whitespace but keep existing hard line breaks as block separators.
    raw_lines = [ln.strip() for ln in context.replace("\r", "").split("\n")]
    pieces = []
    for ln in raw_lines:
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln:
            continue
        if SOURCE_LINE.match(ln) or PAGE_NUM_LINE.match(ln):
            continue  # drop footer / page-number noise
        # Insert breaks before each section-number that begins a new clause.
        ln = SECTION_START.sub(lambda m: "\n" + m.group(0), ln)
        for part in ln.split("\n"):
            part = part.strip()
            if part:
                pieces.append(part)
    return pieces


def format_block(block: str) -> str:
    """Turn a single logical block into a Markdown line."""
    # Keyword headings (ARTICLE / SECTION / ...).
    if KEYWORD_HEADING.match(block):
        return f"## {block}"

    # Numbered blocks: "3. PACKING" or "1.1 This is ...".
    m = re.match(rf"^({SECTION_NUM})\.?\s+(.*)$", block)
    if m:
        num, rest = m.group(1), m.group(2).strip()
        depth = num.count(".")
        # Decide whether this is a real heading: top-level (no dot) numbers whose body
        # starts with an uppercase title (no long sentence) read as headings.
        first_word = rest.split(" ", 1)[0] if rest else ""
        title_part = rest.split(":", 1)[0]
        is_heading = (
            depth == 0
            and rest[:1].isupper()
            and (looks_like_caps_title(title_part) or len(rest) <= 60)
        )
        if is_heading:
            level = "##"
            return f"{level} {num}. {rest}"
        # Otherwise it's a numbered clause: bold the number, keep the text.
        return f"**{num}.** {rest}" if depth == 0 else f"**{num}** {rest}"

    # Standalone ALL-CAPS title line.
    if looks_like_caps_title(block):
        return f"# {block}"

    return block


def context_to_markdown(title: str, context: str) -> str:
    blocks = split_into_blocks(context)
    out_lines = [f"# {title}", ""]
    for block in blocks:
        out_lines.append(format_block(block))
        out_lines.append("")  # blank line between blocks
    # Collapse 3+ blank lines.
    md = "\n".join(out_lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def safe_filename(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", title).strip()
    return (name[:180] if len(name) > 180 else name) + ".md"


def main():
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))["data"]
    OUTPUT_DIR.mkdir(exist_ok=True)
    count = 0
    for entry in data:
        title = entry["title"]
        context = entry["paragraphs"][0]["context"]
        md = context_to_markdown(title, context)
        (OUTPUT_DIR / safe_filename(title)).write_text(md, encoding="utf-8")
        count += 1
    print(f"Wrote {count} markdown files to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
