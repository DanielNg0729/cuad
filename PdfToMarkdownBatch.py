"""Convert the actual source PDFs of the 102 CUAD test contracts to Markdown
with docling, so chunking.py can chunk real document structure instead of the
flat `context` text stored in test.json.

This is the docling path (same technique as ConvertPDFtoMarkdown.py) applied in
batch: for every contract title in test.json we locate its PDF under
dataset/full_contract_pdf/, run docling, and write "<title>.md" into
markdown_docling/.

Mapping title -> PDF:
  * exact: a PDF whose filename stem == the title (100 of 102).
  * prefix: a PDF whose stem starts with the title (e.g. Harpoon's
    "...Development Agreement_Option Agreement.pdf").
  * missing: one contract (Babcock & Wilcox IP) has no PDF in the dataset, so we
    fall back to the test.json context so all 102 .md files still exist.

The run is resumable -- a contract whose .md already exists (and is non-empty)
is skipped -- because docling is ~15-20s per PDF (~30 min for the full set).

Usage:
    python PdfToMarkdownBatch.py                 # convert all, skipping done
    python PdfToMarkdownBatch.py --force         # re-convert everything
    python PdfToMarkdownBatch.py --limit 5       # just the first 5 (smoke test)

Next:
    python chunking.py --strategy markdown       # chunk markdown_docling/*.md
"""

import argparse
import json
import re
import time
from pathlib import Path

TEST_JSON = Path("test.json")
PDF_ROOT = Path("dataset/full_contract_pdf")
OUT_DIR = Path("markdown_docling")


def safe_filename(title: str) -> str:
    """Mirror of chunking.safe_filename so the .md names line up with what the
    'markdown' strategy looks for."""
    name = re.sub(r'[<>:"/\\|?*]', "_", title).strip()
    return (name[:180] if len(name) > 180 else name) + ".md"


def build_pdf_index(root: Path) -> dict[str, list[Path]]:
    """Map lowercased filename stem -> list of PDF paths under `root`."""
    idx: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() == ".pdf":
            idx.setdefault(p.stem.lower(), []).append(p)
    return idx


def find_pdf(title: str, idx: dict[str, list[Path]],
             all_pdfs: list[Path]) -> Path | None:
    """Resolve a contract title to its PDF: exact stem first, then the shortest
    PDF whose stem starts with the title (handles "..._Option Agreement.pdf")."""
    hits = idx.get(title.lower())
    if hits:
        return hits[0]
    cands = [p for p in all_pdfs if p.stem.lower().startswith(title.lower())]
    if cands:
        return min(cands, key=lambda p: len(p.stem))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Re-convert even contracts whose .md already exists.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N contracts (0 = all).")
    args = ap.parse_args()

    data = json.loads(TEST_JSON.read_text(encoding="utf-8"))["data"]
    contexts = {e["title"]: e["paragraphs"][0]["context"] for e in data}
    titles = [e["title"] for e in data]
    if args.limit:
        titles = titles[:args.limit]

    OUT_DIR.mkdir(exist_ok=True)
    idx = build_pdf_index(PDF_ROOT)
    all_pdfs = [p for paths in idx.values() for p in paths]

    # Import + instantiate docling once (model load is the slow part).
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()

    ok = skipped = fallback = errors = 0
    t0 = time.time()
    for i, title in enumerate(titles, 1):
        out = OUT_DIR / safe_filename(title)
        if out.exists() and out.stat().st_size > 0 and not args.force:
            skipped += 1
            print(f"[{i}/{len(titles)}] skip (done): {title[:60]}", flush=True)
            continue

        pdf = find_pdf(title, idx, all_pdfs)
        if pdf is None:
            out.write_text(f"# {title}\n\n{contexts[title]}\n", encoding="utf-8")
            fallback += 1
            print(f"[{i}/{len(titles)}] NO PDF -> context fallback: {title[:50]}",
                  flush=True)
            continue

        t = time.time()
        try:
            result = converter.convert(str(pdf))
            md = result.document.export_to_markdown()
            out.write_text(f"# {title}\n\n{md}\n", encoding="utf-8")
            ok += 1
            print(f"[{i}/{len(titles)}] ok ({time.time() - t:.0f}s, "
                  f"{len(md):,} chars): {title[:45]}", flush=True)
        except Exception as e:
            errors += 1
            print(f"[{i}/{len(titles)}] ERROR {title[:45]}: {e}", flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s. "
          f"converted={ok} skipped={skipped} fallback={fallback} errors={errors}")
    print(f"Markdown in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
