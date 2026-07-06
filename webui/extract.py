"""
Backend pipeline for the CUAD highlighter web UI.

Reuses the project's existing logic:
  - 41 categories + descriptions from category_descriptions.csv
  - chunking via chunking.make_chunks()
  - structured extraction with LangChain (OpenAI *and* Groq), one Pydantic
    model holding an Optional[str] field per category -- the same idea as
    OpenAITest.py, generalised so a single call works for either provider.

Public entry points used by app.py:
  load_categories()                 -> list[dict] (label, description, ... , color)
  parse_pdf_to_text(pdf_bytes)      -> (full_text, chunks)
  extract(text, chunks, model)      -> {"entities": [...], "spans_by_label": {...}, "usage": {...}}
"""

from __future__ import annotations

import asyncio
import colorsys
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import Field, create_model

# --- make the project root importable so we can reuse chunking.py -----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# _split_markdown is the "markdown" chunking strategy: it cuts the docling
# markdown at each heading, keeping each heading attached to its body.
from chunking import _split_markdown  # noqa: E402

CATEGORY_CSV = PROJECT_ROOT / "category_descriptions.csv"

CONCURRENCY = 4

SYSTEM_PROMPT = (
    "You are a legal contract analyst. For each schema field, return the EXACT "
    "verbatim substring from the section that matches that field's description. "
    "Copy character-by-character; never paraphrase or invent. Set a field to "
    "null if its category does not appear in the section."
)
USER_PROMPT = 'Contract section:\n"""\n{chunk}\n"""'


# --- model registry ---------------------------------------------------------
# provider tells app.py / build_chain which LangChain class to use.
MODELS = [
    {"id": "gpt-5.5",                       "label": "GPT-5.5",                   "provider": "openai"},
    {"id": "gpt-5.4",                       "label": "GPT-5.4",                   "provider": "openai"},
    {"id": "gpt-5.4-mini",                  "label": "GPT-5.4-mini",              "provider": "openai"},
    {"id": "gpt-4o-mini",                   "label": "GPT-4o-mini",               "provider": "openai"},
    {"id": "llama-3.3-70b-versatile",       "label": "Groq · Llama-3.3-70B",      "provider": "groq"},
    {"id": "openai/gpt-oss-120b",           "label": "Groq · GPT-OSS-120B",       "provider": "groq"},
    {"id": "qwen/qwen3-32b",                "label": "Groq · Qwen3-32B",          "provider": "groq"},
]
_MODEL_BY_ID = {m["id"]: m for m in MODELS}

# USD per 1M tokens (input, output). Groq rates are rough estimates; OpenAI
# mirrors OpenAITest.py. Unknown models fall back to "default".
PRICING = {
    "gpt-4o-mini":             (0.15, 0.60),
    "gpt-5.4-mini":            (0.25, 2.00),
    "gpt-5.4":                 (1.25, 10.00),
    "gpt-5.5":                 (1.25, 10.00),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "openai/gpt-oss-120b":     (0.15, 0.75),
    "qwen/qwen3-32b":          (0.29, 0.59),
    "default":                 (1.25, 10.00),
}


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = PRICING.get(model, PRICING["default"])
    return in_tokens / 1e6 * pin + out_tokens / 1e6 * pout


# --- categories -------------------------------------------------------------

def _distinct_color(index: int, total: int) -> str:
    """A visually distinct pastel per category, spaced by the golden angle so
    even adjacent indices look different. Returned as an #rrggbb hex string."""
    hue = (index * 0.61803398875) % 1.0          # golden-ratio hue stepping
    sat = 0.55 + 0.12 * (index % 3) / 2.0        # gentle saturation variation
    light = 0.78 - 0.06 * (index % 2)            # alternate light/slightly-darker
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


_CATEGORIES_CACHE: list[dict] | None = None

# Categories the user adds at runtime through the web UI. Kept in memory (like
# the parsed docs / review state) so they survive across requests but reset on
# a server restart. Appended after the 41 CSV categories, so they flow through
# the extraction schema and highlighting exactly like the built-in ones.
_CUSTOM_CATEGORIES: list[dict] = []


def _base_categories() -> list[dict]:
    """Parse category_descriptions.csv into a stable, ordered list of dicts:
    {label, description, answer_format, group, color, custom}. Order matches the
    CSV, so colours stay consistent between the main page and reference page.
    Cached: the CSV is read once."""
    global _CATEGORIES_CACHE
    if _CATEGORIES_CACHE is not None:
        return _CATEGORIES_CACHE

    df = pd.read_csv(CATEGORY_CSV, encoding="utf-8-sig")
    cats: list[dict] = []
    rows = list(df.iterrows())
    for i, (_, row) in enumerate(rows):
        label = str(row.iloc[0]).split("Category:", 1)[-1].strip()
        desc = str(row.iloc[1]).split("Description:", 1)[-1].strip()
        ans = str(row.iloc[2]).split("Answer Format:", 1)[-1].strip() if len(row) > 2 else ""
        group = str(row.iloc[3]).split("Group:", 1)[-1].strip() if len(row) > 3 else "-"
        cats.append({
            "label": label.title(),
            "description": desc,
            "answer_format": ans if ans.lower() != "nan" else "",
            "group": group if group.lower() != "nan" else "-",
            "color": _distinct_color(i, len(rows)),
            "custom": False,
        })
    _CATEGORIES_CACHE = cats
    return cats


def load_categories() -> list[dict]:
    """The 41 built-in categories plus any custom categories added at runtime.
    This is the single source of truth used to build the extraction schema, so
    custom categories are sent to the model just like the built-in ones."""
    return _base_categories() + _CUSTOM_CATEGORIES


def add_custom_category(label: str, description: str,
                        answer_format: str = "") -> dict:
    """Register a user-defined category so the next extraction includes it.

    The description becomes the schema field's description -- i.e. the prompt the
    model uses to decide what to extract -- mirroring how the 41 built-ins work.
    Raises ValueError on empty/duplicate input so the API can return a 400."""
    label = (label or "").strip()
    description = (description or "").strip()
    answer_format = (answer_format or "").strip()
    if not label:
        raise ValueError("Label is required.")
    if not description:
        raise ValueError("Description is required -- it is used as the model prompt.")

    existing = load_categories()
    if any(c["label"].lower() == label.lower() for c in existing):
        raise ValueError(f"A category named {label!r} already exists.")
    fname = _field_name(label)
    if not fname:
        raise ValueError("Label must contain at least one letter or digit.")
    if any(_field_name(c["label"]) == fname for c in existing):
        raise ValueError(f"Label {label!r} is too similar to an existing category.")

    cat = {
        "label": label.title(),
        "description": description,
        "answer_format": answer_format,
        "group": "custom",
        "color": _distinct_color(len(existing), len(existing) + 1),
        "custom": True,
    }
    _CUSTOM_CATEGORIES.append(cat)
    return cat


def remove_custom_category(label: str) -> bool:
    """Drop a custom category by label (case-insensitive). Returns True if one
    was removed. Built-in categories can't be removed."""
    lc = (label or "").strip().lower()
    for i, c in enumerate(_CUSTOM_CATEGORIES):
        if c["label"].lower() == lc:
            del _CUSTOM_CATEGORIES[i]
            return True
    return False


# --- schema -----------------------------------------------------------------

def _field_name(label: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    # Pydantic field names must be valid identifiers; a custom label starting
    # with a digit (none of the 41 built-ins do) would otherwise be invalid.
    if name and name[0].isdigit():
        name = "f_" + name
    return name


def build_schema(categories: list[dict]):
    """One Pydantic model with an Optional[str] field per category.
    Returns (Schema, {field_name: label})."""
    fields, field_to_label = {}, {}
    for c in categories:
        fname = _field_name(c["label"])
        field_to_label[fname] = c["label"]
        fields[fname] = (Optional[str], Field(default=None, description=c["description"]))
    return create_model("ContractExtraction", **fields), field_to_label


# --- PDF -> text ------------------------------------------------------------

# OCR is off by default: the CUAD contracts are native-text PDFs, so OCR just
# wastes time. Flip OCR_ENABLED to True if you ever feed in scanned documents.
OCR_ENABLED = False
_CONVERTER = None
_CONVERTER_FALLBACK = None


def _get_converter(pypdfium: bool = False):
    """Build (once) a docling converter with OCR toggled per OCR_ENABLED.

    pypdfium=True forces the pypdfium2 PDF backend, which avoids the
    docling_parse code path. We keep it as a fallback because some installed
    docling / docling-parse version combos raise inside the default backend
    (e.g. 'DecodePageConfig has no attribute materialize_bitmap_bytes')."""
    global _CONVERTER, _CONVERTER_FALLBACK
    if pypdfium and _CONVERTER_FALLBACK is not None:
        return _CONVERTER_FALLBACK
    if not pypdfium and _CONVERTER is not None:
        return _CONVERTER

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = OCR_ENABLED
    fmt_kwargs = {"pipeline_options": opts}
    if pypdfium:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        fmt_kwargs["backend"] = PyPdfiumDocumentBackend

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(**fmt_kwargs)}
    )
    if pypdfium:
        _CONVERTER_FALLBACK = converter
    else:
        _CONVERTER = converter
    return converter


def parse_pdf_to_text(pdf_bytes: bytes, filename: str = "upload.pdf") -> tuple[str, list[str]]:
    """docling: PDF bytes -> markdown text, then chunk with the 'markdown'
    strategy (split at headings). The markdown is what gets highlighted; the
    chunks are derived from it, so spans found in a chunk can always be located
    back in the full text. OCR is disabled (see OCR_ENABLED)."""
    import tempfile

    import time
    t0 = time.time()
    print(f"\n[parse] docling: converting {filename!r} ({len(pdf_bytes):,} bytes, "
          f"OCR={'on' if OCR_ENABLED else 'off'})...", flush=True)
    suffix = ".pdf" if filename.lower().endswith(".pdf") else Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        try:
            result = _get_converter().convert(tmp_path)
        except Exception as e:  # noqa: BLE001
            # Default backend failed (often a docling/docling-parse version
            # mismatch). Retry once with the pypdfium2 backend.
            print(f"[parse] default backend failed ({str(e)[:120]}); "
                  f"retrying with pypdfium2 backend...", flush=True)
            result = _get_converter(pypdfium=True).convert(tmp_path)
        text = result.document.export_to_markdown()
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    chunks = _split_markdown(text)
    sizes = [len(c) for c in chunks]
    print(f"[parse] done in {time.time() - t0:.1f}s -> {len(text):,} chars markdown, "
          f"{len(chunks)} chunks (sizes: {sizes})", flush=True)
    return text, chunks


# --- LLM chain --------------------------------------------------------------

def build_chain(model_id: str, schema, max_tokens: Optional[int] = None,
                system_prompt: str = SYSTEM_PROMPT, user_prompt: str = USER_PROMPT):
    """Build a LangChain structured-output chain for either provider.

    system_prompt / user_prompt default to the per-chunk extraction prompts but
    can be overridden so the RAG and verification pipelines reuse this same
    provider-selection + structured-output wiring (see rag.py, verify.py)."""
    from langchain_core.prompts import ChatPromptTemplate

    meta = _MODEL_BY_ID.get(model_id)
    provider = meta["provider"] if meta else ("groq" if "/" in model_id or "llama" in model_id else "openai")

    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {"model": model_id, "temperature": 0}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        llm = ChatGroq(**kwargs)
    else:
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model_id, "temperature": 0}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        llm = ChatOpenAI(**kwargs)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )
    return prompt | llm.with_structured_output(schema, include_raw=True)


def _validate_span(span: Optional[str], chunk: str) -> Optional[str]:
    """Keep only spans that literally appear in the chunk (case-insensitive
    fallback), dropping anything the model paraphrased or invented."""
    if not span:
        return None
    span = span.strip().strip('"').strip("'").strip()
    if span and span in chunk:
        return span
    if span and span.lower() in chunk.lower():
        idx = chunk.lower().find(span.lower())
        return chunk[idx:idx + len(span)]
    return None


_RETRY_RE = re.compile(r"try again in\s+(?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


async def _call_with_retry(chain, inputs: dict, retries: int = 4):
    for attempt in range(retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = "429" in msg or "rate" in msg.lower() or "timeout" in msg.lower()
            if attempt == retries or not transient:
                raise
            m = _RETRY_RE.search(msg)
            wait = (float(m.group(2)) + (int(m.group(1)) * 60 if m.group(1) else 0)) if m else 2.0
            await asyncio.sleep(wait + 0.5)


async def _extract_async(chunks: list[str], schema, field_to_label, model_id: str,
                         concurrency: int, progress_cb=None) -> tuple[dict[str, list[str]], dict, dict]:
    """Run one structured-output call per chunk (capped by `concurrency`).
    Prints per-chunk status to the backend console, like the old CLI runs, and
    fires progress_cb(completed, total, ok, failed) as each chunk finishes so
    the UI can show a live progress bar.

    Returns (preds, usage, stats) where stats carries success/failure counts
    and a sample error message so the caller can report a dead model."""
    chain = build_chain(model_id, schema)
    sem = asyncio.Semaphore(concurrency)
    n = len(chunks)
    print(f"[extract] model={model_id} | {n} chunk(s) | concurrency={concurrency}", flush=True)

    prog = {"completed": 0, "ok": 0, "failed": 0}

    async def process(i: int, chunk: str):
        async with sem:
            try:
                res = await _call_with_retry(chain, {"chunk": chunk})
                ok = True
            except Exception as e:  # noqa: BLE001
                res, ok = e, False
            prog["completed"] += 1
            prog["ok" if ok else "failed"] += 1
            status = "OK" if ok else f"FAILED: {str(res)[:120]}"
            print(f"  chunk {prog['completed']}/{n} ({len(chunk):,} chars) -> {status}", flush=True)
            if progress_cb:
                try:
                    progress_cb(prog["completed"], n, prog["ok"], prog["failed"])
                except Exception:  # noqa: BLE001
                    pass
            return res

    results = await asyncio.gather(*(process(i, c) for i, c in enumerate(chunks)))

    preds: dict[str, list[str]] = {label: [] for label in field_to_label.values()}
    usage = {"input": 0, "output": 0}
    n_ok, n_failed, errors = 0, 0, []
    for chunk, res in zip(chunks, results):
        if isinstance(res, Exception):
            n_failed += 1
            errors.append(str(res))
            continue
        if res is None:
            n_failed += 1
            continue
        n_ok += 1
        raw = res.get("raw") if isinstance(res, dict) else None
        parsed = res.get("parsed") if isinstance(res, dict) else None
        um = getattr(raw, "usage_metadata", None) if raw is not None else None
        if um:
            usage["input"] += um.get("input_tokens", 0)
            usage["output"] += um.get("output_tokens", 0)
        if parsed is None:
            continue
        for fname, label in field_to_label.items():
            span = _validate_span(getattr(parsed, fname, None), chunk)
            if span:
                preds[label].append(span)
    # dedupe per label, keep order
    for label in preds:
        preds[label] = list(dict.fromkeys(preds[label]))

    stats = {
        "n_ok": n_ok,
        "n_failed": n_failed,
        # one representative error (the model_decommissioned / 404 / 401 message)
        "error": errors[0] if errors else None,
    }
    return preds, usage, stats


# --- offsets / entities -----------------------------------------------------

def _find_all_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping start/end of `needle` in `haystack`, case-insensitive."""
    if not needle:
        return []
    spans, start = [], 0
    low_h, low_n = haystack.lower(), needle.lower()
    while True:
        idx = low_h.find(low_n, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + len(needle)
    return spans


def _build_entities(text: str, preds: dict[str, list[str]]) -> list[dict]:
    """Turn {label: [span,...]} into a flat, non-overlapping list of
    {start, end, labels:[...], text} entities suitable for inline highlighting.

    Every occurrence of every span is marked. Where ranges overlap we keep the
    longest and merge any labels that share an identical range so a single mark
    can show multiple tags."""
    raw: dict[tuple[int, int], set] = {}
    for label, spans in preds.items():
        for span in spans:
            for (s, e) in _find_all_occurrences(text, span):
                raw.setdefault((s, e), set()).add(label)

    # sort by start, then longer first, and greedily drop overlaps
    ordered = sorted(raw.items(), key=lambda kv: (kv[0][0], -(kv[0][1] - kv[0][0])))
    chosen: list[dict] = []
    last_end = -1
    for (s, e), labels in ordered:
        if s < last_end:
            continue  # overlaps a span we already kept
        chosen.append({"start": s, "end": e, "labels": sorted(labels), "text": text[s:e]})
        last_end = e
    return chosen


def extract(text: str, chunks: list[str], model_id: str,
            concurrency: int = CONCURRENCY, progress_cb=None) -> dict:
    """Full extraction for one already-parsed document. Returns entities (with
    char offsets into `text`), spans grouped by label, token usage, and run
    stats (ok/failed chunk counts + a sample error if a model failed).

    progress_cb(completed, total, ok, failed) fires as each chunk finishes."""
    import time
    t0 = time.time()
    categories = load_categories()
    schema, field_to_label = build_schema(categories)
    preds, usage, stats = asyncio.run(
        _extract_async(chunks, schema, field_to_label, model_id, concurrency, progress_cb)
    )
    entities = _build_entities(text, preds)
    spans_by_label = {label: spans for label, spans in preds.items() if spans}

    tin, tout = usage["input"], usage["output"]
    cost = estimate_cost(model_id, tin, tout)
    # console summary, like the old CLI runs
    print(f"[extract] done in {time.time() - t0:.1f}s | model={model_id}", flush=True)
    print(f"  chunks OK/failed : {stats['n_ok']}/{stats['n_failed']}", flush=True)
    print(f"  clause types hit : {len(spans_by_label)}/41   highlights: {len(entities)}", flush=True)
    print(f"  tokens in/out    : {tin:,} / {tout:,}", flush=True)
    print(f"  est. cost (USD)  : ${cost:.4f}"
          f"{'  [estimate]' if model_id not in PRICING else ''}", flush=True)
    if stats["error"]:
        print(f"  sample error     : {stats['error'][:200]}", flush=True)

    return {
        "entities": entities,
        "spans_by_label": spans_by_label,
        "usage": usage,
        "model": model_id,
        "n_chunks": len(chunks),
        "n_ok": stats["n_ok"],
        "n_failed": stats["n_failed"],
        "error": stats["error"],
        "estimated_cost_usd": round(cost, 6),
        "n_entities": len(entities),
    }
