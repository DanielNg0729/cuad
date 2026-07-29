"""
AI verification of extracted answers.

Second-opinion pass: given a clause category and the answers a model produced for
it, ask a (possibly different) model to judge whether each answer is actually a
correct instance of that category. Each answer is shown with the surrounding
contract text so the verifier can see it in context, not in isolation.

One category = one model call = one verdict per answer:
    {"id": <review item id>, "verdict": "correct|incorrect|unsure", "reason": str}

Reuses extract.build_chain (provider selection + structured output) and
extract._call_with_retry (429 backoff).
"""

from __future__ import annotations

import asyncio

from pydantic import Field, create_model

import extract

CONTEXT_RADIUS = 400   # chars of document text to show on each side of an answer

# Nested structured output: a list of per-answer verdicts.
_Verdict = create_model(
    "AnswerVerdict",
    index=(int, Field(description="0-based index of the answer being judged.")),
    verdict=(str, Field(description="Exactly one of: correct, incorrect, unsure.")),
    reason=(str, Field(description="One short sentence justifying the verdict.")),
)
VerifyResult = create_model(
    "VerifyResult",
    results=(list[_Verdict], Field(
        description="One entry per candidate answer, in the same order given.")),
)

VERIFY_SYSTEM = (
    "You are a senior legal contract analyst auditing another system's work. For "
    "the given clause category, decide whether each candidate answer is a correct "
    "instance of that category as it appears in the contract context provided. "
    "Judge only the correctness of the labelling, not writing quality. Be strict: "
    "mark 'incorrect' if the quoted text does not actually express that category, "
    "and 'unsure' only if the provided context is genuinely insufficient to tell. "
    "Return exactly one verdict per answer index."
)
VERIFY_USER = (
    "Clause category: {label}\n"
    "Category definition: {description}\n\n"
    "Candidate answers to verify (each with its surrounding contract context):\n"
    "{answers}\n\n"
    "For each answer index above, return a verdict (correct / incorrect / unsure) "
    "and a one-sentence reason."
)

_VALID_VERDICTS = {"correct", "incorrect", "unsure"}


def _context_for(text: str, answer: str) -> str:
    """A window of the document around `answer` (case-insensitive), so the verifier
    sees it in situ. Falls back to the answer alone if it can't be located."""
    if not text or not answer:
        return answer or ""
    idx = text.lower().find(answer.lower())
    if idx == -1:
        return f"(exact location not found in document) {answer}"
    start = max(0, idx - CONTEXT_RADIUS)
    end = min(len(text), idx + len(answer) + CONTEXT_RADIUS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def verify_label(label: str, description: str, answers: list[dict],
                 text: str, model_id: str, shared_context: str | None = None) -> list[dict]:
    """Verify every answer for one category in a single model call.

    answers: [{"id", "text"}]. Returns [{"id", "verdict", "reason"}] aligned to
    the input by index. On a model/parse failure every answer comes back
    'unsure' with the error as the reason, so the UI still gets a result.

    `shared_context`: if given, every answer is shown this SAME context instead
    of its own +/- char window -- e.g. the exact top-k chunks retrieval handed
    the extractor for this label (RAG/hybrid mode), so the verifier judges each
    answer against the evidence the extractor actually had."""
    if not answers:
        return []

    blocks = []
    for i, a in enumerate(answers):
        ctx = shared_context if shared_context is not None else _context_for(text, a["text"])
        blocks.append(f'[{i}] ANSWER: "{a["text"]}"\n    CONTEXT: "{ctx}"')
    payload = {"label": label, "description": description, "answers": "\n\n".join(blocks)}

    chain = extract.build_chain(model_id, VerifyResult,
                                system_prompt=VERIFY_SYSTEM, user_prompt=VERIFY_USER)
    try:
        res = asyncio.run(extract._call_with_retry(chain, payload))
        parsed = res.get("parsed") if isinstance(res, dict) else None
    except Exception as e:  # noqa: BLE001
        return [{"id": a["id"], "verdict": "unsure", "reason": f"Verification failed: {e}"[:200]}
                for a in answers]

    by_index: dict[int, dict] = {}
    if parsed is not None:
        for r in (parsed.results or []):
            v = (r.verdict or "").strip().lower()
            if v not in _VALID_VERDICTS:
                v = "unsure"
            by_index[r.index] = {"verdict": v, "reason": (r.reason or "").strip()}

    out = []
    for i, a in enumerate(answers):
        r = by_index.get(i, {"verdict": "unsure", "reason": "No verdict returned for this answer."})
        out.append({"id": a["id"], "verdict": r["verdict"], "reason": r["reason"]})
    return out
