"""User-facing response assembly.

Strict separation between internal RAG structures and the user contract:

  Internal (kept for the Agentic AI layer / diagnostics):
    RetrievalResult, RetrievedContext, RAGResponse (with context/retrieval)

  User-facing (this module):
    build_user_answer(response) -> dict with the predictable contract:
      {response_id, query, answer, language, status, generation_mode,
       sources, verification{status, message}, warnings}

Rules:
  - ``answer`` is the ONLY natural-language field; it never contains raw
    JSON, document IDs, scores, or internal machinery
  - ``sources`` are friendly references (place names + KB source)
  - ``verification`` summarizes reliability in plain language
  - ``warnings`` are user-relevant notes only (no debug codes, no stack traces)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from rag.schemas import RAGResponse, RetrievedContext, VerificationStatus

_WARNING_MESSAGES = {
    "en": {
        "unsupported_numbers": (
            "Some numbers in the answer could not be fully matched to the "
            "knowledge base and were omitted."
        ),
        "no_evidence": (
            "No matching information was found in the knowledge base for part "
            "of the question."
        ),
    },
    "ar": {
        "unsupported_numbers": (
            "بعض الأرقام في الإجابة لم يتم التأكد منها من قاعدة المعرفة وتم "
            "حذفها."
        ),
        "no_evidence": (
            "لم يتم العثور على معلومات مطابقة في قاعدة المعرفة لجزء من السؤال."
        ),
    },
}

_VERIFICATION_MESSAGES = {
    "en": {
        "verified": "This information comes from verified sources in the knowledge base.",
        "mixed": (
            "Part of this information is well-sourced; other parts have not "
            "been fully verified yet."
        ),
        "unverified": (
            "This information is in the knowledge base but has not been fully "
            "verified yet."
        ),
    },
    "ar": {
        "verified": "هذه المعلومة من مصادر موثقة في قاعدة المعرفة.",
        "mixed": (
            "جزء من هذه المعلومة موثق جيداً، وأجزاء أخرى لم يتم التحقق منها "
            "بشكل كافٍ بعد."
        ),
        "unverified": (
            "هذه المعلومة موجودة في قاعدة المعرفة، لكنها لم تُتحقق منها بشكل "
            "كافٍ بعد."
        ),
    },
}


def _norm_name(text: str) -> str:
    """Lowercase, punctuation-light form for name-in-answer matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u0600-\u06FF]", " ", text or "")).strip().lower()


def _mentioned_in_answer(name: str, answer_norm: str) -> bool:
    """True when the name (or a distinctive part of it) appears in the answer."""
    norm = _norm_name(name)
    if not norm:
        return False
    if norm in answer_norm:
        return True
    # long descriptive names may be trimmed in rendering ("Great Pyramid
    # interior (area entry ticket also required)") — a 5+ char token is a
    # sufficient distinctive anchor
    tokens = [t for t in norm.split() if len(t) >= 5]
    return bool(tokens) and all(t in answer_norm for t in tokens[:2])


_SECTION_PREFIXES = ("place:", "المكان:", "suggested plan:", "الخطة المقترحة:")


def _rendered_headers(answer: str) -> List[str]:
    """Normalized bold section headers of a rendered fallback answer.

    The fallback renders every evidence record as a ``**Title**`` (or
    ``**Place: Title**``) header — a record therefore appears in the
    sources list only when it was ACTUALLY rendered as a section, never
    merely because its name occurred inside another record's text.
    """
    headers: List[str] = []
    for raw in re.findall(r"\*\*([^*\n]+)\*\*", answer or ""):
        header = raw.strip()
        low = header.lower()
        for prefix in _SECTION_PREFIXES:
            if low.startswith(prefix):
                header = header[len(prefix):].strip()
                break
        norm = _norm_name(header)
        if norm and norm not in headers:
            headers.append(norm)
    return headers


def _rendered_as_source(name: str, headers: List[str]) -> bool:
    """True when the record's display name matches a rendered header."""
    norm = _norm_name(name)
    if not norm:
        return False
    for header in headers:
        if header == norm:
            return True
    # long names may be shortened in headers — a shared 5+ char token
    # sequence is still a positive match ("Temple of Hatshepsut (Deir
    # el-Bahari)" rendered as "Temple of Hatshepsut")
    name_tokens = [t for t in norm.split() if len(t) >= 5]
    if name_tokens:
        for header in headers:
            header_tokens = set(header.split())
            if all(t in header_tokens for t in name_tokens[:3]):
                return True
    return False


def _friendly_sources(
    context: Optional[RetrievedContext], answer: str, language: str = "en"
) -> List[str]:
    """Source references derived from the evidence ACTUALLY used.

    Fallback answers render every used record as a bold section header, so
    a source is listed only when its section is really present in the
    answer. Groq answers are prose — there, a source is listed when its
    name is mentioned in the answer. Unrelated retrieved documents are
    never listed. Deduplicated by normalized display name; never contains
    internal IDs, chunk IDs or scores.
    """
    if context is None:
        return []
    headers = _rendered_headers(answer)
    answer_norm = _norm_name(answer)
    fallback_rule_label = (
        "إرشاد تخطيطي من رحلة" if language == "ar" else "RehlaAI planning guidance"
    )

    def _used(name: str) -> bool:
        if not name:
            return False
        if headers:
            return _rendered_as_source(name, headers)
        return _mentioned_in_answer(name, answer_norm)

    notes_used = (
        any("ملاحظات مهمة" in header for header in headers)
        if headers
        else "ملاحظات مهمة" in answer
        or "Important notes" in answer
    )
    names: List[str] = []
    for place in context.structured_records:
        display = place.name if language != "ar" or not place.name_arabic else place.name_arabic
        if _used(display):
            names.append(display)
    for extra in context.structured_extras:
        if extra.record_type in ("city", "activity", "itinerary"):
            if _used(extra.title):
                names.append(extra.title)
        elif extra.record_type == "planning_rule":
            # rules render as an "Important notes" section in the answer
            if notes_used:
                names.append(fallback_rule_label)
    for doc in context.documents:
        name = doc.metadata.get("place_name") or doc.metadata.get("title")
        if name and _used(str(name)):
            names.append(str(name))
    # dedupe by normalized name (and containment), preserve order, cap
    seen: set = set()
    out: List[str] = []
    for name in names:
        key = _norm_name(name)
        if key in seen:
            continue
        # skip variants of an already-listed source ("... (العنوان العربي)"
        # doc-title variants of the same rendered record). Containment only
        # counts when the SHORTER key is multi-token, so a one-word city
        # name ("cairo") never swallows a longer, distinct source title.
        def _is_variant(k: str) -> bool:
            shorter, longer = sorted((k, key), key=len)
            if len(shorter.split()) < 2:
                return False
            return shorter in longer

        if any(_is_variant(existing) for existing in seen):
            continue
        seen.add(key)
        out.append(name)
    return out[:5]


def _summarize_verification(
    context: Optional[RetrievedContext], language: str = "en"
) -> Dict[str, str]:
    """Roll field/record-level verification up to a user-facing summary."""
    lang = language if language in ("en", "ar") else "en"
    if context is None or not (
        context.structured_records
        or context.structured_extras
        or context.documents
    ):
        # nothing was found at all — this is a MISSING result, not an
        # unverified claim; do not label non-existent information "unverified"
        return {
            "status": "missing",
            "message": (
                "هذه المعلومة غير موجودة في قاعدة المعرفة."
                if lang == "ar"
                else "This information was not found in the knowledge base."
            ),
        }
    statuses = {p.verification.status for p in context.structured_records}
    statuses |= {e.verification.status for e in context.structured_extras}
    statuses |= {d.verification.status for d in context.documents}

    if statuses == {VerificationStatus.VERIFIED}:
        return {"status": "verified", "message": _VERIFICATION_MESSAGES[lang]["verified"]}
    if VerificationStatus.VERIFIED in statuses and statuses & {
        VerificationStatus.UNVERIFIED, VerificationStatus.ESTIMATED,
    }:
        return {"status": "mixed", "message": _VERIFICATION_MESSAGES[lang]["mixed"]}
    return {"status": "unverified", "message": _VERIFICATION_MESSAGES[lang]["unverified"]}


def _user_warnings(
    codes: List[str], language: str, generated_by: str
) -> List[str]:
    """Map internal warning codes to user-relevant notes (deduplicated)."""
    table = _WARNING_MESSAGES[language if language in ("en", "ar") else "en"]
    out: List[str] = []
    for code in codes:
        base = code.split(":")[0]
        if base in table:
            msg = table[base]
            if msg not in out:
                out.append(msg)
    if generated_by == "fallback":
        msg = (
            "تم تجميع هذه الإجابة مباشرة من قاعدة المعرفة دون صياغة إضافية."
            if language == "ar"
            else "This answer was assembled directly from the knowledge base."
        )
        if msg not in out:
            out.append(msg)
    return out


def build_user_answer(response: RAGResponse) -> Dict:
    """Convert the internal RAGResponse into the user-facing contract dict."""
    status = response.status
    has_structured_evidence = response.context is not None and bool(
        response.context.structured_records or response.context.structured_extras
    )
    if response.generated_by == "fallback" and status == "success":
        # fallback without structured evidence -> clearly unavailable
        if not has_structured_evidence:
            status = "unavailable"

    if status == "unavailable":
        # nothing found: no sources, no warnings, and the verification
        # block must NOT label the (absent) information "unverified"
        verification = _summarize_verification(None, response.language)
        warnings: List[str] = []
    else:
        verification = _summarize_verification(response.context, response.language)
        warnings = _user_warnings(
            response.warnings + response.verification_flags,
            response.language,
            response.generated_by,
        )
    return {
        "response_id": response.response_id,
        "query": response.query,
        "answer": response.answer,
        "language": response.language,
        "status": status,
        "generation_mode": response.generated_by,          # groq | fallback
        "sources": [] if status == "unavailable" else _friendly_sources(
            response.context, response.answer, response.language
        ),
        "verification": verification,
        "warnings": warnings,
    }
