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


def _friendly_sources(context: Optional[RetrievedContext]) -> List[str]:
    """Human-readable source references (place names, no doc IDs)."""
    if context is None:
        return []
    names: List[str] = []
    for place in context.structured_records:
        names.append(place.name)
    for extra in context.structured_extras:
        if extra.record_type in ("city", "activity", "itinerary"):
            names.append(extra.title)
        elif extra.record_type == "planning_rule":
            names.append("RehlaAI planning guidance")
    for doc in context.documents:
        name = doc.metadata.get("place_name") or doc.metadata.get("title")
        if name:
            names.append(str(name))
    # dedupe, preserve order, cap
    return list(dict.fromkeys(names))[:5]


def _summarize_verification(context: Optional[RetrievedContext]) -> Dict[str, str]:
    """Roll field/record-level verification up to a user-facing summary."""
    if context is None:
        return {"status": "unverified", "message": None}
    statuses = {p.verification.status for p in context.structured_records}
    statuses |= {e.verification.status for e in context.structured_extras}
    statuses |= {d.verification.status for d in context.documents}

    lang = "ar" if any("\u0600" <= ch <= "\u06FF" for ch in (context.query or "")) else "en"

    if not statuses:
        return {"status": "unverified", "message": _VERIFICATION_MESSAGES[lang]["unverified"]}
    if statuses == {VerificationStatus.VERIFIED}:
        return {"status": "verified", "message": _VERIFICATION_MESSAGES[lang]["verified"]}
    if VerificationStatus.VERIFIED in statuses:
        return {"status": "mixed", "message": _VERIFICATION_MESSAGES[lang]["mixed"]}
    return {"status": "unverified", "message": _VERIFICATION_MESSAGES[lang]["unverified"]}


def _user_warnings(
    codes: List[str], language: str, generated_by: str
) -> List[str]:
    """Map internal warning codes to user-relevant notes (deduplicated)."""
    lang = _WARNING_MESSAGES if language == "ar" else None
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
    verification = _summarize_verification(response.context)
    # a response that reports no verifiable context at all is unverified
    status = response.status
    if response.generated_by == "fallback" and status == "success":
        # fallback without evidence -> clearly unavailable
        if response.context is not None and not (
            response.context.structured_records
            or response.context.structured_extras
            or response.context.documents
        ):
            status = "unavailable"
    return {
        "response_id": response.response_id,
        "query": response.query,
        "answer": response.answer,
        "language": response.language,
        "status": status,
        "generation_mode": response.generated_by,          # groq | fallback
        "sources": [] if status == "unavailable" else _friendly_sources(response.context),
        "verification": verification,
        "warnings": _user_warnings(
            response.warnings + response.verification_flags,
            response.language,
            response.generated_by,
        ),
    }
