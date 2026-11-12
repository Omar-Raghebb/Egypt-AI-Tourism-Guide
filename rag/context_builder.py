"""Context construction: retrieved evidence -> structured LLM-ready context.

Rules:
  - deduplicate evidence (by doc_id and by content)
  - preserve exact numerical values verbatim (practical sections are JSON)
  - always attach source references, verification status, and freshness
  - clearly mark confirmed / estimated / unverified / unavailable information
  - enforce a configurable character budget; never drop source attribution
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from rag.config import RAGConfig
from rag.retrieval.structured import StructuredRetriever
from rag.schemas import (
    RetrievedContext,
    RetrievedDocument,
    RetrievalResult,
    SourceType,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds the evidence block handed to the Groq LLM (and Agentic layer)."""

    def __init__(
        self,
        structured: Optional[StructuredRetriever] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.structured = structured
        self.config = config or RAGConfig()

    # ------------------------------------------------------------------
    def build(self, result: RetrievalResult) -> RetrievedContext:
        docs = self._select_documents(result)
        sections: List[str] = []
        sources: List[str] = []
        missing: List[str] = []
        unverified: List[str] = []
        estimated: List[str] = []

        # 1) exact structured values first (highest priority evidence)
        for place in result.structured_records:
            if self.structured is not None:
                summary = self.structured.summary_of(place)
            else:
                summary = f"[{place.place_id}] {place.name} ({place.city or ''})"
            sections.append(
                f"### STRUCTURED RECORD — {place.name} (place_id: {place.place_id})\n"
                f"verification: {place.verification.status.value}"
                + (f" (last_verified: {place.verification.last_verified})"
                   if place.verification.last_verified else "")
                + f"\n{summary}"
            )
            source_ref = place.source.source_file or "knowledge base"
            sources.append(f"{place.place_id}: {source_ref}")
            self._collect_flags(place.verification.status, place.place_id,
                                missing, unverified, estimated)

        # 2) structured extras (cities, activities, itineraries, rules, routes)
        for extra in result.structured_extras:
            sections.append(
                f"### STRUCTURED {extra.record_type.upper()} — {extra.title} "
                f"(id: {extra.record_id})\n"
                f"verification: {extra.verification.status.value}"
                + (f" (last_verified: {extra.verification.last_verified})"
                   if extra.verification.last_verified else "")
                + f"\n{json.dumps(extra.data, ensure_ascii=False)}"
            )
            source_ref = extra.source.source_file or "knowledge base"
            sources.append(f"{extra.record_id}: {source_ref}")
            self._collect_flags(extra.verification.status, extra.record_id,
                                missing, unverified, estimated)

        # 3) retrieved documents
        for doc in docs:
            verification = doc.verification.status.value
            lines = [
                f"### DOCUMENT {doc.doc_id}",
                f"place_id: {doc.place_id or 'n/a'} | section: {doc.section or 'n/a'} "
                f"| verification: {verification} | source: {doc.source.source_file or 'knowledge base'}",
                doc.content.strip(),
            ]
            sections.append("\n".join(lines))
            source_ref = doc.source.source_file or doc.metadata.get(
                "source_file", "knowledge base"
            )
            sources.append(f"{doc.doc_id}: {source_ref}")
            self._collect_flags(doc.verification.status, doc.doc_id,
                                missing, unverified, estimated)

        # 4) unavailability warnings derived from the query intent
        if "structured_intent_without_place" in (result.warnings or []):
            missing.append(
                "the query asks for exact structured data but no specific place "
                "could be identified; exact values cannot be given without a place"
            )
        if "no_relevant_results" in (result.warnings or []):
            missing.append("no relevant evidence found in the knowledge base")

        text = self._render(sections, missing, unverified, estimated)
        truncated = False
        if len(text) > self.config.max_context_chars:
            text = self._fit_budget(sections, missing, unverified, estimated)
            truncated = True

        dedup_sources = list(dict.fromkeys(sources))
        return RetrievedContext(
            query=result.query,
            text=text,
            documents=docs,
            structured_records=result.structured_records,
            structured_extras=result.structured_extras,
            sources=dedup_sources,
            missing_info=missing,
            unverified_info=unverified,
            estimated_info=estimated,
            num_chars=len(text),
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    def _select_documents(self, result: RetrievalResult) -> List[RetrievedDocument]:
        """Deduplicate and bound documents to the context budget."""
        docs = result.documents
        docs = [d for d in docs if d.content and d.content.strip()]
        # dedupe by doc_id then by content
        seen_ids, seen_content = set(), set()
        unique: List[RetrievedDocument] = []
        for doc in docs:
            if doc.doc_id in seen_ids or doc.content.strip() in seen_content:
                continue
            seen_ids.add(doc.doc_id)
            seen_content.add(doc.content.strip())
            unique.append(doc)
        return unique[: self.config.max_context_docs]

    @staticmethod
    def _collect_flags(
        status: VerificationStatus,
        ref: str,
        missing: List[str],
        unverified: List[str],
        estimated: List[str],
    ) -> None:
        label_map = {
            VerificationStatus.MISSING: missing,
            VerificationStatus.UNAVAILABLE: missing,
            VerificationStatus.UNVERIFIED: unverified,
            VerificationStatus.ESTIMATED: estimated,
        }
        target = label_map.get(status)
        if target is not None and status != VerificationStatus.VERIFIED:
            target.append(ref)

    # ------------------------------------------------------------------
    def _render(
        self,
        sections: List[str],
        missing: List[str],
        unverified: List[str],
        estimated: List[str],
    ) -> str:
        parts: List[str] = ["RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):"]
        parts.extend(sections)
        if missing:
            parts.append("MISSING / UNAVAILABLE INFORMATION: " + "; ".join(missing))
        if unverified:
            parts.append(
                "UNVERIFIED INFORMATION (present in the KB but without explicit "
                "verification): " + "; ".join(unverified)
            )
        if estimated:
            parts.append("ESTIMATED VALUES (flagged as estimates by the source): "
                         + "; ".join(estimated))
        parts.append(
            "RULES: use ONLY the evidence above. Do not invent prices, hours, "
            "coordinates or transport routes. Mark unverified/estimated values "
            "as such. Cite evidence as [doc_id or place_id]."
        )
        return "\n\n".join(parts)

    def _fit_budget(
        self,
        sections: List[str],
        missing: List[str],
        unverified: List[str],
        estimated: List[str],
    ) -> str:
        """Trim sections from the end while keeping flags + attribution."""
        header = ["RETRIEVED EVIDENCE (from the local RehlaAI knowledge base):"]
        kept: List[str] = []
        current = self._render(header, missing, unverified, estimated)
        for section in sections:
            candidate = self._render(kept + [section], missing, unverified, estimated)
            if len(candidate) > self.config.max_context_chars and kept:
                break
            kept.append(section)
            current = candidate
        return current
