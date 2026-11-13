"""Pydantic schemas shared across the RehlaAI RAG pipeline.

These models are the stable contract between the RAG layer and the Agentic AI
layer: retrieval results always carry source metadata, verification status and
place identifiers, not just plain text.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """How trustworthy a fact in the knowledge base is."""

    VERIFIED = "verified"          # explicit last_verified / official source
    UNVERIFIED = "unverified"      # no verification marker
    ESTIMATED = "estimated"        # the value itself is flagged as an estimate
    UNAVAILABLE = "unavailable"    # field exists but explicitly null/unavailable
    MISSING = "missing"            # field not present in the source data


class SourceType(str, Enum):
    """Where a piece of evidence came from."""

    KNOWLEDGE_BASE = "knowledge_base"
    DENSE = "dense_retrieval"
    SPARSE = "sparse_retrieval"
    STRUCTURED = "structured_retrieval"
    HYBRID = "hybrid_retrieval"


class SourceMetadata(BaseModel):
    """Provenance of a document / structured record."""

    kb_name: Optional[str] = None
    kb_version: Optional[str] = None
    kb_last_updated: Optional[str] = None
    source_file: Optional[str] = None          # e.g. giza_pyramids/info.json
    record_id: Optional[str] = None            # original landmark/place id
    last_updated: Optional[str] = None         # record-level freshness
    disclaimer: Optional[str] = None


class VerificationMetadata(BaseModel):
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    last_verified: Optional[str] = None
    notes: Optional[str] = None


class StructuredPlace(BaseModel):
    """A place record exactly as stored in the structured data layer.

    ``data`` holds the *original* JSON fields (unmodified names and values) so
    downstream consumers always see exact source values.
    """

    place_id: str
    name: str
    name_arabic: Optional[str] = None
    alternate_names: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    category: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    verification: VerificationMetadata = Field(default_factory=VerificationMetadata)
    source: SourceMetadata = Field(default_factory=SourceMetadata)


class StructuredRecord(BaseModel):
    """A non-place structured record (city, activity, itinerary, rule, route).

    ``data`` holds the original JSON fields verbatim.
    """

    record_type: str                  # city | activity | itinerary | planning_rule | transportation_route
    record_id: str
    title: str
    data: Dict[str, Any] = Field(default_factory=dict)
    verification: VerificationMetadata = Field(default_factory=VerificationMetadata)
    source: SourceMetadata = Field(default_factory=SourceMetadata)


class RetrievedDocument(BaseModel):
    """A single retrieved chunk or structured record."""

    doc_id: str
    content: str
    place_id: Optional[str] = None
    collection: str = "document"          # place | faq | structured | ...
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None          # normalized similarity/relevance
    distance: Optional[float] = None       # raw vector distance (dense only)
    rank: Optional[int] = None             # 1-based rank within the result list
    source_type: SourceType = SourceType.KNOWLEDGE_BASE
    verification: VerificationMetadata = Field(default_factory=VerificationMetadata)
    source: SourceMetadata = Field(default_factory=SourceMetadata)


class RetrievalResult(BaseModel):
    """Output of the retrieval layer (never plain text only)."""

    query: str
    documents: List[RetrievedDocument] = Field(default_factory=list)
    structured_records: List[StructuredPlace] = Field(default_factory=list)
    structured_extras: List["StructuredRecord"] = Field(default_factory=list)
    intent: Optional[str] = None
    intents: List[str] = Field(default_factory=list)   # all detected intents
    detected_entities: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    methods_used: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None

    @property
    def is_empty(self) -> bool:
        return not self.documents and not self.structured_records


class RetrievedContext(BaseModel):
    """Evidence bundle handed to the LLM (and to the Agentic AI layer)."""

    context_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str
    text: str                                  # formatted context block
    documents: List[RetrievedDocument] = Field(default_factory=list)
    structured_records: List[StructuredPlace] = Field(default_factory=list)
    structured_extras: List["StructuredRecord"] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)  # source references
    missing_info: List[str] = Field(default_factory=list)
    unverified_info: List[str] = Field(default_factory=list)
    estimated_info: List[str] = Field(default_factory=list)
    num_chars: int = 0
    truncated: bool = False


class UserVerification(BaseModel):
    """User-facing verification summary (no internal status codes)."""

    status: str = "unverified"               # verified | unverified | mixed
    message: Optional[str] = None


class RAGResponse(BaseModel):
    """Final response of ``answer()`` — the Agentic AI contract.

    ``context``/``retrieval`` are INTERNAL evidence structures (for the
    Agentic AI layer and diagnostics). The user-facing answer is produced by
    ``rag.response.build_user_answer`` and must never contain raw evidence.
    """

    response_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str
    answer: str
    language: str = "en"
    generated_by: str = "groq"                 # groq | fallback | error
    status: str = "success"                    # success | partial | unavailable | error
    verification: UserVerification = Field(default_factory=UserVerification)
    warnings: List[str] = Field(default_factory=list)
    context: Optional[RetrievedContext] = None
    retrieval: Optional[RetrievalResult] = None
    sources: List[str] = Field(default_factory=list)
    verification_flags: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
