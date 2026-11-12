"""FastAPI router exposing the RAG pipeline.

Endpoints (mounted by ``rag/server.py`` — see docs/RAG.md):
  POST /rag/retrieve   -> structured retrieval results (no LLM call)
  POST /rag/answer     -> end-to-end RAG answer (local retrieval + Groq)
  GET  /rag/health     -> component status
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline
from rag.response import build_user_answer

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class RetrieveResponse(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    structured_records: List[Dict[str, Any]]
    structured_extras: List[Dict[str, Any]] = Field(default_factory=list)
    intent: Optional[str]
    detected_entities: Dict[str, Any]
    methods_used: List[str]
    warnings: List[str]
    latency_ms: Optional[float]


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    language: Optional[str] = Field(None, pattern="^(en|ar)$")
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class AnswerResponse(BaseModel):
    """User-facing answer contract (no internal evidence or diagnostics).

    Internal structures remain available in the RAGResponse object itself
    (for the Agentic AI layer); they are never serialized here.
    """

    response_id: str
    query: str
    answer: str
    language: str
    status: str                              # success | partial | unavailable | error
    generation_mode: str                     # groq | fallback
    sources: List[str]
    verification: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    kb_loaded: bool
    kb_name: Optional[str] = None
    kb_places: int
    kb_faqs: int
    dense_index: Optional[Dict[str, Any]] = None
    sparse_index: Optional[Dict[str, Any]] = None
    embedding_model: str
    llm_provider: Dict[str, Any] = Field(default_factory=dict)


def create_rag_router(pipeline: RAGPipeline) -> APIRouter:
    """Build the /rag router for an initialized pipeline."""
    router = APIRouter(prefix="/rag", tags=["rag"])

    @router.post("/retrieve", response_model=RetrieveResponse)
    def rag_retrieve(request: RetrieveRequest) -> RetrieveResponse:
        try:
            result = pipeline.retrieve(
                request.query,
                filters=request.filters,
                top_k=request.top_k or pipeline.config.top_k,
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")
        return RetrieveResponse(
            query=result.query,
            documents=[
                {
                    "doc_id": d.doc_id,
                    "content": d.content,
                    "place_id": d.place_id,
                    "collection": d.collection,
                    "section": d.section,
                    "metadata": d.metadata,
                    "score": d.score,
                    "rank": d.rank,
                    "source_type": d.source_type.value,
                    "verification_status": d.verification.status.value,
                    "source_file": d.source.source_file,
                }
                for d in result.documents
            ],
            structured_records=[
                {
                    "place_id": p.place_id,
                    "name": p.name,
                    "name_arabic": p.name_arabic,
                    "city": p.city,
                    "category": p.category,
                    "verification_status": p.verification.status.value,
                    "last_verified": p.verification.last_verified,
                    "data": p.data,
                }
                for p in result.structured_records
            ],
            structured_extras=[
                {
                    "record_type": e.record_type,
                    "record_id": e.record_id,
                    "title": e.title,
                    "verification_status": e.verification.status.value,
                    "last_verified": e.verification.last_verified,
                    "data": e.data,
                }
                for e in result.structured_extras
            ],
            intent=result.intent,
            detected_entities=result.detected_entities,
            methods_used=result.methods_used,
            warnings=result.warnings,
            latency_ms=result.latency_ms,
        )

    @router.post("/answer", response_model=AnswerResponse)
    def rag_answer(request: AnswerRequest) -> AnswerResponse:
        try:
            response = pipeline.answer(
                request.query,
                language=request.language,
                filters=request.filters,
                top_k=request.top_k,
            )
        except Exception:
            logger.exception("RAG answer failed")
            # never leak exception details to the user-facing contract
            raise HTTPException(
                status_code=500,
                detail="The answer service could not process this request. "
                       "Please try again.",
            )
        # The user-facing payload is built by the response layer; internal
        # evidence (context/retrieval) stays out of the serialized contract.
        user = build_user_answer(response)
        return AnswerResponse(**user)

    @router.get("/health", response_model=HealthResponse)
    def rag_health() -> HealthResponse:
        try:
            health = pipeline.health()
        except Exception:
            logger.exception("Health check failed")
            raise HTTPException(
                status_code=500, detail="Health check failed. Check server logs."
            )
        # LLM provider: key presence is *configuration*, not connectivity.
        # Connectivity is only claimed when explicitly probed.
        llm_provider = {
            "provider": "groq",
            "model": health.get("groq_model"),
            "key_configured": bool(health.get("groq_configured")),
            "connectivity": "configured" if health.get("groq_configured") else "not_configured",
        }
        return HealthResponse(
            status="ok" if health.get("kb_loaded") else "degraded",
            kb_loaded=bool(health.get("kb_loaded")),
            kb_name=health.get("kb_name"),
            kb_places=int(health.get("kb_places") or 0),
            kb_faqs=int(health.get("kb_faqs") or 0),
            dense_index={
                "backend": "chromadb",
                "documents": health.get("chroma_docs"),
            },
            sparse_index={
                "backend": "bm25",
                "documents": health.get("bm25_docs"),
            },
            embedding_model=health.get("embedding_model", ""),
            llm_provider=llm_provider,
        )

    return router
