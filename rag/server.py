"""Standalone FastAPI server for the RehlaAI RAG system.

Run with:
    uvicorn rag.server:app --reload

This server does not touch the quarantined `legacy/app.py` (LangChain +
FAISS + Ollama); the /rag router can also be mounted into an existing FastAPI app:

    from rag.api import create_rag_router
    app.include_router(create_rag_router(pipeline))
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from rag.api import create_rag_router
from rag.config import RAGConfig
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RehlaAI Hybrid RAG API",
    description=(
        "Local retrieval (e5 embeddings + ChromaDB + BM25 + structured JSON) "
        "with Groq-based generation."
    ),
    version="1.0.0",
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(config=RAGConfig())
        _pipeline.initialize()
        logger.info("RAG pipeline initialized (health=%s)", _pipeline.health())
    return _pipeline


app.include_router(create_rag_router(get_pipeline()))


@app.get("/")
def home():
    return {
        "service": "RehlaAI Hybrid RAG",
        "endpoints": ["POST /rag/retrieve", "POST /rag/answer", "GET /rag/health"],
    }
