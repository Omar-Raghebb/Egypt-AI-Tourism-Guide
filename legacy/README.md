# Legacy implementation (quarantined)

This folder contains the **original competing implementation** of the
RehlaAI project. It is kept for reference only — it is **not** imported,
tested, documented, or served by the current architecture.

## `app.py` — legacy LangChain + FAISS + Ollama service

What it was:
- A separate FastAPI app exposing `POST /ask {"question": ...}`
- Loads `documents.pkl` and the local FAISS index in `egypt_faiss/`
  (built from the old six-landmark KB)
- Embeds with `sentence-transformers/all-MiniLM-L6-v2`
- Generates with a local **Ollama Mistral** model

Why it was quarantined (not deleted):
1. **Competing pipeline.** It duplicates the RAG job of `rag/` with a
   different stack, a different KB, a different endpoint (`/ask` vs
   `/rag/answer`) and a different response contract (raw string vs the
   user-facing answer contract). Keeping two active implementations
   violates the single-source-of-truth rule.
2. **Local-only generation.** It requires a running Ollama Mistral server,
   while the current system uses Groq for generation with a deterministic
   fallback that works with no LLM at all.
3. **Obsolete knowledge base.** It is wired to the old six-landmark KB,
   not the enhanced RehlaAI JSON (28 places / 9 cities / 9 activities /
   3 rules / 6 itineraries).

What was preserved: the file itself (unchanged), plus the old retrieval
artifacts it depends on — `documents.pkl` and `egypt_faiss/` in the
project root — so the legacy stack can still be inspected or revived by
copying this file back if ever needed.

The active implementation lives in `rag/` (see `docs/RAG.md`); the
Streamlit UI (`streamlit_app.py`) talks to `rag/server.py` only.
