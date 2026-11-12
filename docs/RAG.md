# RehlaAI — Hybrid RAG Pipeline

A complete, executable **Hybrid RAG pipeline** for the Egypt AI Tourism Guide project:
**local retrieval** (multilingual-e5 embeddings + ChromaDB + BM25 + structured JSON
lookups + RRF fusion) with **Groq API** used *only* for LLM answer generation.

All retrieval operations run locally. Groq never receives embedding, keyword-search
or structured-lookup requests — it only receives the assembled evidence context.

---

## 1. Architecture

```
data/egypt_landmarks_kb/          legacy per-landmark JSON KB (backup, no longer indexed)
data/rehlaai_rag_data_enhanced.json  ACTIVE knowledge base (enhanced RehlaAI JSON, read-only)
        │
        ▼
rag/ingestion/loader.py           load + validate JSON (folder or single-file format)
rag/ingestion/chunker.py          semantic section chunking (practical JSON kept intact)
rag/ingestion/indexer.py          embeddings → ChromaDB + BM25 corpus, dedup, incremental
        │
        ▼  (persisted locally: data/chroma/, data/chroma/bm25_corpus.json)
rag/embeddings.py                 intfloat/multilingual-e5-large, local, CPU/CUDA
rag/vector_store.py               ChromaDB persistent client wrapper
rag/retrieval/dense.py            semantic search (Chroma cosine)
rag/retrieval/sparse.py           BM25Okapi keyword search (Arabic-aware tokenizer)
rag/retrieval/structured.py       exact JSON field lookups (prices, hours, coords…)
rag/retrieval/hybrid.py           Reciprocal Rank Fusion of the three methods
rag/query_processing.py           bilingual normalization, entity extraction, intent
rag/context_builder.py            deduped evidence block with verification flags
rag/generation/groq_client.py     Groq chat completion (retries, typed errors)
rag/generation/fallback.py        rule-based evidence responder when Groq is down
rag/pipeline.py                   RAGPipeline.retrieve() / answer() facade
rag/api.py  rag/server.py         FastAPI router + standalone server
```

### Local vs. cloud separation

| Component | Where |
|---|---|
| JSON loading, validation, chunking | local |
| Embeddings (e5-large) | local (HF transformers/sentence-transformers) |
| ChromaDB vector index | local (`./data/chroma`) |
| BM25 index | local (in-memory, corpus persisted to `./data/chroma/bm25_corpus.json`) |
| Structured lookups | local (in-memory structured data layer) |
| RRF fusion, context building | local |
| **Answer generation** | **Groq API only** |

---

## 2. Installation

```bash
cd Egypt-AI-Tourism-Guide
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements-rag.txt

# Optional GPU acceleration (this machine has an NVIDIA GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Configure the environment:

```bash
copy .env.example .env
# edit .env and set GROQ_API_KEY=your_real_key
```

The rest of the defaults work out of the box (e5-large model, `data/chroma`
storage, `TOP_K=4`, chunk 300/45, temperature 0.2, model `openai/gpt-oss-120b`).

---

## 3. Building the indexes (JSON ingestion)

```bash
python -m rag.cli build-index          # incremental (skips unchanged content)
python -m rag.cli build-index --rebuild  # wipe Chroma collection + rebuild
```

What happens:
1. The knowledge base source is resolved: the **enhanced RehlaAI JSON**
   (`data/rehlaai_rag_data_enhanced.json`, 9 cities / 28 places / 9
   activities / 3 planning rules / 6 itineraries / 0 transportation routes)
   is loaded by the dedicated adapter `rag/ingestion/enhanced_loader.py`
   (via `KB_SINGLE_FILE`, now defaulted to this file). The legacy
   per-landmark folder `data/egypt_landmarks_kb/` remains only as a backup
   and is no longer loaded or indexed.
2. Data-quality issues are printed as warnings — they never block ingestion.
3. Records are converted into documents (see §4) and embedded **locally**
   with `intfloat/multilingual-e5-large` (`passage: …` prefix).
4. Vectors + metadata go into the persistent Chroma collection
   `rehlai_knowledge`; the text corpus is persisted for BM25.
5. A manifest of content fingerprints enables duplicate prevention and
   incremental updates: rerunning `build-index` after KB edits re-embeds
   only changed documents.

The source JSON is **never** modified. To swap the KB file, set
`KB_SINGLE_FILE=/path/to/other.json` (the loader tries the enhanced-schema
adapter first, then the generic single-file format, then the folder format).

---

## 4. Chunking strategy

- `place` records are split into semantic sections: `overview`, `practical`,
  `facts`, `accessibility` (enhanced KB; the legacy folder format also had a
  `history` section).
- The **`practical` section serializes opening hours + ticket prices + booking
  info as JSON and is never split** — prices and hours always keep their
  context and exact numeric values (e.g. the full `prices.tickets[]` array is
  preserved verbatim).
- Enhanced-KB non-place records become their own documents: `city`
  (overview chunks), `activity`, `planning_rule`, `itinerary` (full days JSON
  as one document), and an explicit `transportation_routes_empty` document
  when the routes collection is empty.
- Long free-text sections use `RecursiveCharacterTextSplitter`
  (`CHUNK_SIZE=300`, `CHUNK_OVERLAP=45`, configurable).
- Each FAQ pair becomes one document (question and answer are never separated)
  — legacy KB only; the enhanced KB has no FAQ collection.
- Every chunk carries `place_id`/`record_id`, name, `city`, `collection`,
  `section`, `verification_status`, `last_verified`, `source_file`,
  `kb_version`, `kb_last_updated`, `name_arabic`, `kb_disclaimer`.

**Limitations:** practical JSON blocks may exceed `CHUNK_SIZE` (kept whole by
design); section splitting assumes the info.json schema — missing sections are
skipped, unknown fields land in the structured layer instead of text chunks.

---

## 5. Embedding model

- Primary: `intfloat/multilingual-e5-large` (Arabic + English), downloaded
  from Hugging Face on first use, cached afterwards, loaded **once per process**.
- E5 formatting is respected: `query: …` for queries, `passage: …` for documents.
- Device: `EMBEDDING_DEVICE=auto|cpu|cuda`; auto-detects CUDA and falls back
  to CPU on out-of-memory or missing GPU. Batch size configurable
  (`EMBEDDING_BATCH_SIZE`).
- Dimension (1024 for e5-large) is validated against the Chroma collection;
  a mismatch raises a clear error telling you to rebuild.

## 6. ChromaDB

Persistent local storage in `CHROMA_PERSIST_DIRECTORY` (default `./data/chroma`),
cosine similarity, metadata filtering, deterministic document IDs, safe
`--rebuild`, incremental upserts. The Chroma client is cached per path so the
indexer and the runtime never hold conflicting handles.

## 7. Retrieval

- **Dense** (`rag/retrieval/dense.py`): Chroma similarity search, configurable
  `DENSE_TOP_K`, metadata filters (e.g. `{"city": "Giza"}`), returns content +
  metadata + similarity.
- **Sparse** (`rag/retrieval/sparse.py`): `BM25Okapi` with an Arabic-aware
  normalizer (diacritics stripped, alef/ya/hamza/teh-marbuta normalized) —
  finds exact place names and tourism terms in both languages, returns scores,
  empty results handled gracefully.
- **Structured** (`rag/retrieval/structured.py`): exact lookups by
  id / name (EN+AR+aliases) / city / category / coordinates, and field-level
  exact retrieval of `ticket_price`, `opening_hours`,
  `location.coordinates`, `accessibility_notes`, `nearest_airport`, …
  It distinguishes:
  - `verified`  — explicit `last_verified` marker in the source
  - `unverified` — value present, no verification marker
  - `estimated` — source flags the value as an estimate
  - `unavailable` — field present but explicitly null/unavailable
  - `missing` — field does not exist in the source
  It **never guesses** values and never uses semantic similarity as a
  substitute for exact lookup.
- **Hybrid** (`rag/retrieval/hybrid.py`): Reciprocal Rank Fusion
  `score(d) = Σ w_s / (RRF_K + rank_s(d))` over the ranked lists of the three
  methods. Ranks are fused instead of raw scores, so incompatible scales are
  never mixed. Exact structured matches for detected entities are *not*
  discarded for low similarity — they carry a configurable weight
  (`STRUCTURED_WEIGHT`). Documents are deduplicated by ID and content.

## 8. Query processing

`rag/query_processing.py` normalizes Arabic/English input (diacritics,
hamza/alef variants), detects language, matches **place names and alternate
names in both languages**, detects Egyptian cities (English + Arabic aliases,
matching city *and* governorate fields), extracts metadata filters, and
classifies intent (price, hours, coordinates, accessibility, transportation,
indoor/outdoor, age suitability, activities, recommendation, definition)
using pattern rules — **not** a hardcoded list of example sentences. Structured
intents route to the structured retriever in addition to hybrid retrieval.

## 9. Context construction

`rag/context_builder.py` builds the LLM-ready evidence block:
- exact structured records first, then retrieved documents;
- every evidence item keeps doc ID, place ID, verification status, source file;
- missing / unverified / estimated information is collected into explicit
  flag sections;
- configurable budget (`MAX_CONTEXT_CHARS=6000`, `MAX_CONTEXT_DOCS=8`);
  when trimming, source attribution is never removed.

## 10. Groq integration

- Official `groq` Python SDK, key read from `GROQ_API_KEY`, model from
  `GROQ_MODEL` (check <https://console.groq.com/docs/models> for supported ids).
- Generation parameters: temperature 0.2, top-p 0.9, configurable max tokens.
- Typed error handling: auth (401), rate limit (429, retried with exponential
  backoff), network/timeouts (retried), empty responses. Secrets are never
  logged or embedded in error messages.
- The system prompt enforces the user-facing output contract: answer only
  from evidence, never invent prices or hours, mark unverified/estimated
  values, cite sources by **place name** (never internal document IDs),
  output clean labeled sections instead of raw JSON, state clearly when
  evidence is insufficient, and answer in the user's language (AR/EN).

**Adding your Groq API key:** create `.env` from `.env.example`, set
`GROQ_API_KEY=...` (and optionally `GROQ_MODEL=...`). Until a key is present,
`/rag/answer` returns the fallback response (retrieved evidence + an
availability notice) instead of crashing.

## 11. Fallback strategy

If Groq is unavailable (no key, network down, rate-limited beyond retries),
`answer()` uses the **deterministic template-based responder**
(`rag/generation/fallback.py`). It builds a natural, intent-aware answer
from the structured evidence — labeled price/hours/location sections in
Arabic or English, itinerary day-by-day formatting, an explicit
"information not available" statement when nothing matches, a single
verification qualification, and one calm notice that AI wording is offline.
It never dumps raw evidence, JSON, document IDs or validation notes, and it
never pretends an LLM generated the answer (`generation_mode="fallback"`).
The Agentic layer can still consume the internal evidence via
`RAGResponse.context`.

## 12. Validation & guardrails

`rag/generation/validator.py` validates generated answers with **graded
outcomes**: `valid` / `partially_supported` / `insufficient_evidence` /
`invalid`. A partially supported answer is qualified with warnings, not
discarded; only empty answers or answers asserting numbers that cannot be
traced to the evidence (fabrication risk) are replaced by the fallback.
Additional guardrails: empty retrieval handling, low-relevance filtering
(`MIN_RELEVANCE_THRESHOLD` — a ranking guardrail, **not** a factual
confidence score), missing-information detection, verification-status
propagation, context size validation, and API/embedding/Chroma/structured
error handling.

## 12b. User-facing output contract

`rag/response.py::build_user_answer` converts the internal `RAGResponse`
into the serialized API contract. The user-facing layer shows only:

```json
{
  "response_id": "…",
  "query": "…",
  "answer": "natural-language answer",
  "language": "ar",
  "status": "success | partial | unavailable | error",
  "generation_mode": "groq | fallback",
  "sources": ["Giza Pyramids", "…"],
  "verification": {"status": "verified | unverified | mixed", "message": "…"},
  "warnings": ["user-relevant notes only"]
}
```

Never shown to users: raw evidence documents, doc IDs, similarity/BM25
scores, internal prompts/context, validation debug codes, provider errors,
stack traces, configuration. Internal structures stay available on the
`RAGResponse` object for the Agentic layer and diagnostics.

## 13. FastAPI

```bash
uvicorn rag.server:app --reload
```

Endpoints (Pydantic-validated, exceptions mapped to HTTP errors):
- `POST /rag/retrieve` — structured retrieval only, no LLM call
  (internal/advanced endpoint; returns documents + structured evidence)
- `POST /rag/answer`   — end-to-end RAG answer, serialized with the
  **user-facing contract only** (see §12b): never internal evidence
- `GET  /rag/health`   — component status: knowledge base, dense index,
  sparse index, embedding model, and `llm_provider` with
  `key_configured` (configuration) kept distinct from actual connectivity

To mount into an existing FastAPI app instead:

```python
from rag.api import create_rag_router
from rag.pipeline import RAGPipeline
pipeline = RAGPipeline(); pipeline.initialize()
app.include_router(create_rag_router(pipeline))
```

(The legacy Ollama/FAISS implementation is quarantined in `legacy/` and is
not part of this pipeline.)

## 14. Agentic AI integration

The Agentic layer should use `pipeline.retrieve()` — it returns structured
evidence, never plain text:

```python
result = pipeline.retrieve("What is the ticket price of the Pyramids?", top_k=4)
result.documents              # RetrievedDocument: doc_id, place_id, score, rank…
result.structured_records     # StructuredPlace: original JSON data, verification, sources
result.structured_records[0].data["ticket_price"]["general_plateau_entry_foreign_adult"]  # 700
result.structured_records[0].verification.status      # verified / unverified / …
result.structured_records[0].source.source_file       # provenance
result.intent, result.warnings, result.methods_used
```

`pipeline.answer()` returns a `RAGResponse` with `context` (including
`missing_info`, `unverified_info`, `estimated_info`, `sources`) and
`retrieval` (the full RetrievalResult) — everything the planner needs.
Itinerary planning, budgeting, scheduling and replanning stay **outside**
this component by design.

## 15. Evaluation

```bash
python -m evaluation.run_retrieval_eval --top-k 4     # no API key needed
python -m evaluation.run_generation_eval              # requires GROQ_API_KEY
```

`evaluation/dataset.json` holds 26 bilingual cases built for the **legacy**
six-landmark KB (city, place, activity, price, hours, transportation, exact
structured, semantic, missing-information, unverified-information queries).

`evaluation/dataset_enhanced.json` holds 26 bilingual cases built **only**
from `data/rehlaai_rag_data_enhanced.json` (cities, places, activities,
prices, opening hours, coordinates, planning rules, itineraries, missing
transportation routes, unverified information). The two datasets share
neither records nor ground truth, so their scores are not directly
comparable.

Retrieval metrics: Precision@K, Recall@K, MRR, Hit Rate@K, latency, intent
accuracy. Results: `evaluation/results_retrieval.json` (legacy dataset),
`evaluation/results_retrieval_enhanced.json` (enhanced dataset).

Generation metrics (LLM-as-a-judge approximation of RAGAS: faithfulness,
answer relevancy, context precision, context recall, unsupported claim rate)
require a real key; the script exits honestly if the key is missing.

## 16. Testing

`python -m pytest`          # full suite; no Groq key needed
python -m pytest -m model # only the embedding-model-dependent tests
```

171 tests cover the enhanced KB (schema validation, per-field verification
semantics, retrieval aliases, chunking, structured retrieval), loading,
chunking, embeddings, Chroma ingestion, dense/BM25/structured retrieval,
hybrid RRF, deduplication, metadata preservation, Arabic + English queries,
missing/unverified handling, context construction, mocked Groq integration
(auth, rate limits, retries, empty responses, secret redaction), the
template fallback, the graded validator, API endpoints, the Agentic
interface, and API-level **output-quality tests** (`tests/test_output_contract.py`:
no raw JSON/doc IDs/stack traces/provider errors leak, fallback readability,
language correctness, verification labels, fabrication rejection).

## 17. Environment variables

See `.env.example` for the full list (`GROQ_API_KEY`, `GROQ_MODEL`,
`EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `CHROMA_PERSIST_DIRECTORY`,
`CHROMA_COLLECTION_NAME`, `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, weights,
context limits, timeouts…). Never commit `.env`.

## 18. Known limitations & future improvements

- KB coverage: 6 landmarks; no activities/itineraries/transport-routes
  collections yet — structured lookup reports those as *missing* rather
  than guessing. Swap in the enhanced RehlaAI JSON via `KB_SINGLE_FILE`.
- Similarity thresholds are ranking guardrails, not accuracy guarantees;
  tune `MIN_RELEVANCE_THRESHOLD` against real queries.
- Generation metrics need a real Groq key (LLM-judge approximation of RAGAS).
- CPU-only PyTorch by default; install CUDA wheels for GPU acceleration.
- Cross-encoder reranking (e.g. bge-reranker) is a natural next step.
- Arabic BM25 relies on normalization, not stemmization; a dedicated Arabic
  analyzer (e.g. ISRI stemmer) could improve recall.
