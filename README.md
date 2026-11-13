# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July 2026)

> ## 🆕 RehlaAI Hybrid RAG Pipeline
> **RehlaAI (رحلة)** is an AI-powered Egypt tourism assistant built on an independent
> **Hybrid RAG pipeline** in `rag/`:
> **local embeddings (`intfloat/multilingual-e5-large`) + local ChromaDB + local BM25
> + local structured JSON retrieval + RRF fusion**, with **Groq API used only for LLM
> generation** (retrieval works with no API key, and a deterministic template-based
> fallback answers when generation is offline).
> See **[docs/RAG.md](docs/RAG.md)** for architecture, output contract, API contracts,
> setup, testing and evaluation.
>
> Quick start: `pip install -r requirements-rag.txt` → `python -m rag.cli build-index`
> → `uvicorn rag.server:app` → `streamlit run streamlit_app.py`.

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July 2026)**.

## 👤 Participant
Field	Value
Full Name	Nada Walid
Project Name	Egypt AI Tourism Guide
GitHub Username	 Nada-Ishak  “https://github.com/Nada-Ishak ”
Challenge Batch	June–July 2026
Training Program	Large Language Models (LLMs) Program
Organization	Edrak for AI
---

## 📖 Project Overview

RehlaAI answers questions about Egyptian tourism — places, ticket prices, visiting
hours, activities and itineraries — in **Arabic and English**, using
Retrieval-Augmented Generation over an **enhanced knowledge base**
(28 places, 9 cities, 9 activities, 3 planning rules, 6 itineraries) with
**field-level verification tracking** (verified / unverified / estimated /
conflicting / missing / unavailable are never silently merged).

The original LangChain + FAISS + Ollama implementation was removed in the
repository cleanup (see git history for the deleted files); the active
architecture in `rag/` replaces it entirely.

## ✨ Features
- 🏛 Answers about Egyptian landmarks, cities and activities (bilingual AR/EN)
- 💰 Exact structured lookups: ticket prices, visiting hours, coordinates, accessibility
- 📅 Itinerary understanding ("Plan a 3-day itinerary in Cairo")
- 🧭 Hybrid retrieval: dense (e5) + sparse (BM25) + structured, fused with RRF
- ✅ Verification labels: unverified/estimated information is qualified, never invented
- 🌐 Automatic language detection: answers follow the language of the user's latest question (explicit `language` field still overrides)
- 🔒 Strict user-facing output contract (no raw JSON, document IDs or debug output)
- 🤖 Groq generation with deterministic fallback when no LLM is available
- ⚡ FastAPI REST API (`/rag/answer`, `/rag/retrieve`, `/rag/health`)
- 💻 Premium Streamlit tourism interface with chat history

## 🛠️ Technologies Used
- Python 3.11
- Hybrid RAG: sentence-transformers (e5), ChromaDB, rank-bm25
- Groq (LLM generation only)
- FastAPI + Pydantic v2
- Streamlit + requests
- pytest (171 tests)

## ⚙️ Installation (Windows CMD)
```cmd
python -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements-rag.txt
copy .env.example .env
:: put your Groq key in .env (optional — retrieval & fallback work without it)
python -m rag.cli build-index
uvicorn rag.server:app
```

## 🚀 Usage
The user can ask questions such as:
- What is the ticket price of the Giza Pyramids?
- ما مواعيد زيارة معبد الكرنك؟
- Plan a 3-day itinerary in Cairo
- Which attractions are suitable for families in Giza?
- What activities are available in Luxor?

## 🧪 Tests & Evaluation
```cmd
python -m pytest
python -m evaluation.run_retrieval_eval --dataset evaluation/dataset_enhanced.json
```

## 📸 Demo

![alt text](<Screenshot 2026-07-26 235749.png>)

![alt text](<Screenshot 2026-07-26 232229.png>)
![alt text](<Screenshot 2026-07-26 232039.png>)

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July 2026)**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational purposes.
