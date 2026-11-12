"""Shared pytest fixtures.

Tests run WITHOUT a Groq API key and WITHOUT the large e5 model: they use a
small local embedding model and a trimmed copy of the real knowledge base so
the suite stays fast and network-light.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rag.config import RAGConfig
from rag.ingestion.chunker import KBChunker
from rag.ingestion.loader import load_kb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_KB = PROJECT_ROOT / "data" / "egypt_landmarks_kb"

# Small model keeps tests fast; the production default is
# intfloat/multilingual-e5-large (see rag/config.py).
TEST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def mini_kb_dir(tmp_path_factory) -> Path:
    """A trimmed knowledge base (2 landmarks) copied from the real data."""
    tmp = tmp_path_factory.mktemp("kb")
    kb_dir = tmp / "egypt_landmarks_kb"
    kb_dir.mkdir()

    index = json.loads((REAL_KB / "index.json").read_text(encoding="utf-8"))
    keep = [lm for lm in index["landmarks"] if lm["id"] in ("giza_pyramids", "luxor_temple")]
    index["landmarks"] = keep
    (kb_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for lm in keep:
        src = REAL_KB / lm["folder"].rstrip("/")
        dst = kb_dir / lm["folder"].rstrip("/")
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("info.json", "faq.json"):
            shutil.copy(src / name, dst / name)
    return kb_dir


@pytest.fixture(scope="session")
def mini_kb(mini_kb_dir):
    return load_kb(mini_kb_dir)


@pytest.fixture()
def test_config(mini_kb_dir, tmp_path) -> RAGConfig:
    """Config bound to the mini KB, temp storage, small embedding model."""
    config = RAGConfig()
    config.kb_path = mini_kb_dir
    config.kb_single_file = None
    config.chroma_persist_directory = tmp_path / "chroma"
    config.chroma_collection_name = "test_collection"
    config.embedding_model = TEST_EMBEDDING_MODEL
    config.embedding_device = "cpu"
    config.top_k = 4
    config.dense_top_k = 6
    config.sparse_top_k = 6
    config.max_context_chars = 3000
    config.max_context_docs = 6
    return config


@pytest.fixture(scope="session")
def chunker() -> KBChunker:
    return KBChunker(chunk_size=300, chunk_overlap=45)
