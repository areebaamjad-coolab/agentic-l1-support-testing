from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # RAG / KB
    kb_docs_dir: str = os.getenv("KB_DOCS_DIR", "kb_docs")
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", ".vectorstore")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    # Model
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


SETTINGS = Settings()

