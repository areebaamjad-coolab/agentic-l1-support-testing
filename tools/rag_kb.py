from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    score: float | None = None


def retrieve_from_vectorstore(
    *,
    query: str,
    vectorstore_dir: str,
    k: int = 4,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks from a persisted Chroma vector store.

    Notes:
    - If the store doesn't exist yet, returns [] (caller can fallback).
    """

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
    except Exception:
        return []

    try:
        embeddings = OpenAIEmbeddings()
        vs = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        docs: list[Document] = vs.similarity_search(query, k=k)
    except Exception:
        return []

    out: list[RetrievedChunk] = []
    for d in docs:
        meta: dict[str, Any] = d.metadata or {}
        source = str(meta.get("source") or meta.get("id") or meta.get("title") or "kb")
        out.append(RetrievedChunk(content=d.page_content, source=source))
    return out

