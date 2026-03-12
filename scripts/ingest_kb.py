from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_markdown_docs(kb_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for p in kb_dir.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(p), "title": p.stem, "id": p.stem}))
    return docs


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Ingest kb_docs/ into a persisted Chroma vector store.")
    ap.add_argument("--kb-dir", default="kb_docs", help="Directory containing markdown KB docs.")
    ap.add_argument("--out", default=".vectorstore", help="Chroma persist directory.")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    args = ap.parse_args()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.exists():
        raise SystemExit(f"KB dir not found: {kb_dir}")

    docs = _load_markdown_docs(kb_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = OpenAIEmbeddings()
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=args.out)
    vs.persist()

    print(f"Ingested {len(docs)} docs -> {len(chunks)} chunks into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

