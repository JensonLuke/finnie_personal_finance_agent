"""Incremental indexing pipeline: knowledge_base/*.md -> Chroma.

Run manually (`python -m finnie.rag.build_index`) or from `finnie.main`, not
at app startup -- see docs/design/rag-implementation.md §4.

Skips re-embedding/re-upserting any chunk whose content hash already matches
what's stored, so editing one article doesn't re-embed the other ~69.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from finnie.rag.chunker import chunk_document
from finnie.rag.loader import KB_ROOT, load_all

STORE_DIR = Path(__file__).resolve().parent / ".chroma_store"
COLLECTION_NAME = "finance_kb"
EMBEDDING_MODEL = "text-embedding-3-small"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_index(kb_root: Path = KB_ROOT, store_dir: Path = STORE_DIR) -> Chroma:
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(store_dir),
        # OpenAI embeddings are optimized for cosine similarity; Chroma
        # defaults to L2, which pairs badly with
        # similarity_search_with_relevance_scores (produces poorly-separated
        # scores that undermine the retriever's score_threshold cutoff --
        # confirmed empirically while scaffolding this module).
        collection_metadata={"hnsw:space": "cosine"},
    )
    existing = store.get(include=["metadatas"])
    existing_hashes = {
        meta["chunk_id"]: meta.get("content_hash") for meta in existing.get("metadatas", []) if meta
    }

    for doc in load_all(kb_root):
        _upsert_document(store, doc, existing_hashes)

    return store


def _upsert_document(store: Chroma, doc, existing_hashes: dict[str, str]) -> None:
    ids, texts, metadatas = [], [], []
    for i, chunk in enumerate(chunk_document(doc)):
        chunk_id = f"{doc.meta.article_id}::chunk{i}"
        content_hash = _content_hash(chunk)
        if existing_hashes.get(chunk_id) == content_hash:
            continue  # unchanged since last index -- skip re-embedding
        ids.append(chunk_id)
        texts.append(chunk)
        metadatas.append(
            {
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "article_id": doc.meta.article_id,
                "title": doc.meta.title,
                "category": doc.meta.category,
                "difficulty": doc.meta.difficulty,
                "doc_type": doc.meta.doc_type,
                "source_name": doc.meta.source_name or "",
                "source_url": doc.meta.source_url or "",
            }
        )
    if ids:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


if __name__ == "__main__":
    build_index()
    print(f"Knowledge base indexed into {STORE_DIR}")
