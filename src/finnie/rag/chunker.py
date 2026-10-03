"""Markdown-header-aware chunking. See docs/design/rag-implementation.md §4.

Glossary entries are never split -- they're already one atomic ~80-word unit,
and splitting one would defeat the fast-path lookup in glossary.py.
"""

from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from finnie.core.config import get_settings
from finnie.rag.loader import LoadedDocument

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_CHARS_PER_TOKEN = 4  # rough rule of thumb, good enough for chunk sizing


def chunk_document(doc: LoadedDocument) -> list[str]:
    if doc.meta.doc_type == "glossary":
        return [doc.content.strip()]

    settings = get_settings().rag
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
    sections = header_splitter.split_text(doc.content)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens * _CHARS_PER_TOKEN,
        chunk_overlap=settings.chunk_overlap_tokens * _CHARS_PER_TOKEN,
    )
    chunks: list[str] = []
    for section in sections:
        chunks.extend(char_splitter.split_text(section.page_content))
    return chunks or [doc.content.strip()]
