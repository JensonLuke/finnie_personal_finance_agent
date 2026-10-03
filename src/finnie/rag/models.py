"""Shared RAG data contracts. See docs/design/rag-implementation.md."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

Difficulty = Literal["beginner", "intermediate", "advanced"]
DocType = Literal["article", "glossary"]


class Citation(BaseModel):
    """One entry in a response's "Sources" footer. Deduplicated by article,
    per docs/design/rag-implementation.md §7."""

    title: str
    source_name: str
    source_url: str | None = None
    category: str


class DocumentMeta(BaseModel):
    """Frontmatter of one knowledge-base article/glossary entry, per
    docs/design/rag-implementation.md §2-3."""

    id: str
    title: str
    category: str
    difficulty: Difficulty = "beginner"
    doc_type: DocType = "article"
    source_name: str | None = None
    source_url: str | None = None
    last_reviewed: str | None = None
    tags: list[str] = []
    article_id: str

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags_to_strings(cls, value: list) -> list[str]:
        # A bare numeric-looking tag in frontmatter YAML (e.g. `1099` instead
        # of `"1099"`) parses as an int, not a string -- coerce rather than
        # let one mistagged content file crash the whole index build. Found
        # via a real crash while building the knowledge base; see
        # docs/design/rag-implementation.md.
        return [str(item) for item in value]


class RetrievedChunk(BaseModel):
    """One retrieval hit: a chunk of content plus the metadata of the article
    it came from and its (boosted) relevance score."""

    content: str
    metadata: DocumentMeta
    score: float
