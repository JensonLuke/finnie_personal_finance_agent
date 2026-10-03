"""Loads curated knowledge-base markdown files into DocumentMeta + content.

See docs/design/rag-implementation.md §2-3 for the frontmatter schema and
directory layout this reads.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from finnie.rag.models import DocumentMeta

KB_ROOT = Path(__file__).resolve().parent / "knowledge_base"


class LoadedDocument:
    def __init__(self, meta: DocumentMeta, content: str, path: Path):
        self.meta = meta
        self.content = content
        self.path = path


def _load_one(md_path: Path) -> LoadedDocument:
    post = frontmatter.load(md_path)
    article_id = post.get("id", md_path.stem)
    meta = DocumentMeta(
        id=article_id,
        title=post.get("title", md_path.stem),
        category=post.get("category", md_path.parent.name),
        difficulty=post.get("difficulty", "beginner"),
        doc_type=post.get("doc_type", "article"),
        source_name=post.get("source_name"),
        source_url=post.get("source_url"),
        last_reviewed=str(post.get("last_reviewed")) if post.get("last_reviewed") else None,
        tags=list(post.get("tags", [])),
        article_id=article_id,
    )
    return LoadedDocument(meta=meta, content=post.content, path=md_path)


def load_all(kb_root: Path = KB_ROOT) -> list[LoadedDocument]:
    return [_load_one(path) for path in sorted(kb_root.glob("*/*.md"))]
