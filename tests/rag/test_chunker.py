from pathlib import Path

from finnie.rag.chunker import chunk_document
from finnie.rag.loader import LoadedDocument
from finnie.rag.models import DocumentMeta


def _doc(content: str, doc_type: str = "article") -> LoadedDocument:
    meta = DocumentMeta(
        id="test-doc", title="Test", category="stocks", doc_type=doc_type, article_id="test-doc"
    )
    return LoadedDocument(meta=meta, content=content, path=Path("test-doc.md"))


def test_glossary_entries_are_never_split():
    content = "A short definition of a term, well under any chunk-size limit."
    chunks = chunk_document(_doc(content, doc_type="glossary"))
    assert chunks == [content]


def test_article_with_headers_splits_into_multiple_chunks():
    content = "\n\n".join(
        [
            "## Section One\n" + ("word " * 100),
            "## Section Two\n" + ("word " * 100),
            "## Section Three\n" + ("word " * 100),
        ]
    )
    chunks = chunk_document(_doc(content))
    assert len(chunks) >= 3


def test_short_article_still_returns_at_least_one_chunk():
    chunks = chunk_document(_doc("Just one short paragraph."))
    assert len(chunks) >= 1
