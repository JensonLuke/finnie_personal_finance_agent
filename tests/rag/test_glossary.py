from pathlib import Path

from finnie.rag.glossary import find_glossary_match
from finnie.rag.loader import LoadedDocument
from finnie.rag.models import DocumentMeta


def _glossary_doc(title: str, article_id: str) -> LoadedDocument:
    meta = DocumentMeta(
        id=article_id, title=title, category="glossary", doc_type="glossary", article_id=article_id
    )
    return LoadedDocument(meta=meta, content=f"{title} definition.", path=Path(f"{article_id}.md"))


def test_exact_title_match():
    docs = [_glossary_doc("ETF", "glossary-etf"), _glossary_doc("Bond", "glossary-bond")]
    match = find_glossary_match("ETF", docs)
    assert match is not None
    assert match.meta.article_id == "glossary-etf"


def test_fuzzy_match_above_threshold():
    docs = [_glossary_doc("Exchange-Traded Fund", "glossary-etf")]
    match = find_glossary_match("what is an exchange traded fund", docs, threshold=80)
    assert match is not None
    assert match.meta.article_id == "glossary-etf"


def test_no_match_below_threshold_returns_none():
    docs = [_glossary_doc("ETF", "glossary-etf")]
    match = find_glossary_match("how do I file my taxes", docs, threshold=90)
    assert match is None


def test_empty_glossary_returns_none():
    assert find_glossary_match("ETF", []) is None
