"""Retrieval pipeline: category filter (hard) + difficulty (soft boost) +
score threshold + glossary fast-path. See docs/design/rag-implementation.md
§5-7.

TODO: the "one chunk per article, sorted by boosted score" dedup below is a
crude stand-in for real MMR. The design doc's §5 calls for
`max_marginal_relevance_search`, which needs its own score-threshold handling
once `mmr_lambda` (config.yaml) is tuned against real content -- see that
doc's §10 open questions.
"""

from __future__ import annotations

from finnie.core.config import get_settings
from finnie.rag.build_index import build_index
from finnie.rag.glossary import find_glossary_match
from finnie.rag.loader import KB_ROOT, load_all
from finnie.rag.models import Citation, DocumentMeta, RetrievedChunk

# Soft re-rank only -- never a hard filter, so a thin category doesn't return
# zero results just because it lacks content at the user's exact level.
_DIFFICULTY_BOOST = {
    "beginner": {"beginner": 0.05, "intermediate": 0.0, "advanced": -0.05},
    "intermediate": {"beginner": 0.0, "intermediate": 0.03, "advanced": 0.0},
    "advanced": {"beginner": -0.05, "intermediate": 0.0, "advanced": 0.05},
}


def _meta_from_chunk_metadata(metadata: dict) -> DocumentMeta:
    article_id = metadata.get("article_id", "")
    return DocumentMeta(
        id=metadata.get("chunk_id", article_id),
        title=metadata.get("title", ""),
        category=metadata.get("category", ""),
        difficulty=metadata.get("difficulty", "beginner"),
        doc_type=metadata.get("doc_type", "article"),
        source_name=metadata.get("source_name") or None,
        source_url=metadata.get("source_url") or None,
        article_id=article_id,
    )


def retrieve(
    query: str,
    *,
    category: str | list[str] | None = None,
    experience_level: str = "beginner",
    k: int | None = None,
    store=None,
) -> list[RetrievedChunk]:
    settings = get_settings().rag
    k = k or settings.default_k
    boosts = _DIFFICULTY_BOOST.get(experience_level, {})

    glossary_docs = [d for d in load_all(KB_ROOT) if d.meta.doc_type == "glossary"]
    glossary_hit = find_glossary_match(query, glossary_docs)

    store = store or build_index()
    where = None
    if category:
        categories = [category] if isinstance(category, str) else category
        where = {"category": {"$in": categories}}

    raw_hits = store.similarity_search_with_relevance_scores(query, k=max(k * 3, 10), filter=where)

    scored = [
        (doc, score + boosts.get(doc.metadata.get("difficulty", "beginner"), 0.0))
        for doc, score in raw_hits
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # Pre-seed with the glossary hit's article so the loop below skips it --
    # it's inserted once, explicitly, after the loop. Without this, a term
    # that's both an exact glossary match *and* a natural embedding-search
    # hit (e.g. "diversification") would appear twice: once forced to the
    # front at score 1.0, once again wherever it landed naturally.
    seen_articles: set[str] = {glossary_hit.meta.article_id} if glossary_hit else set()
    results: list[RetrievedChunk] = []
    for doc, score in scored:
        if score < settings.score_threshold:
            continue
        article_id = doc.metadata.get("article_id", "")
        if article_id in seen_articles:
            continue  # one citation-worthy chunk per article in the final set
        seen_articles.add(article_id)
        results.append(
            RetrievedChunk(
                content=doc.page_content,
                metadata=_meta_from_chunk_metadata(doc.metadata),
                score=score,
            )
        )
        if len(results) >= k:
            break

    if glossary_hit:
        meta = glossary_hit.meta.model_copy(update={"doc_type": "glossary"})
        results.insert(0, RetrievedChunk(content=glossary_hit.content, metadata=meta, score=1.0))

    return results[:k]


def to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        if chunk.metadata.article_id in seen:
            continue
        seen.add(chunk.metadata.article_id)
        citations.append(
            Citation(
                title=chunk.metadata.title,
                source_name=chunk.metadata.source_name or "Finnie Knowledge Base",
                source_url=chunk.metadata.source_url,
                category=chunk.metadata.category,
            )
        )
    return citations
