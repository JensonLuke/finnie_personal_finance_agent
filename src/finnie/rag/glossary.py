"""Exact/fuzzy term matching so "what's an ETF?" gets the crisp glossary
definition instead of a fragment of a longer article. See
docs/design/rag-implementation.md §6."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from finnie.core.config import get_settings
from finnie.rag.loader import LoadedDocument

# Many glossary titles are "ACRONYM (Full Name)" -- e.g. "IRA (Individual
# Retirement Account)" or "401(k)". partial_ratio scores badly when the query
# is just the acronym and the title's parenthetical is much longer than it
# (the algorithm slides a window the length of the *shorter* string, and a
# short acronym-only query loses almost all its signal against a long
# window of the full title). Scoring the query against several variants of
# the title -- the whole thing, with parens stripped, and the head/inner
# pieces on their own -- and taking the best fixes that without needing a
# different scorer for "acronym" vs "full phrase" queries.
_PAREN_PATTERN = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


_MIN_CANDIDATE_LENGTH = 3  # shorter than this, partial_ratio trivially scores
# ~100 against almost any text containing that substring (e.g. "k" matching
# inside "like") -- found via a real false-positive in the eval suite where
# "401(k)"'s extracted inner group ("k") matched an unrelated weather query.


def _match_candidates(title: str) -> list[str]:
    candidates = [title]
    no_parens = title.replace("(", "").replace(")", "")
    if no_parens != title:
        candidates.append(no_parens)
    match = _PAREN_PATTERN.match(title)
    if match:
        head, inner = match.group(1).strip(), match.group(2).strip()
        candidates.extend(c for c in (head, inner) if len(c) >= _MIN_CANDIDATE_LENGTH)
    return candidates


def find_glossary_match(
    query: str, glossary_docs: list[LoadedDocument], threshold: int | None = None
) -> LoadedDocument | None:
    if not glossary_docs:
        return None
    threshold = threshold or get_settings().rag.glossary_fuzzy_match_threshold
    query_lower = query.lower()

    best_doc: LoadedDocument | None = None
    best_score = -1.0
    for doc in glossary_docs:
        score = max(
            fuzz.partial_ratio(candidate.lower(), query_lower)
            for candidate in _match_candidates(doc.meta.title)
        )
        if score > best_score:
            best_score, best_doc = score, doc

    return best_doc if best_score >= threshold else None
