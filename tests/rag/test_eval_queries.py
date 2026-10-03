"""Retrieval-quality regression suite, per docs/design/rag-implementation.md
§9. Unlike the rest of the test suite, this makes real embedding calls (via
`build_index`/`retrieve` against the real Chroma store) -- it's a fixed,
checked-in eval set specifically meant to catch quality regressions from a
future re-chunking or score_threshold change, not a fast offline unit test.
Skipped automatically when no OpenAI key is configured (e.g. a clean CI
checkout) rather than failing the whole suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finnie.core.config import get_settings
from finnie.rag.build_index import build_index
from finnie.rag.retriever import retrieve

EVAL_PATH = Path(__file__).parent / "eval_queries.jsonl"

pytestmark = pytest.mark.skipif(
    not get_settings().openai_api_key,
    reason="retrieval eval needs a real OpenAI API key to embed queries",
)


def _load_eval_cases() -> list[dict]:
    with EVAL_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def store():
    return build_index()


@pytest.mark.parametrize("case", [c for c in _load_eval_cases() if c["expected_article_id"]])
def test_in_scope_query_finds_expected_article_in_top_3(store, case):
    chunks = retrieve(case["query"], store=store, k=6)
    top_3_article_ids = {c.metadata.article_id for c in chunks[:3]}
    assert case["expected_article_id"] in top_3_article_ids, (
        f"expected {case['expected_article_id']!r} in top 3 for {case['query']!r}, "
        f"got {[c.metadata.article_id for c in chunks[:3]]}"
    )


@pytest.mark.parametrize("case", [c for c in _load_eval_cases() if not c["expected_article_id"]])
def test_out_of_scope_query_returns_nothing_above_threshold(store, case):
    chunks = retrieve(case["query"], store=store)
    assert chunks == [], (
        f"expected no results for out-of-scope query {case['query']!r}, got {chunks}"
    )
