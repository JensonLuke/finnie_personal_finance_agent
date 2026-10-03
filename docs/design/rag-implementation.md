# RAG Implementation Design

Status: design draft, no code yet.
Scope: knowledge base content, indexing pipeline, and retrieval — the piece
referenced but deliberately deferred by
[`langgraph-workflow.md`](./langgraph-workflow.md). Targets rubric line items
RAG Implementation (8%) and Educational Content (8%), and feeds citations into
every core/light agent via the `AgentResult.citations` contract already
defined there.

One correction to that doc: §2's agent table lists the News agent's RAG scope
as a `news` category. News content isn't something you curate as 50-100 static
articles — it's inherently time-sensitive and comes from the live news-fetch
tool instead. §8 below replaces that with the actual design: News's RAG call
retrieves *background/glossary context* for jargon in fetched headlines, not
the headlines themselves.

## 1. Vector store: Chroma, not FAISS

The Milestones doc's default suggestion is FAISS; the FAQ explicitly allows
substituting Chroma if you document why. Going with **Chroma**:

- `chromadb` and `langchain-chroma` are already project dependencies in
  `pyproject.toml` — no new dependency to justify.
- Chroma persists to disk with metadata filtering built in (`where` clauses),
  which the category/difficulty filtering in §5 needs directly. FAISS gives
  you a raw index and leaves metadata filtering, persistence, and ID
  management to you to build.
- Local, single-process, file-backed — matches "local deployment is
  sufficient" from the FAQ; no server to run.

Trade-off accepted: Chroma's ANN search is less battle-tested at large scale
than FAISS, which is irrelevant at this project's size (a few hundred to low
thousands of chunks).

## 2. Content plan

Curated content lives as source-of-truth markdown files with YAML frontmatter,
checked into the repo (not fetched at runtime) — reproducible, offline-capable,
and reviewable in a PR like any other content change.

```
src/rag/knowledge_base/
├── stocks/*.md
├── bonds/*.md
├── etfs_and_funds/*.md
├── retirement_accounts/*.md      # 401k, IRA, Roth — also feeds Tax agent
├── tax/*.md                      # tax-advantaged accounts, capital gains basics
├── risk_and_diversification/*.md
├── budgeting_and_saving/*.md
├── market_mechanics/*.md         # how markets work, order types, indices
└── glossary/*.md                 # one short entry per term, not chunked
```

Target: 60-70 full articles across the first 7 categories (comfortably clears
the "50+" Good-band threshold; 100+ is the Excellent band and is a stretch
goal once the pipeline works, not a v1 requirement) plus ~40-60 glossary
terms. Each article: 400-900 words, beginner-first with a "going deeper"
paragraph — matches the personalization-by-experience-level goal in the
problem statement.

**Sourcing note (worth flagging explicitly):** the Milestones doc names
Investopedia, the Bogleheads wiki, and SEC investor.gov as references. Those
are real, actively-maintained third-party sites — copying their article text
verbatim into this repo would be a copyright/attribution problem, not just a
style issue. The plan is: write original explanations informed by those
sources, cite them (`source_name` + `source_url` in frontmatter) as further
reading, and never bulk-scrape. SEC.gov content is US government work and
generally public domain, but Investopedia and Bogleheads content is not —
treat all three the same way to keep the rule simple.

Frontmatter schema per article:

```yaml
---
id: retirement-accounts-roth-vs-traditional-ira
title: "Roth vs. Traditional IRA: What's the Difference?"
category: retirement_accounts
difficulty: beginner        # beginner | intermediate | advanced
doc_type: article           # article | glossary
source_name: "IRS.gov"      # further-reading attribution, not a copy source
source_url: "https://www.irs.gov/..."
last_reviewed: 2026-09-01
tags: [ira, roth, retirement, tax-advantaged]
---
```

Glossary entries use the same schema with `doc_type: glossary` and a body
capped at ~80 words — deliberately short, since they're for fast lookup
(§6), not for being chunked into a longer retrieval pool.

## 3. Document schema in the vector store

Each Chroma record:

| Field | Source | Used for |
|---|---|---|
| `page_content` | chunked article/glossary body | embedding + returned as context |
| `id` | `{article_id}::chunk{n}` | stable upsert key (see §4) |
| `metadata.category` | frontmatter | agent-scoped filtering |
| `metadata.difficulty` | frontmatter | soft boosting, not a hard filter (§6) |
| `metadata.doc_type` | frontmatter | glossary fast-path (§6) |
| `metadata.title`, `source_name`, `source_url` | frontmatter | citation rendering |
| `metadata.article_id` | derived | grouping chunks back to one citation per article |

## 4. Chunking & embedding

- **Splitting**: `MarkdownHeaderTextSplitter` first (keeps `##` sections
  intact, since articles are hand-written with real structure), then
  `RecursiveCharacterTextSplitter` within any section still over ~500 tokens,
  200-token target chunk size with ~40-token overlap. Glossary entries are
  never split — one entry, one chunk, since splitting an 80-word definition
  defeats the point.
- **Embeddings**: `text-embedding-3-small` via `langchain-openai`, decoupled
  from whichever LLM ends up as the chat/reasoning model (Gemini 2.0 Flash
  per the FAQ's recommendation, or GPT — this repo's `pyproject.toml` already
  pulls `langchain-openai`/`openai`, so OpenAI embeddings need no new
  dependency regardless of which provider wins for chat). Cheap enough that
  the whole KB (a few hundred chunks) costs cents to embed.
- **Indexing pipeline**: `src/rag/build_index.py`, run manually or via a
  `finnie` sub-command, not at app startup. For each source file, computes a
  content hash; skips re-embedding/re-upserting chunks whose hash matches
  what's already in Chroma's metadata, so editing one article doesn't re-embed
  the other 69. Deletes-then-reinserts a file's chunks if its hash changed
  (handles chunk-count drift when an edit adds/removes a section).
- **Persistence**: `src/rag/.chroma_store/` (already covered by
  `chroma_db/`-style patterns in `.gitignore`) — rebuilt by `build_index.py`,
  not shipped in git. README setup instructions will call this out as a
  required first-run step, same class of step as installing dependencies.

## 5. Retrieval pipeline

```python
def retrieve(query: str, *, category: str | list[str] | None, experience_level: str,
             k: int = 6, score_threshold: float = 0.35) -> list[RetrievedChunk]:
    ...
```

1. Embed the query.
2. Similarity search with a **hard metadata filter on category** (each agent
   passes its own scope — see §8) but **no hard filter on difficulty**.
   Difficulty is a soft re-rank signal only: a beginner's results get a small
   score boost for `difficulty: beginner` chunks, an advanced user's get one
   for `intermediate`/`advanced`. Hard-filtering on difficulty risks returning
   zero results for thin categories; boosting doesn't.
3. Apply **MMR (maximal marginal relevance)** over the top ~20 raw candidates
   to select the final `k=6` — avoids returning 4 chunks from the same
   article and nothing else when one article dominates similarity scores.
4. Apply `score_threshold`: drop candidates below it rather than always
   forcing `k` results. If everything is dropped, the agent gets an empty
   `retrieved_docs` list and must say plainly that the knowledge base doesn't
   cover this rather than let the LLM improvise an unsourced answer — this is
   the mechanism that keeps `AgentResult.citations` honest (an agent can
   legitimately return `citations: []` with a lower `confidence`, per the
   contract in the workflow doc).
5. Group returned chunks back to one citation per `article_id` (a user
   shouldn't see the same article listed as 3 separate citations because 3 of
   its chunks matched).

## 6. Glossary fast-path

"What's an ETF?"-style queries are common for the target audience (beginners)
and deserve crisp, exact answers rather than a fragment of a longer article.
Before running embedding search, check the query against glossary term names
(exact + simple fuzzy match, e.g. `rapidfuzz` token match ≥90) restricted to
`doc_type: glossary`. On a hit, return that glossary entry as the sole/primary
chunk (still eligible to be supplemented with 1-2 embedding-search results
for "want more detail" follow-through). This is a small addition with an
outsized effect on perceived answer quality for a beginner-focused product —
worth calling out in the demo.

## 7. Citation format

Matches the `Citation` type referenced by `AgentResult.citations` in the
workflow doc:

```python
class Citation(TypedDict):
    title: str
    source_name: str
    source_url: str | None
    category: str
```

`synthesize` (the workflow's final node) renders these as a short "Sources"
footer per response, deduplicated across agents in the same turn — e.g. if
both Q&A and Tax cite the same `retirement_accounts` article, it appears once.

## 8. Per-agent retrieval scope

| Agent | Category filter | Notes |
|---|---|---|
| Finance Q&A | none (all categories except pulled implicitly by query relevance) | broadest scope, matches its role as general education |
| Tax Education | `tax`, `retirement_accounts` | narrow on purpose — this agent should never answer outside account/tax-structure education |
| News Synthesizer | `market_mechanics`, `glossary` | **not** a `news` category (see correction above) — retrieves only background/definitional context for jargon detected in that turn's fetched headlines (e.g. a headline mentions "yield curve inversion" → pull the glossary/mechanics entry, not a "news article" that doesn't exist in the static KB) |
| Portfolio Analysis | none by default | only calls the retriever opportunistically, e.g. to explain a term it used in its own output ("expense ratio") — otherwise pure calculation, no RAG dependency |
| Goal Planning | `risk_and_diversification`, `budgeting_and_saving` | grounds projection caveats and risk-tolerance explanations |
| Market Analysis | none | pure market-data tool, no RAG dependency |

## 9. Retrieval quality testing

A small fixed eval set checked into `tests/rag/eval_queries.jsonl` — ~20
`(query, expected_category, expected_article_id_or_null)` pairs covering: a
clean single-topic query per category, 2-3 glossary-fast-path queries, and
2-3 deliberately out-of-scope queries (expected: empty/below-threshold
result, not a wrong-category guess). Test asserts top-3 recall on the
in-scope set and asserts the out-of-scope set returns nothing above
`score_threshold`. This is cheap to run (no LLM call, just embedding +
search) and is what catches silent regressions from a future re-chunking or
`score_threshold` tweak — feeds the Testing (5%) and RAG (8%) rubric items
together.

## 10. Open questions

- Final `score_threshold` value is a placeholder (0.35) — needs tuning once
  real content and Chroma's actual similarity metric/distance are in place.
- Whether Market Mechanics content needs its own glossary-style short-entry
  variant for News's use case, separate from the longer articles — likely
  yes, revisit once real headlines are being tested against it.
- MMR's diversity/relevance trade-off (`lambda_mult`) is unset here — default
  to LangChain's `0.5` and adjust only if demo queries show repetitive or
  overly scattered results.
