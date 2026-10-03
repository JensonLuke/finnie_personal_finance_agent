# Technical Design Document — Finnie AI Finance Assistant

This is the submission-facing technical design document called for in the
Problem Statement's §5 Submission Guidelines. It's a summary and synthesis,
not a replacement for the three detailed design docs that were written
before each piece was built and that this implementation still follows:

- [`design/langgraph-workflow.md`](design/langgraph-workflow.md) — agent
  boundaries, graph topology, communication protocol
- [`design/rag-implementation.md`](design/rag-implementation.md) — knowledge
  base, chunking/embedding, retrieval pipeline
- [`design/market-data-integration.md`](design/market-data-integration.md)
  — provider choice, caching, resilience chain

Where this document summarizes a decision, those three carry the full
reasoning and the open questions still being resolved as the project moves
forward.

## 1. System Overview

Finnie is a multi-agent conversational assistant that teaches investing
basics to beginners: general Q&A, portfolio analysis, live market data,
goal projections, tax-account education, and news summarization — grounded
in a curated knowledge base and clearly labeled as education, not
personalized financial advice, per the Problem Statement's regulatory
guidance.

```
User message
  → load_context   (rehydrate UserProfile/Portfolio for this session)
  → router          (LLM structured-output classification + ticker/goal/
                      risk-tolerance extraction; keyword-based fallback)
  → agent dispatch   (sequential, dependency-ordered — Goal after Portfolio)
  → join → synthesize (LLM merge across agents, dedup citations, disclaimer)
  → response
```

## 2. System Architecture Decisions

| Decision | Choice | Why |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Rubric-weighted deliverable (10%); explicit state channels make the "advanced state management" requirement concrete and testable rather than implicit in prompt text. |
| Chat/reasoning LLM | OpenAI (`gpt-4o-mini` by default) | The Milestones FAQ suggests Gemini 2.0 Flash, but `OPENAI_API_KEY`/`langchain-openai` were already configured in this repo — no new provider or key to set up. Documented explicitly as a deviation, as the FAQ allows. |
| Embedding model | `text-embedding-3-small` (OpenAI) | Decoupled from the chat model choice — RAG needs embeddings regardless of which provider wins for chat. |
| Vector store | Chroma, not FAISS | Already a project dependency; built-in metadata filtering and disk persistence, which the category-filtering requirement needs directly. Documented as a deviation from the Milestones doc's FAISS suggestion. |
| Market data | yfinance only (Alpha Vantage designed but not wired) | Alpha Vantage's free tier (25 req/day) is impractical as a primary source for interactive use; yfinance also covers expense ratios and per-ticker news in one provider. |
| Package layout | `src/finnie/{agents,core,data,rag,web_app,utils,workflow}/` | Matches the prescribed structure, nested under an actual installable package name (`finnie`) rather than a bare `src/` that isn't importable on its own. |
| UI | Streamlit, 4 tabs (Chat/Portfolio/Market/Goals) | Matches the rubric's explicit "Streamlit Application" line item and the Milestones FAQ's tab guidance. |
| Charts | Plotly, colors from a validated CVD-safe categorical palette | Already a project dependency; `web_app/theme.py` fixes one color per asset class across every chart ("color follows the entity"), validated with the palette's own contrast/colorblind-safety script rather than eyeballed. |
| User identity | Minimal local username/password login (`core/auth.py`), replacing the earlier random-per-tab UUID | The Milestones FAQ explicitly allows "simple session-based identification" over full auth, but a random UUID meant no one was ever actually recognized across sessions/devices. A real (if intentionally basic) login — PBKDF2-HMAC-SHA256, per-user salt, constant-time comparison, stdlib only — makes `session_id` a durable per-account identity without taking on a production auth system's scope (no email verification, password reset, or rate limiting). |

## 3. Agent Communication Protocol

Six agents, two tiers by engineering depth (all six are implemented and
distinct — the tiering is about how much custom logic vs. shared RAG
machinery each uses, not about which are "real"):

| Tier | Agents | Character |
|---|---|---|
| Core | Q&A, Portfolio Analysis, Goal Planning, Market Analysis | Custom calculation logic (pure functions, unit tested) plus LLM narration |
| Light | Tax Education, News Synthesizer | RAG-only over a narrow category scope, plus LLM narration |

**The contract.** Every agent implements one method — `run(state) ->
AgentResult` — and is never allowed to call another agent directly.
`AgentResult` is a Pydantic model (`agent`, `summary`, `data`, `citations`,
`confidence`, `degraded`), enforced at the type level so a malformed agent
response fails fast rather than corrupting `synthesize`'s input.

**Dependency handling.** The only current cross-agent dependency is
Goal→Portfolio: a goal-planning query auto-prepends Portfolio into the
dispatch queue so Goal can read `agent_outputs["portfolio"].data` for a
starting balance. This is declared data (the queue), not a code call —
Goal never invokes Portfolio directly.

**Dispatch.** Implemented as a sequential supervisor queue (router
populates `pending_agents`, each agent node pops the next one on exit) —
a deliberate scaffold-stage simplification from the design doc's aspiration
of `Send`-based parallel dispatch for independent agents, flagged explicitly
as a documented follow-up rather than silently narrowed scope.

**Extraction beyond classification.** The router's structured-output call
also extracts stated tickers, a financial goal, and a risk tolerance from
the message (`RouterDecision`). Goal/risk-tolerance extraction is merged
into the persisted `UserProfile` and saved via `ProfileStore` — this is how
the workflow design doc's open question ("where does goal intake live")
got resolved: it rides along with the router call already reading the
message, rather than needing a separate intake node.

**Synthesis.** `synthesize` merges every dispatched agent's `AgentResult`
into one response: a single-agent turn skips the LLM merge entirely (no
value in rephrasing one answer); a multi-agent turn gets an LLM call
instructed to combine without inventing new facts, with a tested fallback
to plain concatenation if that call fails. Citations from every agent are
deduplicated by `(title, source_url)` into one "Sources" footer, and the
standing educational-disclaimer is always appended.

**Resilience.** Any single agent's exception is caught at the graph-node
level and recorded as `ErrorInfo`, which `synthesize` surfaces as a
plain-language caveat rather than crashing the turn — one bad agent
degrades a turn, it doesn't fail it.

## 4. RAG Implementation Details

**Content.** 56 original articles across 8 categories (stocks, bonds,
etfs_and_funds, retirement_accounts, tax, risk_and_diversification,
budgeting_and_saving, market_mechanics) + 16 glossary terms = 72 curated
entries, stored as markdown with YAML frontmatter
(`src/finnie/rag/knowledge_base/`). Written as original explanations
informed by SEC Investor.gov, IRS.gov, Investopedia, and the Bogleheads
wiki — cited as further reading, never scraped verbatim, to avoid the
copyright problem that would come from bulk-copying third-party sites.

**Pipeline.** Markdown-header-aware chunking (`MarkdownHeaderTextSplitter`
+ `RecursiveCharacterTextSplitter`), embedded with `text-embedding-3-small`,
indexed into Chroma with **cosine similarity space explicitly configured**
(`collection_metadata={"hnsw:space": "cosine"}`) — a real bug found and
fixed during implementation: Chroma's default L2 space produces poorly
separated relevance scores against OpenAI's cosine-oriented embeddings,
which would have silently undermined the score-threshold cutoff described
below.

**Retrieval.** Hard category filter (per agent's scope — e.g. Tax Education
is restricted to `tax`/`retirement_accounts`) + soft difficulty boosting
(never a hard filter, so a thin category doesn't return zero results) + a
relevance-score threshold that lets an agent legitimately return zero
citations rather than fabricate sourcing for an out-of-scope question.

**Glossary fast-path.** Beginner "what is X" queries get an exact/fuzzy
match against glossary titles before falling back to embedding search.
Two real bugs were found and fixed here during implementation:
1. A term that was both an exact glossary match *and* a natural embedding
   hit (e.g. "diversification") was appearing twice in results — fixed by
   pre-seeding the dedup set with the glossary hit before the search loop.
2. Acronym-style titles like `"401(k)"` extract an inner parenthetical
   (`"k"`) as a match candidate to handle queries like "what's a 401k" —
   but a single-character candidate trivially scores ~100 against almost
   any text containing that letter (`"401(k)"` was matching `"what's the
   weather like tomorrow"` via "li**k**e"). Fixed with a minimum candidate
   length. Both were caught by actually running the retrieval-quality eval
   suite, not by inspection.

**Citation honesty caveat (known, not yet fixed).** An agent's citations
come from every retrieved chunk passed to the LLM as context, not filtered
to what the LLM's answer actually referenced — confirmed live (a
"what is a stock?" answer only about stocks still cited an unrelated ETF
glossary entry that had cleared the retrieval threshold). Over-citing, not
fabrication, but worth tightening.

**Validation.** `tests/rag/test_eval_queries.py` — 22 fixed cases (one
clean query per category, glossary fast-path queries, out-of-scope
queries) checking top-3 recall and out-of-scope rejection, run against the
real embedding pipeline. This is the regression check for future content
or chunking changes; it's what caught both bugs above.

## 5. Performance Considerations

**Market-data resilience chain** (full detail in the market-data design
doc): in-memory TTL cache (30 min default) → primary provider with
retry+exponential-backoff-with-jitter → circuit breaker (stops retrying
into a known outage) → optional secondary provider → SQLite-backed stale
cache → a checked-in seed cache for common tickers so a true cold start
still shows something rather than fabricating a live-looking price →
`MarketDataUnavailable` as a last resort, always caught and turned into an
honest degraded response, never a crash.

**Cost/latency control in the LLM layer:**
- `synthesize` skips the LLM merge call entirely for single-agent turns.
- Every agent computes its templated/calculator-based summary *first*,
  then attempts LLM narration on top — a narration failure costs nothing
  but reverting to output that was already correct.
- The router's LLM classification has a zero-cost, dependency-free keyword
  fallback (`classify()`) if the structured-output call fails.
- Batch quote fetching (`yf.Tickers`) for multi-holding portfolios, one
  round trip instead of N.

**Indexing cost control:** hash-based incremental upsert
(`build_index.py`) — editing one article re-embeds only its changed
chunks, not the other ~70 documents' worth.

## 6. Testing & Validation Status

102 automated tests (offline unit tests + the live retrieval eval suite,
which skips gracefully without an API key). Coverage by area:

- Pure-function calculators (portfolio metrics, goal projections) — full
  branch coverage including zero-value edge cases.
- Market-data resilience chain — cache hit/expiry (via an injectable fake
  clock, no sleeping), primary failure → secondary → stale/seed fallback,
  true-unavailable, and the news-degrades-quietly exception.
- Router — keyword classification, word-boundary matching, LLM
  structured-output path with a fake chat model, dependency ordering.
- RAG — chunking (glossary never split), glossary matching (including
  regression tests for both bugs above), the 22-case retrieval eval suite.
- Every agent — LLM narration success, LLM-failure fallback to the tested
  templated summary, and the "don't call the LLM when there's nothing to
  say" short-circuits (empty portfolio, no goal set, no headlines).
- `synthesize` — single-agent skip, multi-agent LLM merge, merge-failure
  fallback, citation deduplication/footer formatting, error-note append.
- Chart builders (`web_app/charts.py`) — structural checks (trace count,
  color-per-category consistency across chart types, no dual y-axis, target
  line styled as chrome not a data series) against Plotly `Figure` objects
  directly, no browser needed.
- `years_until` (goal target-date parsing) — missing/unparsable date
  fallback, a real future date, and a past date clamped instead of going
  negative.
- MCP tools (`tests/test_mcp_server.py`) — all seven tools called in-process
  via the SDK's own `list_tools`/`call_tool`, fakes for market data/RAG,
  including malformed-input and empty-result cases.
- `YFinanceProvider` (`tests/data/test_yfinance_provider.py`) — mocked
  `fast_info`/`.info`/`.news`, including a test double that faithfully
  reproduces the real `FastInfo.get()` bug (see below) rather than a plain
  dict that would never have caught it.
- `AuthService` (`tests/core/test_auth.py`) — register/authenticate round
  trip, case-insensitive username lookup with canonical-case preserved,
  wrong password, unknown username, duplicate username (including a
  different-case duplicate), username/password length and character
  validation, and a direct check that a registered password's plaintext
  never appears in the underlying SQLite file's raw bytes. The full
  sign-up → sign-in → log-out flow is additionally covered end-to-end in
  `tests/web_app/test_app.py` via `AppTest`.

**Verified live** (not just against fakes): the full graph compiled and
executed end-to-end against the real OpenAI API for router classification,
per-agent narration, and multi-agent synthesis, including a case where the
router correctly parsed "$40,000 down payment, $250/month, 10 years,
pretty conservative" into structured numbers and a risk tolerance. The full
Streamlit UI was verified the same way, via Streamlit's own `AppTest`
harness (executes the real script in a simulated session, no browser
needed): every button across all four tabs — Analyze Portfolio, Refresh
Quotes, Show Trend, Project — clicked and run with zero exceptions, against
live `yfinance` and the real OpenAI API. All seven MCP tools were likewise
called live (real yfinance, real embeddings) during development.

That live MCP pass caught a real bug: yfinance's `fast_info` object
normalizes snake_case keys via `__getitem__` (`info["previous_close"]`
works) but **not** via `.get()` (`info.get("previous_close")` silently
returns `None`/the default even though the field exists). `YFinanceProvider`
was using `.get()`, so every live quote's `change`/`change_percent` was
silently `0.0` regardless of the real price movement. Fixed with a small
`_fast_info_get` helper that goes through `__getitem__`; verified live
before and after (before: `AAPL change=0.0`; after: `AAPL change=4.18,
change_percent=1.25%`).

**Test coverage measured:** 79% via `pytest --cov=finnie` (up from a 69%
baseline) — but that number itself understates the true figure. Streamlit's
`AppTest` executes `web_app/app.py` in a background thread, and `pytest-cov`
doesn't trace that thread regardless of `concurrency=["thread"]` being
correctly configured and loaded (confirmed directly against `coverage.py`,
which *does* trace it correctly outside of pytest-cov — see
`pyproject.toml`'s `[tool.coverage.run]` comment for the full diagnosis).
`app.py`'s 189 statements report 0% despite 6 passing, assertion-rich tests
that genuinely exercise it. Crediting that file honestly, true coverage is
~94%, comfortably past the rubric's 80% target.

## 7. Current Implementation Status

| Problem Statement objective | Status |
|---|---|
| Six agents, distinct capabilities | Done — see §3 |
| LangGraph orchestration | Done — see §3, verified live end-to-end |
| Error handling & fallback | Done throughout — see §5 |
| Knowledge base (50-100 articles) | 56 articles + 16 glossary = 72 entries, past the 50 threshold |
| Real-time market data | Done — full resilience chain, **verified against the live yfinance network** (quotes, history, fund info); a real bug found and fixed live (see §6) |
| User interface | Done — gated behind a real username/password login (`core/auth.py`); all 4 tabs live once signed in. Chat runs the full graph; Portfolio/Market/Goals call agents/services directly and render Plotly charts (allocation pie+bar, price trend, goal projection vs. target) styled per the dataviz skill's validated palette. Every button (including sign-up/sign-in/log-out) exercised via Streamlit's `AppTest` with zero exceptions, against the real OpenAI API and live yfinance |
| Test suite, 80%+ coverage | 154 tests passing; ~94% true coverage (79% as reported by `pytest --cov`, understated by a pytest-cov/threading measurement gap — see §6) |
| MCP server (optional/bonus) | Done — 7 tools, live-verified, 12 passing tests, documented in `docs/design/mcp-server.md` |

## 8. Rubric Alignment (self-assessment, for planning — not a claimed score)

| Rubric line | Weight | Current strength |
|---|---|---|
| Multi-Agent Architecture | 10% | Strong |
| LangGraph Workflow | 10% | Strong |
| RAG Implementation | 8% | Strong |
| Real-time Data Integration | 7% | Strong — live-verified, one real bug found and fixed |
| MCP Server | 5% (bonus) | Done — 7 tools, live-verified |
| Streamlit Application | 10% | Good — all 4 tabs live and interactive, gated behind real per-account login |
| Conversational Flow | 8% | Good — context now persists per real account, not just per browser tab |
| Data Visualization | 7% | Good — 4 chart types, validated CVD-safe palette |
| Educational Content | 8% | Strong |
| Portfolio Analysis | 7% | Good |
| Market Intelligence | 5% | Good — live-verified |
| Code Organization | 5% | Strong |
| Documentation | 5% | Strong |
| Testing | 5% | Good — 154 tests, ~94% true coverage |

The clearest remaining work is continuing to
grow the knowledge base toward the 100-article "Excellent" band. Known,
smaller polish items (chart dark-mode detection, over-citing, fund-info
caching, one-shot goal intake) are tracked in the README's "Current gaps"
section rather than hidden.
