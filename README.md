# Finnie -- AI Finance Assistant

A multi-agent AI finance assistant that teaches investing basics to
beginners through conversation, portfolio analysis, live market data, and
goal planning. Built for the *Applied Agentic AI for SWEs* capstone
(AI Finance Assistant).

**Status: core pipeline working end-to-end, including the UI, login, and an
MCP server.** All six agents are wired with real LLM narration (with a
tested fallback to templated/calculator output on failure), the LangGraph
workflow routes and synthesizes for real, and the knowledge base has 72
curated entries across all 8 categories. The Streamlit app is gated behind
a real (if intentionally minimal) username/password login -- see
`core/auth.py` -- so `session_id` is now a durable per-account identity
instead of a random per-tab UUID: a returning user's risk tolerance and
goals follow them across browser sessions and devices. All four tabs are
live -- Chat runs the full graph; Portfolio, Market, and Goals call
agents/services directly and render real Plotly charts (allocation pie +
bar, price trend, goal projection vs. target), styled per a validated
CVD-safe palette. An MCP server (`finnie-mcp`) exposes the same underlying
tools -- quotes, history, portfolio analysis, goal projection,
knowledge-base search -- to Claude Desktop or any MCP client. 154 tests
passing, ~94% of the codebase covered (`pytest --cov`'s own reported number
understates one file due to a known pytest-cov/threading limitation -- see
`pyproject.toml`'s `[tool.coverage.run]` comment). Verified live end-to-end
against the real OpenAI API and real `yfinance`, not just fakes -- every
button in every UI tab and every MCP tool has been exercised with zero
exceptions; that live pass also caught and fixed a real bug in how
`yfinance`'s `fast_info.get()` silently returns `None` for valid fields. See
[`docs/technical-design.md`](docs/technical-design.md) for the full current
status, and `docs/design/` for the reasoning behind each major choice.

## Architecture

[`docs/technical-design.md`](docs/technical-design.md) is the consolidated
submission-facing design doc -- system architecture, agent communication
protocol, RAG details, performance considerations, and current status all
in one place. The three documents below are the detailed planning notes
written before each piece was built, and remain the source of truth for
*why* specific implementation choices were made -- read them before
changing orchestration, retrieval, or market-data code:

- [`docs/design/langgraph-workflow.md`](docs/design/langgraph-workflow.md) --
  the six agents, their boundaries, the shared state schema, and how the
  LangGraph orchestrates them.
- [`docs/design/rag-implementation.md`](docs/design/rag-implementation.md) --
  knowledge-base content plan, chunking/embedding, and the retrieval pipeline.
- [`docs/design/market-data-integration.md`](docs/design/market-data-integration.md)
  -- provider choice, caching/TTL, retries, circuit breaker, and the
  no-fabricated-prices fallback strategy.
- [`docs/design/mcp-server.md`](docs/design/mcp-server.md) -- MCP tool
  catalog, Claude Desktop setup, and why tools are atomic/unnarrated.

At a glance:

```
Sign in / sign up (core/auth.py)   -- gates the app; username becomes session_id
User message
  -> load_context   (rehydrate UserProfile/Portfolio for this session)
  -> router          (classify intent; order any cross-agent dependency)
  -> agent(s)         Q&A | Portfolio | Market | Goal | News | Tax
  -> join -> synthesize  (merge AgentResults, dedupe citations, add disclaimer)
  -> response
```

## Project layout

```
config.yaml                 non-secret app configuration
.env / .env.example         secrets (API keys) -- never committed
docs/design/                architecture design docs (read these first)
src/finnie/
├── core/                   shared state models, config loading, persistence, auth
├── data/                   market-data provider abstraction + resilience chain
├── rag/                    knowledge base, chunking, retrieval
│   └── knowledge_base/     curated markdown articles + glossary (frontmatter)
├── agents/                 the six agents
├── workflow/               LangGraph graph, router, supervisor
├── web_app/                Streamlit UI (Chat / Portfolio / Market / Goals)
├── mcp_server.py           MCP server exposing tools to Claude Desktop
└── utils/                  logging, shared helpers
tests/                      mirrors the src/finnie/ layout
```

(This intentionally nests the prescribed `agents/ core/ data/ rag/ web_app/
utils/ workflow/` layout one level deeper, under `src/finnie/`, so the
project is a real installable Python package -- `finnie.agents`, not a bare
top-level `agents` module.)

## Setup

Requires Python 3.12-3.14.

1. **Install dependencies** (editable install, so `import finnie` works
   without a build step):

   ```bash
   pip install -e ".[dev]"
   # or, if you use uv:
   uv pip install -e . --group dev
   ```

2. **Configure secrets.** Copy `.env.example` to `.env` and fill in your
   OpenAI key (used for both chat and embeddings -- see `config.yaml`'s `llm`
   section for why OpenAI over the Milestones FAQ's Gemini suggestion):

   ```bash
   cp .env.example .env
   ```

3. **Build the RAG index** (reads `src/finnie/rag/knowledge_base/`, embeds
   into a local Chroma store):

   ```bash
   python -m finnie.rag.build_index
   ```

4. **Run the app** (you'll land on a sign-in/sign-up page first -- create an
   account, there's no seed/demo account):

   ```bash
   streamlit run src/finnie/web_app/app.py
   ```

5. **Run the tests:**

   ```bash
   pytest --cov=finnie
   ```

6. **(Optional) Run the MCP server**, for Claude Desktop or any MCP client
   -- see [`docs/design/mcp-server.md`](docs/design/mcp-server.md) for the
   full tool catalog and Claude Desktop config snippet:

   ```bash
   finnie-mcp
   ```

7. **(Optional) Enable LangSmith tracing.** No code needed -- everything
   already runs on LangChain/LangGraph, so tracing is env-var-only. Get a
   key at [smith.langchain.com](https://smith.langchain.com), then
   uncomment the three `LANGSMITH_*` lines in `.env` (see
   `.env.example`). See `docs/design/langgraph-workflow.md` §8 for what
   gets tagged and how traces are organized (per-agent, router, synthesize,
   and per-entry-point projects).

## Current gaps (tracked, not hidden)

- The graph dispatches agents sequentially through a supervisor queue even
  when they're independent; true parallel dispatch (`Send`) is a documented
  follow-up in the workflow design doc.
- The knowledge base has 72 curated entries (56 articles across all 8
  categories + 16 glossary terms) -- past the 50-article target, with room
  to keep growing toward the 100-article "Excellent" band.
  `tests/rag/test_eval_queries.py` (a live retrieval-quality regression
  suite, skipped without an API key) tracks top-3 recall as this grows.
- Alpha Vantage (secondary market-data provider) is designed but not
  implemented -- see that doc's §1 for why it's not blocking.
- Router-driven goal/risk extraction only fills in a goal the first time it's
  stated and never asks a clarifying question if the amount/contribution is
  missing -- a real intake flow (multi-turn, clarifying) is still a possible
  upgrade, not just the one-shot extraction currently wired in.
- An agent's citations come from every retrieved chunk passed to the LLM as
  context, not filtered down to what the LLM's answer actually ended up
  referencing -- confirmed live (a "what is a stock?" answer that only
  discussed stocks still cited an unrelated ETF glossary entry that happened
  to clear the retrieval threshold). Over-citing, not fabrication, but worth
  tightening.
- Chart light/dark mode (`web_app/theme.py`'s `get_mode()`) reads
  `st.get_option("theme.base")`, which is `None` under Streamlit's default
  "auto" setting -- there's no JS-side signal for the viewer's real
  OS/browser preference exposed server-side. Rather than leave that
  non-deterministic, `.streamlit/config.toml` pins `base = "light"`, so the
  app (and its charts) always render in light mode consistently. This is a
  deliberate "one consistent look" trade-off, not adaptive dark mode --
  `get_mode()`'s dark-mode code path is real and tested, but currently
  unreachable in the shipped app.
- `MarketDataService.get_fund_info` (used for expense ratio / asset class)
  isn't behind the TTL cache the way quotes are -- the Portfolio tab
  re-fetches it from `yfinance` on every "Analyze Portfolio" click rather
  than reusing a recent result. Not incorrect, just an easy win left for
  later.
- The MCP server has no authentication (fine for its local-stdio-process
  usage model) and doesn't validate `search_finance_knowledge_base`'s
  `category` parameter against the known category list -- see
  `docs/design/mcp-server.md`'s "Known limitations" for the full list.
- The Streamlit login (`core/auth.py`) is real (hashed + salted passwords,
  never plaintext) but intentionally minimal: no email verification, no
  password-reset flow, and no login-attempt rate limiting/lockout. Fine for
  a locally-run demo; not a production-grade auth system. Portfolio
  holdings still aren't persisted per-account (only risk tolerance and
  goals are, via `ProfileStore`) -- an easy follow-up now that accounts
  exist.

## Compliance note

This assistant provides financial *education*, not personalized financial
*advice* -- every synthesized response includes that disclaimer, per the
problem statement's regulatory-compliance guidance.
