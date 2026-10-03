# MCP Server

Status: implemented, live-verified (real yfinance + real embeddings), 12
passing tests. This is the Problem Statement's optional bonus deliverable
("Implement MCP server for Claude Desktop integration") — the rubric's
5-point MCP Server bonus line.

## What it's for

Exposes Finnie's finance tools — live market data, portfolio analysis, goal
projection, and the curated knowledge base — to any MCP client (Claude
Desktop, or Claude Code itself) as callable tools, independent of Finnie's
own Streamlit UI or LangGraph workflow.

## Architecture decision: atomic, unnarrated tools

MCP tools return **structured data only** — no LLM-narrated summary. This
was a deliberate choice, not an oversight:

- The calling MCP client (Claude Desktop) brings its own model. Having
  Finnie's tools *also* narrate would mean the client's model is
  interpreting an LLM's prose about the data, one unnecessary layer removed
  from the actual numbers, and it would tie every tool call to an OpenAI API
  key even when the client already has its own model doing the talking.
- It mirrors a pattern already established in this codebase: the Streamlit
  dashboard tabs (Portfolio/Market/Goals) call agents/services directly
  rather than the full LangGraph graph, for the same reason — a dashboard
  view doesn't need an LLM turn in the loop. MCP tools go one layer further
  than that: even the *agent* wrapper is skipped, calling the underlying
  pure functions and services directly (`MarketDataService`,
  `compute_portfolio_metrics`, `project_growth_series`, `retriever.retrieve`).

This means MCP tools share 100% of their logic with the agents and the
Streamlit dashboard — the same resilience chain, the same portfolio math,
the same RAG retrieval and score-threshold honesty (an out-of-scope query
returns an empty list, never a fabricated match). There's exactly one
implementation of "what is a portfolio's diversification score," not three.

## SDK version note

This uses `mcp>=2,<3`. **Important if you're following an older MCP
tutorial**: in `mcp` 1.x, the server class was `FastMCP` from
`mcp.server.fastmcp`. In `mcp` 2.x, it was renamed to `MCPServer` from
`mcp.server.mcpserver` — same API shape (`@server.tool()`, `server.run()`),
different import path. Code written against the old import path fails with
a `ModuleNotFoundError` that names the new path, so it's an easy fix if you
hit it, but worth knowing up front.

## Tool catalog

| Tool | Purpose | Key params | Returns |
|---|---|---|---|
| `get_stock_quote` | Live/cached price quotes | `tickers: list[str]` | `{ticker: {price, change, change_percent, currency, stale, ...}}` |
| `get_stock_history` | Historical daily closes | `ticker`, `period` (e.g. "6mo", "1y") | `[{ticker, date, close}, ...]` |
| `get_fund_info` | Fund/company metadata | `ticker` | `{name, asset_class, expense_ratio, category}` |
| `get_stock_news` | Recent headlines | `ticker`, `limit` | `[{headline, url, published_at}, ...]` |
| `analyze_portfolio` | Value/allocation/diversification | `holdings: [{ticker, shares}, ...]` | `{total_value, allocation_by_asset_class, diversification_score, weighted_expense_ratio}` |
| `project_financial_goal` | Goal projection | `target_amount`, `years`, `monthly_contribution`, `risk_tolerance` | `{projected_value, on_track, assumed_annual_return, yearly_balances}` |
| `search_finance_knowledge_base` | RAG search over curated content | `query`, `category` (optional) | `[{title, source_name, source_url, category, content, relevance_score}, ...]` |

All tools degrade the same way their underlying service does: a
`MarketDataUnavailable` becomes `{"error": "..."}` rather than an unhandled
exception; `search_finance_knowledge_base` returns `[]` for a genuinely
out-of-scope query rather than a low-confidence guess; `analyze_portfolio`
validates its input shape and returns a clear `{"error": ...}` for
malformed holdings rather than a raw `KeyError`.

## Running it

```bash
# Standalone, stdio transport (what Claude Desktop expects):
python -m finnie.mcp_server
# or, after `pip install -e .`:
finnie-mcp
```

## Claude Desktop setup

Add an entry to Claude Desktop's config file (`~/Library/Application
Support/Claude/claude_desktop_config.json` on macOS;
`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "finnie-finance": {
      "command": "/absolute/path/to/finnie_personal_finance_agent/.venv/bin/python",
      "args": ["-m", "finnie.mcp_server"],
      "cwd": "/absolute/path/to/finnie_personal_finance_agent"
    }
  }
}
```

Use the absolute path to this project's `.venv` Python (not a bare
`python`/`python3`), since the server needs the project's dependencies
installed. Restart Claude Desktop after editing the config; the tools
should then appear in its tool picker. `OPENAI_API_KEY` must be set in the
project's `.env` (used for RAG embeddings — the MCP tools don't need it for
market data or goal math, only for `search_finance_knowledge_base`).

## Testing approach

`tests/test_mcp_server.py` calls tools **in-process** via the SDK's own
`server.list_tools()` / `server.call_tool(name, args)` — no stdio
transport, no subprocess, no Claude Desktop needed, the same in-process
testing philosophy as Streamlit's `AppTest` (see
`tests/web_app/test_app.py`). Network/RAG dependencies are monkeypatched to
fakes for the permanent test suite; separately, all seven tools were called
live against the real OpenAI API and real yfinance during development (see
`docs/technical-design.md` §6) — that live run is what caught the
`FastInfo.get()` bug documented in `data/providers/yfinance_provider.py`.

One non-obvious thing the tests had to account for: the MCP SDK's
`CallToolResult` shape differs by return type. A tool returning a `dict`
puts the full JSON in `content[0].text` with `structured_content=None`. A
tool returning a `list` puts the **full** list in
`structured_content={"result": [...]}` and only a partial preview in
`content` — so `structured_content`, when present, is the one to trust.

## Known limitations

- No authentication — this runs as a local stdio server under the
  assumption that whatever spawns it (Claude Desktop, a local dev session)
  is already a trusted context, per MCP's typical local-tool usage model.
- Tools share the same lazily-built, process-global `MarketDataService`
  (one cache/resilience chain across every tool call in the process) — this
  is intentional (see `_market_data`'s docstring) but means a long-running
  server process accumulates cache state over its lifetime rather than
  resetting per call.
- `search_finance_knowledge_base`'s `category` parameter accepts any
  string; an invalid category name silently returns no results (the
  underlying Chroma filter simply matches nothing) rather than validating
  against the known category list. Low-severity — a wrong category name is
  a client-side typo, not a data-integrity issue.
