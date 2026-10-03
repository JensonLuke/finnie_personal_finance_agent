"""MCP server exposing Finnie's finance tools via the Model Context
Protocol -- the Problem Statement's optional bonus deliverable ("Implement
MCP server for Claude Desktop integration"). See
docs/design/mcp-server.md for the protocol/tool documentation and Claude
Desktop setup instructions.

Tools wrap the same underlying data/calculation layer the LangGraph agents
use (MarketDataService, compute_portfolio_metrics, project_growth_series,
the RAG retriever) as standalone, atomic primitives -- deliberately NOT
LLM-narrated. The calling MCP client (Claude Desktop, or any other) brings
its own model to turn these structured results into conversation; Finnie's
own narration layer (agents/narration.py) is specific to the LangGraph
workflow. This mirrors how the Streamlit dashboard tabs call
agents/services directly rather than the full graph -- see
web_app/app.py's module docstring for the same reasoning, applied here one
layer lower (raw data/calculations, not even agent-level).

Run standalone:
    python -m finnie.mcp_server
    # or, after `pip install -e .`:
    finnie-mcp
"""

from __future__ import annotations

import os
from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from finnie.agents.goal_agent import project_growth_series
from finnie.agents.portfolio_agent import compute_portfolio_metrics
from finnie.core.config import get_settings
from finnie.core.state import Holding
from finnie.data.market_data_service import MarketDataService
from finnie.data.models import MarketDataUnavailable
from finnie.data.providers.yfinance_provider import YFinanceProvider
from finnie.data.seed import load_seed_cache
from finnie.rag.retriever import retrieve

# Only applied if the user hasn't already set LANGSMITH_PROJECT themselves
# (see .env.example) -- keeps MCP-server traces grouped separately from the
# web app's in the LangSmith dashboard without overriding an explicit choice.
os.environ.setdefault("LANGSMITH_PROJECT", "finnie-mcp")

server = MCPServer(
    name="finnie-finance",
    instructions=(
        "Finance education tools: live market data, portfolio analysis, goal "
        "projections, and a curated financial-education knowledge base. These "
        "are educational tools, not personalized financial advice -- present "
        "results accordingly, with appropriate uncertainty."
    ),
)

_RISK_TO_ASSET_CLASS = {"conservative": "bond", "moderate": "equity", "aggressive": "equity"}


@lru_cache
def _market_data() -> MarketDataService:
    # Built lazily, once per server process -- same reasoning as
    # web_app/app.py's cached `get_services()`: the cache/resilience layer
    # is meant to be shared across every tool call, not rebuilt per call.
    return MarketDataService(primary=YFinanceProvider(), seed_cache=load_seed_cache())


@server.tool()
def get_stock_quote(tickers: list[str]) -> dict:
    """Get live (or cached/fallback) price quotes for one or more stock/ETF
    tickers, e.g. ["AAPL", "SPY"]."""
    try:
        quotes = _market_data().get_quotes(tickers)
    except MarketDataUnavailable as exc:
        return {"error": str(exc)}
    return {ticker: quote.model_dump(mode="json") for ticker, quote in quotes.items()}


@server.tool()
def get_stock_history(ticker: str, period: str = "6mo") -> list[dict]:
    """Get historical daily closing prices for a ticker. `period` is a
    yfinance-style string, e.g. "1mo", "6mo", "1y", "5y"."""
    try:
        bars = _market_data().get_history(ticker, period)
    except MarketDataUnavailable as exc:
        return [{"error": str(exc)}]
    return [bar.model_dump(mode="json") for bar in bars]


@server.tool()
def get_fund_info(ticker: str) -> dict:
    """Get fund/company metadata for a ticker: name, asset class, and
    expense ratio (funds/ETFs only -- null for individual stocks)."""
    try:
        info = _market_data().get_fund_info(ticker)
    except MarketDataUnavailable as exc:
        return {"error": str(exc)}
    return info.model_dump(mode="json") if info else {"error": f"No fund info found for {ticker}"}


@server.tool()
def get_stock_news(ticker: str, limit: int = 5) -> list[dict]:
    """Get recent news headlines for a ticker. Returns an empty list rather
    than an error if news is temporarily unavailable -- see
    MarketDataService.get_news's docstring for why."""
    news = _market_data().get_news(ticker, limit)
    return [item.model_dump(mode="json") for item in news]


@server.tool()
def analyze_portfolio(holdings: list[dict]) -> dict:
    """Analyze a portfolio's total value, asset allocation, diversification
    score, and weighted expense ratio.

    holdings: a list of objects like {"ticker": "AAPL", "shares": 10}.
    """
    try:
        parsed = [
            Holding(ticker=str(h["ticker"]).upper(), shares=float(h["shares"])) for h in holdings
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"Invalid holdings format: {exc}"}
    if not parsed:
        return {"error": "No holdings provided."}

    market_data = _market_data()
    tickers = [h.ticker for h in parsed]
    try:
        quotes = market_data.get_quotes(tickers)
    except MarketDataUnavailable as exc:
        return {"error": str(exc)}

    fund_infos = {}
    for ticker in tickers:
        try:
            fund_infos[ticker] = market_data.get_fund_info(ticker)
        except MarketDataUnavailable:
            fund_infos[ticker] = None

    prices = {t: q.price for t, q in quotes.items()}
    expense_ratios = {
        t: info.expense_ratio for t, info in fund_infos.items() if info and info.expense_ratio
    }
    for holding in parsed:
        info = fund_infos.get(holding.ticker)
        if info and holding.asset_class is None:
            holding.asset_class = info.asset_class

    metrics = compute_portfolio_metrics(parsed, prices, expense_ratios)
    return metrics.model_dump(mode="json")


@server.tool()
def project_financial_goal(
    target_amount: float,
    years: float,
    monthly_contribution: float,
    risk_tolerance: str = "moderate",
    starting_balance: float = 0.0,
) -> dict:
    """Project whether a savings/investment goal is on track, given a
    monthly contribution and risk tolerance.

    risk_tolerance: "conservative", "moderate", or "aggressive" -- selects a
    static, historical-average annual-return assumption (never a live
    market prediction -- see docs/design/market-data-integration.md §6).
    """
    settings = get_settings().goal_planning
    asset_class = _RISK_TO_ASSET_CLASS.get(risk_tolerance, "equity")
    annual_return = settings.assumed_annual_returns.get(asset_class, 0.05)
    series = project_growth_series(monthly_contribution, years, annual_return, starting_balance)
    projected_value = series[-1][1]
    return {
        "projected_value": projected_value,
        "target_amount": target_amount,
        "on_track": projected_value >= target_amount,
        "assumed_annual_return": annual_return,
        "years": years,
        "yearly_balances": [{"year": year, "balance": balance} for year, balance in series],
    }


@server.tool()
def search_finance_knowledge_base(query: str, category: str | None = None) -> list[dict]:
    """Search Finnie's curated financial-education knowledge base and
    return matching articles/glossary entries with their content and
    source attribution. Returns an empty list if nothing clears the
    relevance threshold -- never fabricates a match.

    category: optionally restrict to one of stocks, bonds, etfs_and_funds,
    retirement_accounts, tax, risk_and_diversification,
    budgeting_and_saving, market_mechanics, or glossary.
    """
    chunks = retrieve(query, category=category)
    return [
        {
            "title": chunk.metadata.title,
            "source_name": chunk.metadata.source_name,
            "source_url": chunk.metadata.source_url,
            "category": chunk.metadata.category,
            "content": chunk.content,
            "relevance_score": chunk.score,
        }
        for chunk in chunks
    ]


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
