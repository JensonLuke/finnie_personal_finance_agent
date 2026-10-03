"""MCP server tests -- call tools in-process via the SDK's own
`list_tools`/`call_tool` (no stdio transport, no subprocess needed, same
spirit as Streamlit's AppTest). Network/RAG dependencies are monkeypatched
to fakes; live verification (real yfinance + real embeddings, including the
FastInfo bug this whole layer sits on top of) happened separately -- see
docs/technical-design.md §6.
"""

from __future__ import annotations

import json

import pytest

import finnie.mcp_server as mcp_server_module
from finnie.data.market_data_service import MarketDataService
from finnie.data.providers.fake_provider import FakeProvider
from finnie.rag.models import DocumentMeta, RetrievedChunk


@pytest.fixture(autouse=True)
def _fake_market_data(monkeypatch):
    monkeypatch.setattr(
        mcp_server_module, "_market_data", lambda: MarketDataService(primary=FakeProvider())
    )


@pytest.fixture(autouse=True)
def _fake_retrieval(monkeypatch):
    chunk = RetrievedChunk(
        content="A stock represents ownership in a company.",
        metadata=DocumentMeta(
            id="stocks-what-is-a-stock",
            title="What Is a Stock?",
            category="stocks",
            article_id="stocks-what-is-a-stock",
            source_name="SEC",
            source_url="https://www.investor.gov/",
        ),
        score=0.8,
    )
    monkeypatch.setattr(mcp_server_module, "retrieve", lambda query, **kwargs: [chunk])


def _result_data(call_tool_result):
    # The MCP SDK's CallToolResult shape differs by return type: a dict-
    # returning tool puts the full JSON in content[0].text with
    # structured_content=None; a list-returning tool puts the *full* list in
    # structured_content={"result": [...]} and only a partial preview in
    # content -- so structured_content, when present, is the authoritative
    # source (confirmed empirically; see the tool call this helper backs).
    if call_tool_result.structured_content is not None:
        structured = call_tool_result.structured_content
        return structured.get("result", structured) if isinstance(structured, dict) else structured
    return json.loads(call_tool_result.content[0].text)


async def test_all_tools_are_registered():
    tools = await mcp_server_module.server.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "get_stock_quote",
        "get_stock_history",
        "get_fund_info",
        "get_stock_news",
        "analyze_portfolio",
        "project_financial_goal",
        "search_finance_knowledge_base",
    }


async def test_get_stock_quote():
    result = await mcp_server_module.server.call_tool("get_stock_quote", {"tickers": ["AAPL"]})
    data = _result_data(result)
    assert data["AAPL"]["price"] == 100.0


async def test_get_stock_history():
    result = await mcp_server_module.server.call_tool("get_stock_history", {"ticker": "AAPL"})
    data = _result_data(result)
    assert len(data) >= 1
    assert data[0]["close"] == 100.0


async def test_get_fund_info():
    result = await mcp_server_module.server.call_tool("get_fund_info", {"ticker": "VOO"})
    data = _result_data(result)
    assert data["asset_class"] == "etf"


async def test_get_stock_news():
    result = await mcp_server_module.server.call_tool("get_stock_news", {"ticker": "AAPL"})
    data = _result_data(result)
    assert len(data) == 1
    assert "AAPL" in data[0]["headline"]


async def test_analyze_portfolio():
    result = await mcp_server_module.server.call_tool(
        "analyze_portfolio",
        {"holdings": [{"ticker": "AAPL", "shares": 10}, {"ticker": "BND", "shares": 10}]},
    )
    data = _result_data(result)
    assert data["total_value"] == 2000.0
    assert (
        "equity" in data["allocation_by_asset_class"] or "etf" in data["allocation_by_asset_class"]
    )


async def test_analyze_portfolio_rejects_malformed_holdings():
    result = await mcp_server_module.server.call_tool(
        "analyze_portfolio",
        {"holdings": [{"ticker": "AAPL"}]},  # missing "shares"
    )
    data = _result_data(result)
    assert "error" in data


async def test_analyze_portfolio_rejects_empty_holdings():
    result = await mcp_server_module.server.call_tool("analyze_portfolio", {"holdings": []})
    data = _result_data(result)
    assert "error" in data


async def test_project_financial_goal_on_track():
    result = await mcp_server_module.server.call_tool(
        "project_financial_goal",
        {
            "target_amount": 100.0,
            "years": 10,
            "monthly_contribution": 100,
            "risk_tolerance": "moderate",
        },
    )
    data = _result_data(result)
    assert data["on_track"] is True
    assert data["assumed_annual_return"] == pytest.approx(0.07)
    assert data["yearly_balances"][0] == {"year": 0.0, "balance": 0.0}


async def test_project_financial_goal_conservative_uses_bond_return():
    result = await mcp_server_module.server.call_tool(
        "project_financial_goal",
        {
            "target_amount": 100.0,
            "years": 10,
            "monthly_contribution": 100,
            "risk_tolerance": "conservative",
        },
    )
    data = _result_data(result)
    assert data["assumed_annual_return"] == pytest.approx(0.03)


async def test_search_finance_knowledge_base():
    result = await mcp_server_module.server.call_tool(
        "search_finance_knowledge_base", {"query": "what is a stock?"}
    )
    data = _result_data(result)
    assert len(data) == 1
    assert data[0]["title"] == "What Is a Stock?"
    assert data[0]["source_url"] == "https://www.investor.gov/"


async def test_search_finance_knowledge_base_empty_for_no_matches(monkeypatch):
    monkeypatch.setattr(mcp_server_module, "retrieve", lambda query, **kwargs: [])
    result = await mcp_server_module.server.call_tool(
        "search_finance_knowledge_base", {"query": "completely unrelated"}
    )
    data = _result_data(result)
    assert data == []
