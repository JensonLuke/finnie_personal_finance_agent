"""News Synthesizer Agent -- light tier. Headlines come from the market-data
tool (yfinance's per-ticker news), not from a curated 'news' KB category --
see the correction at the top of docs/design/rag-implementation.md and its
§8 per-agent retrieval scope.

Ticker comes from the router's extraction, same mechanism as MarketAgent --
see that agent's module docstring.
"""

from __future__ import annotations

from finnie.agents.base import Agent
from finnie.agents.narration import narrate
from finnie.core.llm import get_chat_model
from finnie.core.state import AgentResult, AgentState
from finnie.data.market_data_service import MarketDataService
from finnie.rag.retriever import retrieve, to_citations

NEWS_CONTEXT_CATEGORIES = ["market_mechanics", "glossary"]
_DEFAULT_TICKER = "SPY"

_SYSTEM_PROMPT = """You summarize recent financial news headlines for \
beginners. Use the background context below to explain any jargon in the \
headlines in plain language, but don't editorialize beyond what the \
headlines and context actually say. 2-4 sentences."""


class NewsAgent(Agent):
    name = "news"

    def __init__(self, market_data: MarketDataService, llm=None):
        self.market_data = market_data
        self.llm = llm if llm is not None else get_chat_model()

    def run(self, state: AgentState) -> AgentResult:
        # `.get(..) or [default]`, not `.get(.., [default])`: the router
        # always sets this key, often to an empty list when no ticker was
        # named, and a bare `.get` default only fires when the key is
        # missing entirely -- an empty list would otherwise reach `[0]` below.
        tickers = state.get("market_data", {}).get("requested_tickers") or [_DEFAULT_TICKER]
        ticker = tickers[0]
        try:
            headlines = self.market_data.get_news(ticker)
        except Exception:
            headlines = []

        if not headlines:
            return AgentResult(
                agent="news",
                confidence=0.2,
                summary=f"I couldn't find recent news for {ticker} right now.",
            )

        context_chunks = retrieve(
            headlines[0].headline,
            category=NEWS_CONTEXT_CATEGORIES,
            experience_level=state["user_profile"].experience_level,
        )

        fallback_summary = "; ".join(item.headline for item in headlines[:3])
        context = "\n\n".join(f"({c.metadata.title}) {c.content}" for c in context_chunks)
        summary = narrate(
            self.llm,
            agent_name=self.name,
            system_prompt=_SYSTEM_PROMPT,
            user_content=(
                f"Headlines for {ticker}:\n"
                + "\n".join(f"- {item.headline}" for item in headlines[:3])
                + f"\n\nBackground context:\n{context or '(none available)'}"
            ),
            fallback=fallback_summary,
        )
        return AgentResult(
            agent="news", summary=summary, citations=to_citations(context_chunks), confidence=0.6
        )
