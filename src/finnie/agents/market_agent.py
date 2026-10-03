"""Market Analysis Agent. See docs/design/langgraph-workflow.md's agent table
and docs/design/market-data-integration.md §7.

Requested tickers come from the router's structured-output extraction (see
workflow/router.py's `RouterDecision.tickers`, merged into
`state["market_data"]["requested_tickers"]` by workflow/graph.py's
router_node), falling back to a default index if none were named.
"""

from __future__ import annotations

from finnie.agents.base import Agent
from finnie.agents.narration import narrate
from finnie.core.llm import get_chat_model
from finnie.core.state import AgentResult, AgentState
from finnie.data.market_data_service import MarketDataService
from finnie.data.models import MarketDataUnavailable

_DEFAULT_TICKERS = ["SPY"]

_SYSTEM_PROMPT = """You are a market-analysis assistant. Given the live \
quotes below (JSON: ticker -> price, change_percent, stale), write a brief \
1-2 sentence summary of what's happening. If any quote is marked stale, say \
explicitly that it may not reflect the current price. Don't speculate about \
why prices moved unless it's obvious from the numbers alone."""


class MarketAgent(Agent):
    name = "market"

    def __init__(self, market_data: MarketDataService, llm=None):
        self.market_data = market_data
        self.llm = llm if llm is not None else get_chat_model()

    def run(self, state: AgentState) -> AgentResult:
        tickers = state.get("market_data", {}).get("requested_tickers") or _DEFAULT_TICKERS
        try:
            quotes = self.market_data.get_quotes(tickers)
        except MarketDataUnavailable as exc:
            return AgentResult(
                agent="market",
                confidence=0.1,
                degraded=True,
                summary=f"Market data is unavailable right now: {exc}",
            )

        fallback_lines = [
            f"{ticker}: ${quote.price:,.2f} ({quote.change_percent:+.2f}%)"
            + (" [cached, may be stale]" if quote.stale else "")
            for ticker, quote in quotes.items()
        ]
        degraded = any(quote.stale for quote in quotes.values())
        summary = narrate(
            self.llm,
            agent_name=self.name,
            system_prompt=_SYSTEM_PROMPT,
            user_content="; ".join(
                f"{t}: price={q.price}, change_percent={q.change_percent}, stale={q.stale}"
                for t, q in quotes.items()
            ),
            fallback="; ".join(fallback_lines),
        )
        return AgentResult(
            agent="market",
            summary=summary,
            data={"quotes": {t: q.model_dump(mode="json") for t, q in quotes.items()}},
            confidence=0.6 if degraded else 0.9,
            degraded=degraded,
        )
