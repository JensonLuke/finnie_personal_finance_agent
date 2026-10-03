"""Portfolio Analysis Agent. See docs/design/langgraph-workflow.md's agent
table and docs/design/market-data-integration.md §7.

`compute_portfolio_metrics` is a pure function (no LLM, no network) so it's
cheap to unit test directly -- see tests/agents/test_portfolio_agent.py.
"""

from __future__ import annotations

from datetime import UTC, datetime

from finnie.agents.base import Agent
from finnie.agents.narration import narrate
from finnie.core.llm import get_chat_model
from finnie.core.state import AgentResult, AgentState, Holding, PortfolioMetrics
from finnie.data.market_data_service import MarketDataService

_SYSTEM_PROMPT = """You are a portfolio-analysis assistant for beginners. \
Given the computed metrics below (JSON), write a short, encouraging \
narrative: state the total value, explain what the diversification score \
means in plain language (0 = concentrated in one place, 1 = spread across \
many asset classes), mention the expense ratio if present, and offer ONE \
general observation (e.g. "you might look into other asset classes") \
without telling the user to buy or sell anything specific. 3-5 sentences."""


def compute_portfolio_metrics(
    holdings: list[Holding],
    prices: dict[str, float],
    expense_ratios: dict[str, float] | None = None,
) -> PortfolioMetrics:
    expense_ratios = expense_ratios or {}
    values = {h.ticker: h.shares * prices.get(h.ticker, 0.0) for h in holdings}
    total_value = sum(values.values())

    allocation: dict[str, float] = {}
    if total_value > 0:
        for h in holdings:
            asset_class = h.asset_class or "other"
            allocation[asset_class] = (
                allocation.get(asset_class, 0.0) + values[h.ticker] / total_value * 100
            )

    # Simple diversification heuristic: 1 - Herfindahl-Hirschman index of
    # asset-class weights. 1.0 = spread evenly across many classes,
    # 0.0 = concentrated entirely in one.
    weights = [v / 100 for v in allocation.values()]
    hhi = sum(w**2 for w in weights) if weights else 1.0
    diversification_score = round(1 - hhi, 3)

    weighted_expense_ratio = None
    if total_value > 0 and expense_ratios:
        weighted_expense_ratio = (
            sum(expense_ratios.get(h.ticker, 0.0) * values.get(h.ticker, 0.0) for h in holdings)
            / total_value
        )

    return PortfolioMetrics(
        total_value=round(total_value, 2),
        allocation_by_asset_class={k: round(v, 2) for k, v in allocation.items()},
        diversification_score=diversification_score,
        weighted_expense_ratio=weighted_expense_ratio,
        as_of=datetime.now(UTC),
    )


class PortfolioAgent(Agent):
    name = "portfolio"

    def __init__(self, market_data: MarketDataService, llm=None):
        self.market_data = market_data
        self.llm = llm if llm is not None else get_chat_model()

    def run(self, state: AgentState) -> AgentResult:
        portfolio = state.get("portfolio")
        if not portfolio or not portfolio.holdings:
            return AgentResult(
                agent="portfolio",
                confidence=0.3,
                summary=(
                    "I don't have any holdings on file yet -- tell me what you hold "
                    "(ticker and number of shares) and I can analyze it."
                ),
            )

        tickers = [h.ticker for h in portfolio.holdings]
        try:
            quotes = self.market_data.get_quotes(tickers)
            fund_infos = {t: self.market_data.get_fund_info(t) for t in tickers}
        except Exception as exc:  # market data unavailable -- degrade, don't crash the turn
            return AgentResult(
                agent="portfolio",
                confidence=0.2,
                degraded=True,
                summary=(
                    f"I couldn't reach live pricing right now ({exc}), so I can't "
                    "compute current portfolio value. Try again shortly."
                ),
            )

        prices = {t: q.price for t, q in quotes.items()}
        expense_ratios = {
            t: info.expense_ratio for t, info in fund_infos.items() if info and info.expense_ratio
        }
        for h in portfolio.holdings:
            info = fund_infos.get(h.ticker)
            if info and h.asset_class is None:
                h.asset_class = info.asset_class

        metrics = compute_portfolio_metrics(portfolio.holdings, prices, expense_ratios)

        fallback_summary = (
            f"Your portfolio is worth ${metrics.total_value:,.2f} across "
            f"{len(portfolio.holdings)} holdings. Diversification score: "
            f"{metrics.diversification_score} (0 = concentrated, 1 = highly spread)."
        )
        summary = narrate(
            self.llm,
            agent_name=self.name,
            system_prompt=_SYSTEM_PROMPT,
            user_content=metrics.model_dump_json(),
            fallback=fallback_summary,
        )
        return AgentResult(
            agent="portfolio",
            summary=summary,
            data={"metrics": metrics.model_dump(mode="json")},
            confidence=0.9,
        )
