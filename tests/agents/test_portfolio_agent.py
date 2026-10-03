from finnie.agents.portfolio_agent import PortfolioAgent, compute_portfolio_metrics
from finnie.core.state import Holding, Portfolio, UserProfile
from finnie.data.market_data_service import MarketDataService
from finnie.data.providers.fake_provider import FakeProvider
from finnie.testing.fake_llm import FakeChatModel


def test_total_value_sums_shares_times_price():
    holdings = [Holding(ticker="AAPL", shares=10, asset_class="equity")]
    metrics = compute_portfolio_metrics(holdings, prices={"AAPL": 200.0})
    assert metrics.total_value == 2000.0


def test_allocation_splits_by_asset_class():
    holdings = [
        Holding(ticker="AAPL", shares=10, asset_class="equity"),
        Holding(ticker="BND", shares=10, asset_class="bond"),
    ]
    prices = {"AAPL": 100.0, "BND": 100.0}
    metrics = compute_portfolio_metrics(holdings, prices)
    assert metrics.allocation_by_asset_class == {"equity": 50.0, "bond": 50.0}


def test_diversification_score_is_zero_when_fully_concentrated():
    holdings = [Holding(ticker="AAPL", shares=10, asset_class="equity")]
    metrics = compute_portfolio_metrics(holdings, prices={"AAPL": 100.0})
    assert metrics.diversification_score == 0.0


def test_diversification_score_increases_with_spread():
    concentrated = compute_portfolio_metrics(
        [Holding(ticker="AAPL", shares=10, asset_class="equity")], {"AAPL": 100.0}
    )
    spread = compute_portfolio_metrics(
        [
            Holding(ticker="AAPL", shares=10, asset_class="equity"),
            Holding(ticker="BND", shares=10, asset_class="bond"),
            Holding(ticker="CASH", shares=10, asset_class="cash"),
        ],
        {"AAPL": 100.0, "BND": 100.0, "CASH": 100.0},
    )
    assert spread.diversification_score > concentrated.diversification_score


def test_weighted_expense_ratio_is_value_weighted():
    holdings = [
        Holding(ticker="VOO", shares=10, asset_class="etf"),  # $1000, ER 0.03
        Holding(ticker="ARKK", shares=10, asset_class="etf"),  # $1000, ER 0.75
    ]
    prices = {"VOO": 100.0, "ARKK": 100.0}
    expense_ratios = {"VOO": 0.03, "ARKK": 0.75}
    metrics = compute_portfolio_metrics(holdings, prices, expense_ratios)
    assert metrics.weighted_expense_ratio == 0.39  # equal weight -> simple average here


def test_zero_total_value_does_not_divide_by_zero():
    holdings = [Holding(ticker="AAPL", shares=10, asset_class="equity")]
    metrics = compute_portfolio_metrics(holdings, prices={})  # no price data
    assert metrics.total_value == 0.0
    assert metrics.allocation_by_asset_class == {}


def _state_with_portfolio(portfolio: Portfolio) -> dict:
    return {
        "messages": [],
        "user_profile": UserProfile(session_id="s1"),
        "portfolio": portfolio,
        "agent_outputs": {},
    }


def test_agent_uses_llm_narration_when_available():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(response_text="Your portfolio looks solid and diversified.")
    agent = PortfolioAgent(market_data, llm=llm)
    portfolio = Portfolio(holdings=[Holding(ticker="AAPL", shares=10)])
    result = agent.run(_state_with_portfolio(portfolio))
    assert result.summary == "Your portfolio looks solid and diversified."
    assert len(llm.invoke_calls) == 1


def test_agent_falls_back_to_templated_summary_when_llm_fails():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(raise_error=RuntimeError("down"))
    agent = PortfolioAgent(market_data, llm=llm)
    portfolio = Portfolio(holdings=[Holding(ticker="AAPL", shares=10)])
    result = agent.run(_state_with_portfolio(portfolio))
    assert "Your portfolio is worth $1,000.00" in result.summary


def test_agent_handles_empty_portfolio_without_calling_the_llm():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(response_text="should never be used")
    agent = PortfolioAgent(market_data, llm=llm)
    result = agent.run(_state_with_portfolio(Portfolio()))
    assert "don't have any holdings" in result.summary
    assert llm.invoke_calls == []
