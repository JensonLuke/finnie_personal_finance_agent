from finnie.agents.market_agent import MarketAgent
from finnie.core.state import UserProfile
from finnie.data.market_data_service import MarketDataService
from finnie.data.providers.fake_provider import FakeProvider
from finnie.testing.fake_llm import FakeChatModel


def _state(requested_tickers=None):
    return {
        "user_profile": UserProfile(session_id="s1"),
        "market_data": {"requested_tickers": requested_tickers}
        if requested_tickers is not None
        else {},
    }


def test_defaults_to_spy_when_no_ticker_requested():
    market_data = MarketDataService(primary=FakeProvider())
    agent = MarketAgent(market_data, llm=FakeChatModel(raise_error=RuntimeError("skip llm")))
    result = agent.run(_state())
    assert "SPY" in result.data["quotes"]


def test_empty_ticker_list_from_router_falls_back_to_default_not_a_crash():
    # Regression test: the router always sets `requested_tickers`, often to
    # `[]` when no ticker was named -- this must not crash.
    market_data = MarketDataService(primary=FakeProvider())
    agent = MarketAgent(market_data, llm=FakeChatModel(raise_error=RuntimeError("skip llm")))
    result = agent.run(_state(requested_tickers=[]))
    assert "SPY" in result.data["quotes"]


def test_uses_llm_narration_when_available():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(response_text="SPY is flat today.")
    agent = MarketAgent(market_data, llm=llm)
    result = agent.run(_state(requested_tickers=["SPY"]))
    assert result.summary == "SPY is flat today."


def test_falls_back_to_templated_summary_when_llm_fails():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(raise_error=RuntimeError("down"))
    agent = MarketAgent(market_data, llm=llm)
    result = agent.run(_state(requested_tickers=["AAPL"]))
    assert "AAPL: $100.00" in result.summary
