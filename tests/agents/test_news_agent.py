from finnie.agents.news_agent import NewsAgent
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


def test_empty_ticker_list_from_router_falls_back_to_default_not_a_crash():
    # Same regression as MarketAgent: an empty (not missing) list must not
    # reach `tickers[0]` and raise IndexError.
    market_data = MarketDataService(primary=FakeProvider())
    agent = NewsAgent(market_data, llm=FakeChatModel(raise_error=RuntimeError("skip llm")))
    result = agent.run(_state(requested_tickers=[]))
    assert result.summary  # didn't crash, produced something


def test_no_headlines_returns_a_plain_message_without_calling_the_llm():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(response_text="should never be used")

    class _NoNewsProvider(FakeProvider):
        def get_news(self, ticker, limit=5):
            return []

    market_data = MarketDataService(primary=_NoNewsProvider())
    agent = NewsAgent(market_data, llm=llm)
    result = agent.run(_state(requested_tickers=["AAPL"]))
    assert "couldn't find recent news" in result.summary
    assert llm.invoke_calls == []


def test_uses_llm_narration_when_available():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(response_text="AAPL had some news today.")
    agent = NewsAgent(market_data, llm=llm)
    result = agent.run(_state(requested_tickers=["AAPL"]))
    assert result.summary == "AAPL had some news today."
