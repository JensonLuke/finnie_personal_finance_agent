"""AlphaVantageProvider is an intentional stub -- see
docs/design/market-data-integration.md §1 for why it's not implemented for
v1. These tests just confirm it fails loudly and predictably rather than
silently, so a future caller can't mistake it for a working provider."""

import pytest

from finnie.data.providers.alpha_vantage_provider import AlphaVantageProvider


@pytest.fixture
def provider() -> AlphaVantageProvider:
    return AlphaVantageProvider(api_key="unused")


def test_get_quotes_raises_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        provider.get_quotes(["AAPL"])


def test_get_history_raises_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        provider.get_history("AAPL")


def test_get_fund_info_raises_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        provider.get_fund_info("AAPL")


def test_get_news_raises_not_implemented(provider):
    with pytest.raises(NotImplementedError):
        provider.get_news("AAPL")
