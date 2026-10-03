"""Optional secondary provider -- not wired up for v1.

See docs/design/market-data-integration.md §1 for why: Alpha Vantage's free
tier (5 calls/min, 25/day) makes it impractical as a primary source, and it's
not blocking to add later since MarketDataService already treats "secondary"
as optional. Wiring this in for real needs:
- ALPHA_VANTAGE_API_KEY (not yet in .env.example)
- a token-bucket limiter respecting the 5/min, 25/day quota
- mapping Alpha Vantage's response shape into the shared Quote/FundInfo/etc.
  contracts in finnie.data.models
"""

from __future__ import annotations

from finnie.data.models import FundInfo, HistoricalBar, NewsItem, Quote


class AlphaVantageProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        raise NotImplementedError("Alpha Vantage secondary provider is not implemented yet")

    def get_history(self, ticker: str, period: str = "6mo") -> list[HistoricalBar]:
        raise NotImplementedError("Alpha Vantage secondary provider is not implemented yet")

    def get_fund_info(self, ticker: str) -> FundInfo | None:
        raise NotImplementedError("Alpha Vantage secondary provider is not implemented yet")

    def get_news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        raise NotImplementedError("Alpha Vantage secondary provider is not implemented yet")
