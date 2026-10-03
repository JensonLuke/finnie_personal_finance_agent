"""Deterministic provider for tests -- never touches the network or a real
quota. See docs/design/market-data-integration.md §8."""

from __future__ import annotations

from datetime import UTC, date, datetime

from finnie.data.models import FundInfo, HistoricalBar, NewsItem, Quote


class FakeProvider:
    def __init__(self, quotes: dict[str, Quote] | None = None, fail: bool = False):
        self._quotes = quotes or {}
        self.fail = fail
        self.calls = 0

    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("FakeProvider configured to fail")
        now = datetime.now(UTC)
        return {
            t: self._quotes.get(
                t,
                Quote(
                    ticker=t, price=100.0, change=0.0, change_percent=0.0, as_of=now, source="live"
                ),
            )
            for t in tickers
        }

    def get_history(self, ticker: str, period: str = "6mo") -> list[HistoricalBar]:
        if self.fail:
            raise ConnectionError("FakeProvider configured to fail")
        return [HistoricalBar(ticker=ticker, date=date.today(), close=100.0)]

    def get_fund_info(self, ticker: str) -> FundInfo | None:
        if self.fail:
            raise ConnectionError("FakeProvider configured to fail")
        return FundInfo(ticker=ticker, name=f"{ticker} Fund", asset_class="etf", expense_ratio=0.03)

    def get_news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        if self.fail:
            raise ConnectionError("FakeProvider configured to fail")
        return [
            NewsItem(
                ticker=ticker,
                headline=f"{ticker} news",
                url="https://example.com",
                published_at=datetime.now(UTC),
            )
        ]
