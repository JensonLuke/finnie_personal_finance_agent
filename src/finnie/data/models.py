"""Market-data contracts and the provider interface. See
docs/design/market-data-integration.md §2-3."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol

from pydantic import BaseModel

AssetClass = Literal["equity", "bond", "etf", "mutual_fund", "cash", "other"]


class Quote(BaseModel):
    ticker: str
    price: float
    change: float
    change_percent: float
    currency: str = "USD"
    as_of: datetime
    source: Literal["live", "cache"]
    stale: bool = False


class HistoricalBar(BaseModel):
    ticker: str
    date: date
    close: float


class FundInfo(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    expense_ratio: float | None = None
    category: str | None = None


class NewsItem(BaseModel):
    ticker: str
    headline: str
    url: str
    published_at: datetime


class MarketDataUnavailable(Exception):
    """Raised when no provider succeeds and no cached value (including the
    seed cache) exists -- the "true cold start" case in
    docs/design/market-data-integration.md §4, step 6. Deliberately never
    caught silently: the caller (an agent) must turn this into an honest
    degraded response rather than a fabricated price."""


class MarketDataProvider(Protocol):
    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]: ...

    def get_history(self, ticker: str, period: str = "6mo") -> list[HistoricalBar]: ...

    def get_fund_info(self, ticker: str) -> FundInfo | None: ...

    def get_news(self, ticker: str, limit: int = 5) -> list[NewsItem]: ...
