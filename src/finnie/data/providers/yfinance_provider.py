"""Primary market-data provider. See docs/design/market-data-integration.md §1.

Verified against the live network (see docs/technical-design.md §6). That
verification caught a real bug, now fixed: yfinance's `fast_info` is a
`FastInfo` object whose `__getitem__` normalizes snake_case keys
("previous_close") to its actual camelCase fields ("previousClose"), but
its `.get()` does *not* apply that normalization -- `.get("previous_close")`
silently returns `None` (or your default) even though the field exists and
`info["previous_close"]` returns it correctly. `_fast_info_get` below is the
one place that inconsistency is worked around, rather than every call site
needing to remember it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import yfinance as yf

from finnie.data.models import FundInfo, HistoricalBar, NewsItem, Quote


def _fast_info_get(fast_info: Any, key: str, default: Any = None) -> Any:
    try:
        return fast_info[key]
    except KeyError:
        return default


class YFinanceProvider:
    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        if not tickers:
            return {}
        # Batched: one round trip for all tickers rather than N, per
        # docs/design/market-data-integration.md §3.
        batch = yf.Tickers(" ".join(tickers))
        now = datetime.now(UTC)
        quotes: dict[str, Quote] = {}
        for ticker in tickers:
            info = batch.tickers[ticker].fast_info
            price = float(info["last_price"])
            prev_close = float(_fast_info_get(info, "previous_close") or price)
            change = price - prev_close
            change_percent = (change / prev_close * 100) if prev_close else 0.0
            quotes[ticker] = Quote(
                ticker=ticker,
                price=price,
                change=change,
                change_percent=change_percent,
                currency=str(_fast_info_get(info, "currency", "USD")),
                as_of=now,
                source="live",
            )
        return quotes

    def get_history(self, ticker: str, period: str = "6mo") -> list[HistoricalBar]:
        history = yf.Ticker(ticker).history(period=period)
        return [
            HistoricalBar(ticker=ticker, date=index.date(), close=float(row["Close"]))
            for index, row in history.iterrows()
        ]

    def get_fund_info(self, ticker: str) -> FundInfo | None:
        info = yf.Ticker(ticker).info
        if not info:
            return None
        quote_type = str(info.get("quoteType", "")).lower()
        asset_class = (
            "etf"
            if quote_type == "etf"
            else "mutual_fund"
            if quote_type == "mutualfund"
            else "equity"
        )
        return FundInfo(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName") or ticker,
            asset_class=asset_class,
            expense_ratio=info.get("annualReportExpenseRatio"),
            category=info.get("category"),
        )

    def get_news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        items = yf.Ticker(ticker).news or []
        return [
            NewsItem(
                ticker=ticker,
                headline=item.get("title", ""),
                url=item.get("link", ""),
                published_at=datetime.fromtimestamp(item.get("providerPublishTime", 0)),
            )
            for item in items[:limit]
        ]
