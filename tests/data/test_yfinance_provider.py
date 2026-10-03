"""Mocked tests for YFinanceProvider -- no real network calls. Live
verification (real yfinance) happened separately during development, see
docs/technical-design.md §6; that's also where the `FastInfo.get()` bug
these tests guard against was actually found.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finnie.data.providers.yfinance_provider import YFinanceProvider


class FakeFastInfo:
    """Faithfully replicates the real bug that shipped: yfinance's
    `FastInfo.__getitem__` normalizes snake_case keys to its actual
    camelCase fields, but its `.get()` does NOT apply that same
    normalization and always returns the default. A test double that (like
    a plain dict) made `.get()` "just work" would never have caught this --
    this one only supports the real object's actual (broken) behavior.
    """

    def __init__(self, **camel_case_fields):
        self._fields = camel_case_fields
        self._key_map = {
            "last_price": "lastPrice",
            "previous_close": "previousClose",
            "currency": "currency",
        }

    def __getitem__(self, key):
        real_key = self._key_map.get(key, key)
        if real_key not in self._fields:
            raise KeyError(key)
        return self._fields[real_key]

    def get(self, key, default=None):
        return default  # faithfully broken, like the real thing


def test_get_quotes_computes_a_nonzero_change_from_previous_close():
    fake_info = FakeFastInfo(lastPrice=110.0, previousClose=100.0, currency="USD")
    fake_ticker = MagicMock(fast_info=fake_info)
    fake_batch = MagicMock(tickers={"AAPL": fake_ticker})

    with patch("finnie.data.providers.yfinance_provider.yf.Tickers", return_value=fake_batch):
        quotes = YFinanceProvider().get_quotes(["AAPL"])

    quote = quotes["AAPL"]
    assert quote.price == 110.0
    # This is the regression this test exists for: with the old `.get()`
    # call, `previous_close` would silently come back None -> fall back to
    # `price` -> change always 0.0, even though a real previous_close exists.
    assert quote.change == 10.0
    assert quote.change_percent == pytest.approx(10.0)
    assert quote.currency == "USD"


def test_get_quotes_falls_back_to_price_when_previous_close_is_truly_absent():
    fake_info = FakeFastInfo(lastPrice=50.0, currency="USD")  # no previousClose at all
    fake_ticker = MagicMock(fast_info=fake_info)
    fake_batch = MagicMock(tickers={"NEWSTOCK": fake_ticker})

    with patch("finnie.data.providers.yfinance_provider.yf.Tickers", return_value=fake_batch):
        quotes = YFinanceProvider().get_quotes(["NEWSTOCK"])

    quote = quotes["NEWSTOCK"]
    assert quote.price == 50.0
    assert quote.change == 0.0  # no previous close to compare against -- flat is the honest answer
    assert quote.change_percent == 0.0


def test_get_quotes_returns_empty_dict_for_no_tickers():
    assert YFinanceProvider().get_quotes([]) == {}


def test_get_history_maps_yfinance_dataframe_to_historical_bars():
    import pandas as pd

    fake_df = pd.DataFrame(
        {"Close": [100.0, 101.5]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"])
    )
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = fake_df

    with patch("finnie.data.providers.yfinance_provider.yf.Ticker", return_value=fake_ticker):
        bars = YFinanceProvider().get_history("AAPL", period="1mo")

    assert len(bars) == 2
    assert bars[0].close == 100.0
    assert bars[1].close == 101.5
    fake_ticker.history.assert_called_once_with(period="1mo")


def test_get_fund_info_classifies_etf_asset_class():
    fake_ticker = MagicMock()
    fake_ticker.info = {
        "quoteType": "ETF",
        "longName": "Vanguard S&P 500 ETF",
        "annualReportExpenseRatio": 0.0003,
        "category": "Large Blend",
    }

    with patch("finnie.data.providers.yfinance_provider.yf.Ticker", return_value=fake_ticker):
        info = YFinanceProvider().get_fund_info("VOO")

    assert info.asset_class == "etf"
    assert info.name == "Vanguard S&P 500 ETF"
    assert info.expense_ratio == 0.0003


def test_get_fund_info_defaults_to_equity_for_a_stock():
    fake_ticker = MagicMock()
    fake_ticker.info = {"quoteType": "EQUITY", "longName": "Apple Inc."}

    with patch("finnie.data.providers.yfinance_provider.yf.Ticker", return_value=fake_ticker):
        info = YFinanceProvider().get_fund_info("AAPL")

    assert info.asset_class == "equity"
    assert info.expense_ratio is None


def test_get_fund_info_returns_none_for_empty_info():
    fake_ticker = MagicMock()
    fake_ticker.info = {}

    with patch("finnie.data.providers.yfinance_provider.yf.Ticker", return_value=fake_ticker):
        info = YFinanceProvider().get_fund_info("BADTICKER")

    assert info is None


def test_get_news_maps_yfinance_news_items():
    fake_ticker = MagicMock()
    fake_ticker.news = [
        {
            "title": "Apple announces new product",
            "link": "https://example.com/1",
            "providerPublishTime": 1_700_000_000,
        },
        {
            "title": "Apple stock rises",
            "link": "https://example.com/2",
            "providerPublishTime": 1_700_000_100,
        },
    ]

    with patch("finnie.data.providers.yfinance_provider.yf.Ticker", return_value=fake_ticker):
        news = YFinanceProvider().get_news("AAPL", limit=1)

    assert len(news) == 1  # limit respected
    assert news[0].headline == "Apple announces new product"
    assert news[0].url == "https://example.com/1"


def test_get_news_returns_empty_list_when_no_news():
    fake_ticker = MagicMock()
    fake_ticker.news = None

    with patch("finnie.data.providers.yfinance_provider.yf.Ticker", return_value=fake_ticker):
        news = YFinanceProvider().get_news("AAPL")

    assert news == []
