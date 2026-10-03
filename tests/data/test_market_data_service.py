"""Exercises the resilience chain from docs/design/market-data-integration.md
§4 and §8: cache hit, primary failure falling back to stale/seed data, and a
true cold start raising MarketDataUnavailable.
"""

from datetime import datetime

import pytest

from finnie.core.store import SqliteKeyValueStore
from finnie.data.market_data_service import MarketDataService
from finnie.data.models import MarketDataUnavailable, Quote
from finnie.data.providers.fake_provider import FakeProvider


class _FakeClock:
    """Manual clock injected as TTLCache's `timer`, so TTL expiry can be
    tested without sleeping -- see MarketDataService's `timer` parameter."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_live_quote_is_served_and_cached():
    provider = FakeProvider()
    service = MarketDataService(primary=provider)

    quotes = service.get_quotes(["AAPL"])
    assert quotes["AAPL"].source == "live"
    assert provider.calls == 1

    # Second call within the TTL should hit the in-memory cache, not the provider.
    service.get_quotes(["AAPL"])
    assert provider.calls == 1


def test_cache_expires_after_ttl_and_refetches():
    clock = _FakeClock()
    provider = FakeProvider()
    service = MarketDataService(primary=provider, timer=clock)

    service.get_quotes(["AAPL"])
    assert provider.calls == 1

    clock.advance(31 * 60)  # past the default 30-minute TTL
    service.get_quotes(["AAPL"])
    assert provider.calls == 2


def test_primary_failure_falls_back_to_stale_store(tmp_path):
    store = SqliteKeyValueStore(db_path=tmp_path / "cache.sqlite3")
    stale_quote = Quote(
        ticker="AAPL",
        price=150.0,
        change=0.0,
        change_percent=0.0,
        as_of=datetime(2026, 1, 1),
        source="live",
    )
    store.set("market_quotes", "AAPL", stale_quote.model_dump(mode="json"))

    failing_provider = FakeProvider(fail=True)
    service = MarketDataService(primary=failing_provider, store=store)

    quotes = service.get_quotes(["AAPL"])
    assert quotes["AAPL"].stale is True
    assert quotes["AAPL"].source == "cache"
    assert quotes["AAPL"].price == 150.0


def test_seed_cache_is_used_when_nothing_else_is_available():
    seed_quote = Quote(
        ticker="SPY",
        price=560.0,
        change=0.0,
        change_percent=0.0,
        as_of=datetime(2026, 1, 1),
        source="cache",
        stale=True,
    )
    service = MarketDataService(primary=FakeProvider(fail=True), seed_cache={"SPY": seed_quote})
    quotes = service.get_quotes(["SPY"])
    assert quotes["SPY"].stale is True
    assert quotes["SPY"].price == 560.0


def test_cold_start_with_no_fallback_raises():
    service = MarketDataService(primary=FakeProvider(fail=True))
    with pytest.raises(MarketDataUnavailable):
        service.get_quotes(["UNKNOWNTICKER"])


def test_news_failure_degrades_to_empty_list_instead_of_raising():
    service = MarketDataService(primary=FakeProvider(fail=True))
    assert service.get_news("AAPL") == []


def test_secondary_provider_is_used_when_primary_fails():
    failing_primary = FakeProvider(fail=True)
    working_secondary = FakeProvider()
    service = MarketDataService(primary=failing_primary, secondary=working_secondary)

    quotes = service.get_quotes(["AAPL"])
    assert quotes["AAPL"].source == "live"
    assert working_secondary.calls == 1
