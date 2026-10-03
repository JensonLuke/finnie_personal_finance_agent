"""Cache + retry + circuit-breaker + fallback chain in front of one or more
MarketDataProvider implementations.

Implements the resilience chain diagrammed in
docs/design/market-data-integration.md §4: fresh cache -> primary (retried
with backoff) -> circuit breaker -> optional secondary -> stale cache/seed ->
MarketDataUnavailable. `get_news` is the one deliberate exception: per that
doc's agent-mapping table, news is best-effort supplementary context, so a
failure there degrades to an empty list rather than raising.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from cachetools import TTLCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from finnie.core.config import get_settings
from finnie.core.store import SqliteKeyValueStore
from finnie.data.models import (
    FundInfo,
    HistoricalBar,
    MarketDataProvider,
    MarketDataUnavailable,
    NewsItem,
    Quote,
)

# Failures worth retrying/falling back on. A provider that raises anything
# else (e.g. a programming error) is left to propagate rather than masked.
_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)


class _CircuitBreaker:
    """Opens after repeated failures so a known outage doesn't cost a full
    retry budget on every subsequent call. See design doc §4."""

    def __init__(self, cooldown_seconds: int, failure_threshold: int = 3):
        self.cooldown_seconds = cooldown_seconds
        self.failure_threshold = failure_threshold
        self._failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            self._failures = 0
            return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


class MarketDataService:
    def __init__(
        self,
        primary: MarketDataProvider,
        secondary: MarketDataProvider | None = None,
        store: SqliteKeyValueStore | None = None,
        seed_cache: dict[str, Quote] | None = None,
        timer: Callable[[], float] = time.monotonic,
    ):
        """`timer` is injectable so tests can advance the cache's clock
        without sleeping -- see tests/data/test_market_data_service.py."""
        settings = get_settings().market_data
        self.primary = primary
        self.secondary = secondary
        self.store = store
        self.max_retries = settings.max_retries
        self.backoff_base = settings.backoff_base_seconds
        self._cache: TTLCache[str, Quote] = TTLCache(
            maxsize=512, ttl=settings.cache_ttl_minutes * 60, timer=timer
        )
        self._breaker = _CircuitBreaker(settings.circuit_breaker_cooldown_seconds)
        self._seed = seed_cache or {}

    # -- quotes: the full resilience chain -------------------------------

    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        results: dict[str, Quote] = {}
        missing: list[str] = []
        for ticker in tickers:
            cached = self._cache.get(ticker)
            if cached is not None:
                results[ticker] = cached
            else:
                missing.append(ticker)
        if missing:
            results.update(self._fetch_with_fallback(missing))
        return results

    def _fetch_with_fallback(self, tickers: list[str]) -> dict[str, Quote]:
        if not self._breaker.is_open():
            try:
                quotes = self._call_with_retry(self.primary.get_quotes, tickers)
                self._breaker.record_success()
                for ticker, quote in quotes.items():
                    self._cache[ticker] = quote
                    if self.store:
                        self.store.set("market_quotes", ticker, quote.model_dump(mode="json"))
                return quotes
            except _TRANSIENT_ERRORS:
                self._breaker.record_failure()

        if self.secondary is not None:
            try:
                quotes = self.secondary.get_quotes(tickers)
                for ticker, quote in quotes.items():
                    self._cache[ticker] = quote
                return quotes
            except Exception:
                pass  # secondary is best-effort; fall through to stale/seed

        return self._stale_or_unavailable(tickers)

    def _stale_or_unavailable(self, tickers: list[str]) -> dict[str, Quote]:
        results: dict[str, Quote] = {}
        for ticker in tickers:
            stale = self._lookup_stale(ticker)
            if stale is None:
                raise MarketDataUnavailable(f"No live, cached, or seed data available for {ticker}")
            results[ticker] = stale.model_copy(update={"source": "cache", "stale": True})
        return results

    def _lookup_stale(self, ticker: str) -> Quote | None:
        if self.store:
            raw = self.store.get("market_quotes", ticker)
            if raw:
                return Quote.model_validate(raw)
        return self._seed.get(ticker)

    # -- history / fund info: simpler primary-then-secondary fallback ----

    def get_history(self, ticker: str, period: str = "6mo") -> list[HistoricalBar]:
        try:
            return self._call_with_retry(self.primary.get_history, ticker, period)
        except _TRANSIENT_ERRORS:
            if self.secondary is not None:
                return self.secondary.get_history(ticker, period)
            raise MarketDataUnavailable(f"History unavailable for {ticker}") from None

    def get_fund_info(self, ticker: str) -> FundInfo | None:
        try:
            return self._call_with_retry(self.primary.get_fund_info, ticker)
        except _TRANSIENT_ERRORS:
            if self.secondary is not None:
                return self.secondary.get_fund_info(ticker)
            raise MarketDataUnavailable(f"Fund info unavailable for {ticker}") from None

    def get_news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        try:
            return self._call_with_retry(self.primary.get_news, ticker, limit)
        except _TRANSIENT_ERRORS:
            if self.secondary is not None:
                try:
                    return self.secondary.get_news(ticker, limit)
                except Exception:
                    pass
            return []  # news is supplementary -- degrade quietly, don't fail the turn

    # -- shared retry wrapper ---------------------------------------------

    def _call_with_retry(self, fn, *args, **kwargs):
        retrying = retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=self.backoff_base, max=10),
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        )(fn)
        return retrying(*args, **kwargs)
