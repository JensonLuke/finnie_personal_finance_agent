"""Loads the checked-in seed cache -- honestly-stale sample quotes so a demo
still shows something for common tickers on a true cold start, instead of
fabricating a live-looking price. See docs/design/market-data-integration.md
§4, step 6."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from finnie.data.models import Quote

SEED_PATH = Path(__file__).resolve().parent / "seed_cache.json"


@lru_cache
def load_seed_cache(path: Path = SEED_PATH) -> dict[str, Quote]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {ticker: Quote.model_validate(payload) for ticker, payload in raw.items()}
