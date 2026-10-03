"""One place to configure logging, per the problem statement's call for
"error handling and logging throughout." Call `configure_logging()` once, at
process start (CLI entry point, Streamlit app, or a test fixture)."""

from __future__ import annotations

import logging

from finnie.core.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = get_settings().log_level
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
