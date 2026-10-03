"""Loads config.yaml + .env into a validated Settings object.

Non-secret tunables come from config.yaml; secrets (API keys) come only from
the environment/.env, never from config.yaml, so config.yaml is safe to commit
and review like any other source file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# src/finnie/core/config.py -> core -> finnie -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config.yaml"


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2


class RAGConfig(BaseModel):
    score_threshold: float = 0.35
    default_k: int = 6
    mmr_lambda: float = 0.5
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40
    glossary_fuzzy_match_threshold: int = 90


class MarketDataConfig(BaseModel):
    primary_provider: str = "yfinance"
    secondary_provider: str | None = None
    cache_ttl_minutes: int = 30
    request_timeout_seconds: int = 10
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    circuit_breaker_cooldown_seconds: int = 300
    alpha_vantage_rate_limit_per_minute: int = 5
    alpha_vantage_daily_limit: int = 25


class GoalPlanningConfig(BaseModel):
    assumed_annual_returns: dict[str, float] = Field(
        default_factory=lambda: {"equity": 0.07, "bond": 0.03, "cash": 0.01}
    )


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    goal_planning: GoalPlanningConfig = Field(default_factory=GoalPlanningConfig)
    log_level: str = "INFO"
    openai_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    raw: dict = {}
    if CONFIG_PATH.exists():
        raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}

    llm = LLMConfig(**raw.get("llm", {}))
    if model_override := os.getenv("OPENAI_MODEL"):
        llm.model = model_override

    return Settings(
        llm=llm,
        rag=RAGConfig(**raw.get("rag", {})),
        market_data=MarketDataConfig(**raw.get("market_data", {})),
        goal_planning=GoalPlanningConfig(**raw.get("goal_planning", {})),
        log_level=raw.get("logging", {}).get("level", os.getenv("LOG_LEVEL", "INFO")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
