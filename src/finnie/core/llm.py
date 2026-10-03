"""One place to construct the chat model every LLM-grounded piece uses:
the router's structured-output classifier, each agent's narration, and
`synthesize`'s multi-agent merge. See docs/design/langgraph-workflow.md.

Decoupled from the RAG embedding model on purpose (see
docs/design/rag-implementation.md §4) -- this is the *chat* model only.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from finnie.core.config import get_settings


@lru_cache
def get_chat_model(temperature: float | None = None) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm.model,
        temperature=temperature if temperature is not None else settings.llm.temperature,
        api_key=settings.openai_api_key,
    )
