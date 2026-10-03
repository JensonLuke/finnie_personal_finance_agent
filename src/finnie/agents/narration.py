"""Shared LLM-narration helper used by every agent.

Deliberate design point, consistent with the resilience philosophy in
docs/design/market-data-integration.md: an LLM call here augments an
already-correct templated summary, it doesn't replace the only path to one.
Every agent computes its templated summary first (from real data / real
retrieval, per docs/design/langgraph-workflow.md's agent table), then tries
to have the LLM narrate it more naturally; if that call fails for any reason
(rate limit, network, bad key), the templated summary -- already tested,
already correct -- is what the user sees. A turn never fails just because the
narration call did.

Citations are never touched by this: they stay a code-computed field (see
docs/design/rag-implementation.md §7), so the LLM narrating a summary can
never fabricate sourcing.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from finnie.utils.logging import get_logger

logger = get_logger(__name__)


def narrate(
    llm, *, system_prompt: str, user_content: str, fallback: str, agent_name: str | None = None
) -> str:
    """`agent_name`, if given, tags the call for LangSmith (see .env.example's
    LangSmith section) -- a `run_name`/`tags` pair so a trace reads
    "portfolio_narration" instead of an anonymous ChatOpenAI call, whether it
    was reached via the graph or a direct dashboard-tab call. No-op when
    tracing isn't enabled; LangChain just ignores unused config otherwise."""
    config = (
        {"run_name": f"{agent_name}_narration", "tags": [f"agent:{agent_name}"]}
        if agent_name
        else None
    )
    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)],
            config=config,
        )
        text = (response.content or "").strip()
        return text or fallback
    except Exception:
        logger.warning("LLM narration failed, falling back to templated summary", exc_info=True)
        return fallback
