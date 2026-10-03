"""The graph's non-agent nodes: context loading, the join point, and final
response synthesis. See docs/design/langgraph-workflow.md §4.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from finnie.core.profile_store import ProfileStore
from finnie.core.state import AgentResult, AgentState
from finnie.rag.models import Citation
from finnie.utils.logging import get_logger

logger = get_logger(__name__)

DISCLAIMER = "\n\n_This is educational information, not personalized financial advice._"

_MERGE_SYSTEM_PROMPT = """You merge answers from several specialist finance \
agents into ONE coherent reply for the user. Each agent's answer is given \
below, labeled by which agent produced it. Combine them into a natural, \
non-repetitive response -- remove duplicate points, keep it organized, and \
NEVER invent a fact, number, or claim that isn't already present in one of \
the agents' answers below. If two agents disagree or overlap, prefer the more \
specific one and don't repeat the same point twice."""


def make_load_context(profile_store: ProfileStore):
    def load_context(state: AgentState) -> dict:
        session_id = state["user_profile"].session_id
        return {"user_profile": profile_store.load(session_id)}

    return load_context


def join(state: AgentState) -> dict:
    # Pure convergence point -- every agent node writes into agent_outputs
    # directly, so there's nothing to merge here today. Kept as its own node
    # (rather than wiring every agent straight to synthesize) so a real
    # Send-based parallel fan-out for independent agents (a follow-up
    # optimization -- see workflow/graph.py's module docstring) can converge
    # here later without changing synthesize.
    return {}


def _collect_citations(outputs: dict[str, AgentResult]) -> list[Citation]:
    seen: set[tuple[str, str | None]] = set()
    citations: list[Citation] = []
    for result in outputs.values():
        for citation in result.citations:
            key = (citation.title, citation.source_url)
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
    return citations


def _format_sources_footer(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["\n\n**Sources:**"]
    for citation in citations:
        if citation.source_url:
            lines.append(f"- [{citation.title}]({citation.source_url}) -- {citation.source_name}")
        else:
            lines.append(f"- {citation.title} ({citation.source_name})")
    return "\n".join(lines)


def _fallback_merge(outputs: dict[str, AgentResult]) -> str:
    return "\n\n".join(result.summary for result in outputs.values())


def make_synthesize(llm=None):
    def synthesize(state: AgentState) -> dict:
        outputs = state.get("agent_outputs", {})
        error = state.get("error")

        if not outputs:
            message = "I'm not sure how to help with that yet."
            if error:
                message = f"Something went wrong ({error.message}). Please try again."
            return {"final_response": message}

        # A single agent's answer needs no merging -- skip the LLM call
        # entirely rather than paying for a no-op rephrase.
        if len(outputs) == 1:
            merged = next(iter(outputs.values())).summary
        else:
            merged = _fallback_merge(outputs)
            if llm is not None:
                labeled = "\n\n".join(
                    f"[{name}]: {result.summary}" for name, result in outputs.items()
                )
                try:
                    response = llm.invoke(
                        [
                            SystemMessage(content=_MERGE_SYSTEM_PROMPT),
                            HumanMessage(content=labeled),
                        ],
                        config={"run_name": "synthesize_merge", "tags": ["synthesize"]},
                    )
                    merged = (response.content or "").strip() or merged
                except Exception:
                    logger.warning(
                        "LLM synthesis failed, falling back to concatenated summaries",
                        exc_info=True,
                    )

        if error:
            merged += (
                f"\n\n(Note: {error.agent} hit an issue and its part of this "
                "answer may be incomplete.)"
            )

        footer = _format_sources_footer(_collect_citations(outputs))
        return {"final_response": merged + footer + DISCLAIMER}

    return synthesize
