"""Tax Education Agent -- light tier (RAG-only over a narrow category scope),
per docs/design/langgraph-workflow.md §1. Deliberately never answers outside
account/tax-structure education."""

from __future__ import annotations

from finnie.agents.base import Agent
from finnie.agents.narration import narrate
from finnie.core.llm import get_chat_model
from finnie.core.state import AgentResult, AgentState
from finnie.rag.retriever import retrieve, to_citations

# Matches docs/design/rag-implementation.md §8's per-agent retrieval scope.
TAX_CATEGORIES = ["tax", "retirement_accounts"]

_SYSTEM_PROMPT = """You explain tax-advantaged account types (401k, IRA, \
Roth, etc.) and general tax concepts for education only, using ONLY the \
context provided below. Never tell the user what to do with their specific \
filing situation -- if the question needs that, say a tax professional or \
irs.gov is the right place, not this assistant. Keep it to 3-5 sentences."""


class TaxAgent(Agent):
    name = "tax"

    def __init__(self, llm=None):
        self.llm = llm if llm is not None else get_chat_model()

    def run(self, state: AgentState) -> AgentResult:
        query = state["messages"][-1].content if state["messages"] else ""
        chunks = retrieve(
            query,
            category=TAX_CATEGORIES,
            experience_level=state["user_profile"].experience_level,
        )

        if not chunks:
            return AgentResult(
                agent="tax",
                confidence=0.2,
                summary=(
                    "I don't have curated tax/account material covering that -- for "
                    "anything filing-specific, a tax professional or the IRS site is "
                    "the right source, not this assistant."
                ),
            )

        fallback_summary = chunks[0].content[:500]
        context = "\n\n".join(f"({c.metadata.title}) {c.content}" for c in chunks)
        summary = narrate(
            self.llm,
            agent_name=self.name,
            system_prompt=_SYSTEM_PROMPT,
            user_content=f"Question: {query}\n\nContext:\n{context}",
            fallback=fallback_summary,
        )
        return AgentResult(
            agent="tax", summary=summary, citations=to_citations(chunks), confidence=0.75
        )
