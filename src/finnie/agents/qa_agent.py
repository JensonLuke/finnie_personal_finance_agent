"""Finance Q&A Agent -- broadest RAG scope of any agent, per
docs/design/rag-implementation.md §8."""

from __future__ import annotations

from finnie.agents.base import Agent
from finnie.agents.narration import narrate
from finnie.core.llm import get_chat_model
from finnie.core.state import AgentResult, AgentState
from finnie.rag.retriever import retrieve, to_citations

_SYSTEM_PROMPT = """You are a friendly financial-education assistant for \
beginners. Answer the user's question using ONLY the context provided below \
-- if it doesn't fully answer the question, say what it does cover and note \
the rest isn't something you have sourced material on. Avoid jargon without \
explaining it. Keep it to 3-6 sentences. This is education, not personalized \
financial advice -- never tell the user what to buy or sell."""


class QAAgent(Agent):
    name = "qa"

    def __init__(self, llm=None):
        self.llm = llm if llm is not None else get_chat_model()

    def run(self, state: AgentState) -> AgentResult:
        query = state["messages"][-1].content if state["messages"] else ""
        chunks = retrieve(query, experience_level=state["user_profile"].experience_level)

        if not chunks:
            return AgentResult(
                agent="qa",
                confidence=0.3,
                summary=(
                    "I don't have curated material on that yet -- happy to give a "
                    "general answer, but treat it as a starting point, not a sourced one."
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
            agent="qa", summary=summary, citations=to_citations(chunks), confidence=0.8
        )
