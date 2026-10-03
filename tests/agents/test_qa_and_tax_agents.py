"""QAAgent/TaxAgent narration tests. `retrieve` is monkeypatched to a stub so
these stay fast/offline unit tests -- real retrieval quality is exercised
separately (tests/rag/, plus the manual RAG smoke test in
docs/design/rag-implementation.md's testing section), not re-verified here.
"""

from langchain_core.messages import HumanMessage

from finnie.agents import qa_agent as qa_agent_module
from finnie.agents import tax_agent as tax_agent_module
from finnie.agents.qa_agent import QAAgent
from finnie.agents.tax_agent import TaxAgent
from finnie.core.state import UserProfile
from finnie.rag.models import Citation, DocumentMeta, RetrievedChunk
from finnie.testing.fake_llm import FakeChatModel

_CHUNK = RetrievedChunk(
    content="A stock represents ownership in a company.",
    metadata=DocumentMeta(
        id="stocks-what-is-a-stock",
        title="What Is a Stock?",
        category="stocks",
        article_id="stocks-what-is-a-stock",
        source_name="SEC",
    ),
    score=0.8,
)


def _state():
    return {
        "messages": [HumanMessage(content="what is a stock?")],
        "user_profile": UserProfile(session_id="s1"),
    }


def test_qa_agent_returns_no_citations_message_when_retrieval_is_empty(monkeypatch):
    monkeypatch.setattr(qa_agent_module, "retrieve", lambda *a, **k: [])
    agent = QAAgent(llm=FakeChatModel(response_text="should never be used"))
    result = agent.run(_state())
    assert "don't have curated material" in result.summary
    assert agent.llm.invoke_calls == []


def test_qa_agent_uses_llm_narration_grounded_in_retrieved_chunks(monkeypatch):
    monkeypatch.setattr(qa_agent_module, "retrieve", lambda *a, **k: [_CHUNK])
    llm = FakeChatModel(response_text="A stock is a piece of company ownership.")
    agent = QAAgent(llm=llm)
    result = agent.run(_state())
    assert result.summary == "A stock is a piece of company ownership."
    assert result.citations == [
        Citation(title="What Is a Stock?", source_name="SEC", source_url=None, category="stocks")
    ]


def test_qa_agent_falls_back_to_raw_chunk_when_llm_fails(monkeypatch):
    monkeypatch.setattr(qa_agent_module, "retrieve", lambda *a, **k: [_CHUNK])
    agent = QAAgent(llm=FakeChatModel(raise_error=RuntimeError("down")))
    result = agent.run(_state())
    assert result.summary == _CHUNK.content[:500]


def test_tax_agent_scopes_retrieval_to_tax_categories(monkeypatch):
    captured = {}

    def fake_retrieve(query, *, category=None, **kwargs):
        captured["category"] = category
        return [_CHUNK]

    monkeypatch.setattr(tax_agent_module, "retrieve", fake_retrieve)
    agent = TaxAgent(llm=FakeChatModel(response_text="IRAs are tax-advantaged accounts."))
    result = agent.run(_state())
    assert captured["category"] == tax_agent_module.TAX_CATEGORIES
    assert result.summary == "IRAs are tax-advantaged accounts."
