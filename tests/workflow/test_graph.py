"""End-to-end graph tests -- builds and invokes the real compiled
LangGraph, with FakeChatModel/FakeProvider so it stays fast and offline.
This is the pytest-covered counterpart to the manual live smoke tests run
during development (see docs/technical-design.md §6) -- those proved it
works against the real APIs; these are what catch a regression on every
future change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finnie.agents.goal_agent import GoalAgent
from finnie.agents.market_agent import MarketAgent
from finnie.agents.news_agent import NewsAgent
from finnie.agents.portfolio_agent import PortfolioAgent
from finnie.agents.qa_agent import QAAgent
from finnie.agents.tax_agent import TaxAgent
from finnie.core.profile_store import ProfileStore
from finnie.core.state import FinancialGoal, Holding, Portfolio, UserProfile
from finnie.core.store import SqliteKeyValueStore
from finnie.data.market_data_service import MarketDataService
from finnie.data.providers.fake_provider import FakeProvider
from finnie.rag import retriever as retriever_module
from finnie.rag.models import DocumentMeta, RetrievedChunk
from finnie.testing.fake_llm import FakeChatModel
from finnie.workflow.graph import build_graph
from finnie.workflow.router import RouterDecision


@pytest.fixture
def profile_store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(SqliteKeyValueStore(db_path=tmp_path / "profiles.sqlite3"))


@pytest.fixture
def agents():
    market_data = MarketDataService(primary=FakeProvider())
    llm = FakeChatModel(raise_error=RuntimeError("no narration in this test -- fallback path only"))
    return {
        "qa": QAAgent(llm=llm),
        "portfolio": PortfolioAgent(market_data, llm=llm),
        "market": MarketAgent(market_data, llm=llm),
        "goal": GoalAgent(llm=llm),
        "news": NewsAgent(market_data, llm=llm),
        "tax": TaxAgent(llm=llm),
    }


def _base_state(query: str, *, portfolio=None) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "user_profile": UserProfile(session_id="graph-test"),
        "portfolio": portfolio,
        "intent": [],
        "pending_agents": [],
        "agent_outputs": {},
        "retrieved_docs": [],
        "market_data": {},
        "error": None,
        "final_response": None,
    }


@pytest.fixture(autouse=True)
def _stub_retrieval(monkeypatch):
    # Keep these graph tests offline -- no real embeddings/Chroma. QAAgent's
    # own narration+retrieval-quality is covered separately (test_qa_and_tax_agents.py,
    # test_eval_queries.py); this fixture just needs `retrieve` to return
    # something plausible so QA/Tax/News agents behave normally in a full
    # graph run.
    chunk = RetrievedChunk(
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

    def fake_retrieve(query, **kwargs):
        return [chunk]

    monkeypatch.setattr(retriever_module, "retrieve", fake_retrieve)
    # Agents import `retrieve` into their own namespace, so patch there too.
    # `to_citations` is left real -- it's pure logic over the chunks above,
    # no network involved, no need to stub it.
    import finnie.agents.news_agent as news_module
    import finnie.agents.qa_agent as qa_module
    import finnie.agents.tax_agent as tax_module

    monkeypatch.setattr(qa_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(tax_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(news_module, "retrieve", fake_retrieve)


def test_single_agent_turn_qa(agents, profile_store):
    router_llm = FakeChatModel(structured_response=RouterDecision(intents=["qa"]))
    graph = build_graph(agents, profile_store, llm=router_llm)

    result = graph.invoke(_base_state("What is a stock?"))

    assert result["intent"] == ["qa"]
    assert "ownership" in result["final_response"]
    assert "educational information" in result["final_response"]


def test_single_agent_turn_portfolio(agents, profile_store):
    router_llm = FakeChatModel(structured_response=RouterDecision(intents=["portfolio"]))
    graph = build_graph(agents, profile_store, llm=router_llm)
    portfolio = Portfolio(holdings=[Holding(ticker="AAPL", shares=10)])

    result = graph.invoke(_base_state("How is my portfolio?", portfolio=portfolio))

    assert "$1,000.00" in result["final_response"]


def test_compound_query_dispatches_portfolio_before_goal(agents, profile_store):
    # This is the dependency-ordering behavior the workflow design doc
    # describes: a 'goal' intent auto-prepends 'portfolio' so Goal can read
    # its starting balance from agent_outputs this same turn.
    router_llm = FakeChatModel(structured_response=RouterDecision(intents=["goal"]))
    graph = build_graph(agents, profile_store, llm=router_llm)
    portfolio = Portfolio(holdings=[Holding(ticker="AAPL", shares=10)])
    profile_store.save(
        UserProfile(
            session_id="graph-test",
            goals=[FinancialGoal(name="retire", target_amount=5000, monthly_contribution=100)],
        )
    )

    result = graph.invoke(_base_state("Am I on track?", portfolio=portfolio))

    assert result["pending_agents"] == []  # queue fully drained
    assert "portfolio" in result["agent_outputs"]
    assert "goal" in result["agent_outputs"]
    # Goal's starting balance came from Portfolio's output this same turn.
    assert result["agent_outputs"]["goal"].data["starting_balance"] == 1000.0


def test_router_llm_failure_falls_back_to_keyword_classifier(agents, profile_store):
    router_llm = FakeChatModel(raise_error=RuntimeError("router down"))
    graph = build_graph(agents, profile_store, llm=router_llm)

    result = graph.invoke(_base_state("Can you look at my portfolio?"))

    assert result["intent"] == ["portfolio"]  # keyword fallback still classified correctly


def test_router_extracted_goal_and_risk_are_persisted(agents, profile_store):
    decision = RouterDecision(
        intents=["goal"],
        extracted_goal=FinancialGoal(name="house", target_amount=40000, monthly_contribution=250),
        extracted_risk_tolerance="conservative",
    )
    router_llm = FakeChatModel(structured_response=decision)
    graph = build_graph(agents, profile_store, llm=router_llm)

    graph.invoke(_base_state("I want to save for a house"))

    reloaded = profile_store.load("graph-test")
    assert reloaded.risk_tolerance == "conservative"
    assert reloaded.goals[0].name == "house"


def test_agent_exception_degrades_the_turn_instead_of_crashing(profile_store):
    class ExplodingAgent(QAAgent):
        def run(self, state):
            raise RuntimeError("boom")

    router_llm = FakeChatModel(structured_response=RouterDecision(intents=["qa"]))
    graph = build_graph({"qa": ExplodingAgent()}, profile_store, llm=router_llm)

    result = graph.invoke(_base_state("What is a stock?"))

    assert result["error"].agent == "qa"
    assert "went wrong" in result["final_response"]


def test_unclassified_agent_name_is_simply_skipped(profile_store):
    # Router might classify 'news', but if 'news' isn't in the agents dict
    # passed to build_graph, the queue should just skip it, not crash.
    router_llm = FakeChatModel(structured_response=RouterDecision(intents=["news"]))
    qa_agent = QAAgent(llm=FakeChatModel(raise_error=RuntimeError("x")))
    graph = build_graph({"qa": qa_agent}, profile_store, llm=router_llm)

    result = graph.invoke(_base_state("any news?"))

    assert result["final_response"] == "I'm not sure how to help with that yet."
