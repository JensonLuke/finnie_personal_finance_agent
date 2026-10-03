from finnie.core.state import AgentResult
from finnie.rag.models import Citation
from finnie.testing.fake_llm import FakeChatModel
from finnie.workflow.nodes import make_synthesize


def _state(**agent_outputs):
    return {"agent_outputs": agent_outputs, "error": None}


def test_single_agent_output_skips_the_llm_entirely():
    llm = FakeChatModel(response_text="should never be used")
    synthesize = make_synthesize(llm)
    result = synthesize(_state(qa=AgentResult(agent="qa", summary="A stock is ownership.")))
    assert "A stock is ownership." in result["final_response"]
    assert llm.invoke_calls == []  # no LLM call made for a single result


def test_multi_agent_output_uses_llm_merge():
    llm = FakeChatModel(response_text="Merged: here's your portfolio and goal projection.")
    synthesize = make_synthesize(llm)
    result = synthesize(
        _state(
            portfolio=AgentResult(agent="portfolio", summary="Worth $1,000."),
            goal=AgentResult(agent="goal", summary="On track for retirement."),
        )
    )
    assert "Merged: here's your portfolio and goal projection." in result["final_response"]
    assert len(llm.invoke_calls) == 1


def test_multi_agent_falls_back_to_concatenation_on_llm_failure():
    llm = FakeChatModel(raise_error=RuntimeError("down"))
    synthesize = make_synthesize(llm)
    result = synthesize(
        _state(
            portfolio=AgentResult(agent="portfolio", summary="Worth $1,000."),
            goal=AgentResult(agent="goal", summary="On track for retirement."),
        )
    )
    assert "Worth $1,000." in result["final_response"]
    assert "On track for retirement." in result["final_response"]


def test_disclaimer_is_always_appended():
    synthesize = make_synthesize(None)
    result = synthesize(_state(qa=AgentResult(agent="qa", summary="Some answer.")))
    assert "educational information" in result["final_response"]


def test_citations_are_deduplicated_across_agents_into_a_sources_footer():
    citation = Citation(
        title="What Is a Stock?",
        source_name="SEC",
        source_url="https://sec.gov/x",
        category="stocks",
    )
    synthesize = make_synthesize(None)
    result = synthesize(
        _state(
            qa=AgentResult(agent="qa", summary="Stocks are ownership.", citations=[citation]),
            tax=AgentResult(agent="tax", summary="IRAs are tax-advantaged.", citations=[citation]),
        )
    )
    response = result["final_response"]
    assert response.count("What Is a Stock?") == 1  # deduplicated, not listed twice
    assert "**Sources:**" in response


def test_no_citations_means_no_sources_footer():
    synthesize = make_synthesize(None)
    result = synthesize(_state(qa=AgentResult(agent="qa", summary="No sourced content here.")))
    assert "**Sources:**" not in result["final_response"]


def test_error_note_is_appended_when_present():
    from finnie.core.state import ErrorInfo

    synthesize = make_synthesize(None)
    state = _state(qa=AgentResult(agent="qa", summary="Partial answer."))
    state["error"] = ErrorInfo(agent="market", message="timeout")
    result = synthesize(state)
    assert "market hit an issue" in result["final_response"]
