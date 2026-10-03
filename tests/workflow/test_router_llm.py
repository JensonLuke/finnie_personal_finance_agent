from finnie.core.state import FinancialGoal
from finnie.testing.fake_llm import FakeChatModel
from finnie.workflow.router import RouterDecision, route


def test_route_uses_llm_decision_when_available():
    decision = RouterDecision(intents=["market"], tickers=["AAPL"])
    llm = FakeChatModel(structured_response=decision)
    result = route("what's AAPL trading at?", llm=llm)
    assert result.intents == ["market"]
    assert result.tickers == ["AAPL"]


def test_route_falls_back_to_keyword_classifier_on_llm_failure():
    llm = FakeChatModel(raise_error=RuntimeError("API down"))
    result = route("Can you look at my portfolio?", llm=llm)
    assert result.intents == ["portfolio"]
    assert result.tickers == []


def test_route_falls_back_to_keyword_classifier_without_an_llm():
    result = route("What is a stock?", llm=None)
    assert result.intents == ["qa"]


def test_route_falls_back_when_llm_returns_no_intents():
    # A malformed/empty decision (bypassing validation, to simulate a bad
    # response) should not be trusted over the fallback.
    empty_decision = RouterDecision.model_construct(intents=[], tickers=[])
    llm = FakeChatModel(structured_response=empty_decision)
    result = route("Can you look at my portfolio?", llm=llm)
    assert result.intents == ["portfolio"]


def test_route_extracts_goal_and_risk_tolerance():
    decision = RouterDecision(
        intents=["goal"],
        extracted_goal=FinancialGoal(
            name="house down payment", target_amount=40000, monthly_contribution=500
        ),
        extracted_risk_tolerance="conservative",
    )
    llm = FakeChatModel(structured_response=decision)
    result = route(
        "I want to save $40,000 for a house down payment, $500/month, I'm cautious", llm=llm
    )
    assert result.extracted_goal.name == "house down payment"
    assert result.extracted_risk_tolerance == "conservative"
