from finnie.core.state import AgentResult, FinancialGoal, Portfolio, UserProfile


def test_user_profile_defaults():
    profile = UserProfile(session_id="abc123")
    assert profile.experience_level == "beginner"
    assert profile.risk_tolerance is None
    assert profile.goals == []


def test_user_profile_round_trips_through_json():
    profile = UserProfile(
        session_id="abc123",
        experience_level="advanced",
        risk_tolerance="aggressive",
        goals=[FinancialGoal(name="retire", target_amount=1_000_000)],
    )
    restored = UserProfile.model_validate(profile.model_dump(mode="json"))
    assert restored == profile


def test_agent_result_defaults_are_safe():
    result = AgentResult(agent="qa", summary="hello")
    assert result.data == {}
    assert result.citations == []
    assert result.degraded is False


def test_portfolio_defaults_to_no_holdings():
    portfolio = Portfolio()
    assert portfolio.holdings == []
    assert portfolio.metrics is None
