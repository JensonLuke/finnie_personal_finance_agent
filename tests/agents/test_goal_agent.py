from datetime import date, timedelta

from finnie.agents.goal_agent import GoalAgent, project_growth, project_growth_series, years_until
from finnie.core.state import FinancialGoal, UserProfile
from finnie.testing.fake_llm import FakeChatModel


def test_zero_return_is_pure_accumulation():
    # No growth: 12 months x $100 = $1200 exactly.
    assert project_growth(monthly_contribution=100, years=1, annual_return=0.0) == 1200.0


def test_positive_return_beats_pure_accumulation():
    no_growth = project_growth(monthly_contribution=100, years=10, annual_return=0.0)
    with_growth = project_growth(monthly_contribution=100, years=10, annual_return=0.07)
    assert with_growth > no_growth


def test_starting_balance_compounds_too():
    with_start = project_growth(
        monthly_contribution=0, years=10, annual_return=0.07, starting_balance=1000
    )
    assert with_start > 1000


def test_series_starts_at_year_zero_with_the_starting_balance():
    series = project_growth_series(
        monthly_contribution=0, years=5, annual_return=0.05, starting_balance=1000
    )
    assert series[0] == (0.0, 1000.0)


def test_series_final_point_matches_project_growth():
    series = project_growth_series(monthly_contribution=100, years=10, annual_return=0.07)
    final_from_series = series[-1][1]
    final_from_scalar = project_growth(monthly_contribution=100, years=10, annual_return=0.07)
    assert final_from_series == final_from_scalar


def test_series_includes_a_partial_final_year_not_just_whole_years():
    # 3.5 years -- the final point must reflect the full 3.5 years' growth,
    # not silently stop at the year-3 checkpoint. This was a real bug caught
    # by this test while refactoring project_growth to reuse the series.
    series = project_growth_series(monthly_contribution=100, years=3.5, annual_return=0.07)
    assert series[-1][0] == 3.5
    full_scalar = project_growth(monthly_contribution=100, years=3.5, annual_return=0.07)
    assert series[-1][1] == full_scalar


def test_series_is_monotonically_increasing_with_positive_contributions():
    series = project_growth_series(monthly_contribution=100, years=5, annual_return=0.05)
    balances = [balance for _, balance in series]
    assert balances == sorted(balances)


def test_years_until_defaults_when_no_target_date():
    goal = FinancialGoal(name="g", target_amount=1000)
    assert years_until(goal) == 10.0


def test_years_until_defaults_on_unparsable_target_date():
    goal = FinancialGoal(name="g", target_amount=1000, target_date="not-a-date")
    assert years_until(goal) == 10.0


def test_years_until_computes_from_a_real_future_date():
    target = date.today() + timedelta(days=365 * 5)
    goal = FinancialGoal(name="g", target_amount=1000, target_date=target.isoformat())
    # Allow a little slack for day-of-test rounding.
    assert 4.9 <= years_until(goal) <= 5.1


def test_years_until_clamps_a_past_date_instead_of_going_negative():
    past = date.today() - timedelta(days=365)
    goal = FinancialGoal(name="g", target_amount=1000, target_date=past.isoformat())
    assert years_until(goal) > 0


def _state_with_goal(profile: UserProfile) -> dict:
    return {"user_profile": profile, "agent_outputs": {}}


def test_agent_prompts_for_a_goal_without_calling_the_llm_when_none_is_set():
    llm = FakeChatModel(response_text="should never be used")
    agent = GoalAgent(llm=llm)
    result = agent.run(_state_with_goal(UserProfile(session_id="s1")))
    assert "Tell me a goal" in result.summary
    assert llm.invoke_calls == []


def test_agent_uses_llm_narration_when_available():
    llm = FakeChatModel(response_text="You're on track if you keep saving $300/month.")
    agent = GoalAgent(llm=llm)
    profile = UserProfile(
        session_id="s1",
        goals=[FinancialGoal(name="retire", target_amount=1000, monthly_contribution=300)],
    )
    result = agent.run(_state_with_goal(profile))
    assert result.summary == "You're on track if you keep saving $300/month."


def test_agent_falls_back_to_templated_summary_when_llm_fails():
    llm = FakeChatModel(raise_error=RuntimeError("down"))
    agent = GoalAgent(llm=llm)
    profile = UserProfile(
        session_id="s1",
        goals=[FinancialGoal(name="retire", target_amount=1000, monthly_contribution=300)],
    )
    result = agent.run(_state_with_goal(profile))
    assert "Projecting $" in result.summary
    assert "'retire'" in result.summary
