"""Goal Planning Agent. Per the boundary rule in
docs/design/langgraph-workflow.md's agent table, this agent never calls the
market-data tool itself -- it reads Portfolio's already-computed metrics from
`agent_outputs` when the supervisor has run Portfolio first this turn.

`project_growth` is a pure function; the return-rate assumption it uses comes
from static config, never a live API -- see
docs/design/market-data-integration.md §6 for why that's a deliberate choice.

Goal/risk-tolerance intake is handled by the router (see workflow/router.py
and workflow/graph.py's router_node) rather than by this agent -- by the time
this agent runs, `profile.goals` is already populated if the user has ever
stated one.
"""

from __future__ import annotations

import json
from datetime import date

from finnie.agents.base import Agent
from finnie.agents.narration import narrate
from finnie.core.config import get_settings
from finnie.core.llm import get_chat_model
from finnie.core.state import AgentResult, AgentState, FinancialGoal

_DEFAULT_HORIZON_YEARS = 10.0  # used when a goal has no target_date -- e.g.
# a goal extracted by the router from chat (see workflow/router.py), which
# doesn't yet prompt for a specific date. The Goals tab UI always supplies
# one (see web_app/app.py's render_goals_tab), so this default is the
# chat-only fallback, not a hardcoded ceiling on every goal.

_MIN_HORIZON_YEARS = 1 / 12  # a target_date in the past or this month still
# gets a projection rather than a division-by-zero/negative-months crash.

_RISK_TO_ASSET_CLASS = {
    "conservative": "bond",
    "moderate": "equity",
    "aggressive": "equity",
    None: "equity",
}

_SYSTEM_PROMPT = """You are a goal-planning assistant for beginners. Given \
the projection below (JSON), explain in plain language whether the user is \
on track for their goal, using the actual numbers given. Add exactly ONE \
clear caveat that this assumes a constant historical-average return and real \
results will vary year to year -- don't overstate certainty either way. \
3-4 sentences."""


def project_growth_series(
    monthly_contribution: float,
    years: float,
    annual_return: float,
    starting_balance: float = 0.0,
) -> list[tuple[float, float]]:
    """Year-by-year (year, balance) trajectory -- the single source of truth
    for the compounding math. Used both by `project_growth` (final value
    only) and by the Goals tab's projection chart, so the chart and the
    agent's narrated number can never drift apart. Year 0 is the starting
    balance, included so a chart has a real point to start the line from."""
    months = int(years * 12)
    monthly_rate = annual_return / 12
    balance = starting_balance
    series = [(0.0, round(balance, 2))]
    for month in range(1, months + 1):
        balance = balance * (1 + monthly_rate) + monthly_contribution
        # Sample every 12 months for the chart, and always include the final
        # month even if `years` isn't a whole number -- otherwise a partial
        # last year's growth would be silently dropped from both the chart
        # and (via `project_growth` below) the agent's narrated final value.
        if month % 12 == 0 or month == months:
            series.append((month / 12, round(balance, 2)))
    return series


def project_growth(
    monthly_contribution: float,
    years: float,
    annual_return: float,
    starting_balance: float = 0.0,
) -> float:
    return project_growth_series(monthly_contribution, years, annual_return, starting_balance)[-1][
        1
    ]


def years_until(goal: FinancialGoal) -> float:
    """Years remaining until `goal.target_date`, or `_DEFAULT_HORIZON_YEARS`
    if the goal doesn't carry one (or it fails to parse) -- resolves the
    TODO that used to sit on `_DEFAULT_HORIZON_YEARS` unconditionally."""
    if not goal.target_date:
        return _DEFAULT_HORIZON_YEARS
    try:
        target = date.fromisoformat(goal.target_date)
    except ValueError:
        return _DEFAULT_HORIZON_YEARS
    days_remaining = (target - date.today()).days
    return max(days_remaining / 365.25, _MIN_HORIZON_YEARS)


class GoalAgent(Agent):
    name = "goal"

    def __init__(self, llm=None):
        self.llm = llm if llm is not None else get_chat_model()

    def run(self, state: AgentState) -> AgentResult:
        profile = state["user_profile"]
        if not profile.goals:
            return AgentResult(
                agent="goal",
                confidence=0.3,
                summary=(
                    "Tell me a goal (e.g. 'save $50,000 in 10 years') and your risk "
                    "tolerance, and I'll project whether you're on track."
                ),
            )

        settings = get_settings().goal_planning
        asset_class = _RISK_TO_ASSET_CLASS.get(profile.risk_tolerance, "equity")
        annual_return = settings.assumed_annual_returns.get(asset_class, 0.05)

        goal = profile.goals[0]
        years = years_until(goal)
        contribution = goal.monthly_contribution or 0.0

        starting_balance = 0.0
        portfolio_result = state.get("agent_outputs", {}).get("portfolio")
        if portfolio_result:
            starting_balance = portfolio_result.data.get("metrics", {}).get("total_value", 0.0)

        projected = project_growth(contribution, years, annual_return, starting_balance)
        on_track = projected >= goal.target_amount

        fallback_summary = (
            f"Projecting ${projected:,.2f} toward '{goal.name}' in {years:.0f} years at an "
            f"assumed {annual_return:.0%} annual return -- "
            f"{'on track for' if on_track else 'short of'} your ${goal.target_amount:,.2f} target."
        )
        summary = narrate(
            self.llm,
            agent_name=self.name,
            system_prompt=_SYSTEM_PROMPT,
            user_content=json.dumps(
                {
                    "goal_name": goal.name,
                    "target_amount": goal.target_amount,
                    "years": years,
                    "monthly_contribution": contribution,
                    "assumed_annual_return": annual_return,
                    "projected_value": projected,
                    "on_track": on_track,
                }
            ),
            fallback=fallback_summary,
        )
        return AgentResult(
            agent="goal",
            summary=summary,
            data={
                "projected_value": projected,
                "on_track": on_track,
                "assumed_annual_return": annual_return,
                "years": years,
                "monthly_contribution": contribution,
                "starting_balance": starting_balance,
                "target_amount": goal.target_amount,
            },
            confidence=0.6,  # projections are inherently uncertain -- reflected
            # honestly in the score, not hidden.
        )
