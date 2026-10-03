"""The router: an LLM structured-output classifier with the original
rule-based `classify` kept as its fallback (see docs/design/langgraph-workflow.md
§4), plus the dependency-ordering logic that's independent of which
classifier produced the intents.

`route()` also does one more thing beyond classification: it extracts a
stated goal or risk tolerance from the message, if any. This is the resolved
answer to that design doc's §7 open question ("where does the risk-tolerance
/ goal-setting intake flow live") -- rather than a separate intake node, it
rides along with the classification call that already reads the message.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from finnie.core.state import AgentName, FinancialGoal
from finnie.utils.logging import get_logger

logger = get_logger(__name__)

_KEYWORDS: dict[AgentName, list[str]] = {
    "portfolio": ["my portfolio", "my holdings", "i own", "diversif", "allocation"],
    "market": ["price of", "quote", "stock price", "how is the market", "trading at"],
    "goal": ["save for", "retirement", "goal", "how much should i", "on track"],
    "tax": ["401k", "401(k)", "ira", "roth", "capital gains", "tax"],
    "news": ["news", "headline", "latest on"],
}

# The only known cross-agent dependency so far: Goal Planning reads
# Portfolio's metrics if available. See docs/design/langgraph-workflow.md §4/§7.
DEPENDENCY_PAIRS: dict[AgentName, AgentName] = {"goal": "portfolio"}

# Short enough to risk a false match inside another word (e.g. "tax" inside
# "syntax", "ira" inside "inspiration") -- these get word-boundary matching;
# everything else is a plain substring check.
_WORD_BOUNDARY_KEYWORDS = {"tax", "ira"}

_SYSTEM_PROMPT = """You route a user's message to one or more of six finance \
assistant agents, and pull out two optional pieces of structured data if the \
message states them.

Agents (choose all that apply):
- qa: general financial education questions
- portfolio: the user wants their current holdings analyzed
- market: the user wants a live price/quote/trend for a specific ticker or the market generally
- goal: the user wants help planning toward a savings/investment goal
- tax: questions about tax-advantaged accounts (401k, IRA, Roth) or tax concepts
- news: the user wants recent news about a company or the market

If the message names one or more stock/ETF tickers or company names whose \
ticker you're confident of, include them in `tickers` (as ticker symbols, \
uppercase). If the message states a concrete savings/investment goal (a name, \
a target dollar amount, and ideally a monthly contribution), fill in \
`extracted_goal`. If it states a risk tolerance (conservative, moderate, or \
aggressive -- infer from words like "safe", "cautious", "aggressive", \
"willing to take risk"), fill in `extracted_risk_tolerance`. Leave anything \
not clearly stated as null/empty -- never guess a ticker or a goal amount \
that wasn't actually said."""


class RouterDecision(BaseModel):
    intents: list[AgentName] = Field(min_length=1)
    tickers: list[str] = Field(default_factory=list)
    extracted_goal: FinancialGoal | None = None
    extracted_risk_tolerance: str | None = None


def _contains(text: str, keyword: str) -> bool:
    if keyword in _WORD_BOUNDARY_KEYWORDS:
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def classify(query: str) -> list[AgentName]:
    """Rule-based fallback classifier -- see module docstring."""
    text = query.lower()
    matched = [
        agent
        for agent, keywords in _KEYWORDS.items()
        if any(_contains(text, keyword) for keyword in keywords)
    ]
    return matched or ["qa"]


def resolve_dependencies(intents: list[AgentName]) -> list[AgentName]:
    """Orders the dispatch queue so a dependency always runs before the agent
    that needs it -- a no-op reordering when no dependency is present."""
    ordered: list[AgentName] = []
    for agent in intents:
        dependency = DEPENDENCY_PAIRS.get(agent)
        if dependency and dependency not in ordered:
            ordered.append(dependency)
        if agent not in ordered:
            ordered.append(agent)
    return ordered


def route(query: str, llm=None) -> RouterDecision:
    """Tries the LLM structured-output classifier; falls back to the
    keyword-based `classify` (with no ticker/goal/risk extraction) on any
    failure, so a routing call never blocks the turn."""
    if llm is not None:
        try:
            structured_llm = llm.with_structured_output(RouterDecision)
            decision = structured_llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=query)],
                config={"run_name": "router_classify", "tags": ["router"]},
            )
            if decision.intents:
                return decision
        except Exception:
            logger.warning("LLM router failed, falling back to keyword classifier", exc_info=True)

    return RouterDecision(intents=classify(query))
