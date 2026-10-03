"""The shared graph state and its component models.

Matches docs/design/langgraph-workflow.md §3 exactly, with one deliberate
implementation choice beyond what that doc spells out: `AgentResult` (and its
nested models) are Pydantic `BaseModel`s, not bare `TypedDict`s, because
docs/design/rag-implementation.md §5 calls for the inter-agent contract to be
"enforced with a Pydantic model, not just a TypedDict hint" -- a malformed
agent output should fail fast in `synthesize`, not corrupt it silently.
`AgentState` itself stays a `TypedDict` because that's what LangGraph's
`StateGraph` expects as a channel schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from finnie.rag.models import Citation

AgentName = Literal["qa", "portfolio", "market", "goal", "news", "tax"]
AssetClass = Literal["equity", "bond", "etf", "mutual_fund", "cash", "other"]


class FinancialGoal(BaseModel):
    name: str
    target_amount: float
    target_date: str | None = None  # ISO date string; kept as a string for now,
    # see docs/design/langgraph-workflow.md §7 open questions on the intake flow.
    monthly_contribution: float | None = None


class UserProfile(BaseModel):
    """Persisted separately from conversation memory -- see §3's rationale for
    keeping this out of the LangGraph checkpointer."""

    session_id: str
    experience_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = None
    goals: list[FinancialGoal] = Field(default_factory=list)


class Holding(BaseModel):
    ticker: str
    shares: float
    cost_basis: float | None = None
    asset_class: AssetClass | None = None


class PortfolioMetrics(BaseModel):
    total_value: float
    allocation_by_asset_class: dict[str, float]  # asset_class -> % of total_value
    diversification_score: float  # 0 (concentrated) - 1 (highly spread)
    weighted_expense_ratio: float | None = None
    as_of: datetime


class Portfolio(BaseModel):
    holdings: list[Holding] = Field(default_factory=list)
    metrics: PortfolioMetrics | None = None


class ErrorInfo(BaseModel):
    agent: str
    message: str
    degraded: bool = True
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentResult(BaseModel):
    """The only thing one agent may hand to another (via `synthesize`) -- see
    docs/design/langgraph-workflow.md §5, communication protocol rule 1."""

    agent: AgentName
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.5
    degraded: bool = False


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    portfolio: Portfolio | None
    intent: list[AgentName]
    pending_agents: list[AgentName]
    agent_outputs: dict[str, AgentResult]
    retrieved_docs: list[Citation]
    market_data: dict[str, Any]
    error: ErrorInfo | None
    final_response: str | None
