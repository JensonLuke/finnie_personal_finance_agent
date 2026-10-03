"""Dependency-ordered dispatch queue. See
docs/design/langgraph-workflow.md §4's description of the supervisor node.
"""

from __future__ import annotations

from finnie.core.state import AgentName, AgentState


def peek_next_agent(state: AgentState) -> AgentName | None:
    pending = state.get("pending_agents") or []
    return pending[0] if pending else None


def advance_queue(state: AgentState) -> dict:
    pending = state.get("pending_agents") or []
    return {"pending_agents": pending[1:]}
