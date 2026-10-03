"""Builds the LangGraph StateGraph described in
docs/design/langgraph-workflow.md §4.

Scaffold-stage simplification, flagged explicitly: dispatch is always
sequential through the supervisor queue, even for agents with no dependency
on each other (e.g. a query that hits both Q&A and Market runs Q&A then
Market, not both at once). The design doc's `Send`-based parallel fan-out for
independent agents is a follow-up optimization once this sequential path is
proven correct in practice -- see that doc's §7 open questions. Every other
piece of the topology (load_context -> router -> queue-driven agent dispatch
-> join -> synthesize, with a dependency such as Goal->Portfolio resolved
into dispatch order up front) matches the design doc as drawn.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from finnie.agents.base import Agent
from finnie.core.llm import get_chat_model
from finnie.core.profile_store import ProfileStore
from finnie.core.state import AgentState, ErrorInfo
from finnie.workflow.nodes import join, make_load_context, make_synthesize
from finnie.workflow.router import resolve_dependencies, route
from finnie.workflow.supervisor import advance_queue, peek_next_agent

_VALID_RISK_TOLERANCES = {"conservative", "moderate", "aggressive"}


def _last_user_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    return messages[-1].content if messages else ""


def _route_by_queue(state: AgentState) -> str:
    return peek_next_agent(state) or "join"


def build_graph(agents: dict[str, Agent], profile_store: ProfileStore | None = None, llm=None):
    """`agents` maps agent name -> Agent instance (see agents/base.py). Only
    agents present in this dict are wired into the graph; a name the router
    classifies but that isn't in `agents` is filtered out of the dispatch
    queue in `router_node` below -- LangGraph's conditional-edge routing
    hard-crashes (`KeyError`) on a destination that isn't a declared edge
    target, so this filtering has to happen before `pending_agents` is ever
    set, not be left to each consumer to handle defensively.

    `llm` (optional) is used by the router's structured-output classifier and
    by `synthesize`'s multi-agent merge; defaults to `get_chat_model()`. Pass
    a test double (see finnie.testing.fake_llm) to keep tests off the network.
    """
    profile_store = profile_store or ProfileStore()
    llm = llm if llm is not None else get_chat_model()
    graph = StateGraph(AgentState)

    def router_node(state: AgentState) -> dict:
        decision = route(_last_user_text(state), llm=llm)
        # `intent` keeps the router's raw classification (useful for
        # debugging/observability even if an intent isn't dispatchable);
        # `pending_agents` is the actual execution queue and must only ever
        # contain names this graph instance actually registered.
        dispatchable = [name for name in resolve_dependencies(decision.intents) if name in agents]
        update: dict = {
            "intent": decision.intents,
            "pending_agents": dispatchable,
            "market_data": {"requested_tickers": decision.tickers},
        }

        profile = state["user_profile"]
        changed = False
        if (
            decision.extracted_risk_tolerance in _VALID_RISK_TOLERANCES
            and profile.risk_tolerance is None
        ):
            profile = profile.model_copy(
                update={"risk_tolerance": decision.extracted_risk_tolerance}
            )
            changed = True
        if decision.extracted_goal and not any(
            g.name == decision.extracted_goal.name for g in profile.goals
        ):
            profile = profile.model_copy(
                update={"goals": [*profile.goals, decision.extracted_goal]}
            )
            changed = True
        if changed:
            profile_store.save(profile)
            update["user_profile"] = profile

        return update

    def make_agent_node(agent: Agent):
        def node(state: AgentState) -> dict:
            update = advance_queue(state)
            try:
                result = agent.run(state)
            except Exception as exc:  # any agent failure degrades this turn, never crashes it
                update["error"] = ErrorInfo(agent=agent.name, message=str(exc))
                return update
            outputs = dict(state.get("agent_outputs", {}))
            outputs[agent.name] = result
            update["agent_outputs"] = outputs
            return update

        return node

    graph.add_node("load_context", make_load_context(profile_store))
    graph.add_node("router", router_node)
    graph.add_node("join", join)
    graph.add_node("synthesize", make_synthesize(llm))
    for name, agent in agents.items():
        graph.add_node(name, make_agent_node(agent))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "router")
    graph.add_conditional_edges("router", _route_by_queue, [*agents.keys(), "join"])
    for name in agents:
        graph.add_conditional_edges(name, _route_by_queue, [*agents.keys(), "join"])
    graph.add_edge("join", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
