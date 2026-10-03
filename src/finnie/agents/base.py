"""Shared agent interface. Every agent node in the graph wraps one of these.

Per docs/design/langgraph-workflow.md §5: an agent reads whatever it needs
from state and returns exactly one AgentResult. It never calls another
agent's code directly and never mutates state itself -- the graph node
wrapper (workflow/graph.py) is what writes the result into `agent_outputs`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from finnie.core.state import AgentResult, AgentState


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, state: AgentState) -> AgentResult:
        raise NotImplementedError
