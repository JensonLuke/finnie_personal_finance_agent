"""A minimal chat-model double, shipped in the package (not tests/) so every
test that needs one imports the same implementation rather than
reimplementing it. Mimics just enough of LangChain's chat-model interface
(`.invoke(messages) -> object with .content`, `.with_structured_output`) for
the LLM layer's fallback logic to be testable without a real API call.
"""

from __future__ import annotations

from typing import Any


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChatModel:
    def __init__(
        self,
        response_text: str | None = None,
        structured_response: Any = None,
        raise_error: Exception | None = None,
    ):
        self.response_text = response_text
        self.structured_response = structured_response
        self.raise_error = raise_error
        self.invoke_calls: list[Any] = []

    def invoke(self, messages: Any, config: Any = None) -> FakeMessage:
        # `config` mirrors real LangChain chat models' `.invoke(input,
        # config=None)` signature (used for LangSmith run tags/names -- see
        # agents/narration.py) -- accepted and ignored here, not acted on.
        self.invoke_calls.append(messages)
        if self.raise_error:
            raise self.raise_error
        return FakeMessage(self.response_text or "")

    def with_structured_output(self, schema: type) -> _StructuredFakeChatModel:
        return _StructuredFakeChatModel(self)


class _StructuredFakeChatModel:
    """Returned by `with_structured_output` -- LangChain's real chat models
    return a new runnable rather than mutating themselves in place."""

    def __init__(self, parent: FakeChatModel):
        self._parent = parent

    def invoke(self, messages: Any, config: Any = None) -> Any:
        self._parent.invoke_calls.append(messages)
        if self._parent.raise_error:
            raise self._parent.raise_error
        return self._parent.structured_response
