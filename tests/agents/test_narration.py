from finnie.agents.narration import narrate
from finnie.testing.fake_llm import FakeChatModel


def test_narrate_returns_llm_text_on_success():
    llm = FakeChatModel(response_text="A nicer, LLM-written answer.")
    result = narrate(llm, system_prompt="sys", user_content="hi", fallback="fallback text")
    assert result == "A nicer, LLM-written answer."


def test_narrate_falls_back_on_exception():
    llm = FakeChatModel(raise_error=RuntimeError("rate limited"))
    result = narrate(llm, system_prompt="sys", user_content="hi", fallback="fallback text")
    assert result == "fallback text"


def test_narrate_falls_back_on_empty_response():
    llm = FakeChatModel(response_text="   ")
    result = narrate(llm, system_prompt="sys", user_content="hi", fallback="fallback text")
    assert result == "fallback text"
