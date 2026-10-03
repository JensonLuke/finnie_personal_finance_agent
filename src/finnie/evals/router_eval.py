"""LangSmith dataset + evaluation for `workflow/router.py`'s LLM
classification -- built to answer the follow-up to
docs/design/langgraph-workflow.md §8's tracing note: tracing shows you what
happened on one call, an eval dataset tells you whether it's *right* across
many calls, tracked over time as the router prompt changes.

Chosen over RAG or end-to-end response quality as the first eval target
because: RAG retrieval already has its own eval set
(tests/rag/eval_queries.jsonl, run via pytest); end-to-end response quality
needs an LLM-judge design decision that's a separate piece of work; router
classification has objective ground truth (which agent(s), which tickers,
whether a goal/risk-tolerance was stated) and is the most architecturally
central piece not yet covered by any eval.

Run standalone (needs LANGSMITH_API_KEY and OPENAI_API_KEY in .env):
    python -m finnie.evals.router_eval

This creates the dataset once (skipped on subsequent runs if it already
exists -- see `ensure_dataset`'s docstring for the re-run caveat that
follows from that) and runs a new experiment against it every time.
"""

from __future__ import annotations

from langsmith import Client
from langsmith.evaluation import evaluate

from finnie.core.llm import get_chat_model
from finnie.workflow.router import route

DATASET_NAME = "finnie-router-classification"
DATASET_DESCRIPTION = (
    "Ground truth for workflow/router.py's LLM classification: which "
    "agent(s) a message should dispatch to, plus ticker/goal/risk-tolerance "
    "extraction. See docs/design/langgraph-workflow.md §8."
)

# Each example's "outputs" is compared against route_target()'s output by
# the evaluators below. `extracted_goal_name` is included for visibility in
# the LangSmith UI, but goal_presence_match only checks presence/absence --
# not the exact name -- since a stated goal's name is free text an LLM can
# reasonably paraphrase (e.g. "house" vs "house down payment").
EXAMPLES: list[dict] = [
    {
        "inputs": {"query": "What is a stock, and how does it work?"},
        "outputs": {
            "intents": ["qa"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "What is compound interest and why does it matter?"},
        "outputs": {
            "intents": ["qa"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "Tell me about diversification."},
        "outputs": {
            "intents": ["qa"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "Can you look at my portfolio and tell me my diversification score?"},
        "outputs": {
            "intents": ["portfolio"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "What's my total portfolio value right now?"},
        "outputs": {
            "intents": ["portfolio"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "What's AAPL trading at right now?"},
        "outputs": {
            "intents": ["market"],
            "tickers": ["AAPL"],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "How is Tesla stock doing today?"},
        "outputs": {
            "intents": ["market"],
            "tickers": ["TSLA"],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {
            "query": "I want to save $50,000 for a house down payment in 10 years, putting in $400 a month."
        },
        "outputs": {
            "intents": ["goal"],
            "tickers": [],
            "extracted_goal_name": "house down payment",
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "How much should I save monthly to retire with $1,000,000 in 30 years?"},
        "outputs": {
            "intents": ["goal"],
            "tickers": [],
            "extracted_goal_name": "retirement",
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "I'm a pretty conservative investor -- is my current portfolio too risky?"},
        "outputs": {
            "intents": ["portfolio"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": "conservative",
        },
    },
    {
        "inputs": {
            "query": (
                "I'm aggressive with risk and want to retire on $800,000 in 25 years, "
                "saving $800 a month."
            )
        },
        "outputs": {
            "intents": ["goal"],
            "tickers": [],
            "extracted_goal_name": "retirement",
            "extracted_risk_tolerance": "aggressive",
        },
    },
    {
        "inputs": {"query": "What's the latest news on Amazon?"},
        "outputs": {
            "intents": ["news"],
            "tickers": ["AMZN"],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "Any recent news about Microsoft?"},
        "outputs": {
            "intents": ["news"],
            "tickers": ["MSFT"],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "What's a Roth IRA?"},
        "outputs": {
            "intents": ["tax"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "How does capital gains tax work?"},
        "outputs": {
            "intents": ["tax"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "What's the difference between a 401k and an IRA?"},
        "outputs": {
            "intents": ["tax"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "I have $50,000 in my portfolio -- should I put more into a Roth IRA?"},
        "outputs": {
            "intents": ["portfolio", "tax"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        "inputs": {"query": "What's a stock split, and how has AAPL's price moved this week?"},
        "outputs": {
            "intents": ["qa", "market"],
            "tickers": ["AAPL"],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
    {
        # Genuinely out-of-domain -- the router only has 6 finance categories
        # to choose from, so this checks it falls back to the closest
        # reasonable one (qa) rather than refusing or crashing, matching
        # classify()'s own fallback behavior. Best-effort ground truth, not
        # a hard requirement -- there's no "none of the above" option.
        "inputs": {"query": "What's the weather like today?"},
        "outputs": {
            "intents": ["qa"],
            "tickers": [],
            "extracted_goal_name": None,
            "extracted_risk_tolerance": None,
        },
    },
]


def route_target(inputs: dict) -> dict:
    """The `target` LangSmith's evaluate() calls per example -- wraps the
    real router with the real LLM, serialized to a plain dict so it's
    directly comparable to an example's "outputs"."""
    decision = route(inputs["query"], llm=get_chat_model())
    return {
        "intents": decision.intents,
        "tickers": decision.tickers,
        "extracted_goal_name": decision.extracted_goal.name if decision.extracted_goal else None,
        "extracted_risk_tolerance": decision.extracted_risk_tolerance,
    }


def intents_exact_match(run, example) -> dict:
    actual = set(run.outputs.get("intents") or [])
    expected = set(example.outputs.get("intents") or [])
    return {"key": "intents_exact_match", "score": float(actual == expected)}


def intents_recall(run, example) -> dict:
    """Partial credit for a compound query: what fraction of the expected
    agents were actually included, even if the set wasn't exact."""
    actual = set(run.outputs.get("intents") or [])
    expected = set(example.outputs.get("intents") or [])
    if not expected:
        return {"key": "intents_recall", "score": 1.0 if not actual else 0.0}
    return {"key": "intents_recall", "score": len(actual & expected) / len(expected)}


def tickers_match(run, example) -> dict:
    actual = {t.upper() for t in (run.outputs.get("tickers") or [])}
    expected = {t.upper() for t in (example.outputs.get("tickers") or [])}
    return {"key": "tickers_match", "score": float(actual == expected)}


def risk_tolerance_match(run, example) -> dict:
    actual = run.outputs.get("extracted_risk_tolerance")
    expected = example.outputs.get("extracted_risk_tolerance")
    return {"key": "risk_tolerance_match", "score": float(actual == expected)}


def goal_presence_match(run, example) -> dict:
    actual_present = run.outputs.get("extracted_goal_name") is not None
    expected_present = example.outputs.get("extracted_goal_name") is not None
    return {"key": "goal_presence_match", "score": float(actual_present == expected_present)}


EVALUATORS = [
    intents_exact_match,
    intents_recall,
    tickers_match,
    risk_tolerance_match,
    goal_presence_match,
]


def ensure_dataset(client: Client | None = None) -> None:
    """Creates the dataset + uploads EXAMPLES only if the dataset doesn't
    already exist. Caveat: if you edit EXAMPLES later, re-running this
    won't push the changes to an already-existing dataset -- delete it in
    the LangSmith UI first (or give it a new DATASET_NAME) to pick up
    edits. Kept simple on purpose rather than diffing/upserting examples."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    client.create_dataset(dataset_name=DATASET_NAME, description=DATASET_DESCRIPTION)
    client.create_examples(dataset_name=DATASET_NAME, examples=EXAMPLES)


def run_evaluation(client: Client | None = None):
    client = client or Client()
    ensure_dataset(client)
    return evaluate(
        route_target,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix="router-eval",
        client=client,
        metadata={"component": "workflow.router"},
    )


if __name__ == "__main__":
    results = run_evaluation()
    print(results)
