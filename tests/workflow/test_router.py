from finnie.workflow.router import classify, resolve_dependencies


def test_classify_defaults_to_qa_for_unmatched_query():
    assert classify("What's the meaning of life?") == ["qa"]


def test_classify_matches_portfolio_keywords():
    assert "portfolio" in classify("Can you look at my portfolio?")


def test_classify_matches_tax_keyword_with_word_boundary():
    assert "tax" in classify("What's a Roth IRA?")


def test_classify_does_not_false_positive_on_substring():
    # "tax" must not fire on "syntax" -- word-boundary matching, not substring.
    assert classify("I have a syntax error in my code") == ["qa"]


def test_resolve_dependencies_orders_portfolio_before_goal():
    ordered = resolve_dependencies(["goal"])
    assert ordered == ["portfolio", "goal"]


def test_resolve_dependencies_is_a_no_op_without_dependencies():
    assert resolve_dependencies(["market"]) == ["market"]


def test_resolve_dependencies_does_not_duplicate_existing_portfolio():
    assert resolve_dependencies(["portfolio", "goal"]) == ["portfolio", "goal"]
