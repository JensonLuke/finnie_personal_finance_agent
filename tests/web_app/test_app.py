"""Full-app UI tests via Streamlit's official `AppTest` harness -- executes
the real `app.py` script in a simulated session (no browser needed) and lets
us inspect it for exceptions and content. This is the pytest-covered
counterpart to the manual live verification done during development (real
OpenAI + real yfinance, zero exceptions across every button in every tab --
see docs/technical-design.md §6); these tests use fakes so they run fast,
offline, and don't write into the project's real SQLite file.

Four things need isolating before each run, since `AppTest` exec's the
script fresh but the things it constructs are process-global:
1. `st.cache_resource` -- `get_services()` is cached process-wide, so a
   fake built in one test would otherwise leak into the next.
2. Network -- `YFinanceProvider`/`get_chat_model` are patched to fakes.
3. Filesystem -- `ProfileStore()`'s and `AuthService()`'s default SQLite
   path is the real `data/finnie.sqlite3`; both patched to a temp path so
   tests never write there (and never see another test's accounts).
4. The app now gates everything behind a login (core/auth.py) -- every test
   below signs up and signs in first via `_sign_up_and_sign_in`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import finnie.core.auth as auth_module
import finnie.core.llm as llm_module
import finnie.core.profile_store as profile_store_module
import finnie.data.providers.yfinance_provider as yfinance_module
from finnie.core.store import SqliteKeyValueStore
from finnie.data.providers.fake_provider import FakeProvider
from finnie.testing.fake_llm import FakeChatModel

APP_PATH = Path(__file__).resolve().parents[2] / "src" / "finnie" / "web_app" / "app.py"
TEST_USERNAME = "testuser"
TEST_PASSWORD = "correct-password-123"


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(yfinance_module, "YFinanceProvider", lambda: FakeProvider())
    monkeypatch.setattr(
        llm_module,
        "get_chat_model",
        lambda *a, **k: FakeChatModel(response_text="Narrated by a fake LLM for testing."),
    )
    profiles_db = tmp_path / "profiles.sqlite3"
    monkeypatch.setattr(
        profile_store_module,
        "SqliteKeyValueStore",
        lambda *a, **k: SqliteKeyValueStore(db_path=profiles_db),
    )
    auth_db = tmp_path / "auth.sqlite3"
    monkeypatch.setattr(
        auth_module,
        "SqliteKeyValueStore",
        lambda *a, **k: SqliteKeyValueStore(db_path=auth_db),
    )
    yield AppTest.from_file(APP_PATH)
    st.cache_resource.clear()


def _sign_up_and_sign_in(
    at: AppTest, username: str = TEST_USERNAME, password: str = TEST_PASSWORD
) -> None:
    """Drives the login page's two forms exactly as a user would, leaving
    `at` on the signed-in, 4-tab view."""
    at.run(timeout=30)
    assert not at.exception

    at.text_input(key="sign_up_username").set_value(username)
    at.text_input(key="sign_up_password").set_value(password)
    at.text_input(key="sign_up_confirm_password").set_value(password)
    at.button(key="FormSubmitter:sign_up_form-Create Account").click().run(timeout=30)
    assert not at.exception

    at.text_input(key="sign_in_username").set_value(username)
    at.text_input(key="sign_in_password").set_value(password)
    at.button(key="FormSubmitter:sign_in_form-Sign In").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["authenticated_user"] == username


def test_login_page_shown_when_not_authenticated(isolated_app):
    at = isolated_app
    at.run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 2  # Sign In, Sign Up -- not the 4 main tabs yet


def test_sign_up_then_sign_in_reaches_all_four_tabs(isolated_app):
    at = isolated_app
    _sign_up_and_sign_in(at)
    assert len(at.tabs) == 4


def test_sign_in_with_wrong_password_shows_an_error_and_stays_logged_out(isolated_app):
    at = isolated_app
    at.run(timeout=30)
    at.text_input(key="sign_up_username").set_value(TEST_USERNAME)
    at.text_input(key="sign_up_password").set_value(TEST_PASSWORD)
    at.text_input(key="sign_up_confirm_password").set_value(TEST_PASSWORD)
    at.button(key="FormSubmitter:sign_up_form-Create Account").click().run(timeout=30)

    at.text_input(key="sign_in_username").set_value(TEST_USERNAME)
    at.text_input(key="sign_in_password").set_value("wrong-password")
    at.button(key="FormSubmitter:sign_in_form-Sign In").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["authenticated_user"] is None
    assert any("Incorrect username or password" in e.value for e in at.error)


def test_log_out_returns_to_the_login_page(isolated_app):
    at = isolated_app
    _sign_up_and_sign_in(at)
    at.sidebar.button[0].click().run(timeout=30)  # the sidebar's only button: "Log out"
    assert not at.exception
    assert at.session_state["authenticated_user"] is None
    assert len(at.tabs) == 2


def test_chat_tab_full_round_trip(isolated_app):
    at = isolated_app
    _sign_up_and_sign_in(at)
    at.chat_input[0].set_value("What is a stock?").run(timeout=30)
    assert not at.exception
    assert len(at.chat_message) >= 2  # the user's message + the assistant's reply


def test_portfolio_tab_analyze_button(isolated_app):
    at = isolated_app
    _sign_up_and_sign_in(at)
    portfolio_tab = at.tabs[1]
    (button,) = [b for b in portfolio_tab.button if b.label == "Analyze Portfolio"]
    button.click().run(timeout=30)
    assert not at.exception
    metrics = at.tabs[1].metric
    assert len(metrics) == 3  # total value, diversification score, expense ratio
    assert metrics[0].value.startswith("$")


def test_market_tab_refresh_and_trend(isolated_app):
    at = isolated_app
    _sign_up_and_sign_in(at)
    market_tab = at.tabs[2]
    for b in market_tab.button:
        if b.label == "Refresh Quotes":
            b.click().run(timeout=30)
    assert not at.exception
    for b in at.tabs[2].button:
        if b.label == "Show Trend":
            b.click().run(timeout=30)
    assert not at.exception


def test_goals_tab_project_button(isolated_app):
    at = isolated_app
    _sign_up_and_sign_in(at)
    goals_tab = at.tabs[3]
    (button,) = [b for b in goals_tab.button if b.label == "Project"]
    button.click().run(timeout=30)
    assert not at.exception
    metrics = at.tabs[3].metric
    assert len(metrics) == 1  # projected value
    assert metrics[0].value.startswith("$")


def test_goals_tab_persists_the_goal_for_the_chat_tab_to_see(isolated_app):
    # This is the cross-tab consistency the profile-store persistence design
    # is for (docs/design/langgraph-workflow.md §3) -- a goal set in the
    # Goals tab should be visible to Chat via the same identity (now the
    # signed-in username, not a random UUID), without the UI needing to pass
    # it manually between tabs.
    at = isolated_app
    _sign_up_and_sign_in(at)
    username = at.session_state["authenticated_user"]

    goals_tab = at.tabs[3]
    (button,) = [b for b in goals_tab.button if b.label == "Project"]
    button.click().run(timeout=30)

    from finnie.core.profile_store import ProfileStore

    reloaded = ProfileStore().load(username)
    assert reloaded.goals
    assert reloaded.goals[0].name == "Retirement"  # the form's default goal name
