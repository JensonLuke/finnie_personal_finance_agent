"""Streamlit entry point: `streamlit run src/finnie/web_app/app.py`.

Four tabs per the Milestones FAQ's UI guidance (Chat, Portfolio, Market,
Goals). Chat runs the full LangGraph workflow; the other three call agents
directly for a faster, more dashboard-like interaction (see each render_*
function's docstring for why) but reuse the *same* agent instances -- so
"How is my portfolio doing?" in Chat and clicking "Analyze" in the Portfolio
tab run the identical calculation/narration code, not two implementations
of the same thing.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from finnie.agents.goal_agent import GoalAgent, project_growth_series
from finnie.agents.market_agent import MarketAgent
from finnie.agents.news_agent import NewsAgent
from finnie.agents.portfolio_agent import PortfolioAgent
from finnie.agents.qa_agent import QAAgent
from finnie.agents.tax_agent import TaxAgent
from finnie.core.auth import AuthError, AuthService
from finnie.core.profile_store import ProfileStore
from finnie.core.state import FinancialGoal, Holding, Portfolio, UserProfile
from finnie.data.market_data_service import MarketDataService
from finnie.data.models import MarketDataUnavailable
from finnie.data.providers.yfinance_provider import YFinanceProvider
from finnie.data.seed import load_seed_cache
from finnie.utils.logging import configure_logging, get_logger
from finnie.web_app import charts
from finnie.web_app.theme import get_chrome
from finnie.workflow.graph import build_graph

st.set_page_config(page_title="Finnie", page_icon="\U0001f4b0", layout="wide")
configure_logging()
logger = get_logger(__name__)

# Only applied if the user hasn't already set LANGSMITH_PROJECT themselves
# (see .env.example) -- keeps web-app traces grouped separately from the MCP
# server's in the LangSmith dashboard without overriding an explicit choice.
os.environ.setdefault("LANGSMITH_PROJECT", "finnie-webapp")

_DEFAULT_WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT"]
_RISK_LEVELS = ["conservative", "moderate", "aggressive"]


@st.cache_resource
def get_services() -> dict:
    """Built once per server process (`st.cache_resource`, not per session)
    -- the market-data cache and RAG index are meant to be shared across
    everyone using this app instance, not rebuilt per browser tab."""
    market_data = MarketDataService(primary=YFinanceProvider(), seed_cache=load_seed_cache())
    profile_store = ProfileStore()
    auth = AuthService()
    agents = {
        "qa": QAAgent(),
        "portfolio": PortfolioAgent(market_data),
        "market": MarketAgent(market_data),
        "goal": GoalAgent(),
        "news": NewsAgent(market_data),
        "tax": TaxAgent(),
    }
    graph = build_graph(agents, profile_store)
    return {
        "graph": graph,
        "market_data": market_data,
        "profile_store": profile_store,
        "auth": auth,
        "agents": agents,
    }


def _init_session_state() -> None:
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = Portfolio(
            holdings=[Holding(ticker="AAPL", shares=10), Holding(ticker="BND", shares=5)]
        )


def _current_profile() -> UserProfile:
    services = get_services()
    return services["profile_store"].load(st.session_state.authenticated_user)


def render_login_page() -> None:
    """Gates the app behind a minimal username/password login (see
    core/auth.py for exactly how minimal, and why). Once signed in, the
    username itself becomes the persistence key (`session_id`) for
    `ProfileStore` -- see main()'s note on that -- so a returning user is
    recognized across browser sessions and devices, not just within one tab.
    """
    st.title("Finnie -- AI Finance Assistant")
    st.caption("Sign in or create an account to continue.")
    auth: AuthService = get_services()["auth"]

    sign_in_tab, sign_up_tab = st.tabs(["Sign In", "Sign Up"])

    with sign_in_tab, st.form("sign_in_form"):
        username = st.text_input("Username", key="sign_in_username")
        password = st.text_input("Password", type="password", key="sign_in_password")
        if st.form_submit_button("Sign In", type="primary"):
            try:
                canonical_username = auth.authenticate(username, password)
            except AuthError as exc:
                st.error(str(exc))
            else:
                st.session_state.authenticated_user = canonical_username
                st.rerun()

    with sign_up_tab, st.form("sign_up_form"):
        new_username = st.text_input("Choose a username", key="sign_up_username")
        new_password = st.text_input("Choose a password", type="password", key="sign_up_password")
        confirm_password = st.text_input(
            "Confirm password", type="password", key="sign_up_confirm_password"
        )
        if st.form_submit_button("Create Account", type="primary"):
            if new_password != confirm_password:
                st.error("Passwords don't match.")
            else:
                try:
                    auth.register(new_username, new_password)
                except AuthError as exc:
                    st.error(str(exc))
                else:
                    st.success("Account created -- you can now sign in from the Sign In tab.")


def render_chat_tab() -> None:
    st.header("Chat")
    st.caption("Runs the full multi-agent workflow -- routing, dependency ordering, and synthesis.")
    for role, content in st.session_state.chat_history:
        st.chat_message(role).write(content)

    prompt = st.chat_input("Ask about investing, your portfolio, or the market...")
    if not prompt:
        return

    st.session_state.chat_history.append(("user", prompt))
    st.chat_message("user").write(prompt)

    services = get_services()
    state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_profile": UserProfile(session_id=st.session_state.authenticated_user),
        "portfolio": st.session_state.portfolio,
        "intent": [],
        "pending_agents": [],
        "agent_outputs": {},
        "retrieved_docs": [],
        "market_data": {},
        "error": None,
        "final_response": None,
    }
    with st.spinner("Thinking..."):
        try:
            result = services["graph"].invoke(
                state,
                config={
                    "run_name": "chat_turn",
                    "tags": ["web_app", "chat_tab"],
                    "metadata": {"username": st.session_state.authenticated_user},
                },
            )
            response = result.get("final_response", "Sorry, something went wrong.")
        except Exception:
            logger.exception("Chat turn failed")
            response = "Something went wrong on my end -- please try again."
    st.session_state.chat_history.append(("assistant", response))
    st.chat_message("assistant").write(response)


def _holdings_editor_df(holdings: list[Holding]) -> pd.DataFrame:
    if not holdings:
        return pd.DataFrame([{"ticker": "", "shares": 0.0}])
    return pd.DataFrame([{"ticker": h.ticker, "shares": h.shares} for h in holdings])


def _df_to_holdings(df: pd.DataFrame) -> list[Holding]:
    holdings = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        shares = row.get("shares", 0) or 0
        if ticker and shares > 0:
            holdings.append(Holding(ticker=ticker, shares=float(shares)))
    return holdings


def render_portfolio_tab() -> None:
    """Calls PortfolioAgent directly (not the full graph) -- a dashboard
    view where the user is editing structured holdings, not typing a
    message, so there's no router classification to run first."""
    st.header("Portfolio Analysis")
    st.caption("Edit your holdings below, then analyze. Ticker + number of shares.")

    edited = st.data_editor(
        _holdings_editor_df(st.session_state.portfolio.holdings),
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
        },
        key="holdings_editor",
        width="stretch",
    )

    if st.button("Analyze Portfolio", type="primary"):
        st.session_state.portfolio = Portfolio(holdings=_df_to_holdings(edited))
        st.session_state.portfolio_analyzed = True

    if not st.session_state.get("portfolio_analyzed") or not st.session_state.portfolio.holdings:
        st.info("Add at least one holding and click **Analyze Portfolio** to see results.")
        return

    services = get_services()
    state = {
        "messages": [],
        "user_profile": _current_profile(),
        "portfolio": st.session_state.portfolio,
        "agent_outputs": {},
    }
    with st.spinner("Analyzing..."):
        result = services["agents"]["portfolio"].run(state)

    st.markdown(result.summary)
    if result.degraded:
        st.warning("Pricing data may be stale or incomplete -- see note above.")

    metrics = result.data.get("metrics")
    if not metrics:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", f"${metrics['total_value']:,.2f}")
    col2.metric(
        "Diversification Score",
        f"{metrics['diversification_score']:.2f}",
        help="0 = concentrated, 1 = highly spread",
    )
    expense_ratio = metrics.get("weighted_expense_ratio")
    col3.metric(
        "Weighted Expense Ratio", f"{expense_ratio:.2%}" if expense_ratio is not None else "n/a"
    )

    allocation = metrics.get("allocation_by_asset_class") or {}
    if allocation:
        chrome = get_chrome()
        chart_col1, chart_col2 = st.columns(2)
        chart_col1.plotly_chart(charts.allocation_pie_chart(allocation, chrome), width="stretch")
        chart_col2.plotly_chart(charts.allocation_bar_chart(allocation, chrome), width="stretch")


def render_market_tab() -> None:
    """Calls MarketDataService directly -- a live-quotes dashboard doesn't
    need the router or an LLM call, just the same resilience-chain-backed
    service the agents use."""
    st.header("Market Overview")

    watchlist_input = st.text_input(
        "Watchlist (comma-separated tickers)", value=", ".join(_DEFAULT_WATCHLIST)
    )
    tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

    services = get_services()
    market_data: MarketDataService = services["market_data"]

    if tickers and st.button("Refresh Quotes", type="primary"):
        st.session_state.market_quotes = None  # force a re-fetch below

    if "market_quotes" not in st.session_state:
        st.session_state.market_quotes = None

    if tickers and st.session_state.market_quotes is None:
        with st.spinner("Fetching quotes..."):
            try:
                st.session_state.market_quotes = market_data.get_quotes(tickers)
            except MarketDataUnavailable as exc:
                st.error(f"Market data unavailable: {exc}")
                st.session_state.market_quotes = {}

    quotes = st.session_state.market_quotes or {}
    if quotes:
        cols = st.columns(len(quotes))
        for col, (ticker, quote) in zip(cols, quotes.items(), strict=False):
            freshness = " (cached)" if quote.stale else ""
            col.metric(
                ticker + freshness,
                f"${quote.price:,.2f}",
                f"{quote.change_percent:+.2f}%",
            )
        if any(q.stale for q in quotes.values()):
            st.caption("🟡 Some quotes are cached/stale -- live data was temporarily unavailable.")

    st.divider()
    st.subheader("Price Trend")
    trend_ticker = st.selectbox("Ticker", options=tickers or _DEFAULT_WATCHLIST)
    if st.button("Show Trend"):
        with st.spinner(f"Loading {trend_ticker} history..."):
            try:
                bars = market_data.get_history(trend_ticker)
            except MarketDataUnavailable as exc:
                st.error(f"History unavailable for {trend_ticker}: {exc}")
                bars = []
        if bars:
            st.plotly_chart(
                charts.price_trend_chart(bars, trend_ticker, get_chrome()), width="stretch"
            )
        else:
            st.info("No history available for this ticker.")


def render_goals_tab() -> None:
    """Calls GoalAgent directly, but mirrors the real Goal->Portfolio
    dependency: if the user has holdings, PortfolioAgent runs first so the
    projection starts from the actual current portfolio value, exactly like
    the graph's supervisor queue does for a chat message."""
    st.header("Goal Planning")

    with st.form("goal_form"):
        name = st.text_input("Goal name", value="Retirement")
        target_amount = st.number_input(
            "Target amount ($)", min_value=0.0, value=50000.0, step=1000.0
        )
        years = st.number_input("Years", min_value=1.0, max_value=60.0, value=10.0, step=1.0)
        monthly_contribution = st.number_input(
            "Monthly contribution ($)", min_value=0.0, value=300.0, step=50.0
        )
        risk_tolerance = st.selectbox("Risk tolerance", options=_RISK_LEVELS, index=1)
        submitted = st.form_submit_button("Project", type="primary")

    if not submitted:
        return

    services = get_services()
    target_date = (date.today() + timedelta(days=years * 365.25)).isoformat()
    profile = UserProfile(
        session_id=st.session_state.authenticated_user,
        risk_tolerance=risk_tolerance,
        goals=[
            FinancialGoal(
                name=name,
                target_amount=target_amount,
                monthly_contribution=monthly_contribution,
                target_date=target_date,
            )
        ],
    )
    services["profile_store"].save(profile)  # persists for Chat tab too, same session_id

    agent_outputs = {}
    if st.session_state.portfolio.holdings:
        portfolio_state = {
            "messages": [],
            "user_profile": profile,
            "portfolio": st.session_state.portfolio,
            "agent_outputs": {},
        }
        agent_outputs["portfolio"] = services["agents"]["portfolio"].run(portfolio_state)

    goal_state = {
        "messages": [],
        "user_profile": profile,
        "agent_outputs": agent_outputs,
    }
    with st.spinner("Projecting..."):
        result = services["agents"]["goal"].run(goal_state)

    st.markdown(result.summary)

    data = result.data
    if "projected_value" not in data:
        return

    on_track = data["on_track"]
    st.metric(
        "Projected Value",
        f"${data['projected_value']:,.2f}",
        f"{'On track' if on_track else 'Short'} of ${data['target_amount']:,.2f} target",
    )

    series = project_growth_series(
        monthly_contribution=data["monthly_contribution"],
        years=data["years"],
        annual_return=data["assumed_annual_return"],
        starting_balance=data["starting_balance"],
    )
    st.plotly_chart(
        charts.goal_projection_chart(
            series, target_amount=data["target_amount"], chrome=get_chrome()
        ),
        width="stretch",
    )
    st.caption(
        f"Assumes a {data['assumed_annual_return']:.0%} annual return based on your risk "
        "tolerance -- a simplified, static assumption, not a guarantee."
    )


def main() -> None:
    _init_session_state()

    if not st.session_state.authenticated_user:
        render_login_page()
        return

    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.authenticated_user}**")
        if st.button("Log out"):
            st.session_state.authenticated_user = None
            st.session_state.chat_history = []
            st.rerun()

    st.title("Finnie -- AI Finance Assistant")
    chat_tab, portfolio_tab, market_tab, goals_tab = st.tabs(
        ["Chat", "Portfolio", "Market", "Goals"]
    )
    with chat_tab:
        render_chat_tab()
    with portfolio_tab:
        render_portfolio_tab()
    with market_tab:
        render_market_tab()
    with goals_tab:
        render_goals_tab()


main()
