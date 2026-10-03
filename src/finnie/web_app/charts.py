"""Pure chart-builder functions -- each returns a Plotly `Figure`, so they're
easy to unit-test structurally (trace count, colors, axis titles) without a
running Streamlit app. Mark specs (thin 2px lines, >=8px markers, hover
tooltips, transparent background so charts blend with Streamlit's own
light/dark theme, no dual y-axis) follow the dataviz skill's guidance;
colors come from `theme.py`'s fixed categorical order.
"""

from __future__ import annotations

import plotly.graph_objects as go

from finnie.data.models import HistoricalBar
from finnie.web_app.theme import ASSET_CLASS_SLOT, color_for_asset_class, get_chrome

_FRIENDLY_ASSET_CLASS = {
    "equity": "Equity",
    "bond": "Bond",
    "etf": "ETF",
    "mutual_fund": "Mutual Fund",
    "cash": "Cash",
    "other": "Other",
}


def _sorted_allocation(allocation: dict[str, float]) -> list[tuple[str, float]]:
    # Fixed category order (not insertion order) so the same asset classes
    # always appear in the same order/color across renders and across the
    # pie and bar chart together.
    return sorted(allocation.items(), key=lambda item: ASSET_CLASS_SLOT.get(item[0], 99))


def _base_layout(chrome: dict, title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=chrome["primary_ink"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=chrome["secondary_ink"],
            family="system-ui, -apple-system, Segoe UI, sans-serif",
        ),
        margin=dict(l=10, r=10, t=48, b=10),
    )


def allocation_pie_chart(allocation: dict[str, float], chrome: dict | None = None) -> go.Figure:
    chrome = chrome or get_chrome()
    items = _sorted_allocation(allocation)
    labels = [_FRIENDLY_ASSET_CLASS.get(k, k.title()) for k, _ in items]
    values = [v for _, v in items]
    colors = [color_for_asset_class(k, chrome) for k, _ in items]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color=chrome["gridline"], width=2)),
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            textinfo="label+percent",
            textfont=dict(color=chrome["primary_ink"]),
            hole=0.45,  # donut -- easier to read labels around than a solid pie
            sort=False,
        )
    )
    fig.update_layout(**_base_layout(chrome, "Portfolio Allocation"), showlegend=len(items) > 4)
    return fig


def allocation_bar_chart(allocation: dict[str, float], chrome: dict | None = None) -> go.Figure:
    chrome = chrome or get_chrome()
    items = _sorted_allocation(allocation)
    labels = [_FRIENDLY_ASSET_CLASS.get(k, k.title()) for k, _ in items]
    values = [v for _, v in items]
    colors = [color_for_asset_class(k, chrome) for k, _ in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color=chrome["primary_ink"]),
        )
    )
    fig.update_layout(
        **_base_layout(chrome, "Allocation by Asset Class"),
        showlegend=False,
        xaxis=dict(
            title="% of portfolio",
            gridcolor=chrome["gridline"],
            zerolinecolor=chrome["baseline"],
            color=chrome["muted"],
        ),
        yaxis=dict(color=chrome["muted"], automargin=True),
    )
    return fig


def price_trend_chart(
    bars: list[HistoricalBar], ticker: str, chrome: dict | None = None
) -> go.Figure:
    chrome = chrome or get_chrome()
    dates = [b.date for b in bars]
    closes = [b.close for b in bars]

    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=closes,
            mode="lines",
            line=dict(color=chrome["sequential"], width=2),
            hovertemplate="%{x|%b %d, %Y}: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(chrome, f"{ticker} -- Price Trend"),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(gridcolor=chrome["gridline"], color=chrome["muted"]),
        yaxis=dict(
            title="Price ($)",
            gridcolor=chrome["gridline"],
            zerolinecolor=chrome["baseline"],
            color=chrome["muted"],
        ),
    )
    return fig


def goal_projection_chart(
    series: list[tuple[float, float]], target_amount: float, chrome: dict | None = None
) -> go.Figure:
    chrome = chrome or get_chrome()
    years = [point[0] for point in series]
    balances = [point[1] for point in series]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=balances,
            mode="lines+markers",
            name="Projected balance",
            line=dict(color=chrome["sequential"], width=2),
            marker=dict(size=8),
            hovertemplate="Year %{x:.0f}: $%{y:,.2f}<extra></extra>",
        )
    )
    # Target is chart chrome (a reference line), not a data series -- muted,
    # dashed ink rather than a categorical/sequential hue, per the skill's
    # "text wears text tokens, color carries identity" rule extended to
    # reference lines.
    fig.add_hline(
        y=target_amount,
        line=dict(color=chrome["muted"], width=2, dash="dash"),
        annotation_text=f"Target: ${target_amount:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color=chrome["muted"]),
    )
    fig.update_layout(
        **_base_layout(chrome, "Goal Projection"),
        showlegend=False,
        xaxis=dict(title="Years", gridcolor=chrome["gridline"], color=chrome["muted"]),
        yaxis=dict(
            title="Balance ($)",
            gridcolor=chrome["gridline"],
            zerolinecolor=chrome["baseline"],
            color=chrome["muted"],
        ),
    )
    return fig
