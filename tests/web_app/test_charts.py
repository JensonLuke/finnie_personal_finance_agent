"""Structural tests for the chart builders -- Plotly figures are plain
Python objects, so these check trace count/type, color assignment, and axis
config without needing a running Streamlit app or a browser."""

from finnie.data.models import HistoricalBar
from finnie.web_app import charts
from finnie.web_app.theme import get_chrome

CHROME = get_chrome("light")


def test_allocation_pie_chart_has_one_trace_per_category_in_fixed_order():
    allocation = {"bond": 30.0, "equity": 70.0}  # deliberately out of canonical order
    fig = charts.allocation_pie_chart(allocation, chrome=CHROME)
    assert len(fig.data) == 1
    pie = fig.data[0]
    # Canonical order is equity before bond (theme.ASSET_CLASS_SLOT), regardless
    # of the input dict's order.
    assert list(pie.labels) == ["Equity", "Bond"]
    assert list(pie.values) == [70.0, 30.0]


def test_allocation_pie_and_bar_charts_use_the_same_color_per_asset_class():
    allocation = {"equity": 60.0, "bond": 40.0}
    pie = charts.allocation_pie_chart(allocation, chrome=CHROME)
    bar = charts.allocation_bar_chart(allocation, chrome=CHROME)
    # "color follows the entity" -- equity is the same color in both charts.
    assert list(pie.data[0].marker.colors) == list(bar.data[0].marker.color)


def test_allocation_charts_never_reuse_a_slot_for_two_categories():
    allocation = {"equity": 25.0, "bond": 25.0, "etf": 25.0, "cash": 25.0}
    fig = charts.allocation_pie_chart(allocation, chrome=CHROME)
    colors = list(fig.data[0].marker.colors)
    assert len(colors) == len(set(colors))


def test_price_trend_chart_is_a_single_series_line_no_dual_axis():
    bars = [
        HistoricalBar(ticker="AAPL", date=__import__("datetime").date(2026, 1, i), close=100.0 + i)
        for i in range(1, 6)
    ]
    fig = charts.price_trend_chart(bars, "AAPL", chrome=CHROME)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].line.width == 2
    # Exactly one y-axis -- never a second y-axis layered on the same chart.
    assert not any(key.startswith("yaxis2") for key in fig.layout.to_plotly_json())


def test_goal_projection_chart_has_a_line_and_a_target_reference_line():
    series = [(0.0, 1000.0), (1.0, 2200.0), (2.0, 3500.0)]
    fig = charts.goal_projection_chart(series, target_amount=5000.0, chrome=CHROME)
    assert len(fig.data) == 1  # the target line is layout chrome, not a second data series
    assert fig.data[0].mode == "lines+markers"
    assert fig.data[0].marker.size >= 8
    shapes = fig.layout.shapes
    assert len(shapes) == 1  # the hline
    assert shapes[0].y0 == shapes[0].y1 == 5000.0


def test_goal_projection_target_line_is_muted_not_a_categorical_color():
    series = [(0.0, 0.0), (1.0, 1000.0)]
    fig = charts.goal_projection_chart(series, target_amount=2000.0, chrome=CHROME)
    assert fig.layout.shapes[0].line.color == CHROME["muted"]
    assert fig.layout.shapes[0].line.dash == "dash"
