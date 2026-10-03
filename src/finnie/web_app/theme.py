"""Chart color/style constants, per the dataviz skill's reference palette.

Categorical hues are used in a **fixed order** (never cycled/regenerated),
per the skill's core rule -- color follows the entity (asset class), and the
same entity gets the same color in every chart it appears in (e.g. "equity"
is always slot 1's blue, whether in the allocation pie or the allocation
bar). This module is the one place that order is defined.
"""

from __future__ import annotations

# The 8-slot categorical order from the dataviz skill's reference palette
# (references/palette.md) -- validated for CVD-safety in this fixed order,
# both light and dark surfaces. Never reassign a slot to a different
# category, and never add a 9th generated hue (fold into "Other" instead).
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# Fixed assignment: an asset class always gets the same slot, in every
# chart, regardless of which classes happen to be present this time.
ASSET_CLASS_SLOT = {
    "equity": 0,
    "bond": 1,
    "etf": 2,
    "mutual_fund": 3,
    "cash": 4,
    "other": 5,
}

SEQUENTIAL_HUE_LIGHT = "#2a78d6"  # blue -- single-series trend lines
SEQUENTIAL_HUE_DARK = "#3987e5"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


def get_mode() -> str:
    """Best-effort light/dark detection from Streamlit's own theme config.
    Falls back to "light" -- there's no reliable way to read the viewer's
    OS/browser preference from server-side Streamlit code."""
    try:
        import streamlit as st

        base = st.get_option("theme.base")
        return "dark" if base == "dark" else "light"
    except Exception:
        return "light"


def get_chrome(mode: str | None = None) -> dict[str, str]:
    mode = mode or get_mode()
    if mode == "dark":
        return {
            "primary_ink": "#ffffff",
            "secondary_ink": "#c3c2b7",
            "muted": "#898781",
            "gridline": "#2c2c2a",
            "baseline": "#383835",
            "categorical": CATEGORICAL_DARK,
            "sequential": SEQUENTIAL_HUE_DARK,
        }
    return {
        "primary_ink": "#0b0b0b",
        "secondary_ink": "#52514e",
        "muted": "#898781",
        "gridline": "#e1e0d9",
        "baseline": "#c3c2b7",
        "categorical": CATEGORICAL_LIGHT,
        "sequential": SEQUENTIAL_HUE_LIGHT,
    }


def color_for_asset_class(asset_class: str, chrome: dict[str, str]) -> str:
    slot = ASSET_CLASS_SLOT.get(asset_class, len(ASSET_CLASS_SLOT) - 1)
    palette = chrome["categorical"]
    return palette[slot % len(palette)]
