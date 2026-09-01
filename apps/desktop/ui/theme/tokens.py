"""Design tokens — the single source of truth for all visual values.

Mirrors docs/architecture/theme.md exactly. No widget may hard-code a color,
radius, or spacing value; everything references a token from here.
"""

from __future__ import annotations

# --- Surfaces -------------------------------------------------------------
BG_BASE = "#12161C"
BG_PANEL = "#1D222B"
BG_PANEL_RAISED = "#242A34"
BORDER = "#2A303B"
BORDER_STRONG = "#3A4150"

# --- Text -----------------------------------------------------------------
TEXT_PRIMARY = "#E8E9ED"
TEXT_SECONDARY = "#D5D8DE"
TEXT_MUTED = "#8B92A0"
TEXT_DISABLED = "#4B525E"

# --- Accent (pitch green) -------------------------------------------------
ACCENT = "#4A8C72"
ACCENT_HOVER = "#5A9C82"
ACCENT_BG = "rgba(74, 140, 114, 0.15)"
ACCENT_TEXT_ON_FILL = "#0D1A15"
ACCENT_TEXT = "#6FBFA0"

# --- Semantic -------------------------------------------------------------
INFO = "#5B7FA6"
WARNING = "#C99A4A"
DANGER = "#B5544B"
DANGER_BG = "rgba(181, 84, 75, 0.15)"

# --- Data / ranking ramp (single hue, never traffic-light) ----------------
RANK_COLORS = ["#4A8C72", "#3E7862", "#325F4F", "#26493C"]


def rank_color(index: int) -> str:
    """Color for a candidate at rank `index` (0 = best). Clamps to lowest."""
    return RANK_COLORS[min(index, len(RANK_COLORS) - 1)]


# --- Loader colors (deliberate exception: referee cards stay recognizable) -
CARD_YELLOW = "#C99A4A"
CARD_RED = "#B5544B"
BALL_WHITE = "#D5D8DE"
BALL_PANEL = "#1D222B"

# --- Typography -----------------------------------------------------------
FONT_FAMILY = "Segoe UI"
FONT_SIZE_HEADING = 16
FONT_SIZE_BODY = 13
FONT_SIZE_META = 12

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500

# --- Geometry -------------------------------------------------------------
SPACING_UNIT = 8
RADIUS_PANEL = 12
RADIUS_CONTROL = 6
RADIUS_BADGE = 4
PANEL_PADDING = 16
