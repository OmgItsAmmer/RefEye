"""Design tokens — the single source of truth for all visual values.

Mirrors docs/architecture/theme.md exactly. No widget may hard-code a color,
radius, or spacing value; everything references a token from here.

Cloned from sample_ui/ (a Tauri/React reference console): monochrome
off-white primary on true near-black surfaces, sharp small radii, and a
two-typeface system (Outfit for UI chrome, Space Mono for data/technical
values). See theme.md for the full rationale.
"""

from __future__ import annotations

# --- Surfaces ---------------------------------------------------------------
BG_BASE = "#121212"
BG_SURFACE = "#1A1A1A"
BG_SURFACE_2 = "#222222"
BORDER = "#2A2A2A"
BORDER_STRONG = "#3A3A3A"

# --- Text --------------------------------------------------------------------
TEXT_PRIMARY = "#FAFAFA"  # same value as PRIMARY — active/emphasized text
TEXT_MAIN = "#E0E0E0"
TEXT_MUTED = "#525252"
TEXT_DISABLED = "#3A3A3A"

# --- Primary (monochrome accent) ---------------------------------------------
PRIMARY = "#FAFAFA"
PRIMARY_BG = "rgba(250, 250, 250, 0.10)"
PRIMARY_BG_HOVER = "rgba(250, 250, 250, 0.16)"
PRIMARY_BORDER = "rgba(250, 250, 250, 0.35)"
PRIMARY_TEXT_ON_FILL = "#121212"

# --- Semantic — reserved for exactly two states, nowhere else ---------------
ALERT = "#FF3366"
ALERT_BG = "rgba(255, 51, 102, 0.10)"
SUCCESS = "#00FF66"

# --- Data / ranking ramp (single hue, opacity-ramped, never traffic-light) ---
RANK_COLORS = [
    "rgba(250, 250, 250, 0.90)",
    "rgba(250, 250, 250, 0.65)",
    "rgba(250, 250, 250, 0.40)",
    "rgba(250, 250, 250, 0.22)",
]


def rank_color(index: int) -> str:
    """Color for a candidate at rank `index` (0 = best). Clamps to lowest."""
    return RANK_COLORS[min(index, len(RANK_COLORS) - 1)]


# --- Loader colors (deliberate exception: referee cards stay recognizable) --
CARD_YELLOW = "#E8C158"
CARD_RED = "#FF3366"
BALL_WHITE = "#FAFAFA"
BALL_PANEL = "#1A1A1A"

# --- Typography ---------------------------------------------------------------
# Vendored locally in fonts/ and registered via QFontDatabase at startup —
# the app must render identically offline. Space Mono is the "data face":
# timestamps, ids, source/codec strings, diagnostic values.
FONT_FAMILY = "Outfit"
FONT_FAMILY_DATA = "Space Mono"

FONT_SIZE_TITLE = 15       # screen/section title — UPPERCASE, tracked
FONT_SIZE_HEADING = 18     # panel content heading — sentence case
FONT_SIZE_BODY = 13
FONT_SIZE_META = 10        # micro caption — UPPERCASE, tracked
FONT_SIZE_DATA = 11        # Space Mono data values
FONT_SIZE_BUTTON = 10      # UPPERCASE, tracked
FONT_SIZE_ICON = 20        # sidebar nav glyphs — legible at a glance, unlike body text

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600

TRACKING_TITLE = 1.5    # px, absolute letter-spacing
TRACKING_META = 1.0
TRACKING_BUTTON = 1.5

# --- Geometry ------------------------------------------------------------------
SPACING_UNIT = 8
RADIUS_PANEL = 8
RADIUS_CONTROL = 4
RADIUS_BADGE = 2
RADIUS_PILL = 999
PANEL_PADDING = 20
