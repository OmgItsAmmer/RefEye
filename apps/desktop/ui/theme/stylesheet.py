"""Application-wide QSS, generated from design tokens.

All styling lives here. Widgets opt into styling via `objectName` or a Qt
dynamic property (`role`), never via inline setStyleSheet calls.

Dynamic properties used:
    role="title"     UPPERCASE tracked screen/section title (panel header)
    role="meta"       sentence-case secondary/descriptive text
    role="label"      UPPERCASE tracked muted micro-caption (chrome, not content)
    role="data"       Space Mono technical value (timestamp, id, codec)
    role="primary"    outlined high-emphasis button (max one per view)
    role="secondary"  outlined default button
    role="danger"     outlined destructive/retry button
    role="ghost"      borderless low-emphasis button
    variant="accent"|"info"|"warning"|"danger"|"muted"  StatusBadge state (see common.py)

QSS supports neither text-transform, letter-spacing, nor box-shadow — case,
tracking, and elevation are handled in Python (common.py) instead. This
system is intentionally flat: no drop shadows or glow anywhere in UI chrome
(theme.md section 1), depth comes only from the base -> surface -> surface-2
step.
"""

from __future__ import annotations

from apps.desktop.ui.theme import tokens as t


def build_stylesheet() -> str:
    return f"""
/* ---------- base ----------
   Deliberately no background here. A blanket QWidget background cascades
   into every nested child and repaints panel interiors with the base color,
   which flattens the card surfaces. Only the window paints the ground; each
   surface below opts in explicitly. */
QWidget {{
    color: {t.TEXT_MAIN};
    font-family: "{t.FONT_FAMILY}";
    font-size: {t.FONT_SIZE_BODY}px;
    font-weight: {t.WEIGHT_REGULAR};
    selection-background-color: {t.PRIMARY};
    selection-color: {t.PRIMARY_TEXT_ON_FILL};
}}

QMainWindow, QDialog {{
    background-color: {t.BG_BASE};
}}

QToolTip {{
    background-color: {t.BG_SURFACE_2};
    color: {t.TEXT_MAIN};
    border: 1px solid {t.BORDER_STRONG};
    border-radius: 2px;
    padding: 5px 9px;
    font-size: {t.FONT_SIZE_DATA}px;
}}

/* ---------- panels ---------- */
QFrame#Panel {{
    background-color: {t.BG_SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PANEL}px;
}}

/* Recessed secondary surface (diagnostics) — de-emphasized vs main panels. */
QFrame#PanelRecessed {{
    background-color: {t.BG_BASE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PANEL}px;
}}

/* ---------- typography ---------- */
QLabel {{
    background-color: transparent;
}}

QLabel[role="title"] {{
    font-size: {t.FONT_SIZE_TITLE}px;
    font-weight: {t.WEIGHT_SEMIBOLD};
    color: {t.TEXT_MAIN};
}}

QLabel[role="meta"] {{
    font-size: {t.FONT_SIZE_BODY}px;
    color: {t.TEXT_MUTED};
}}

QLabel[role="label"] {{
    font-size: {t.FONT_SIZE_META}px;
    font-weight: {t.WEIGHT_MEDIUM};
    color: {t.TEXT_MUTED};
}}

QLabel[role="data"] {{
    font-family: "{t.FONT_FAMILY_DATA}";
    font-size: {t.FONT_SIZE_DATA}px;
    color: {t.TEXT_MAIN};
}}

QLabel:disabled {{
    color: {t.TEXT_DISABLED};
}}

/* ---------- buttons ----------
   Every button in this system is outlined, never filled — emphasis comes
   from border/label brightness and hover-tint strength, not a solid fill
   (theme.md section 4). Label case/tracking is applied in code. */
QPushButton {{
    background-color: transparent;
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px;
    color: {t.TEXT_MAIN};
    font-weight: {t.WEIGHT_SEMIBOLD};
    font-size: {t.FONT_SIZE_BUTTON}px;
    padding: 8px 16px;
}}

QPushButton:hover {{
    background-color: {t.PRIMARY_BG};
    border-color: {t.PRIMARY};
    color: {t.PRIMARY};
}}

QPushButton:pressed {{
    background-color: {t.PRIMARY_BG_HOVER};
    border-color: {t.PRIMARY};
}}

QPushButton:focus {{
    border: 1px solid {t.PRIMARY};
    outline: none;
}}

QPushButton:disabled {{
    color: {t.TEXT_DISABLED};
    background-color: transparent;
    border-color: {t.BORDER};
}}

/* Primary — brighter outline + brighter label, still not filled. */
QPushButton[role="primary"] {{
    border: 1px solid {t.PRIMARY_BORDER};
    color: {t.TEXT_PRIMARY};
}}

QPushButton[role="primary"]:hover {{
    background-color: {t.PRIMARY_BG_HOVER};
    border-color: {t.PRIMARY};
}}

QPushButton[role="primary"]:pressed {{
    background-color: {t.PRIMARY_BG};
    border-color: {t.PRIMARY};
}}

QPushButton[role="primary"]:disabled {{
    border-color: {t.BORDER};
    color: {t.TEXT_DISABLED};
}}

/* Secondary — the plain default outline, explicit for symmetry with
   primary/danger when several buttons share a row. */
QPushButton[role="secondary"] {{
    border: 1px solid {t.BORDER};
    color: {t.TEXT_MAIN};
}}

/* Danger — stop recording / retry-after-failure. */
QPushButton[role="danger"] {{
    border: 1px solid {t.ALERT};
    color: {t.ALERT};
}}

QPushButton[role="danger"]:hover {{
    background-color: {t.ALERT_BG};
    border-color: {t.ALERT};
    color: {t.ALERT};
}}

/* Ghost — lowest emphasis, e.g. panel-header utility toggles. */
QPushButton[role="ghost"] {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {t.TEXT_MUTED};
    padding: 5px 10px;
}}

QPushButton[role="ghost"]:hover {{
    background-color: {t.BG_SURFACE_2};
    border-color: transparent;
    color: {t.TEXT_MAIN};
}}

/* ---------- status dot (common.py's StatusBadge) ---------- */
QLabel#StatusDot {{
    border-radius: 4px;
}}

/* ---------- sidebar ---------- */
QWidget#Sidebar {{
    background-color: {t.BG_SURFACE};
    border-right: 1px solid {t.BORDER};
}}

QPushButton#SidebarNavItem {{
    background-color: transparent;
    border: none;
    border-left: 2px solid transparent;
    border-radius: 0px;
    padding: 10px 0px;
}}

QPushButton#SidebarNavItem:hover {{
    background-color: {t.BG_SURFACE_2};
}}

QPushButton#SidebarNavItem[active="true"] {{
    background-color: {t.PRIMARY_BG};
    border-left: 2px solid {t.PRIMARY};
}}

QPushButton#SidebarNavItem QLabel#SidebarNavIcon {{
    font-size: {t.FONT_SIZE_ICON}px;
    color: {t.TEXT_MUTED};
    background: transparent;
}}

QPushButton#SidebarNavItem QLabel#SidebarNavLabel {{
    font-size: {t.FONT_SIZE_BODY}px;
    color: {t.TEXT_MUTED};
    background: transparent;
}}

QPushButton#SidebarNavItem:hover QLabel#SidebarNavIcon,
QPushButton#SidebarNavItem:hover QLabel#SidebarNavLabel {{
    color: {t.TEXT_MAIN};
}}

QPushButton#SidebarNavItem[active="true"] QLabel#SidebarNavIcon,
QPushButton#SidebarNavItem[active="true"] QLabel#SidebarNavLabel {{
    color: {t.TEXT_PRIMARY};
}}

/* ---------- detection-overlay toggle (Live Grid) ---------- */
QPushButton#OverlayToggleButton {{
    background-color: transparent;
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px;
    color: {t.TEXT_MUTED};
    font-weight: {t.WEIGHT_SEMIBOLD};
    font-size: {t.FONT_SIZE_BUTTON}px;
    padding: 6px 12px;
}}

QPushButton#OverlayToggleButton:hover {{
    background-color: {t.BG_SURFACE_2};
    border-color: {t.BORDER_STRONG};
    color: {t.TEXT_MAIN};
}}

QPushButton#OverlayToggleButton:checked {{
    background-color: {t.PRIMARY_BG};
    border-color: {t.PRIMARY};
    color: {t.TEXT_PRIMARY};
}}

/* ---------- grid-mode toggle (Live Grid) ---------- */
QPushButton#GridModeButton {{
    background-color: transparent;
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px;
    padding: 0px;
}}

QPushButton#GridModeButton:hover {{
    background-color: {t.BG_SURFACE_2};
    border-color: {t.BORDER_STRONG};
}}

QPushButton#GridModeButton:checked {{
    background-color: {t.PRIMARY_BG};
    border-color: {t.PRIMARY};
}}

/* ---------- camera dropdown (Live Grid, 1-camera mode) ---------- */
QComboBox#CameraDropdown {{
    background-color: {t.BG_SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px;
    color: {t.TEXT_MAIN};
    padding: 8px 12px;
    min-width: 100px;
}}

QComboBox#CameraDropdown:hover {{
    border-color: {t.PRIMARY};
}}

QComboBox#CameraDropdown::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox#CameraDropdown QAbstractItemView {{
    background-color: {t.BG_SURFACE_2};
    border: 1px solid {t.BORDER_STRONG};
    color: {t.TEXT_MAIN};
    selection-background-color: {t.PRIMARY_BG};
    selection-color: {t.TEXT_PRIMARY};
    outline: none;
}}

/* ---------- camera tile (Live Grid) ---------- */
QFrame#CameraTile {{
    background-color: {t.BG_SURFACE};
    border: none;
    border-radius: {t.RADIUS_PANEL}px;
}}

/* On-video overlay badge — the one place a translucent "glassy" surface is
   correct (theme.md §4); everywhere else surfaces are opaque. */
QWidget#OnVideoBadge {{
    background-color: rgba(26, 26, 26, 0.80);
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_BADGE}px;
}}

/* ---------- bottom transport bar & review control deck ---------- */
QFrame#ReviewControlDeck {{
    background-color: {t.BG_SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PANEL}px;
    padding: 2px 4px;
}}

QFrame#TransportBar {{
    background-color: {t.BG_BASE};
    border: 1px solid {t.BORDER};
    border-radius: 18px;
    padding: 2px 4px;
}}

QPushButton#TransportButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 14px;
    color: {t.TEXT_MAIN};
    font-weight: {t.WEIGHT_SEMIBOLD};
    font-size: 13px;
    min-width: 28px;
    min-height: 28px;
    padding: 3px 10px;
}}

QPushButton#TransportButton:hover {{
    background-color: {t.BG_SURFACE_2};
    border-color: {t.BORDER_STRONG};
    color: {t.TEXT_PRIMARY};
}}

QPushButton#TransportButton:pressed {{
    background-color: {t.PRIMARY_BG};
    border-color: {t.PRIMARY};
}}

/* BEST sits centered among the transport buttons:
   Modern glowing accent pill */
QPushButton#TransportButton[best="true"] {{
    border: 1.5px solid {t.PRIMARY};
    color: {t.TEXT_PRIMARY};
    background-color: {t.PRIMARY_BG};
    font-weight: {t.WEIGHT_SEMIBOLD};
    border-radius: 14px;
    padding: 3px 14px;
}}

QPushButton#TransportButton[best="true"]:hover {{
    background-color: {t.PRIMARY_BG_HOVER};
    border-color: {t.PRIMARY_BORDER};
    color: #ffffff;
}}

/* ---------- candidate rows (horizontal pills) ---------- */
QWidget#CandidateRow {{
    background-color: {t.BG_BASE};
    border: 1px solid {t.BORDER};
    border-radius: 12px;
    padding: 2px 8px;
}}

QWidget#CandidateRow:hover {{
    background-color: {t.BG_SURFACE_2};
    border-color: {t.PRIMARY};
}}

/* Selected candidate pill */
QWidget#CandidateRow[selected="true"] {{
    background-color: {t.PRIMARY_BG};
    border: 1.5px solid {t.PRIMARY};
}}

/* ---------- offside review (M2.7) ---------- */
QProgressBar#OffsideProgressBar {{
    background-color: {t.BG_BASE};
    border: none;
    border-radius: 1px;
}}

QProgressBar#OffsideProgressBar::chunk {{
    background-color: {t.PRIMARY};
    border-radius: 1px;
}}

QWidget#StageChecklistRow {{
    background-color: transparent;
    border-radius: {t.RADIUS_CONTROL}px;
}}

QWidget#StageChecklistRow[running="true"] {{
    background-color: {t.PRIMARY_BG};
    border-left: 2px solid {t.PRIMARY};
}}

QPushButton[selected="true"] {{
    background-color: {t.PRIMARY_BG_HOVER};
    border: 1px solid {t.PRIMARY};
    color: {t.TEXT_PRIMARY};
}}

QWidget#SignalRow {{
    background-color: transparent;
}}

QWidget#SignalRow:hover {{
    background-color: {t.BG_SURFACE_2};
}}

QWidget#RequirementRow {{
    background-color: transparent;
}}

QWidget#RequirementRow:hover {{
    background-color: {t.BG_SURFACE_2};
}}

/* The verdict word: the one thing this panel exists to say, sized to read
   from across the room. Neutral colour deliberately — see offside_review.py
   for why this is never coloured by verdict. */
QLabel#VerdictWord {{
    color: {t.TEXT_PRIMARY};
}}

/* Prominent Suggestions Box */
QWidget#SuggestionBox {{
    background-color: rgba(35, 55, 40, 0.55);
    border: 1px solid rgba(0, 255, 102, 0.35);
    border-left: 3px solid {t.SUCCESS};
    border-radius: {t.RADIUS_PANEL}px;
}}

QLabel#SuggestionItem {{
    color: {t.TEXT_PRIMARY};
    font-size: {t.FONT_SIZE_BODY}px;
}}

/* Prominent Why & Reasoning Card */
QFrame#ReasoningBox {{
    background-color: {t.PRIMARY_BG};
    border: 1px solid {t.PRIMARY_BORDER};
    border-left: 3px solid {t.PRIMARY};
    border-radius: {t.RADIUS_PANEL}px;
}}

QLabel#ReasonHeadline {{
    color: {t.TEXT_PRIMARY};
    font-weight: {t.WEIGHT_SEMIBOLD};
    font-size: {t.FONT_SIZE_BODY}px;
}}

QLabel#ReasonDetail {{
    color: {t.TEXT_MUTED};
    font-size: {t.FONT_SIZE_DATA}px;
}}

/* ---------- dividers ---------- */
QFrame[role="divider"] {{
    background-color: {t.BORDER};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

/* ---------- status bar ---------- */
QStatusBar {{
    background-color: {t.BG_SURFACE};
    border-top: 1px solid {t.BORDER};
    color: {t.TEXT_MUTED};
    font-size: {t.FONT_SIZE_DATA}px;
    padding: 2px 8px;
}}

QStatusBar::item {{
    border: none;
}}

/* Live event ticker — the operator's window into what the app is doing
   right now, pushed the instant each backend event fires. */
QLabel#EventTicker {{
    font-family: "{t.FONT_FAMILY_DATA}";
    font-size: {t.FONT_SIZE_DATA}px;
    color: {t.TEXT_MUTED};
    background: transparent;
    padding-right: 12px;
}}

/* ---------- Info screen ---------- */
QLabel#HelpStepBadge {{
    background-color: {t.PRIMARY_BG};
    border: 1px solid {t.PRIMARY_BORDER};
    border-radius: 14px;
    color: {t.TEXT_PRIMARY};
    font-family: "{t.FONT_FAMILY_DATA}";
    font-weight: {t.WEIGHT_SEMIBOLD};
    font-size: {t.FONT_SIZE_DATA}px;
}}

QFrame#StorySpine {{
    background-color: {t.BORDER};
    border: none;
    max-width: 1px;
    min-width: 1px;
}}

QFrame#DisclaimerBanner {{
    background-color: {t.ALERT_BG};
    border: 1px solid {t.ALERT};
    border-radius: {t.RADIUS_PANEL}px;
}}

QLabel#DisclaimerHeading {{
    color: {t.ALERT};
    font-weight: {t.WEIGHT_SEMIBOLD};
    font-size: {t.FONT_SIZE_BODY}px;
}}

/* ---------- scroll areas ---------- */
/* QScrollArea paints an opaque (usually white) background behind its
   viewport by default; without this rule a scrollable panel on a dark
   screen shows a bright rectangle the instant it needs to scroll. */
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {t.BORDER_STRONG};
    border-radius: 2px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {t.TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""
