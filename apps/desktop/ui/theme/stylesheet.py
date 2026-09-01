"""Application-wide QSS, generated from design tokens.

All styling lives here. Widgets opt into styling via `objectName` or a Qt
dynamic property (`role`), never via inline setStyleSheet calls.

Dynamic properties used:
    role="heading"      panel title text
    role="body"         default body text (implicit)
    role="meta"         muted caption/metadata text
    role="primary"      solid accent button (max one per view)
    role="secondary"    outlined button
    variant="accent"    status badge tinted with the accent hue
    variant="info"      status badge tinted with the info hue
    variant="warning"   status badge tinted with the warning hue
    variant="danger"    status badge tinted with the danger hue
"""

from __future__ import annotations

from apps.desktop.ui.theme import tokens as t


def build_stylesheet() -> str:
    return f"""
/* ---------- base ---------- */
QWidget {{
    background-color: {t.BG_BASE};
    color: {t.TEXT_PRIMARY};
    font-family: "{t.FONT_FAMILY}";
    font-size: {t.FONT_SIZE_BODY}px;
    font-weight: {t.WEIGHT_REGULAR};
}}

QMainWindow, QDialog {{
    background-color: {t.BG_BASE};
}}

QToolTip {{
    background-color: {t.BG_PANEL_RAISED};
    color: {t.TEXT_PRIMARY};
    border: 1px solid {t.BORDER};
    padding: 4px 8px;
}}

/* ---------- panels ---------- */
QFrame#Panel {{
    background-color: {t.BG_PANEL};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PANEL}px;
}}

/* Recessed secondary surface (diagnostics) — de-emphasized vs main panels. */
QFrame#PanelRecessed {{
    background-color: {t.BG_BASE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PANEL}px;
}}

QFrame#Panel > QWidget, QFrame#PanelRecessed > QWidget {{
    background-color: transparent;
}}

/* ---------- typography ---------- */
QLabel {{
    background-color: transparent;
}}

QLabel[role="heading"] {{
    font-size: {t.FONT_SIZE_HEADING}px;
    font-weight: {t.WEIGHT_MEDIUM};
    color: {t.TEXT_PRIMARY};
}}

QLabel[role="meta"] {{
    font-size: {t.FONT_SIZE_META}px;
    color: {t.TEXT_MUTED};
}}

QLabel[role="secondary"] {{
    color: {t.TEXT_SECONDARY};
}}

QLabel:disabled {{
    color: {t.TEXT_DISABLED};
}}

/* ---------- buttons ---------- */
QPushButton {{
    background-color: transparent;
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px;
    color: {t.TEXT_PRIMARY};
    font-weight: {t.WEIGHT_MEDIUM};
    padding: 7px 14px;
}}

QPushButton:hover {{
    background-color: {t.BG_PANEL_RAISED};
}}

QPushButton:pressed {{
    background-color: {t.BORDER};
}}

QPushButton:focus {{
    border: 1px solid {t.ACCENT};
    outline: none;
}}

/* Disabled: drop the label, keep the control surface (theme.md section 4). */
QPushButton:disabled {{
    color: {t.TEXT_DISABLED};
    background-color: transparent;
}}

QPushButton[role="primary"] {{
    background-color: {t.ACCENT};
    border: 1px solid {t.ACCENT};
    color: {t.ACCENT_TEXT_ON_FILL};
}}

QPushButton[role="primary"]:hover {{
    background-color: {t.ACCENT_HOVER};
    border-color: {t.ACCENT_HOVER};
}}

QPushButton[role="primary"]:disabled {{
    color: {t.TEXT_DISABLED};
}}

/* ---------- status badges ---------- */
QLabel#StatusBadge {{
    border-radius: {t.RADIUS_BADGE}px;
    font-size: {t.FONT_SIZE_META}px;
    font-weight: {t.WEIGHT_MEDIUM};
    padding: 3px 8px;
}}

QLabel#StatusBadge[variant="accent"] {{
    background-color: {t.ACCENT_BG};
    color: {t.ACCENT_TEXT};
}}

QLabel#StatusBadge[variant="info"] {{
    background-color: rgba(91, 127, 166, 0.15);
    color: {t.INFO};
}}

QLabel#StatusBadge[variant="warning"] {{
    background-color: rgba(201, 154, 74, 0.15);
    color: {t.WARNING};
}}

QLabel#StatusBadge[variant="danger"] {{
    background-color: {t.DANGER_BG};
    color: {t.DANGER};
}}

QLabel#StatusBadge[variant="muted"] {{
    background-color: rgba(139, 146, 160, 0.12);
    color: {t.TEXT_MUTED};
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
    background-color: {t.BG_PANEL};
    border-top: 1px solid {t.BORDER};
    color: {t.TEXT_MUTED};
    font-size: {t.FONT_SIZE_META}px;
}}

QStatusBar::item {{
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
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""
