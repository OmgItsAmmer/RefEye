"""Small shared widgets: panels, status dots, headings, dividers.

Every visual value comes from the stylesheet via objectName/dynamic property.
No widget here calls setStyleSheet — that is the rule theme.md sets out and
the reason all styling stays in one readable place.

Qt style sheets support neither `text-transform` nor `letter-spacing`
(theme.md section 3), so every uppercase/tracked label in this system is
produced here in code: `.upper()` on the string plus `QFont.setLetterSpacing`
on the widget's font, not a stylesheet trick.
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.ui.theme import tokens as t


def _tracked(widget: QWidget, px: float) -> None:
    """Apply absolute letter-spacing to `widget`'s current font, in place."""
    font = widget.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, px)
    widget.setFont(font)


class AnimatedButton(QPushButton):
    """The system's one button widget: UPPERCASE tracked label (theme.md §3),
    pointing-hand cursor.

    theme.md section 5 is explicit that pressed/hover feedback is a color
    step, never a scale transform — so this intentionally does *not*
    animate geometry on press. The name is kept (rather than a plain
    QPushButton) so a future real per-state color animation has one place
    to live without another pass over every call site.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text.upper(), parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        _tracked(self, t.TRACKING_BUTTON)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt override
        super().setText(text.upper())


class BadgeVariant(str, Enum):
    ACCENT = "accent"  # success/active (green dot)
    INFO = "info"  # in-progress (pulsing primary dot)
    WARNING = "warning"  # non-fatal, no color — text alone carries it
    DANGER = "danger"  # error/failed (red dot)
    MUTED = "muted"  # idle/neutral


_DOT_COLOR = {
    BadgeVariant.ACCENT: t.SUCCESS,
    BadgeVariant.INFO: t.PRIMARY,
    BadgeVariant.WARNING: t.TEXT_MUTED,
    BadgeVariant.DANGER: t.ALERT,
    BadgeVariant.MUTED: t.TEXT_MUTED,
}


class _PulsingDot(QLabel):
    """An 8px circular dot; pulses opacity when `pulsing=True` (theme.md §5:
    `bg-primary animate-pulse` — a plain opacity loop, distinct from the
    ring-expand pulse reserved for an active-recording indicator, which this
    app has no equivalent state for).
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StatusDot")
        self.setFixedSize(8, 8)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(1000)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.35)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)

    def set_color(self, hex_color: str) -> None:
        self.setStyleSheet(
            f"background-color: {hex_color}; border-radius: 4px;"
        )

    def set_pulsing(self, pulsing: bool) -> None:
        if pulsing:
            self._anim.setDirection(QPropertyAnimation.Direction.Forward)
            self._anim.setLoopCount(-1)
            if self._anim.state() != QPropertyAnimation.State.Running:
                self._anim.start()
        else:
            self._anim.stop()
            self._effect.setOpacity(1.0)


class StatusBadge(QWidget):
    """A colored status dot + an uppercase tracked label (theme.md §4).

    Replaces the previous tinted-pill badge: this system reserves filled
    pills for count badges only (unread-alert style numerals), and uses a
    dot + label for every state indicator (stream/model/analysis status).
    """

    def __init__(self, text: str = "", variant: BadgeVariant = BadgeVariant.MUTED, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusBadge")
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._dot = _PulsingDot(self)
        layout.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._label = QLabel(self)
        self._label.setProperty("role", "label")
        _tracked(self._label, t.TRACKING_META)
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.set_status(text, variant)

    def set_status(self, text: str, variant: BadgeVariant) -> None:
        self._label.setText(text.upper())
        self._dot.set_color(_DOT_COLOR[variant])
        self._dot.set_pulsing(variant == BadgeVariant.INFO)
        self.updateGeometry()
        self.adjustSize()

    # Kept for callers that only ever set a tooltip/etc on the whole badge.
    def setToolTip(self, tip: str) -> None:  # noqa: N802 — Qt override
        super().setToolTip(tip)
        self._dot.setToolTip(tip)
        self._label.setToolTip(tip)


class _TrackedLabel(QLabel):
    """A QLabel that stays UPPERCASE + tracked across later `setText()` calls.

    Several chrome captions (loader status text, diagnostics values) get
    updated after construction — a plain `.upper()` applied once at creation
    would silently stop applying once the caller sets new text at runtime.
    """

    def __init__(self, text: str, tracking_px: float, parent: QWidget | None = None):
        super().__init__(text.upper(), parent)
        self._tracking_px = tracking_px
        _tracked(self, tracking_px)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt override
        super().setText(text.upper())
        _tracked(self, self._tracking_px)


def simple_dot(hex_color: str) -> QLabel:
    """A plain 8px circular dot, no pulse — for a static status indicator
    (e.g. a camera tile's live/idle marker) that doesn't warrant the full
    StatusBadge (dot + label) treatment."""
    dot = QLabel()
    dot.setObjectName("StatusDot")
    dot.setFixedSize(8, 8)
    dot.setStyleSheet(f"background-color: {hex_color}; border-radius: 4px;")
    return dot


def title(text: str) -> QLabel:
    """Screen/section title: 15px, semibold, UPPERCASE, wide tracking."""
    lbl = _TrackedLabel(text, t.TRACKING_TITLE)
    lbl.setProperty("role", "title")
    return lbl


# Panel section headers use the title style, not a separate "heading" — this
# codebase has no large sentence-case content heading anywhere yet (theme.md
# reserves that style for content, e.g. a candidate's action name, which
# currently renders via StatusBadge instead). Kept as an alias so existing
# call sites (`Panel(title=...)`) read naturally.
heading = title


def meta(text: str = "") -> QLabel:
    """Sentence-case secondary text — descriptions, dynamic values, captions
    that are *content*, not chrome (theme.md's case rule, section 3)."""
    label = QLabel(text)
    label.setProperty("role", "meta")
    return label


def label(text: str = "") -> QLabel:
    """Uppercase, tracked, muted micro-caption — chrome naming a value
    (e.g. a diagnostics row's caption), not the value itself."""
    lbl = _TrackedLabel(text, t.TRACKING_META)
    lbl.setProperty("role", "label")
    return lbl


def data_value(text: str = "") -> QLabel:
    """Space Mono value text — timestamps, ids, codec/source strings,
    diagnostic numbers (theme.md's "data face" rule)."""
    lbl = QLabel(text)
    lbl.setProperty("role", "data")
    return lbl


def divider() -> QFrame:
    line = QFrame()
    line.setProperty("role", "divider")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


class Panel(QFrame):
    """Standard card surface: surface background, hairline border, 8px
    sharp radius (theme.md §4 — flat, no drop shadow).

    `recessed=True` selects the de-emphasized variant used for diagnostics
    and other secondary content.
    """

    def __init__(self, title: str | None = None, recessed: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("PanelRecessed" if recessed else "Panel")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            t.PANEL_PADDING, t.PANEL_PADDING, t.PANEL_PADDING, t.PANEL_PADDING
        )
        self._layout.setSpacing(t.SPACING_UNIT)

        self._header: QHBoxLayout | None = None
        if title is not None:
            self._header = QHBoxLayout()
            self._header.setContentsMargins(0, 0, 0, 0)
            self._header.setSpacing(t.SPACING_UNIT)
            self._header.addWidget(heading(title))
            self._header.addStretch(1)
            self._layout.addLayout(self._header)

    def add_header_widget(self, widget: QWidget) -> None:
        """Place a widget (usually a status badge) at the right of the title row."""
        if self._header is None:
            raise ValueError("Panel was created without a title, so it has no header row")
        self._header.addWidget(widget)

    def body(self) -> QVBoxLayout:
        return self._layout
