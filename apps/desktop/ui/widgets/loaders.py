"""The two loaders defined in theme.md section 6.

Which loader appears is deterministic, never a per-screen judgement call:

    CardSwapLoader      — an AnalysisRequest is QUEUED/PREPARING/.../RANKING.
                          The AI is deciding. Referee-card metaphor.
    FootballSpinLoader  — any other blocking wait: startup, model warm-up,
                          opening or buffering video, reconnect attempts.

Never show both at once — they represent mutually exclusive app states.

Both are QPainter-driven because QSS has no animation support; the CSS in
theme.md is the visual specification, reproduced here in Qt terms.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QConicalGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from apps.desktop.ui.theme import tokens as t

_FRAME_INTERVAL_MS = 16  # ~60fps


class _AnimatedWidget(QWidget):
    """Base for the loaders: runs a timer only while visible.

    Stopping the timer when hidden matters — a loader left ticking behind a
    hidden panel burns UI-thread time for nothing.
    """

    def __init__(self, duration_ms: int, size: int, parent: QWidget | None = None):
        super().__init__(parent)
        self._duration_ms = duration_ms
        self._elapsed_ms = 0
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

    @property
    def progress(self) -> float:
        """Loop position in [0, 1)."""
        return (self._elapsed_ms % self._duration_ms) / self._duration_ms

    def _tick(self) -> None:
        self._elapsed_ms += _FRAME_INTERVAL_MS
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self._timer.stop()
        super().hideEvent(event)


class CardSwapLoader(_AnimatedWidget):
    """Yellow and red cards continuously swapping position with a 3D tilt.

    Reserved for the single highest-stakes wait: the AI is deciding.

    The cards keep their true referee colors — theme.md's one deliberate
    exception to the desaturated palette, because the metaphor only reads if
    the cards are recognizable.
    """

    CARD_W = 34
    CARD_H = 48
    TRAVEL = 26
    MAX_TILT_DEG = 25

    def __init__(self, parent: QWidget | None = None):
        super().__init__(duration_ms=1600, size=90, parent=parent)

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(t.BG_SURFACE))

        # Ease the raw linear progress through a sine-based curve so the
        # swing lingers slightly at the extremes instead of moving at
        # constant velocity — reads as weightier, less mechanical.
        eased = math.sin(2 * math.pi * self.progress)
        phase = 2 * math.pi * self.progress
        swing = math.cos(phase)

        center = QPointF(self.width() / 2, self.height() / 2)

        # Soft ambient glow beneath the cards, brightest when they cross.
        # Monochrome (theme.md §1: color only for alert/success), not the
        # previous green pitch-accent glow.
        glow_strength = 0.35 + 0.25 * (1 - abs(eased))
        glow = QRadialGradient(center, self.width() / 2)
        glow_color = QColor(t.PRIMARY)
        glow_color.setAlphaF(glow_strength * 0.12)
        glow.setColorAt(0.0, glow_color)
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, self.width() / 2, self.height() / 2)

        # Both cards run the same loop exactly out of phase, so they cross at
        # the centre twice per cycle.
        cards = [
            (QColor(t.CARD_YELLOW), -self.TRAVEL * swing, self.MAX_TILT_DEG * swing),
            (QColor(t.CARD_RED), self.TRAVEL * swing, -self.MAX_TILT_DEG * swing),
        ]
        # Draw the receding card first so the advancing one overlaps it.
        cards.sort(key=lambda c: c[2])

        for color, offset_x, tilt_deg in cards:
            # rotateY foreshortening: width collapses by cos(angle).
            squeeze = abs(math.cos(math.radians(tilt_deg)))
            # Cheap perspective: the card tilting toward the viewer reads
            # slightly larger.
            depth = 1.0 + 0.07 * math.sin(math.radians(tilt_deg))

            width = self.CARD_W * squeeze * depth
            height = self.CARD_H * depth

            rect = QRectF(
                center.x() + offset_x - width / 2,
                center.y() - height / 2,
                width,
                height,
            )

            # Subtle drop shadow under each card for lift, drawn before the
            # card face so it only peeks out at the edges.
            shadow_rect = rect.translated(0, 3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 70))
            painter.drawRoundedRect(shadow_rect, 5, 5)

            # Slight top-to-bottom sheen on the card face for a glossier,
            # more premium material than a flat fill.
            sheen = QRadialGradient(rect.center() - QPointF(0, height * 0.3), width * 1.1)
            sheen.setColorAt(0.0, color.lighter(122))
            sheen.setColorAt(1.0, color)
            painter.setBrush(sheen)
            painter.drawRoundedRect(rect, 5, 5)


class FootballSpinLoader(_AnimatedWidget):
    """A ball in constant rotation — neutral 'something is loading'.

    Used for infrastructure/IO waits only. Linear timing, constant speed, no
    easing: the emphasis of the card swap is reserved for AI decisions.
    """

    DIAMETER = 44

    def __init__(self, parent: QWidget | None = None):
        super().__init__(duration_ms=1200, size=self.DIAMETER + 8, parent=parent)

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(t.BG_SURFACE))

        center = QPointF(self.width() / 2, self.height() / 2)
        radius = self.DIAMETER / 2

        # A faint rotating conical sweep behind the ball — a lightweight
        # "spinner" cue that reads as motion even in a still frame, on top
        # of the ball's own rotation. Monochrome border-tone ring, matching
        # the reference app's plain border-t-primary spinner rather than the
        # previous green glow (theme.md §7).
        sweep = QConicalGradient(center, self.progress * -360.0)
        ring_edge = QColor(t.BORDER_STRONG)
        ring_edge.setAlpha(0)
        ring_bright = QColor(t.PRIMARY)
        ring_bright.setAlpha(110)
        sweep.setColorAt(0.0, ring_bright)
        sweep.setColorAt(0.28, ring_edge)
        sweep.setColorAt(1.0, ring_edge)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(sweep)
        painter.drawEllipse(center, radius + 5, radius + 5)

        painter.translate(center)
        painter.rotate(self.progress * 360.0)

        # Soft contact shadow beneath the ball for lift off the panel.
        painter.setBrush(QColor(0, 0, 0, 60))
        painter.drawEllipse(QPointF(0, radius * 0.75), radius * 0.85, radius * 0.28)

        ball_gradient = QRadialGradient(QPointF(-radius * 0.3, -radius * 0.3), radius * 1.6)
        ball_gradient.setColorAt(0.0, QColor(t.BALL_WHITE).lighter(108))
        ball_gradient.setColorAt(1.0, QColor(t.BALL_WHITE).darker(112))
        painter.setBrush(ball_gradient)
        painter.drawEllipse(QPointF(0, 0), radius, radius)

        # Panel pattern, drawn in the panel color so it rotates as one unit
        # with the ball (matching the inline-SVG approach in theme.md).
        scale = self.DIAMETER / 100.0
        painter.scale(scale, scale)
        painter.translate(-50, -50)

        pentagon = [(50, 20), (62, 32), (58, 48), (42, 48), (38, 32)]
        path = QPainterPath(QPointF(*pentagon[0]))
        for point in pentagon[1:]:
            path.lineTo(QPointF(*point))
        path.closeSubpath()

        painter.setBrush(QColor(t.BALL_PANEL))
        painter.drawPath(path)

        seams = [
            ((50, 20), (50, 5)),
            ((62, 32), (80, 25)),
            ((58, 48), (70, 65)),
            ((42, 48), (30, 65)),
            ((38, 32), (20, 25)),
        ]
        pen = QPen(QColor(t.BALL_PANEL))
        pen.setWidthF(3.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        for (x1, y1), (x2, y2) in seams:
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
