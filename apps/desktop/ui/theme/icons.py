"""Small vector icons for transport controls, drawn at runtime with QPainter.

Same rationale as app_icon.py: no icon font or binary asset is bundled
(theme.md §8), so a glyph that needs real vector shape — not a paintable
character like '‹'/'›' — is built as a QPainterPath and rendered into a
QIcon, in the same monochrome palette as the rest of the UI, instead of
reaching for a platform emoji glyph.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from apps.desktop.ui.theme import tokens as t

_SIZES = (16, 24, 32)
_VIEWBOX = 24.0


def _skip_path(forward: bool) -> QPainterPath:
    """Media-transport 'skip to next/previous' glyph: one triangle against an
    end bar (the same shape Material's skip_next/skip_previous use), on a
    24x24 canvas."""
    path = QPainterPath()

    if forward:
        bar = QRectF(16.0, 6.0, 2.0, 12.0)
        tip = [(6.0, 18.0), (6.0, 6.0), (14.5, 12.0)]
    else:
        bar = QRectF(6.0, 6.0, 2.0, 12.0)
        tip = [(18.0, 18.0), (18.0, 6.0), (9.5, 12.0)]

    path.addRect(bar)
    path.moveTo(QPointF(*tip[0]))
    path.lineTo(QPointF(*tip[1]))
    path.lineTo(QPointF(*tip[2]))
    path.closeSubpath()
    return path


def _render(path: QPainterPath, size: int, color: str) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / _VIEWBOX, size / _VIEWBOX)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawPath(path)
    painter.end()
    return pixmap


def skip_icon(forward: bool) -> QIcon:
    """Previous/next-candidate icon, with a dimmed variant for the disabled
    state (theme.md's TEXT_DISABLED token)."""
    path = _skip_path(forward)
    icon = QIcon()
    for size in _SIZES:
        icon.addPixmap(_render(path, size, t.TEXT_MAIN), QIcon.Mode.Normal)
        icon.addPixmap(_render(path, size, t.TEXT_DISABLED), QIcon.Mode.Disabled)
    return icon


def _grid_cells(cells: int) -> list[QRectF]:
    """Cell rectangles for a 1/2/4-way camera-grid glyph, on a 24x24 canvas."""
    if cells == 1:
        return [QRectF(3.0, 3.0, 18.0, 18.0)]
    if cells == 2:
        return [QRectF(3.0, 3.0, 8.0, 18.0), QRectF(13.0, 3.0, 8.0, 18.0)]
    return [
        QRectF(3.0, 3.0, 8.0, 8.0),
        QRectF(13.0, 3.0, 8.0, 8.0),
        QRectF(3.0, 13.0, 8.0, 8.0),
        QRectF(13.0, 13.0, 8.0, 8.0),
    ]


def _render_stroke(rects: list[QRectF], size: int, color: str) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / _VIEWBOX, size / _VIEWBOX)
    pen = QPen(QColor(color))
    pen.setWidthF(2.2)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for rect in rects:
        painter.drawRoundedRect(rect, 2.0, 2.0)
    painter.end()
    return pixmap


def grid_mode_icon(cells: int) -> QIcon:
    """1/2/4-camera grid-layout toggle glyph (outlined cells, matching the
    camera tiles' own outline style rather than a filled block)."""
    rects = _grid_cells(cells)
    icon = QIcon()
    for size in _SIZES:
        icon.addPixmap(_render_stroke(rects, size, t.TEXT_MAIN), QIcon.Mode.Normal)
        icon.addPixmap(_render_stroke(rects, size, t.TEXT_DISABLED), QIcon.Mode.Disabled)
    return icon
