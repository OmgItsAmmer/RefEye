"""Application icon, drawn at runtime.

Generated with QPainter rather than shipped as a binary asset so it stays in
version control as readable source, scales to any DPI, and uses the same
palette tokens as the rest of the UI.

The mark is a camera — this is a camera/video analysis tool, not a referee
mascot; the glyph should say so at a glance.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from apps.desktop.ui.theme import tokens as t

_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _draw(size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    scale = size / 64.0
    painter.scale(scale, scale)

    # Rounded dark tile so the mark reads on any taskbar background.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(t.BG_SURFACE))
    painter.drawRoundedRect(QRectF(2, 2, 60, 60), 14, 14)

    pen = QPen(QColor(t.PRIMARY))
    pen.setWidthF(3.0)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    # Viewfinder bump, top-left of the body (a camera's shutter housing).
    painter.drawRoundedRect(QRectF(22, 14, 15, 9), 3, 3)

    # Camera body.
    painter.drawRoundedRect(QRectF(9, 21, 46, 30), 6, 6)

    # Lens.
    painter.drawEllipse(QPointF(32, 36), 10.5, 10.5)

    # Aperture dot at the lens center — same "ball" accent as the rest of
    # the UI's monochrome palette (t.BALL_WHITE), not a new color.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(t.BALL_WHITE))
    painter.drawEllipse(QPointF(32, 36), 3.5, 3.5)

    # Shutter-release button, only where it stays legible.
    if size >= 32:
        painter.setBrush(QColor(t.PRIMARY))
        painter.drawRoundedRect(QRectF(41, 25, 8, 4), 1.5, 1.5)

    painter.end()
    return pixmap


def build_app_icon() -> QIcon:
    icon = QIcon()
    for size in _SIZES:
        icon.addPixmap(_draw(size))
    return icon


def save_icon(path: str) -> None:
    """Write a .ico for PyInstaller, which needs a real file on disk."""
    pixmaps = [_draw(size) for size in _SIZES]
    # QPixmap cannot write multi-resolution .ico, so emit the largest and let
    # Windows downscale; the in-app QIcon still carries every size.
    pixmaps[-1].save(path, "ICO")
