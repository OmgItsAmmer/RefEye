"""Startup splash screen.

Shown as the very first thing `main()` does, before importing the rest of
the application (the full widget tree, and everything it pulls in
transitively). A freshly installed build's first launch can take a while —
antivirus/SmartScreen scanning a new multi-gigabyte executable and its DLLs,
or a cold OS file cache — and a blank screen during that gap reads as "did
this even open?" rather than "it's loading." This fills that gap with
visible feedback for whatever part of the delay is inside Python's control.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from apps.desktop.ui.theme import tokens as t
from apps.desktop.ui.theme.app_icon import build_app_icon

_WIDTH = 420
_HEIGHT = 260


def build_splash() -> QSplashScreen:
    pixmap = QPixmap(_WIDTH, _HEIGHT)
    pixmap.fill(QColor(t.BG_BASE))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    icon = build_app_icon().pixmap(72, 72)
    painter.drawPixmap((_WIDTH - 72) // 2, 54, icon)

    painter.setPen(QColor(t.TEXT_MAIN))
    title_font = QFont(t.FONT_FAMILY, 20)
    title_font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(title_font)
    painter.drawText(QRect(0, 138, _WIDTH, 32), Qt.AlignmentFlag.AlignCenter, "RefEye")

    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    set_splash_message(splash, "Starting…")
    return splash


def set_splash_message(splash: QSplashScreen, text: str) -> None:
    splash.showMessage(
        text,
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        QColor(t.TEXT_MUTED),
    )
