"""Live event ticker for the status bar.

The app has no HTTP layer, so there is no literal SSE stream — but the
viewmodel already pushes every backend event to the UI thread as a Qt
signal the instant it happens (main_viewmodel.py's docstring), which is the
same push model SSE gives a web client. This widget is where that stream
surfaces: main_window.py calls `push()` from every signal handler, so the
operator always sees the latest thing the app did, timestamped, sitting
just left of the AI-ready badge.
"""

from __future__ import annotations

import time

from PySide6.QtWidgets import QLabel, QWidget


class EventTicker(QLabel):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("EventTicker")
        self.setMinimumWidth(260)
        self.push("Waiting for the first event…")

    def push(self, message: str) -> None:
        text = f"{time.strftime('%H:%M:%S')}  ·  {message}"
        self.setText(text)
        self.setToolTip(text)
