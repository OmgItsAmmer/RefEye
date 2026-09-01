"""Main application window shell.

The UI thread must never decode video or run inference (architecture.md
section 4.1) — this window only renders state pushed to it via Qt signals
from background workers/viewmodels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soccer Analysis MVP")
        self.resize(1440, 900)

        self._build_layout()
        self._build_status_bar()

    def _build_layout(self) -> None:
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        live_panel = self._make_panel("Live Preview")
        review_panel = self._make_panel("Candidate Review")

        root_layout.addWidget(live_panel, stretch=3)
        root_layout.addWidget(review_panel, stretch=2)

        self.setCentralWidget(central)

    def _make_panel(self, title: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        heading = QLabel(title)
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        placeholder = QLabel("Not yet wired to video/analysis pipeline (Phase M1.1 scaffold).")
        placeholder.setProperty("role", "secondary")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder, stretch=1)

        return panel

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Stream: not connected  |  Model: not loaded")
