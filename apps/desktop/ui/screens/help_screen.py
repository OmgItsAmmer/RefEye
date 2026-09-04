"""Info screen — a plain-language walkthrough of what the app does, plus a
live readout of which models are actually loaded in this build.

Leads with a disclaimer: this is an unpaid trial/demo build, not a shipped
product — it must not be mistaken for something validated and supported for
production use.

No new visual language otherwise: every container here is the same
Panel/label/badge vocabulary the rest of the app uses (theme.md), just
arranged as explanatory copy instead of live controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ai.model_registry.registry import ModelRegistry
from apps.desktop.ui.widgets.common import Panel, data_value, divider, heading, label, meta, title
from core.config.schema import AppSettings

_STEPS = [
    (
        "1",
        "Live capture",
        (
            "The configured video source streams continuously into a rolling frame "
            "buffer. Nothing is sent to the AI yet — this is just playback."
        ),
    ),
    (
        "2",
        "Detect & track",
        (
            "In the background, a detector locates players and the ball on a "
            "sampled subset of frames; a tracker keeps consistent identities "
            "across frames so movement can be read over time."
        ),
    ),
    (
        "3",
        "Spot & rank candidates",
        (
            "Pressing Analyse (or picking a camera/Best on Live Grid) hands the "
            "action spotter the recent buffered window. It finds ball-contact "
            "moments, refines the exact contact frame, and ranks the strongest "
            "candidates."
        ),
    ),
    (
        "4",
        "Operator review",
        (
            "You step through candidates and individual frames, compare "
            "alternatives, and confirm the exact frame yourself — the AI "
            "proposes, you decide."
        ),
    ),
]


class HelpScreen(QWidget):
    def __init__(
        self,
        settings: AppSettings,
        registry: ModelRegistry,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._settings = settings
        self._registry = registry

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16)
        root.setSpacing(16)
        root.addWidget(title("Info"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 4, 0)
        inner_layout.setSpacing(16)
        inner_layout.addWidget(self._build_disclaimer_banner())
        inner_layout.addWidget(self._build_flow_panel())
        inner_layout.addWidget(self._build_models_panel())
        inner_layout.addStretch(1)

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

    # -- disclaimer -----------------------------------------------------------

    @staticmethod
    def _build_disclaimer_banner() -> QFrame:
        banner = QFrame()
        banner.setObjectName("DisclaimerBanner")
        layout = QVBoxLayout(banner)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        heading_label = QLabel("Not a production product")
        heading_label.setObjectName("DisclaimerHeading")
        layout.addWidget(heading_label)

        body = QLabel(
            "This build is an unpaid trial made for demonstration purposes only. "
            "It has not been validated on real client footage, is not licensed or "
            "supported for production use, and its analysis must not be relied on "
            "for any real match or officiating decision."
        )
        body.setWordWrap(True)
        body.setProperty("role", "meta")
        layout.addWidget(body)
        return banner

    # -- how it works -------------------------------------------------------

    def _build_flow_panel(self) -> Panel:
        panel = Panel("How this app works")
        for index, (number, heading_text, body_text) in enumerate(_STEPS):
            panel.body().addWidget(self._build_step_row(number, heading_text, body_text))
            if index != len(_STEPS) - 1:
                panel.body().addWidget(divider())
        return panel

    @staticmethod
    def _build_step_row(number: str, heading_text: str, body_text: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(14)

        badge = QLabel(number)
        badge.setObjectName("HelpStepBadge")
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(heading(heading_text))
        body_label = meta(body_text)
        body_label.setWordWrap(True)
        text_col.addWidget(body_label)
        layout.addLayout(text_col, 1)
        return row

    # -- models in this build ------------------------------------------------

    def _build_models_panel(self) -> Panel:
        panel = Panel("Models in this build")
        self._models_layout = QVBoxLayout()
        self._models_layout.setSpacing(6)
        panel.body().addLayout(self._models_layout)
        self.refresh()
        return panel

    def refresh(self) -> None:
        """Re-read the registry. Called on startup and again every time
        model_state_changed fires (main_window.py), so this panel reflects
        the real loaded weights, not just the config file's intent."""
        while self._models_layout.count():
            item = self._models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        info = self._registry.model_info()
        if not info:
            spotter = self._settings.ai.action_spotter.provider
            detector = self._settings.ai.detector.provider
            self._models_layout.addWidget(
                meta(f"Loading — configured for '{detector}' detection and '{spotter}' action spotting…")
            )
            return

        for role, entry in info.items():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            role_label = label(role.replace("_", " "))
            role_label.setFixedWidth(110)
            row_layout.addWidget(role_label)

            value = f"{entry['name']}  ·  v{entry['version']}  ·  {entry['provider']}"
            row_layout.addWidget(data_value(value))
            row_layout.addStretch(1)
            self._models_layout.addWidget(row)

        device = self._settings.runtime.device
        self._models_layout.addWidget(divider())
        self._models_layout.addWidget(meta(f"Running on: {device}"))
