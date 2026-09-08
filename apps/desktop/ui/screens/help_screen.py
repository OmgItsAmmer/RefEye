"""Info screen — a client-presentable walkthrough of what the app does, plus
a live readout of which models are actually loaded in this build.

Leads with a disclaimer: this is an unpaid trial/demo build, not a shipped
product — it must not be mistaken for something validated and supported for
production use.

The pipeline story (`PipelineStoryPanel`, in widgets/pipeline_story.py) is
the one place in this screen that goes beyond plain Panel/label vocabulary —
a staggered reveal animation, built specifically because this screen is meant
to be shown to a client, not just read once by an operator. Everything else
here stays inside the same vocabulary the rest of the app uses (theme.md).
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ai.model_registry.registry import ModelRegistry
from apps.desktop.ui.widgets.common import Panel, data_value, divider, label, meta, title
from apps.desktop.ui.widgets.pipeline_story import PipelineStoryPanel
from core.config.schema import AppSettings


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
        inner_layout.addWidget(PipelineStoryPanel())
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
