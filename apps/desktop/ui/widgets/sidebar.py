"""Collapsible left sidebar — cloned from sample_ui's Sidebar.tsx.

Collapsed to an icon rail at rest, expands to show labels on hover, exactly
like the reference app (`isHovered` driving `w-16` <-> `w-64` with a 300ms
transition). Three destinations exist in this app (no People/Vehicles/
Alerts/Recordings — this isn't a VMS): Live Grid, Analyzer, and Help.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from apps.desktop.ui.theme.app_icon import build_app_icon
from apps.desktop.ui.widgets.common import title

_COLLAPSED_WIDTH = 56
_EXPANDED_WIDTH = 200
_ANIM_MS = 220

# (screen name, glyph, label) — no icon font is bundled (theme.md §8), so
# nav items use a plain text glyph, rendered in its own larger-font label
# (FONT_SIZE_ICON) the same way frame/candidate nav already does ('‹', '›').
_ITEMS = [
    ("live_grid", "▦", "Live Grid"),
    ("analyzer", "▶", "Analyzer"),
    ("help", "?", "Info"),
]


class Sidebar(QWidget):
    screen_requested = Signal(str)  # "live_grid" | "analyzer" | "help"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self._expanded = False
        self._active = "live_grid"

        self.setFixedWidth(_COLLAPSED_WIDTH)
        self._anim = QPropertyAnimation(self, b"minimumWidth", self)
        self._anim.setDuration(_ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim_max = QPropertyAnimation(self, b"maximumWidth", self)
        self._anim_max.setDuration(_ANIM_MS)
        self._anim_max.setEasingCurve(QEasingCurve.Type.OutCubic)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(4)

        # Collapsed: just the mark, no letters — a logo, not an abbreviation.
        self._brand_icon = QLabel()
        self._brand_icon.setPixmap(build_app_icon().pixmap(28, 28))
        self._brand_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._brand_icon.setContentsMargins(0, 0, 0, 16)
        layout.addWidget(self._brand_icon)

        self._brand = title("REFEYE")
        self._brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._brand.setContentsMargins(0, 0, 0, 16)
        layout.addWidget(self._brand)

        self._buttons: dict[str, QPushButton] = {}
        self._icons: dict[str, QLabel] = {}
        self._labels: dict[str, QLabel] = {}
        for name, glyph, label_text in _ITEMS:
            button = QPushButton()
            button.setObjectName("SidebarNavItem")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, n=name: self._on_clicked(n))

            row = QHBoxLayout(button)
            row.setContentsMargins(14, 0, 14, 0)
            row.setSpacing(12)

            icon = QLabel(glyph)
            icon.setObjectName("SidebarNavIcon")
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(icon)
            self._icons[name] = icon

            item_label = QLabel(label_text)
            item_label.setObjectName("SidebarNavLabel")
            row.addWidget(item_label, 1)
            self._labels[name] = item_label

            layout.addWidget(button)
            self._buttons[name] = button

        layout.addStretch(1)
        self._apply_active()
        self._update_labels()

    def set_active(self, name: str) -> None:
        """Called externally (e.g. a Live Grid camera button navigated us to
        the Analyzer screen) so the sidebar's highlight stays in sync even
        when the click didn't originate from the sidebar itself."""
        if name == self._active:
            return
        self._active = name
        self._apply_active()

    def _on_clicked(self, name: str) -> None:
        self._active = name
        self._apply_active()
        self.screen_requested.emit(name)

    def _apply_active(self) -> None:
        for name, button in self._buttons.items():
            button.setProperty("active", "true" if name == self._active else "false")
            for widget in (button, self._icons[name], self._labels[name]):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def _update_labels(self) -> None:
        self._brand_icon.setVisible(not self._expanded)
        self._brand.setVisible(self._expanded)
        for name, _glyph, _label_text in _ITEMS:
            self._labels[name].setVisible(self._expanded)

    def enterEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().enterEvent(event)
        self._expanded = True
        self._update_labels()
        self._animate_to(_EXPANDED_WIDTH)

    def leaveEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().leaveEvent(event)
        self._expanded = False
        self._update_labels()
        self._animate_to(_COLLAPSED_WIDTH)

    def _animate_to(self, width: int) -> None:
        for anim, prop in ((self._anim, self.minimumWidth), (self._anim_max, self.maximumWidth)):
            anim.stop()
            anim.setStartValue(prop())
            anim.setEndValue(width)
        self._anim.start()
        self._anim_max.start()
