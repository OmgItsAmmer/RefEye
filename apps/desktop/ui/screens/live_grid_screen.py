"""Live Grid screen — a camera dashboard cloned from sample_ui's
LiveGridScreen.tsx, plus a Spotify-style bottom transport bar.

Four independent local video files play across the four tiles (config's
video.local_file for Cam 1, video.preview_cameras for Cams 2-4 — see
core/config/schema.py). All four decode and display continuously, but only
one camera is ever "focused" at a time: that's the only tile that gets a
detection/tracking overlay, and the only one "Analyse recent play" ever
analyzes (main_viewmodel.py's `set_focused_camera` / `focused_camera_changed`
— one analysis pipeline, one inference cost, regardless of grid size, per
architecture.md section 5). Selecting a camera (or Best, which keeps
whichever camera is already focused) moves the focus there. A tile whose
camera has no configured file just stays "No signal".

A grid-mode toggle (1/2/4 cameras) sits top-right, next to the stream badge.
Each mode swaps in its own bottom bar:
  * 4-up (default) — the full Cam1/Cam2/Best/Cam3/Cam4 transport bar.
  * 2-up            — a reduced Cam1/Best/Cam2 bar.
  * 1-up            — no transport bar at all (nothing to switch between);
                       instead an Analyse button plus a dropdown that picks
                       which camera's label the single tile wears.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.ui.theme.icons import grid_mode_icon
from apps.desktop.ui.widgets.common import (
    AnimatedButton,
    BadgeVariant,
    StatusBadge,
    data_value,
    simple_dot,
    title,
)
from apps.desktop.ui.widgets.video_panel import VideoSurface

CAMERA_NAMES = ["Cam 1", "Cam 2", "Cam 3", "Cam 4"]

_MODE_QUAD = 0
_MODE_DUO = 1
_MODE_SOLO = 2

_FADE_OUT_MS = 110
_FADE_IN_MS = 180


class _CameraTile(QFrame):
    """One grid tile: a live video surface with an on-video name badge,
    top-left, in the same glassy translucent style as the reference app's
    camera-info overlay (theme.md's one exception to opaque surfaces)."""

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("CameraTile")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        self.surface = VideoSurface()
        self.surface.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.surface.set_placeholder("No signal")
        layout.addWidget(self.surface)

        self._badge = QWidget(self)
        self._badge.setObjectName("OnVideoBadge")
        badge_layout = QHBoxLayout(self._badge)
        badge_layout.setContentsMargins(8, 4, 8, 4)
        badge_layout.setSpacing(6)
        self._dot = simple_dot("#525252")
        badge_layout.addWidget(self._dot)
        self._name_value = data_value(name)
        badge_layout.addWidget(self._name_value)
        self._badge.adjustSize()
        self._badge.move(8, 8)
        self._badge.raise_()

    def set_live(self, live: bool) -> None:
        self._dot.setStyleSheet(
            f"background-color: {'#00FF66' if live else '#525252'}; border-radius: 4px;"
        )

    def set_name(self, name: str) -> None:
        self._name_value.setText(name)
        self._badge.adjustSize()


class LiveGridScreen(QWidget):
    camera_selected = Signal(int)  # 1..4
    best_selected = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._single_cam_index = 1

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16)
        root.setSpacing(16)

        top_bar = QHBoxLayout()
        top_bar.addWidget(title("Live Grid Dashboard"))
        top_bar.addStretch(1)
        top_bar.addWidget(self._build_overlay_toggle())
        top_bar.addSpacing(12)
        top_bar.addLayout(self._build_grid_mode_toggle())
        top_bar.addSpacing(12)
        self._stream_badge = StatusBadge("Connecting", BadgeVariant.MUTED)
        top_bar.addWidget(self._stream_badge)
        root.addLayout(top_bar)

        self._grid = QGridLayout()
        self._grid.setSpacing(12)
        self._tiles: list[_CameraTile] = [_CameraTile(name) for name in CAMERA_NAMES]
        self._grid_widget = QWidget()
        self._grid_widget.setLayout(self._grid)
        root.addWidget(self._grid_widget, stretch=1)

        self._bottom_stack = QStackedWidget()
        self._bottom_stack.addWidget(self._build_quad_bar())   # _MODE_QUAD
        self._bottom_stack.addWidget(self._build_duo_bar())    # _MODE_DUO
        self._bottom_stack.addWidget(self._build_solo_bar())   # _MODE_SOLO
        root.addWidget(self._bottom_stack)

        self._grid_mode = 4
        self._mode_anim: QParallelAnimationGroup | None = None
        self._relayout(4)

    # -- detection overlay toggle --------------------------------------------

    def _build_overlay_toggle(self) -> QPushButton:
        button = AnimatedButton("Detections")
        button.setObjectName("OverlayToggleButton")
        button.setCheckable(True)
        button.setChecked(True)
        button.setToolTip("Show/hide outlines around detected players and the ball")
        button.toggled.connect(self._on_overlay_toggled)
        return button

    # -- grid-mode toggle --------------------------------------------------

    def _build_grid_mode_toggle(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)

        self._grid_mode_group = QButtonGroup(self)
        self._grid_mode_group.setExclusive(True)
        self._grid_mode_buttons: dict[int, QPushButton] = {}

        for cells in (1, 2, 4):
            button = QPushButton()
            button.setObjectName("GridModeButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setIcon(grid_mode_icon(cells))
            button.setIconSize(QSize(15, 15))
            button.setFixedSize(30, 30)
            button.setToolTip(f"{cells}-camera grid")
            button.clicked.connect(lambda _checked=False, c=cells: self._apply_grid_mode(c))
            self._grid_mode_group.addButton(button)
            self._grid_mode_buttons[cells] = button
            row.addWidget(button)

        self._grid_mode_buttons[4].setChecked(True)
        return row

    def _apply_grid_mode(self, cells: int) -> None:
        """Crossfade into the new grid/bar arrangement rather than snapping
        tiles to their new size instantly — a hard cut there reads as a
        glitch, not a mode switch."""
        self._grid_mode = cells
        self._grid_mode_buttons[cells].setChecked(True)
        self._fade_swap(lambda: self._relayout(cells))

    def _relayout(self, cells: int) -> None:
        for tile in self._tiles:
            self._grid.removeWidget(tile)
            tile.setVisible(False)

        if cells == 4:
            for index, tile in enumerate(self._tiles):
                self._grid.addWidget(tile, index // 2, index % 2)
                tile.setVisible(True)
            self._bottom_stack.setCurrentIndex(_MODE_QUAD)
        elif cells == 2:
            for index in (0, 1):
                self._grid.addWidget(self._tiles[index], 0, index)
                self._tiles[index].setVisible(True)
            self._bottom_stack.setCurrentIndex(_MODE_DUO)
        else:
            tile = self._tiles[self._single_cam_index - 1]
            self._grid.addWidget(tile, 0, 0)
            tile.setVisible(True)
            self._bottom_stack.setCurrentIndex(_MODE_SOLO)

    def _fade_swap(self, mutate) -> None:
        targets = [self._grid_widget, self._bottom_stack]
        effects = [QGraphicsOpacityEffect(w) for w in targets]
        for widget, effect in zip(targets, effects):
            widget.setGraphicsEffect(effect)

        fade_out = QParallelAnimationGroup(self)
        for effect in effects:
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(_FADE_OUT_MS)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.Type.InCubic)
            fade_out.addAnimation(anim)

        def _swap_then_fade_in() -> None:
            mutate()

            fade_in = QParallelAnimationGroup(self)
            for effect in effects:
                anim = QPropertyAnimation(effect, b"opacity", self)
                anim.setDuration(_FADE_IN_MS)
                anim.setStartValue(0.0)
                anim.setEndValue(1.0)
                anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                fade_in.addAnimation(anim)
            fade_in.finished.connect(lambda: [w.setGraphicsEffect(None) for w in targets])
            fade_in.start()
            self._mode_anim = fade_in  # keep the group alive until it finishes

        fade_out.finished.connect(_swap_then_fade_in)
        fade_out.start()
        self._mode_anim = fade_out  # keep the group alive until it finishes

    # -- bottom bars ---------------------------------------------------

    def _build_quad_bar(self) -> QWidget:
        wrapper = QHBoxLayout()
        wrapper.addStretch(1)

        bar = QFrame()
        bar.setObjectName("TransportBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 10, 10, 10)
        bar_layout.setSpacing(8)

        for i in (1, 2):
            bar_layout.addWidget(self._transport_button(f"Cam {i}", lambda _c=False, n=i: self._select_camera(n)))

        best_button = self._transport_button("Best", lambda: self._select_best())
        best_button.setProperty("best", "true")
        bar_layout.addWidget(best_button)

        for i in (3, 4):
            bar_layout.addWidget(self._transport_button(f"Cam {i}", lambda _c=False, n=i: self._select_camera(n)))

        wrapper.addWidget(bar)
        wrapper.addStretch(1)
        container = QWidget()
        container.setLayout(wrapper)
        return container

    def _build_duo_bar(self) -> QWidget:
        wrapper = QHBoxLayout()
        wrapper.addStretch(1)

        bar = QFrame()
        bar.setObjectName("TransportBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 10, 10, 10)
        bar_layout.setSpacing(8)

        bar_layout.addWidget(self._transport_button("Cam 1", lambda: self._select_camera(1)))
        best_button = self._transport_button("Best", lambda: self._select_best())
        best_button.setProperty("best", "true")
        bar_layout.addWidget(best_button)
        bar_layout.addWidget(self._transport_button("Cam 2", lambda: self._select_camera(2)))

        wrapper.addWidget(bar)
        wrapper.addStretch(1)
        container = QWidget()
        container.setLayout(wrapper)
        return container

    def _build_solo_bar(self) -> QWidget:
        wrapper = QHBoxLayout()
        wrapper.addStretch(1)

        self._solo_dropdown = QComboBox()
        self._solo_dropdown.setObjectName("CameraDropdown")
        self._solo_dropdown.addItems(CAMERA_NAMES)
        self._solo_dropdown.currentIndexChanged.connect(self._on_solo_camera_changed)
        wrapper.addWidget(self._solo_dropdown)

        analyze_button = AnimatedButton(f"Analyse  ({CAMERA_NAMES[0]})")
        analyze_button.setObjectName("SoloAnalyzeButton")
        analyze_button.setProperty("role", "primary")
        analyze_button.clicked.connect(lambda: self._select_camera(self._single_cam_index))
        self._solo_analyze_button = analyze_button
        wrapper.addWidget(analyze_button)

        wrapper.addStretch(1)
        container = QWidget()
        container.setLayout(wrapper)
        return container

    def _on_solo_camera_changed(self, index: int) -> None:
        self._single_cam_index = index + 1
        self._solo_analyze_button.setText(f"Analyse  ({CAMERA_NAMES[index]})")
        if self._bottom_stack.currentIndex() == _MODE_SOLO:
            self._apply_grid_mode(1)

    @staticmethod
    def _transport_button(text: str, handler) -> QPushButton:
        button = QPushButton(text.upper())
        button.setObjectName("TransportButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(handler)
        return button

    def _select_camera(self, index: int) -> None:
        self.camera_selected.emit(index)

    def _select_best(self) -> None:
        self.best_selected.emit()

    # Public entry points for shortcut bindings (main_window.py) — identical
    # effect to clicking the corresponding transport button.
    def trigger_camera(self, index: int) -> None:
        self._select_camera(index)

    def trigger_best(self) -> None:
        self._select_best()

    # -- viewmodel wiring -------------------------------------------------
    # Each camera is now a genuinely independent feed (module docstring), so
    # every update targets one tile (index 1..4) rather than broadcasting.

    def set_camera_frame(self, index: int, image) -> None:
        tile = self._tiles[index - 1]
        tile.surface.set_frame(image)
        tile.set_live(True)

    def set_camera_overlay(self, index: int, tracks, ball_center, ball_confidence) -> None:
        """What the focused camera is actually detecting right now
        (architecture.md section 14) — per-tile, not broadcast, since only
        one camera is ever focused/detected at a time (module docstring)."""
        self._tiles[index - 1].surface.set_overlay(tracks, ball_center, ball_confidence)

    def clear_camera(self, index: int) -> None:
        tile = self._tiles[index - 1]
        tile.surface.clear()
        tile.set_live(False)

    def set_stream_status(self, text: str, variant: BadgeVariant) -> None:
        self._stream_badge.set_status(text, variant)

    def _on_overlay_toggled(self, checked: bool) -> None:
        for tile in self._tiles:
            tile.surface.set_overlay_enabled(checked)
