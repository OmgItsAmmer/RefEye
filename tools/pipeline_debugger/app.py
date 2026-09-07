"""RefEye — Pipeline Inspector. Watch a video go through every stage.

    python -m tools.pipeline_debugger [path/to/clip.mp4]
    make debug-ui

Open any video, step through it, and see what each stage of the offside
pipeline actually did to it: which players were found, where their feet are,
what the pitch calibration can and cannot support, what colour each shirt
measured as, who ended up on which team, and what still needs a human look.

## What this window is for

**Trust.** An operator cannot be asked to accept an offside verdict from a
system whose working is invisible; a verdict with no visible reasoning is
just an assertion. So every stage shows its own evidence, in the operator's
terms, and every number on screen sits next to the thing it was measured
from. A stage that has not been built yet still appears, marked, rather than
being silently absent — a gap in the pipeline should look like a gap.

It loads the same config and the same models through the same
`ModelRegistry` as the product, so what it shows is what the product
computes. That is the whole point: this is not a mock-up of the pipeline, it
is the pipeline.

## Layout

Left  — the video with toggleable overlays, and the transport.
Right — three views over the same frame:
        Pipeline  every stage in order, with state, facts and warnings
        Teams     the measured kit colours as two grids of swatches, each
                  beside the player crop it came from, with corrections
        Pitch     the calibration, the landmark marking, the top-down map

## Marking the pitch

Pick a landmark, click where it is in the frame, repeat four times. That is
what turns the calibration from "directional" (offside lines only) into
"metric" (positions in metres, and the top-down map fills in). Marks persist
while the camera stays on the same shot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ai.model_registry.registry import ModelRegistry
from core.config.loader import load_settings
from core.config.paths import resolve
from core.domain.models import FramePacket
from offside.team_assignment.teams import TEAM_A, TEAM_B, PlayerRole
from tools.pipeline_debugger import overlays, presenter
from tools.pipeline_debugger.panels import (
    ACTION_CLEAR,
    ACTION_CONFIRM,
    ACTION_EXCLUDE,
    ACTION_GOALKEEPER,
    ACTION_TEAM_A,
    ACTION_TEAM_B,
    BG,
    BORDER,
    MUTED,
    SURFACE,
    TEXT,
    FlowPanel,
    TeamsPanel,
    bgr_to_pixmap,
)
from tools.pipeline_debugger.pipeline import FrameAnalysis, OffsidePipeline
from vision.features.feature_cache import FeatureCache

DEFAULT_CLIP = "data/videos/client_m2_test_video.mp4"
VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.avi *.m4v);;All files (*)"


class FrameWorker(QObject):
    """Decodes and analyses frames off the UI thread."""

    ready = Signal(object)
    failed = Signal(str)
    opened = Signal(int, str)

    def __init__(self, pipeline: OffsidePipeline):
        super().__init__()
        self._pipeline = pipeline
        self._capture: cv2.VideoCapture | None = None
        self._clip_path: str | None = None
        self._fps = 30.0

    @Slot(str)
    def open(self, clip_path: str) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = cv2.VideoCapture(clip_path)
        if not self._capture.isOpened():
            self.failed.emit(f"could not open {clip_path}")
            return
        self._clip_path = clip_path
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.opened.emit(int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)), clip_path)

    @Slot(int)
    def analyse(self, frame_index: int) -> None:
        if self._capture is None:
            return

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = self._capture.read()
        if not ok:
            self.failed.emit(f"could not read frame {frame_index}")
            return

        height, width = image.shape[:2]
        timestamp = int(frame_index * 1000 / max(1.0, self._fps))
        frame = FramePacket(
            frame_id=frame_index,
            pts=frame_index,
            timestamp_ms=timestamp,
            capture_timestamp_ms=timestamp,
            width=width,
            height=height,
            source_id="inspector",
            image=image,
        )

        try:
            self.ready.emit(self._pipeline.analyse(frame, frame_index))
        except Exception as exc:  # noqa: BLE001 — an inspector must not die
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FrameCanvas(QLabel):
    """The frame, scaled to fit, reporting clicks in *image* coordinates."""

    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background:#000; border:1px solid {BORDER};")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image: np.ndarray | None = None

    def show_image(self, image: np.ndarray) -> None:
        self._image = image
        self._repaint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._repaint()

    def _repaint(self) -> None:
        if self._image is None:
            return
        self.setPixmap(
            bgr_to_pixmap(self._image).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or self._image is None:
            return

        # Undo the letterboxing that KeepAspectRatio introduced, so a click
        # lands on the pixel the operator actually pointed at.
        drawn_w, drawn_h = pixmap.width(), pixmap.height()
        pad_x = (self.width() - drawn_w) / 2
        pad_y = (self.height() - drawn_h) / 2

        x = event.position().x() - pad_x
        y = event.position().y() - pad_y
        if not (0 <= x < drawn_w and 0 <= y < drawn_h):
            return

        height, width = self._image.shape[:2]
        self.clicked.emit(x * width / drawn_w, y * height / drawn_h)


class PitchMapCanvas(QLabel):
    """The top-down map, clickable so a landmark can be picked by pointing.

    The map is the instruction: it shows *where on a pitch* the app wants you
    to click next. Being able to choose the point here as well closes the loop
    — nobody can pick "penalty left top" out of a dropdown of thirty names
    without a picture of a pitch in front of them.
    """

    picked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(320)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"background:#0E0E0E; border:1px solid {BORDER};")
        self._size = (760, 520)

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None:
            return
        drawn_w, drawn_h = pixmap.width(), pixmap.height()
        x = event.position().x() - (self.width() - drawn_w) / 2
        y = event.position().y() - (self.height() - drawn_h) / 2
        if not (0 <= x < drawn_w and 0 <= y < drawn_h):
            return
        self.picked.emit(
            x * self._size[0] / drawn_w, y * self._size[1] / drawn_h
        )


class InspectorWindow(QWidget):
    request_frame = Signal(int)
    request_open = Signal(str)

    def __init__(self, clip_path: str):
        super().__init__()
        self.setWindowTitle("RefEye — Pipeline Inspector")
        self.resize(1720, 980)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}; font-size:12px;")

        self._settings = load_settings()
        self._registry = ModelRegistry(
            self._settings,
            FeatureCache(max_frames=self._settings.ai.feature_cache.max_frames),
        )
        self._pipeline = OffsidePipeline(self._settings, self._registry)

        self._analysis: FrameAnalysis | None = None
        self._clip_path = clip_path
        self._frame_index = 0
        self._frame_count = 0
        self._busy = False
        self._playing = False

        self._build_ui()

        self._thread = QThread(self)
        self._worker = FrameWorker(self._pipeline)
        self._worker.moveToThread(self._thread)
        self.request_open.connect(self._worker.open)
        self.request_frame.connect(self._worker.analyse)
        self._worker.ready.connect(self._on_analysis)
        self._worker.failed.connect(self._on_failure)
        self._worker.opened.connect(self._on_opened)
        self._thread.start()

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._advance_playback)

        self._on_landmark_changed(self._landmark_box.currentIndex())

        # Let the window paint before the (slow) model load blocks the thread.
        QTimer.singleShot(50, self._load_models)

    # -- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addLayout(self._build_source_bar())

        body = QHBoxLayout()
        body.setSpacing(10)

        left = QVBoxLayout()
        self._canvas = FrameCanvas()
        self._canvas.clicked.connect(self._on_canvas_click)
        left.addWidget(self._canvas, stretch=1)
        left.addLayout(self._build_transport())
        left.addWidget(self._build_overlay_box())

        self._status = QLabel("loading models…")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{MUTED}; padding:4px;")
        left.addWidget(self._status)

        body.addLayout(left, stretch=3)
        body.addWidget(self._build_tabs(), stretch=2)
        root.addLayout(body, stretch=1)

    def _build_source_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()

        open_button = QPushButton("Open video…")
        open_button.clicked.connect(self._choose_video)
        open_button.setStyleSheet(self._button_style())
        row.addWidget(open_button)

        self._source_label = QLabel(Path(self._clip_path).name)
        self._source_label.setStyleSheet(f"color:{TEXT}; font-weight:600;")
        row.addWidget(self._source_label)

        self._models_label = QLabel("")
        self._models_label.setStyleSheet(f"color:{MUTED}; font-family:monospace;")
        row.addStretch()
        row.addWidget(self._models_label)
        return row

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._play_button = QPushButton("▶ Play")
        self._play_button.clicked.connect(self._toggle_play)
        previous = QPushButton("◀ Frame")
        previous.clicked.connect(lambda: self._goto(self._frame_index - 1))
        following = QPushButton("Frame ▶")
        following.clicked.connect(lambda: self._goto(self._frame_index + 1))

        for button in (self._play_button, previous, following):
            button.setStyleSheet(self._button_style())

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.sliderMoved.connect(self._goto)

        self._frame_label = QLabel("frame 0")
        self._frame_label.setStyleSheet(f"color:{MUTED}; font-family:monospace;")

        row.addWidget(self._play_button)
        row.addWidget(previous)
        row.addWidget(following)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._frame_label)
        return row

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabWidget::pane {{ border:1px solid {BORDER}; }}"
            f"QTabBar::tab {{ background:{SURFACE}; color:{MUTED};"
            f" padding:7px 14px; border:1px solid {BORDER}; }}"
            f"QTabBar::tab:selected {{ background:#262626; color:{TEXT}; }}"
        )

        self._flow_panel = FlowPanel()
        tabs.addTab(self._flow_panel, "Pipeline")

        self._teams_panel = TeamsPanel()
        self._teams_panel.action_requested.connect(self._on_team_action)
        self._teams_panel.selection_changed.connect(lambda _: self._render())
        tabs.addTab(self._teams_panel, "Teams")

        tabs.addTab(self._build_pitch_tab(), "Pitch")
        return tabs

    def _build_pitch_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_calibration_box())

        self._top_down = PitchMapCanvas()
        self._top_down.picked.connect(self._on_pitch_map_click)
        layout.addWidget(self._top_down, stretch=1)
        return page

    def _build_overlay_box(self) -> QGroupBox:
        box = QGroupBox("What to draw on the frame")
        box.setStyleSheet(self._group_style())
        layout = QHBoxLayout(box)

        self._toggles: dict[str, QCheckBox] = {}
        left = QVBoxLayout()
        right = QVBoxLayout()
        for position, (key, label, default) in enumerate(
            (
                ("detections", "Player / ball boxes (M1)", True),
                ("poses", "Body skeletons (M2.2)", True),
                ("ground", "Foot positions (M2.2)", True),
                ("teams", "Team colours & roles (M2.3)", True),
                ("identities", "Player identities & trails (M2.4)", True),
                ("offside", "Offside line & verdict (M2.5)", True),
                ("mask", "What the line finder sees (M2.1)", False),
                ("lines", "Detected pitch markings (M2.1)", True),
                ("offside_dir", "Goal-line direction through each player (M2.1)", False),
                ("marks", "Marked landmarks (M2.1)", True),
            )
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(default)
            checkbox.setStyleSheet(f"color:{TEXT};")
            checkbox.stateChanged.connect(self._render)
            (left if position % 2 == 0 else right).addWidget(checkbox)
            self._toggles[key] = checkbox

        layout.addLayout(left)
        layout.addLayout(right)
        return box

    def _build_calibration_box(self) -> QGroupBox:
        box = QGroupBox("Pitch calibration (M2.1)")
        box.setStyleSheet(self._group_style())
        layout = QVBoxLayout(box)

        hint = QLabel(
            "The camera sees the pitch at an angle, so screen positions are not "
            "pitch positions. Show the app four points whose real place on a "
            "pitch is already known and it can work out the camera, and measure "
            "everything else from there.  "
            "Pick a point on the map below (or from the list), then click that "
            "same point in the video. Four of them turns the calibration metric."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        layout.addWidget(hint)

        self._landmark_box = QComboBox()
        for label, name in presenter.landmark_choices(self._pipeline.pitch):
            self._landmark_box.addItem(label, name)
        self._landmark_box.currentIndexChanged.connect(self._on_landmark_changed)
        self._landmark_box.setStyleSheet(self._combo_style())
        layout.addWidget(self._landmark_box)

        self._landmark_hint = QLabel("")
        self._landmark_hint.setWordWrap(True)
        self._landmark_hint.setStyleSheet(
            f"color:{TEXT}; font-size:11px; padding:2px 0;"
        )
        layout.addWidget(self._landmark_hint)

        row = QHBoxLayout()
        clear = QPushButton("Clear marks")
        clear.clicked.connect(self._clear_marks)
        clear.setStyleSheet(self._button_style())
        row.addWidget(clear)

        self._family_box = QComboBox()
        self._family_box.addItem("goal-line group: unconfirmed", None)
        self._family_box.currentIndexChanged.connect(self._on_family_changed)
        self._family_box.setStyleSheet(self._combo_style())
        row.addWidget(self._family_box, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        for label, handler in (
            ("Swap team A / B", self._swap_teams),
            ("Clear my corrections", self._clear_team_overrides),
            ("Re-measure kits", self._reset_team_colors),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            button.setStyleSheet(self._button_style())
            row.addWidget(button)
        layout.addLayout(row)

        self._direction_box = QComboBox()
        self._direction_box.addItem("attack direction: work it out (auto)", None)
        self._direction_box.addItem("attack direction: towards the left goal", -1.0)
        self._direction_box.addItem("attack direction: towards the right goal", 1.0)
        self._direction_box.currentIndexChanged.connect(self._on_direction_changed)
        self._direction_box.setStyleSheet(self._combo_style())
        layout.addWidget(self._direction_box)

        self._attacking_box = QComboBox()
        self._attacking_box.addItem("attacking side: from the ball (auto)", None)
        self._attacking_box.addItem("attacking side: team A", TEAM_A)
        self._attacking_box.addItem("attacking side: team B", TEAM_B)
        self._attacking_box.currentIndexChanged.connect(self._on_attacking_changed)
        self._attacking_box.setStyleSheet(self._combo_style())
        layout.addWidget(self._attacking_box)

        self._marks_label = QLabel("no landmarks marked yet — 4 needed")
        self._marks_label.setWordWrap(True)
        self._marks_label.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        layout.addWidget(self._marks_label)

        return box

    def _button_style(self) -> str:
        return (
            f"QPushButton {{ background:{SURFACE}; color:{TEXT};"
            f" border:1px solid {BORDER}; padding:6px 12px; }}"
            "QPushButton:hover { background:#262626; }"
        )

    def _combo_style(self) -> str:
        return (
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:4px;"
        )

    def _group_style(self) -> str:
        return (
            f"QGroupBox {{ color:{MUTED}; border:1px solid {BORDER};"
            " margin-top:8px; padding:8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left:8px; }"
        )

    # -- lifecycle ----------------------------------------------------------

    def _load_models(self) -> None:
        state = self._registry.load_all()
        self._models_label.setText(
            f"models: {state.status.value}"
            + (f" · {len(state.warnings)} warning(s)" if state.warnings else "")
        )
        self._status.setText(state.message)
        self.request_open.emit(self._clip_path)

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if not path:
            return
        self._load_clip(path)

    def _load_clip(self, path: str) -> None:
        if self._playing:
            self._toggle_play()
        # A different clip shares nothing with the previous one: kit colours,
        # player identities and pitch marks all describe footage that is gone.
        self._pipeline.clear_landmarks()
        self._pipeline.clear_team_overrides()
        self._pipeline.reset_team_colors()
        self._pipeline.reset_identities()
        self._clip_path = path
        self._source_label.setText(Path(path).name)
        self._status.setText(f"opening {Path(path).name}…")
        self._busy = False
        self.request_open.emit(path)

    @Slot(int, str)
    def _on_opened(self, frame_count: int, clip_path: str) -> None:
        self._frame_count = frame_count
        self._slider.setEnabled(True)
        self._slider.setRange(0, max(0, frame_count - 1))
        self._status.setText(f"{Path(clip_path).name} — {frame_count} frames")
        self._goto(0)

    @Slot(str)
    def _on_failure(self, message: str) -> None:
        self._busy = False
        self._status.setText(f"error: {message}")

    @Slot(object)
    def _on_analysis(self, analysis: FrameAnalysis) -> None:
        self._busy = False
        self._analysis = analysis
        self._frame_index = analysis.frame_index
        self._slider.setValue(analysis.frame_index)
        self._frame_label.setText(
            f"frame {analysis.frame_index} / {max(0, self._frame_count - 1)}"
        )

        self._sync_family_box(analysis)
        self._flow_panel.update_from(presenter.flow_steps(analysis))
        self._teams_panel.update_from(
            analysis,
            presenter.team_columns(analysis),
            presenter.confirmation_summary(analysis),
            presenter.kit_summary(analysis) + presenter.identity_summary(analysis),
        )
        self._render()

    def _goto(self, frame_index: int) -> None:
        if self._frame_count == 0 or self._busy:
            return
        frame_index = max(0, min(frame_index, self._frame_count - 1))
        self._busy = True
        self.request_frame.emit(frame_index)

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._play_button.setText("⏸ Pause" if self._playing else "▶ Play")
        if self._playing:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _advance_playback(self) -> None:
        # Self-paced: never queue a frame while one is still being analysed,
        # or the queue grows unboundedly and the UI drifts behind reality.
        if self._busy:
            return
        if self._frame_index >= self._frame_count - 1:
            self._toggle_play()
            return
        self._goto(self._frame_index + 1)

    # -- interaction --------------------------------------------------------

    def _selected_landmark(self) -> str:
        """The technical name behind the plain label the operator picked."""
        return self._landmark_box.currentData() or self._landmark_box.currentText()

    def _on_landmark_changed(self, _index: int) -> None:
        """Point the map at whatever was just chosen.

        Without this the dropdown and the map disagreed: the operator was told
        to click one point while the diagram highlighted another, which is a
        worse instruction than none at all.
        """
        name = self._selected_landmark()
        self._landmark_hint.setText(presenter.landmark_description(name))
        self._status.setText(
            f"find this in the video and click it: {presenter.landmark_label(name)}"
        )
        self._render()

    @Slot(float, float)
    def _on_canvas_click(self, x: float, y: float) -> None:
        landmark = self._selected_landmark()
        self._pipeline.mark_landmark((x, y), landmark)
        marked = len(self._pipeline.manual_correspondences)
        self._status.setText(
            f"marked {presenter.landmark_label(landmark)} at ({x:.0f}, {y:.0f}) — "
            + (
                f"{marked} of 4 needed"
                if marked < 4
                else f"{marked} marked; the pitch is mapped"
            )
        )
        self._advance_to_next_landmark()
        self._goto(self._frame_index)

    @Slot(float, float)
    def _on_pitch_map_click(self, x: float, y: float) -> None:
        """Pick the landmark to mark by pointing at it on the pitch diagram."""
        target = overlays.canvas_to_pitch(self._pipeline.pitch, (x, y))
        landmarks = self._pipeline.pitch.landmarks()
        nearest = min(
            landmarks,
            key=lambda name: (landmarks[name][0] - target[0]) ** 2
            + (landmarks[name][1] - target[1]) ** 2,
        )
        index = self._landmark_box.findData(nearest)
        if index >= 0:
            self._landmark_box.setCurrentIndex(index)  # redraws via the signal

    def _advance_to_next_landmark(self) -> None:
        """Move the selection on to a landmark that has not been marked yet.

        Small thing, but without it the obvious next click re-marks the point
        just marked, and the operator has to work out for themselves why the
        count is not going up.
        """
        marked = {c.landmark for c in self._pipeline.manual_correspondences}
        for index in range(self._landmark_box.count()):
            if self._landmark_box.itemData(index) not in marked:
                self._landmark_box.setCurrentIndex(index)
                return

    def _on_team_action(self, action: str, indices) -> None:
        analysis = self._analysis
        if analysis is None or analysis.teams is None:
            return

        applied = 0
        for index in indices:
            player = analysis.teams.by_index(index)
            if player is None:
                continue
            anchor = player.anchor_xy

            if action == ACTION_CLEAR:
                self._pipeline.clear_player_pin(anchor)
            elif action == ACTION_TEAM_A:
                self._pipeline.pin_player(anchor, team_id=TEAM_A)
            elif action == ACTION_TEAM_B:
                self._pipeline.pin_player(anchor, team_id=TEAM_B)
            elif action == ACTION_GOALKEEPER:
                self._pipeline.pin_player(anchor, role=PlayerRole.GOALKEEPER)
            elif action == ACTION_EXCLUDE:
                self._pipeline.pin_player(anchor, role=PlayerRole.UNKNOWN)
            elif action == ACTION_CONFIRM:
                # "Confirm as shown" pins whatever is on screen, including
                # "not on either team" — the operator is agreeing with the
                # picture in front of them, whatever it says.
                if player.team_id is not None:
                    self._pipeline.pin_player(anchor, team_id=player.team_id)
                elif player.role is PlayerRole.GOALKEEPER:
                    self._pipeline.pin_player(anchor, role=PlayerRole.GOALKEEPER)
                else:
                    self._pipeline.pin_player(anchor, role=PlayerRole.UNKNOWN)
            applied += 1

        self._status.setText(f"applied '{action}' to {applied} player(s)")
        self._goto(self._frame_index)

    def _clear_marks(self) -> None:
        self._pipeline.clear_landmarks()
        self._status.setText("cleared marked landmarks")
        self._marks_label.setText("no landmarks marked yet — 4 needed")
        self._goto(self._frame_index)

    def _on_family_changed(self, index: int) -> None:
        self._pipeline.goal_line_family_index = self._family_box.itemData(index)
        self._goto(self._frame_index)

    def _on_direction_changed(self, index: int) -> None:
        """Which end is being defended decides the whole verdict — the same
        picture read from the other end gives the opposite answer — so the
        operator can state it rather than living with an inference."""
        self._pipeline.set_attack_direction(self._direction_box.itemData(index))
        self._goto(self._frame_index)

    def _on_attacking_changed(self, index: int) -> None:
        self._pipeline.set_attacking_team(self._attacking_box.itemData(index))
        self._goto(self._frame_index)

    def _swap_teams(self) -> None:
        swapped = self._pipeline.swap_teams()
        self._status.setText(
            "team labels swapped" if swapped else "team labels back to as measured"
        )
        self._goto(self._frame_index)

    def _clear_team_overrides(self) -> None:
        self._pipeline.clear_team_overrides()
        self._attacking_box.blockSignals(True)
        self._attacking_box.setCurrentIndex(0)
        self._attacking_box.blockSignals(False)
        self._status.setText("cleared all team corrections")
        self._goto(self._frame_index)

    def _reset_team_colors(self) -> None:
        self._pipeline.reset_team_colors()
        self._pipeline.reset_identities()
        self._status.setText("kit colours and player identities will be re-measured")
        self._goto(self._frame_index)

    def _sync_family_box(self, analysis: FrameAnalysis) -> None:
        calibration = analysis.calibration
        families = calibration.families if calibration else []

        self._family_box.blockSignals(True)
        self._family_box.clear()
        self._family_box.addItem("goal-line group: unconfirmed", None)
        for index, family in enumerate(families):
            self._family_box.addItem(
                f"group {index}: {len(family.lines)} lines @ {family.mean_angle_deg:.0f}°",
                index,
            )
        position = self._family_box.findData(self._pipeline.goal_line_family_index)
        self._family_box.setCurrentIndex(max(0, position))
        self._family_box.blockSignals(False)

    # -- rendering ----------------------------------------------------------

    def _render(self) -> None:
        analysis = self._analysis
        if analysis is None or analysis.frame.image is None:
            return

        image = analysis.frame.image.copy()
        enabled = {key: box.isChecked() for key, box in self._toggles.items()}
        min_confidence = (
            self._settings.offside.body_keypoints.keypoint_confidence_threshold
        )

        if enabled["mask"]:
            overlays.draw_line_mask(image, analysis.line_mask)
        if enabled["lines"]:
            overlays.draw_pitch_lines(image, analysis.calibration)
        if enabled["offside_dir"]:
            overlays.draw_offside_direction(image, analysis.calibration, analysis.poses)
        if enabled["detections"]:
            overlays.draw_detections(image, analysis)
        if enabled["poses"]:
            overlays.draw_poses(image, analysis.poses, min_confidence)
        if enabled["ground"]:
            overlays.draw_ground_points(image, analysis.poses)
        if enabled["teams"]:
            overlays.draw_team_assignment(image, analysis.poses, analysis.teams)
        if enabled["identities"]:
            overlays.draw_identities(image, analysis.poses, analysis.identities)
        if enabled["offside"]:
            overlays.draw_offside_line(image, analysis.offside, analysis.explanation)
        if enabled["marks"]:
            overlays.draw_marked_landmarks(image, self._pipeline.manual_correspondences)

        overlays.highlight_player(
            image, analysis.poses, self._teams_panel.selected_index
        )

        self._canvas.show_image(image)

        marked = {
            correspondence.landmark: correspondence.image_xy
            for correspondence in self._pipeline.manual_correspondences
        }
        top_down = overlays.render_top_down(
            self._pipeline.pitch,
            analysis.calibration,
            analysis.poses,
            analysis.teams if enabled["teams"] else None,
            highlight_landmark=self._selected_landmark(),
            highlight_label=presenter.landmark_label(self._selected_landmark()),
            highlight_description=presenter.landmark_description(
                self._selected_landmark()
            ),
            marked_landmarks=marked,
        )
        self._marks_label.setText(
            "no landmarks marked yet — 4 needed"
            if not marked
            else f"marked {len(marked)} of 4: "
            + ", ".join(presenter.landmark_label(name) for name in marked)
        )
        self._top_down.setPixmap(
            bgr_to_pixmap(top_down).scaled(
                self._top_down.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self._play_timer.stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    clip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLIP
    clip_path = resolve(clip)
    if not clip_path.exists():
        print(f"clip not found: {clip_path}")
        return 1

    app = QApplication(sys.argv)
    window = InspectorWindow(str(clip_path))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
