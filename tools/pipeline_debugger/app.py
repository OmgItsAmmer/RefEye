"""RefEye pipeline debugger — see every offside stage on real footage.

A developer/operator tool, not part of the shipped app. It loads the same
config and the same models through the same `ModelRegistry`, so what it draws
is what the product computes.

    python -m tools.pipeline_debugger [path/to/clip.mp4]

Layout: the frame with toggleable overlays on the left, the pipeline panel on
the right. The panel lists **every** stage in the plan, including the ones
that do not exist yet — a phase that is missing should look missing, not be
absent from the page.

Marking the pitch: pick a landmark, click where it is in the frame, repeat
four times. That is what turns the calibration from "directional" (offside
lines only) into "metric" (positions in metres, and the top-down map fills
in). Marks persist while the camera stays on the same shot.

Correcting the teams: pick a correction in the team box, then click the player
it applies to. Kit colours are measured from the footage, never configured, so
each player's box is drawn in the colour that was actually measured — a box
whose colour does not match the shirt inside it is the failure, shown directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ai.model_registry.registry import ModelRegistry
from core.config.loader import load_settings
from core.config.paths import resolve
from core.domain.models import FramePacket
from offside.team_assignment.teams import TEAM_A, TEAM_B, PlayerRole
from tools.pipeline_debugger import overlays
from tools.pipeline_debugger.pipeline import (
    FrameAnalysis,
    OffsidePipeline,
    StageState,
)
from vision.features.feature_cache import FeatureCache

DEFAULT_CLIP = "data/videos/client_m2_test_video.mp4"

BG = "#121212"
SURFACE = "#1A1A1A"
BORDER = "#2A2A2A"
TEXT = "#E0E0E0"
MUTED = "#8A8A8A"

STATE_COLORS = {
    StageState.OK: "#00FF66",
    StageState.DEGRADED: "#FFB020",
    StageState.UNAVAILABLE: "#FF3366",
    StageState.PENDING: "#525252",
}
STATE_LABELS = {
    StageState.OK: "OK",
    StageState.DEGRADED: "PARTIAL",
    StageState.UNAVAILABLE: "UNAVAILABLE",
    StageState.PENDING: "NOT BUILT",
}


class FrameWorker(QObject):
    """Decodes and analyses frames off the UI thread."""

    ready = Signal(object)
    failed = Signal(str)
    opened = Signal(int)

    def __init__(self, clip_path: str, pipeline: OffsidePipeline):
        super().__init__()
        self._clip_path = clip_path
        self._pipeline = pipeline
        self._capture: cv2.VideoCapture | None = None

    @Slot()
    def open(self) -> None:
        self._capture = cv2.VideoCapture(self._clip_path)
        if not self._capture.isOpened():
            self.failed.emit(f"could not open {self._clip_path}")
            return
        self.opened.emit(int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))

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
        frame = FramePacket(
            frame_id=frame_index,
            pts=frame_index,
            timestamp_ms=int(frame_index * 1000 / 30),
            capture_timestamp_ms=int(frame_index * 1000 / 30),
            width=width,
            height=height,
            source_id="debugger",
            image=image,
        )

        try:
            self.ready.emit(self._pipeline.analyse(frame, frame_index))
        except Exception as exc:  # noqa: BLE001 — a debug tool must not die
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
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimage).scaled(
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


class StageCard(QFrame):
    """One pipeline stage, with its state, summary and reasons."""

    def __init__(self, report):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background:{SURFACE}; border:1px solid {BORDER}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        header = QHBoxLayout()
        title = QLabel(f"{report.phase} · {report.title}")
        title.setStyleSheet(f"color:{TEXT}; font-weight:600; border:none;")
        badge = QLabel(STATE_LABELS[report.state])
        badge.setStyleSheet(
            f"color:{STATE_COLORS[report.state]}; font-family:monospace;"
            " font-size:10px; border:none;"
        )
        header.addWidget(title)
        header.addStretch()
        header.addWidget(badge)
        layout.addLayout(header)

        summary = QLabel(report.summary)
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color:{MUTED}; border:none;")
        layout.addWidget(summary)

        for detail in report.details:
            is_warning = detail.startswith("WARNING:")
            label = QLabel(("⚠ " + detail[8:].strip()) if is_warning else "· " + detail)
            label.setWordWrap(True)
            label.setStyleSheet(
                f"color:{'#FFB020' if is_warning else '#6E6E6E'};"
                " font-size:11px; border:none;"
            )
            layout.addWidget(label)


class DebuggerWindow(QWidget):
    request_frame = Signal(int)
    request_open = Signal()

    def __init__(self, clip_path: str):
        super().__init__()
        self.setWindowTitle(f"RefEye — pipeline debugger — {Path(clip_path).name}")
        self.resize(1640, 940)
        self.setStyleSheet(f"background:{BG}; color:{TEXT}; font-size:12px;")

        self._settings = load_settings()
        self._registry = ModelRegistry(
            self._settings,
            FeatureCache(max_frames=self._settings.ai.feature_cache.max_frames),
        )
        self._pipeline = OffsidePipeline(self._settings, self._registry)

        self._analysis: FrameAnalysis | None = None
        self._frame_index = 0
        self._frame_count = 0
        self._busy = False
        self._playing = False

        self._build_ui()

        self._thread = QThread(self)
        self._worker = FrameWorker(clip_path, self._pipeline)
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

        # Let the window paint before the (slow) model load blocks the thread.
        QTimer.singleShot(50, self._load_models)

    # -- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QVBoxLayout()
        self._canvas = FrameCanvas()
        self._canvas.clicked.connect(self._on_canvas_click)
        left.addWidget(self._canvas, stretch=1)
        left.addLayout(self._build_transport())

        self._status = QLabel("loading models…")
        self._status.setStyleSheet(f"color:{MUTED}; padding:4px;")
        left.addWidget(self._status)

        root.addLayout(left, stretch=3)
        root.addWidget(self._build_side_panel(), stretch=2)

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._play_button = QPushButton("▶ Play")
        self._play_button.clicked.connect(self._toggle_play)
        previous = QPushButton("◀ Frame")
        previous.clicked.connect(lambda: self._goto(self._frame_index - 1))
        following = QPushButton("Frame ▶")
        following.clicked.connect(lambda: self._goto(self._frame_index + 1))

        for button in (self._play_button, previous, following):
            button.setStyleSheet(
                f"QPushButton {{ background:{SURFACE}; color:{TEXT};"
                f" border:1px solid {BORDER}; padding:6px 12px; }}"
                f"QPushButton:hover {{ background:#262626; }}"
            )

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

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_overlay_box())
        layout.addWidget(self._build_calibration_box())
        layout.addWidget(self._build_team_box())

        self._stage_container = QVBoxLayout()
        self._stage_container.setSpacing(6)
        stages_holder = QWidget()
        stages_holder.setLayout(self._stage_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(stages_holder)
        scroll.setStyleSheet(f"border:1px solid {BORDER};")
        layout.addWidget(scroll, stretch=1)

        self._top_down = QLabel()
        self._top_down.setFixedHeight(300)
        self._top_down.setAlignment(Qt.AlignCenter)
        self._top_down.setStyleSheet(f"background:#0E0E0E; border:1px solid {BORDER};")
        layout.addWidget(self._top_down)

        return panel

    def _build_overlay_box(self) -> QGroupBox:
        box = QGroupBox("Overlays")
        box.setStyleSheet(
            f"QGroupBox {{ color:{MUTED}; border:1px solid {BORDER};"
            " margin-top:8px; padding:8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left:8px; }"
        )
        layout = QVBoxLayout(box)

        self._toggles: dict[str, QCheckBox] = {}
        for key, label, default in (
            ("detections", "Player / ball boxes (M1)", True),
            ("poses", "Body skeletons (M2.2)", True),
            ("ground", "Foot positions (M2.2)", True),
            ("teams", "Team colours & roles (M2.3)", True),
            ("mask", "What the line finder sees (M2.1)", False),
            ("lines", "Detected pitch markings (M2.1)", True),
            ("offside_dir", "Goal-line direction through each player (M2.1)", False),
            ("marks", "Marked landmarks (M2.1)", True),
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(default)
            checkbox.setStyleSheet(f"color:{TEXT};")
            checkbox.stateChanged.connect(self._render)
            layout.addWidget(checkbox)
            self._toggles[key] = checkbox

        return box

    def _build_calibration_box(self) -> QGroupBox:
        box = QGroupBox("Pitch calibration (M2.1)")
        box.setStyleSheet(
            f"QGroupBox {{ color:{MUTED}; border:1px solid {BORDER};"
            " margin-top:8px; padding:8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left:8px; }"
        )
        layout = QVBoxLayout(box)

        hint = QLabel(
            "Pick a landmark, then click it in the frame. Four marks give "
            "metric calibration and fill in the top-down map."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        layout.addWidget(hint)

        self._landmark_box = QComboBox()
        self._landmark_box.addItems(sorted(self._pipeline.pitch.landmarks()))
        self._landmark_box.setStyleSheet(
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:4px;"
        )
        layout.addWidget(self._landmark_box)

        row = QHBoxLayout()
        clear = QPushButton("Clear marks")
        clear.clicked.connect(self._clear_marks)
        clear.setStyleSheet(
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:5px;"
        )
        row.addWidget(clear)

        self._family_box = QComboBox()
        self._family_box.addItem("goal-line group: unconfirmed", None)
        self._family_box.currentIndexChanged.connect(self._on_family_changed)
        self._family_box.setStyleSheet(
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:4px;"
        )
        row.addWidget(self._family_box, stretch=1)
        layout.addLayout(row)

        return box

    def _build_team_box(self) -> QGroupBox:
        """Operator corrections for M2.3.

        Every control here drives `TeamOverrides` on the assigner — the same
        object the M2.7 review panel will drive — rather than reaching into
        the assignment result. A correction made in this window is therefore
        the product's correction path, tested here first.
        """
        box = QGroupBox("Team assignment (M2.3)")
        box.setStyleSheet(
            f"QGroupBox {{ color:{MUTED}; border:1px solid {BORDER};"
            " margin-top:8px; padding:8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left:8px; }"
        )
        layout = QVBoxLayout(box)

        hint = QLabel(
            "Kit colours are measured from this footage, never configured. "
            "Pick a correction below, then click the player it applies to; "
            "leave it on 'off' to keep clicking for pitch landmarks."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        layout.addWidget(hint)

        self._pin_mode = QComboBox()
        self._pin_mode.addItem("click: mark pitch landmarks (off)", None)
        self._pin_mode.addItem("click: this player is team A", ("team", TEAM_A))
        self._pin_mode.addItem("click: this player is team B", ("team", TEAM_B))
        self._pin_mode.addItem("click: this player is a goalkeeper", ("role", "gk"))
        self._pin_mode.addItem("click: not a player — ignore them", ("role", "unknown"))
        self._pin_mode.addItem("click: remove my correction", ("clear", None))
        self._pin_mode.setStyleSheet(
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:4px;"
        )
        layout.addWidget(self._pin_mode)

        self._attacking_box = QComboBox()
        self._attacking_box.addItem("attacking side: from the ball (auto)", None)
        self._attacking_box.addItem("attacking side: team A", TEAM_A)
        self._attacking_box.addItem("attacking side: team B", TEAM_B)
        self._attacking_box.currentIndexChanged.connect(self._on_attacking_changed)
        self._attacking_box.setStyleSheet(
            f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER}; padding:4px;"
        )
        layout.addWidget(self._attacking_box)

        row = QHBoxLayout()
        for label, handler in (
            ("Swap A / B", self._swap_teams),
            ("Clear corrections", self._clear_team_overrides),
            ("Re-measure kits", self._reset_team_colors),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            button.setStyleSheet(
                f"background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER};"
                " padding:5px;"
            )
            row.addWidget(button)
        layout.addLayout(row)

        return box

    # -- lifecycle ----------------------------------------------------------

    def _load_models(self) -> None:
        state = self._registry.load_all()
        self._status.setText(
            f"models: {state.status.value} — {state.message}"
            + (f"  |  {len(state.warnings)} warning(s)" if state.warnings else "")
        )
        self.request_open.emit()

    @Slot(int)
    def _on_opened(self, frame_count: int) -> None:
        self._frame_count = frame_count
        self._slider.setEnabled(True)
        self._slider.setRange(0, max(0, frame_count - 1))
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
        self._frame_label.setText(f"frame {analysis.frame_index} / {max(0, self._frame_count - 1)}")

        self._sync_family_box(analysis)
        self._rebuild_stages(analysis)
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

    @Slot(float, float)
    def _on_canvas_click(self, x: float, y: float) -> None:
        mode = self._pin_mode.currentData()
        if mode is not None:
            self._pin_player((x, y), mode)
            return

        landmark = self._landmark_box.currentText()
        self._pipeline.mark_landmark((x, y), landmark)
        self._status.setText(
            f"marked {landmark.replace('_', ' ')} at ({x:.0f}, {y:.0f}) — "
            f"{len(self._pipeline.manual_correspondences)} of 4 needed"
        )
        self._goto(self._frame_index)

    def _pin_player(self, point: tuple[float, float], mode) -> None:
        kind, value = mode
        if kind == "clear":
            removed = self._pipeline.clear_player_pin(point)
            self._status.setText(
                "removed a correction" if removed else "no correction near that click"
            )
        elif kind == "team":
            self._pipeline.pin_player(point, team_id=value)
            self._status.setText(f"pinned the nearest player to team {value[-1].upper()}")
        elif value == "gk":
            self._pipeline.pin_player(point, role=PlayerRole.GOALKEEPER)
            self._status.setText("pinned the nearest player as a goalkeeper")
        else:
            self._pipeline.pin_player(point, role=PlayerRole.UNKNOWN)
            self._status.setText("excluded the nearest player from both teams")
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
        self._status.setText("kit colours will be re-measured from this frame")
        self._goto(self._frame_index)

    def _clear_marks(self) -> None:
        self._pipeline.clear_landmarks()
        self._status.setText("cleared marked landmarks")
        self._goto(self._frame_index)

    def _on_family_changed(self, index: int) -> None:
        self._pipeline.goal_line_family_index = self._family_box.itemData(index)
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
        wanted = self._pipeline.goal_line_family_index
        position = self._family_box.findData(wanted)
        self._family_box.setCurrentIndex(max(0, position))
        self._family_box.blockSignals(False)

    def _rebuild_stages(self, analysis: FrameAnalysis) -> None:
        while self._stage_container.count():
            item = self._stage_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for report in analysis.reports:
            self._stage_container.addWidget(StageCard(report))
        self._stage_container.addStretch()

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
        if enabled["teams"]:
            overlays.draw_team_assignment(image, analysis.poses, analysis.teams)
        if enabled["poses"]:
            overlays.draw_poses(image, analysis.poses, min_confidence)
        if enabled["ground"]:
            overlays.draw_ground_points(image, analysis.poses)
        if enabled["marks"]:
            overlays.draw_marked_landmarks(image, self._pipeline.manual_correspondences)

        self._canvas.show_image(image)

        top_down = overlays.render_top_down(
            self._pipeline.pitch,
            analysis.calibration,
            analysis.poses,
            analysis.teams if enabled["teams"] else None,
        )
        rgb = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self._top_down.setPixmap(
            QPixmap.fromImage(
                QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
            ).scaled(self._top_down.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
    window = DebuggerWindow(str(clip_path))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
