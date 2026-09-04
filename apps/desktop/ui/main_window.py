"""RefEye main window.

The UI thread does exactly three things: paint, route input, and react to
signals from the viewmodel. No decoding, no inference, no polling of worker
internals (architecture.md sections 4.1, 32, 33).

Navigation model: a persistent Sidebar (collapsed icon rail, expands on
hover — cloned from sample_ui's Sidebar.tsx) plus a QStackedWidget holding
three screens:

    Live Grid   — a 1/2/4-camera dashboard (toggle top-right) with a
                  Spotify-style bottom transport bar that adapts to the
                  chosen grid size. Selecting a camera or Best crossfades
                  into the Analyzer screen.
    Analyzer    — 70% left: the candidate review workspace (frame nav,
                  alternatives, confirm/retry). 30% right: a camera picker
                  (no camera chosen yet) or a looping "last N seconds"
                  preview, plus diagnostics.
    Help        — a static walkthrough of the pipeline plus which models
                  are actually loaded in this build.

This window owns the viewmodel wiring; the screens themselves are dumb
containers exposing the widgets/signals main_window needs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import cv2
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Slot
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from ai.model_registry.registry import ModelStatus
from apps.desktop.shortcuts.registry import ShortcutRegistry
from apps.desktop.ui.screens.analyzer_screen import (
    PAGE_BUSY,
    PAGE_IDLE,
    PAGE_REVIEW,
    AnalyzerScreen,
)
from apps.desktop.ui.screens.help_screen import HelpScreen
from apps.desktop.ui.screens.live_grid_screen import LiveGridScreen
from apps.desktop.ui.widgets.common import BadgeVariant, StatusBadge, data_value
from apps.desktop.ui.widgets.event_ticker import EventTicker
from apps.desktop.ui.widgets.sidebar import Sidebar
from apps.desktop.viewmodels.main_viewmodel import MainViewModel
from core.config.paths import resolve
from core.config.schema import AppSettings
from core.domain.models import FramePacket
from observability.logging.setup import get_logger
from video.frame_access.video_service import StreamState

logger = get_logger(__name__)

_TRANSITION_MS = 220

_SCREEN_INDEX = {"live_grid": 0, "analyzer": 1, "help": 2}


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings, viewmodel: MainViewModel):
        super().__init__()
        self._settings = settings
        self._vm = viewmodel
        self._transition_anim: QPropertyAnimation | None = None

        self.setWindowTitle(settings.application.name)
        self.resize(1560, 940)
        self.setMinimumSize(1180, 720)

        self._build_ui()
        self._bind_shortcuts()
        self._connect_signals()

    # -- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.screen_requested.connect(self._go_to_screen)
        root.addWidget(self._sidebar)

        self._live_grid = LiveGridScreen()
        self._live_grid.camera_selected.connect(self._on_camera_selected)
        self._live_grid.best_selected.connect(self._on_best_selected)

        self._analyzer = AnalyzerScreen(
            video_service=self._vm.video,
            recent_window_seconds=self._settings.buffer.recent_window_seconds,
            analyze_shortcut_label=self._settings.shortcuts.analyze,
            feature_cache=self._vm.feature_cache,
        )
        self._analyzer.analyze_button.clicked.connect(
            lambda: self._vm.trigger_analysis("button")
        )
        self._analyzer.review.confirmed.connect(self._on_confirmed)
        self._analyzer.review.retry_requested.connect(
            lambda: self._vm.trigger_analysis("retry")
        )
        self._analyzer.camera_pick_requested.connect(lambda: self._go_to_screen("live_grid"))

        self._help = HelpScreen(settings=self._settings, registry=self._vm.registry)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._live_grid)  # _SCREEN_INDEX["live_grid"]
        self._stack.addWidget(self._analyzer)   # _SCREEN_INDEX["analyzer"]
        self._stack.addWidget(self._help)       # _SCREEN_INDEX["help"]
        root.addWidget(self._stack, stretch=1)

        self.setCentralWidget(central)
        self._build_status_bar()

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        # Bottom-left: empty until model load resolves the actual device
        # (GPU is preferred; this only appears on a CPU fallback — see
        # ModelRegistry._resolve_device and _on_model_state below).
        self._device_badge = data_value("")
        bar.addWidget(self._device_badge)

        self._event_ticker = EventTicker()
        bar.addPermanentWidget(self._event_ticker)

        self._model_info_badge = data_value("Models: loading…")
        bar.addPermanentWidget(self._model_info_badge)

        self._model_badge = StatusBadge("AI: loading", BadgeVariant.INFO)
        bar.addPermanentWidget(self._model_badge)

        keys = self._settings.shortcuts
        self._default_status = (
            f"{keys.select_camera_1}-{keys.select_camera_4} camera   ·   "
            f"{keys.select_best} best   ·   "
            f"{keys.analyze} analyse   ·   "
            f"{keys.previous_frame}/{keys.next_frame} frame   ·   "
            f"{keys.previous_candidate}/{keys.next_candidate} candidate   ·   "
            f"{keys.jump_to_best} best   ·   {keys.confirm_frame} confirm"
        )
        bar.showMessage(self._default_status)

    # -- navigation -------------------------------------------------------

    def _on_camera_selected(self, index: int) -> None:
        self._vm.set_focused_camera(index)
        self._enter_analyzer_and_analyze()

    def _on_best_selected(self) -> None:
        self._enter_analyzer_and_analyze()
        self._analyzer.review.jump_to_best()

    def _enter_analyzer_and_analyze(self) -> None:
        """Selecting any Live Grid transport button (a camera or Best) both
        navigates to the Analyzer and immediately kicks off analysis of the
        recent play — no separate click needed."""
        self._go_to_screen("analyzer", camera_context=True)
        self._vm.trigger_analysis("live_grid")

    @Slot(str)
    def _go_to_screen(self, name: str, camera_context: bool = False) -> None:
        target_index = _SCREEN_INDEX[name]
        target_widget = self._stack.widget(target_index)

        if self._stack.currentIndex() == target_index:
            self._sidebar.set_active(name)
            if name == "analyzer" and camera_context:
                self._analyzer.on_shown(camera_context=True)
            return

        self._stack.setCurrentIndex(target_index)
        self._sidebar.set_active(name)
        if name == "analyzer":
            self._analyzer.on_shown(camera_context)
        self._fade_in(target_widget)

    def _fade_in(self, widget: QWidget) -> None:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(_TRANSITION_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        anim.start()
        self._transition_anim = anim  # keep a reference alive until it finishes

    # -- wiring -----------------------------------------------------------

    def _bind_shortcuts(self) -> None:
        self._shortcuts = ShortcutRegistry(self, self._settings.shortcuts)
        review = self._analyzer.review
        bindings = [
            ("analyze", lambda: self._vm.trigger_analysis("shortcut"), "Analyse recent play"),
            ("previous_frame", review.previous_frame, "Previous frame"),
            ("next_frame", review.next_frame, "Next frame"),
            ("previous_candidate", review.previous_candidate, "Previous candidate"),
            ("next_candidate", review.next_candidate, "Next candidate"),
            ("jump_to_best", review.jump_to_best, "Jump to best candidate"),
            ("confirm_frame", review.confirm, "Confirm current frame"),
            ("select_camera_1", lambda: self._live_grid.trigger_camera(1), "Select Cam 1"),
            ("select_camera_2", lambda: self._live_grid.trigger_camera(2), "Select Cam 2"),
            ("select_camera_3", lambda: self._live_grid.trigger_camera(3), "Select Cam 3"),
            ("select_camera_4", lambda: self._live_grid.trigger_camera(4), "Select Cam 4"),
            ("select_best", self._live_grid.trigger_best, "Select best"),
        ]
        for action, handler, description in bindings:
            self._shortcuts.bind(action, handler, description)

    def _connect_signals(self) -> None:
        self._vm.frame_ready.connect(self._on_frame_ready)
        self._vm.stream_state_changed.connect(self._on_stream_state)
        self._vm.camera_frame_ready.connect(self._on_camera_preview_frame)
        self._vm.camera_stream_state_changed.connect(self._on_camera_preview_state)
        self._vm.focused_camera_changed.connect(self._on_focused_camera_changed)
        self._vm.model_state_changed.connect(self._on_model_state)
        self._vm.analysis_state_changed.connect(self._on_analysis_state)
        self._vm.analysis_busy_changed.connect(self._on_analysis_busy)
        self._vm.analysis_completed.connect(self._on_analysis_completed)
        self._vm.analysis_failed.connect(self._on_analysis_failed)

    # -- slots --------------------------------------------------------------

    @Slot(object)
    def _on_frame_ready(self, frame: FramePacket) -> None:
        if frame.image is None:
            return
        self._live_grid.set_camera_frame(1, frame.image)
        self._update_camera_overlay(1)

    @Slot(int, object)
    def _on_camera_preview_frame(self, index: int, frame: FramePacket) -> None:
        if frame.image is None:
            return
        self._live_grid.set_camera_frame(index, frame.image)
        self._update_camera_overlay(index)

    def _update_camera_overlay(self, index: int) -> None:
        """Only the focused camera has anything to show — detection runs on
        one camera at a time (main_viewmodel.py's `focused_camera_changed`)."""
        if index != self._vm.focused_camera:
            return

        # The live CV pipeline runs a few frames behind by design
        # (architecture.md section 4.5) — the freshest cached entry is the
        # best available outline for what's on screen right now, not
        # necessarily this exact frame_id.
        cached = self._vm.feature_cache.latest()
        if cached is not None:
            self._live_grid.set_camera_overlay(
                index, cached.tracks, cached.features.ball_center, cached.features.ball_confidence
            )
        else:
            self._live_grid.set_camera_overlay(index, [], None, None)

    def _on_focused_camera_changed(self, index: int) -> None:
        # Clear every other tile's overlay immediately rather than letting
        # stale boxes from the previously-focused camera linger until
        # something happens to overwrite them (which, for a now-unfocused
        # camera, is never).
        for other in (1, 2, 3, 4):
            if other != index:
                self._live_grid.set_camera_overlay(other, [], None, None)

        # The "Last Ns" preview must follow the focused camera too — it was
        # pinned to camera 1's feed regardless of which camera got analyzed,
        # so picking camera 2/3/4 showed the right analysis but the wrong
        # looping preview clip.
        self._analyzer.recent_clip.set_video_service(self._vm.video_for_camera(index))

    @Slot(int, str, str)
    def _on_camera_preview_state(self, index: int, state: str, message: str) -> None:
        if state in (StreamState.ERROR.value, StreamState.STOPPED.value):
            # A camera failing stays local to its tile — it must not affect
            # the main stream badge or another camera's playback (section 49).
            self._live_grid.clear_camera(index)

    @Slot(str, str)
    def _on_stream_state(self, state: str, message: str) -> None:
        mapping = {
            StreamState.OPENING.value: ("Connecting", BadgeVariant.INFO),
            StreamState.RUNNING.value: ("Live", BadgeVariant.ACCENT),
            StreamState.STOPPED.value: ("Stopped", BadgeVariant.MUTED),
            StreamState.ERROR.value: ("Unavailable", BadgeVariant.DANGER),
        }
        text, variant = mapping.get(state, ("Unknown", BadgeVariant.MUTED))
        self._live_grid.set_stream_status(text, variant)
        self._event_ticker.push(f"Stream: {text.lower()}")

        if state == StreamState.ERROR.value:
            # The app stays usable when a source fails (section 49).
            self._live_grid.clear_camera(1)

    @Slot(str, str)
    def _on_model_state(self, status: str, message: str) -> None:
        mapping = {
            ModelStatus.READY.value: ("AI ready", BadgeVariant.ACCENT),
            ModelStatus.DEGRADED.value: ("AI degraded", BadgeVariant.WARNING),
            ModelStatus.UNAVAILABLE.value: ("AI unavailable", BadgeVariant.DANGER),
            ModelStatus.LOADING.value: ("AI loading", BadgeVariant.INFO),
        }
        text, variant = mapping.get(status, ("AI unknown", BadgeVariant.MUTED))
        self._model_badge.set_status(text, variant)
        self._model_badge.setToolTip(message)
        self._event_ticker.push(message or text)

        if self._vm.registry.resolved_device == "cpu":
            self._device_badge.setText("CPU detected — the app may run slowly")
            self._device_badge.setToolTip(
                "No GPU was found (or GPU use is disabled in config); AI analysis is "
                "running on the CPU instead, which is significantly slower."
            )
        else:
            self._device_badge.setText("")
            self._device_badge.setToolTip("")

        info = self._vm.registry.model_info()
        if info:
            names = " · ".join(entry["name"] for entry in info.values())
            self._model_info_badge.setText(f"Models: {names}")
            self._model_info_badge.setToolTip(
                "\n".join(f"{role}: {entry['name']} v{entry['version']}" for role, entry in info.items())
            )
        else:
            self._model_info_badge.setText("Models: —")
        self._help.refresh()

        if status in (ModelStatus.DEGRADED.value, ModelStatus.UNAVAILABLE.value) and message:
            self._analyzer.analysis_message.setText(message)

    @Slot(str, str)
    def _on_analysis_state(self, request_id: str, state: str) -> None:
        caption = _stage_caption(state)
        self._analyzer.analysis_stage.setText(caption)
        self._event_ticker.push(caption)
        if request_id:
            self._analyzer.request_label.setText(f"Request {request_id} · {state.lower()}")

    @Slot(bool)
    def _on_analysis_busy(self, busy: bool) -> None:
        # Deterministic: the card swap is bound to AnalysisRequest state, not
        # to a per-screen decision (theme.md section 6).
        review = self._analyzer.review
        if busy:
            self._analyzer.analysis_stack.setCurrentIndex(PAGE_BUSY)
            self._analyzer.analysis_badge.set_status("Analysing", BadgeVariant.INFO)
        elif review.has_candidates:
            self._analyzer.analysis_stack.setCurrentIndex(PAGE_REVIEW)
        else:
            self._analyzer.analysis_stack.setCurrentIndex(PAGE_IDLE)
        self._analyzer.analyze_button.setEnabled(not busy)

    @Slot(str, object)
    def _on_analysis_completed(self, request_id: str, result) -> None:
        session = self._vm.take_review_session()
        self._analyzer.review.set_session(session)

        count = len(result.candidates)
        if count:
            self._analyzer.analysis_badge.set_status(f"{count} candidates", BadgeVariant.ACCENT)
            self._analyzer.analysis_stack.setCurrentIndex(PAGE_REVIEW)
            self._event_ticker.push(f"Analysis complete — {count} candidate(s) found")
        else:
            self._analyzer.analysis_badge.set_status("No candidates", BadgeVariant.MUTED)
            self._analyzer.analysis_stack.setCurrentIndex(PAGE_IDLE)
            self._analyzer.analysis_message.setText(
                result.warnings[0]
                if result.warnings
                else "No ball-contact actions were found in the recent play."
            )
            self._event_ticker.push("Analysis complete — no candidates found")

        if result.warnings and count:
            self.statusBar().showMessage(result.warnings[0], 8000)

    @Slot(str, str)
    def _on_analysis_failed(self, request_id: str, message: str) -> None:
        self._analyzer.analysis_badge.set_status("Failed", BadgeVariant.DANGER)
        self._analyzer.analysis_message.setText(message)
        self._event_ticker.push(f"Analysis failed — {message}")
        # Live video keeps running; only the analysis panel reports the problem.
        if not self._analyzer.review.has_candidates:
            self._analyzer.analysis_stack.setCurrentIndex(PAGE_IDLE)

    def _on_confirmed(self, candidate_index: int, frame_id: int) -> None:
        session = self._analyzer.review._session  # noqa: SLF001 — sibling widget state
        candidate = session.current_candidate if session else None

        logger.info(
            "candidate_confirmed",
            request_id=session.request_id if session else None,
            candidate_index=candidate_index,
            frame_id=frame_id,
            action_type=candidate.action_type if candidate else None,
            frame_shift_from_ai=session.frame_offset_from_candidate if session else 0,
        )

        self._analyzer.analysis_badge.set_status("Confirmed", BadgeVariant.ACCENT)

        saved_path = self._save_confirmed_frame(session, candidate, frame_id)
        if saved_path is not None:
            message = f"Frame {frame_id} confirmed — saved to exports/{saved_path.name}"
        else:
            message = f"Frame {frame_id} confirmed, but saving to disk failed — see logs."
        self.statusBar().showMessage(message, 8000)
        self._event_ticker.push(message)

    def _save_confirmed_frame(self, session, candidate, frame_id: int):
        """Write the confirmed frame to disk as a JPEG so the operator has
        the exact evidence frame, not just an on-screen toast that vanishes
        (persistence.exports_directory in config/default.yaml)."""
        if session is None:
            return None

        current = session.current_frame()
        if current is None:
            logger.warning("confirmed_frame_unavailable", frame_id=frame_id)
            return None
        current_frame_id, image = current

        exports_dir = resolve(self._settings.persistence.exports_directory)
        try:
            exports_dir.mkdir(parents=True, exist_ok=True)

            request_id = _safe_filename_part(session.request_id) or "request"
            action_type = _safe_filename_part(candidate.action_type) if candidate else "frame"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"{stamp}_{request_id}_frame{current_frame_id}_{action_type}.jpg"
            path = exports_dir / filename

            ok = cv2.imwrite(str(path), image)
            if not ok:
                logger.warning("confirmed_frame_write_failed", path=str(path))
                return None

            logger.info("confirmed_frame_saved", path=str(path), frame_id=current_frame_id)
            return path
        except OSError as exc:
            logger.warning("confirmed_frame_save_error", error=str(exc), directory=str(exports_dir))
            return None

    # -- lifecycle ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self._vm.shutdown()
        super().closeEvent(event)


def _stage_caption(state: str) -> str:
    return {
        "QUEUED": "Queued…",
        "PREPARING": "Preparing clip…",
        "SPOTTING_ACTIONS": "Spotting actions…",
        "REFINING": "Refining contact frame…",
        "RANKING": "Ranking candidates…",
    }.get(state, "Analysing recent play…")


def _safe_filename_part(value: str | None) -> str:
    """Strip a value down to characters safe in a filename on every OS."""
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
