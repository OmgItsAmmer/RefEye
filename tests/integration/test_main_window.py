"""UI smoke tests: signal wiring and the analyze hotkey path.

These do not start real video ingest — they drive the viewmodel's signals
directly, which is the same contract the background workers use.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402

from apps.desktop.ui.main_window import MainWindow  # noqa: E402
from apps.desktop.ui.theme.stylesheet import build_stylesheet  # noqa: E402
from apps.desktop.viewmodels.main_viewmodel import MainViewModel  # noqa: E402
from analysis.request_manager.states import RequestState  # noqa: E402
from core.config.loader import load_settings  # noqa: E402
from core.domain.models import FramePacket  # noqa: E402


@pytest.fixture
def settings():
    return load_settings("config/default.yaml", local_path=None, apply_env=False)


@pytest.fixture
def window(qtbot, settings, qapp):
    qapp.setStyleSheet(build_stylesheet())
    viewmodel = MainViewModel(settings)
    win = MainWindow(settings, viewmodel)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win, viewmodel
    viewmodel.shutdown()


def make_frame(frame_id: int = 1) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id * 40,
        capture_timestamp_ms=frame_id * 40,
        width=64,
        height=36,
        source_id="test",
        image=np.full((36, 64, 3), 60, dtype=np.uint8),
    )


def test_window_opens_with_configured_title(window, settings):
    win, _ = window
    assert win.windowTitle() == settings.application.name


def test_frame_signal_renders_without_touching_workers(window, qtbot):
    """The UI reacts to a signal; it never polls worker internals."""
    win, vm = window
    vm.frame_ready.emit(make_frame())
    qtbot.wait(50)

    assert win._live_stack.currentIndex() == 1  # video page, not the loader


def test_stream_error_keeps_the_app_alive(window, qtbot):
    win, vm = window
    vm.stream_state_changed.emit("error", "Video file not found: nope.mp4")
    qtbot.wait(50)

    assert win.isVisible()
    assert "not found" in win._stream_loader_caption.text()


def test_analysis_busy_shows_the_card_loader(window, qtbot):
    """theme.md: the card swap is bound to AnalysisRequest state, deterministically."""
    win, vm = window

    vm.analysis_busy_changed.emit(True)
    qtbot.wait(50)
    assert win._analysis_stack.currentIndex() == 1
    assert not win._analyze_button.isEnabled()

    vm.analysis_busy_changed.emit(False)
    qtbot.wait(50)
    assert win._analysis_stack.currentIndex() == 0
    assert win._analyze_button.isEnabled()


def test_analysis_failure_shows_an_operator_friendly_message(window, qtbot):
    win, vm = window
    vm.analysis_failed.emit("abc123", "AI analysis could not complete. Please retry.")
    qtbot.wait(50)

    message = win._analysis_message.text()
    assert "could not complete" in message
    # No stack traces or technical jargon leak into the UI.
    assert "Traceback" not in message


def test_trigger_without_buffered_video_explains_why(window, qtbot):
    """The operator gets a clear reason, never a crash or a silent no-op."""
    win, vm = window

    failures = []
    vm.analysis_failed.connect(lambda rid, msg: failures.append(msg))
    qtbot.keyClick(win, Qt.Key.Key_F8)
    qtbot.wait(50)

    assert failures and "No video buffered" in failures[0]


def test_trigger_before_models_are_ready_explains_why(window, qtbot):
    """AI still loading is a normal state, not a failure to hide."""
    win, vm = window
    vm.video._decoded_buffer.append(make_frame(10))

    failures = []
    vm.analysis_failed.connect(lambda rid, msg: failures.append(msg))
    qtbot.keyClick(win, Qt.Key.Key_F8)
    qtbot.wait(50)

    assert failures
    assert "AI" in failures[0] or "loading" in failures[0].lower()
    # Nothing was queued, so no request is left dangling in a live state.
    assert vm.requests.latest() is None


def test_analyze_hotkey_creates_a_request_with_an_id(window, qtbot):
    """Exit criterion: the hotkey fires and produces a tracked request id."""
    win, vm = window

    # The fixture detector + kinematic spotter load instantly and need no GPU.
    vm.registry.load_all()
    assert vm.registry.is_ready

    vm.video._decoded_buffer.append(make_frame(10))
    qtbot.keyClick(win, Qt.Key.Key_F8)
    qtbot.wait(150)

    latest = vm.requests.latest()
    assert latest is not None
    assert latest.request_id
    assert latest.request.trigger_source == "shortcut"


def test_rebinding_the_key_in_config_rebinds_the_ui(qtbot, qapp, settings):
    """Change the config, and the whole UI follows — no hard-coded keys.

    Activation itself is covered by the F8 test above; Qt's offscreen platform
    does not deliver synthetic modifier chords to QShortcut, so this test
    verifies the binding and the advertised label rather than pretending to
    press Ctrl+Shift+A.
    """
    settings = settings.model_copy(deep=True)
    settings.shortcuts.analyze = "Ctrl+Shift+A"

    vm = MainViewModel(settings)
    win = MainWindow(settings, vm)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)

    assert win._shortcuts.key_for("analyze") == "Ctrl+Shift+A"
    bound = {b.action: b.key for b in win._shortcuts.bindings()}
    assert bound["analyze"] == "Ctrl+Shift+A"

    # Every surface that advertises the key reads it from config.
    assert "Ctrl+Shift+A" in win._analyze_button.text()
    assert "Ctrl+Shift+A" in win._analysis_message.text()
    assert "Ctrl+Shift+A" in win.statusBar().currentMessage()

    vm.shutdown()


def test_request_walks_the_full_state_machine(window, qtbot):
    """A trigger runs the real pipeline and reaches COMPLETED via every stage."""
    win, vm = window
    vm.registry.load_all()

    # Not enough features to spot anything — the point here is the lifecycle,
    # and "found nothing" must still complete rather than report a failure.
    for i in range(1, 40):
        vm.video._decoded_buffer.append(make_frame(i))

    completed = []
    vm.analysis_completed.connect(lambda rid, res: completed.append(res))

    vm._scheduler.start()  # model work is serialised through the scheduler
    vm._load_models()      # installs the pipeline on the request manager
    vm.requests.start()
    vm.trigger_analysis("test")

    qtbot.waitUntil(lambda: len(completed) == 1, timeout=20000)

    tracked = vm.requests.latest()
    assert tracked.state == RequestState.COMPLETED
    assert tracked.history[:2] == [RequestState.CREATED, RequestState.QUEUED]
    for stage in (
        RequestState.PREPARING,
        RequestState.SPOTTING_ACTIONS,
        RequestState.REFINING,
        RequestState.RANKING,
    ):
        assert stage in tracked.history
