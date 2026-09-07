"""M2.7 — confirming a frame in the real window actually triggers M2.

Everything else about this wiring (the runner's threading contract, the
checklist panel, the review panel) is covered elsewhere in isolation. What is
only visible with the real `MainWindow` wired up is the join at the top: does
confirming a candidate frame in the M1 review flow actually reach
`MainViewModel.check_offside`, and does a result that lands later actually
reach the screen. This is the test that would have caught the panel being
built with nothing behind it, which is exactly the gap the user found before
this file existed.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QApplication  # noqa: E402

from analysis.results.review_session import ReviewSession, ReviewStrip  # noqa: E402
from apps.desktop.ui.main_window import MainWindow  # noqa: E402
from apps.desktop.viewmodels.main_viewmodel import MainViewModel  # noqa: E402
from core.config.loader import load_settings  # noqa: E402
from core.domain.models import AnalysisResult, RefinedCandidate  # noqa: E402

_ok, _ENCODED_FRAME = cv2.imencode(".jpg", np.zeros((360, 640, 3), dtype=np.uint8))
_ENCODED_FRAME = _ENCODED_FRAME.tobytes()


def _one_candidate_session(frame_id: int) -> ReviewSession:
    """A real (but minimal) `ReviewSession` — one candidate, one frame — built
    the same way `tests/unit/test_review_session.py` builds one. A hand-rolled
    stand-in kept breaking every time the review panel's redraw touched one
    more field of the session it renders from; the real class doesn't have
    that problem."""
    candidate = RefinedCandidate(
        candidate_id="c0",
        action_type="pass",
        original_frame_id=frame_id,
        refined_frame_id=frame_id,
        final_score=0.9,
        model_score=0.9,
        contact_score=0.9,
        trajectory_score=0.9,
        proximity_score=0.9,
        temporal_score=0.9,
    )
    result = AnalysisResult(
        request_id="test-request",
        candidates=[candidate],
        selected_candidate_index=0,
        status="completed",
    )
    strip = ReviewStrip({frame_id: _ENCODED_FRAME}, center_frame_id=frame_id)
    return ReviewSession(request_id="test-request", result=result, strips=[strip])


@pytest.fixture
def settings():
    return load_settings("config/default.yaml", local_path=None, apply_env=False)


@pytest.fixture
def window(qtbot, settings, qapp):
    viewmodel = MainViewModel(settings)
    win = MainWindow(settings, viewmodel)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    # The app opens on Live Grid; the review panel (and the offside panel
    # beneath it) only exists on the Analyzer screen, and a widget on a
    # QStackedWidget page that isn't current reports isVisible() == False
    # regardless of its own setVisible(True) — confirming a frame from the
    # wrong screen would be an odd thing to test anyway.
    win._go_to_screen("analyzer")
    yield win, viewmodel
    viewmodel.shutdown()


def wait_for(condition, timeout=8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def _confirm(win, frame_id: int, monkeypatch) -> None:
    # set_session (not a raw attribute write) is what the real app calls once
    # analysis completes — it also builds the candidate-row widgets the panel
    # redraws whenever a verdict lands.
    win._analyzer.review.set_session(_one_candidate_session(frame_id))
    monkeypatch.setattr(win, "_save_confirmed_frame", lambda *a, **k: None)
    win._analyzer.review.confirmed.emit(0, frame_id)


def test_confirming_a_frame_triggers_the_offside_pipeline(window, monkeypatch):
    win, viewmodel = window
    calls = []
    monkeypatch.setattr(viewmodel, "check_offside", lambda fid, img: calls.append(fid))

    _confirm(win, 77, monkeypatch)

    assert calls == [77]


def test_the_progress_panel_appears_the_instant_a_run_starts(window, monkeypatch):
    win, _viewmodel = window
    _confirm(win, 5, monkeypatch)
    assert wait_for(lambda: win._analyzer.offside_progress.isVisible())


def test_a_completed_run_reaches_the_review_panel_and_the_frame(window, monkeypatch):
    """The failure this test exists for: the checklist finishes, and the
    verdict panel and the line on the video never actually update."""
    win, _viewmodel = window
    _confirm(win, 9, monkeypatch)

    assert wait_for(lambda: win._analyzer.offside.explanation is not None, timeout=15.0)
    assert win._analyzer.review._offside_decision is not None  # noqa: SLF001
    assert "COMPLETE" in win._analyzer.offside_progress._caption.text()  # noqa: SLF001


def test_a_completed_run_also_reaches_the_pitch_map(window, monkeypatch):
    """The map is driven by a separate call (`set_pitch_analysis`) from the
    verdict panel (`set_offside_decision`) — this is the one test that would
    catch `MainWindow` wiring one and forgetting the other."""
    win, _viewmodel = window
    _confirm(win, 11, monkeypatch)

    assert wait_for(
        lambda: "COMPLETE" in win._analyzer.offside_progress._caption.text(),  # noqa: SLF001
        timeout=15.0,
    )
    assert win._analyzer.pitch_map._canvas._pixmap is not None  # noqa: SLF001
