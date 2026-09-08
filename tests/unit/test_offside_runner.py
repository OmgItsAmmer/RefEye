"""M2.7 wiring — the offside pipeline actually running in the shipped app.

Before this, `OffsidePipeline` was only ever driven by the standalone
debugger and the test suite. `OffsideRunner` is what the app calls once a
frame is confirmed, off the UI thread. These tests use a registry with no
models loaded — every stage reports UNAVAILABLE — because the behaviour worth
protecting here is entirely about the *runner*, not about what any one stage
concludes: that a run reports every stage exactly once and in pipeline order,
that a stale result never overwrites a newer confirm, and that a crash in the
pipeline reaches the operator as a message rather than as a dead thread.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from apps.desktop.viewmodels.offside_runner import OffsideRunner  # noqa: E402
from core.config.loader import load_settings  # noqa: E402
from offside.pipeline import StageState  # noqa: E402
from tools.pipeline_debugger.pipeline import FrameAnalysis  # noqa: E402


class _EmptyRegistry:
    """No models loaded — every stage should still report, just UNAVAILABLE."""

    def get_detector(self):
        return None

    def get_pose_estimator(self):
        return None


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings():
    return load_settings("config/default.yaml", local_path=None, apply_env=False)


@pytest.fixture
def runner(app, settings):
    return OffsideRunner(settings, _EmptyRegistry())


def blank_frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


def wait_for(condition, timeout=20.0) -> bool:
    """Pump the Qt event loop until `condition()` is true or time runs out.

    The pipeline runs on a background `threading.Thread`; its signals are
    only delivered once this thread's event loop gets to process them, so a
    plain `time.sleep` without pumping events would never see them arrive.

    20s, not 5s: each `runner` fixture builds a fresh `OffsidePipeline`, and
    the auto-landmark model it loads on its first pitch-calibration stage
    (`offside/pitch_calibration/auto_landmarks.py`) is a real checkpoint read
    from disk — a cold read comfortably exceeds 5s, well before this test's
    own logic (which uses no models at all, `_EmptyRegistry`) even runs.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


# -- the run itself -----------------------------------------------------


def test_a_run_reports_started_then_every_stage_then_completed(runner):
    events = []
    runner.started.connect(lambda frame_id: events.append(("started", frame_id)))
    runner.stage_progress.connect(lambda report: events.append(("stage", report.key)))
    runner.completed.connect(lambda analysis: events.append(("completed", analysis)))

    runner.analyse(42, blank_frame())
    assert wait_for(lambda: any(e[0] == "completed" for e in events))

    assert events[0] == ("started", 42)
    stage_keys = [e[1] for e in events if e[0] == "stage"]
    assert stage_keys == [
        "detection",
        "body_keypoints",
        "tracking",
        "pitch_calibration",
        "team_assignment",
        "offside_line",
        "decision_support",
    ]
    assert events[-1][0] == "completed"


def test_the_completed_analysis_carries_a_decision_and_an_explanation(runner):
    results = []
    runner.completed.connect(results.append)
    runner.analyse(1, blank_frame())
    assert wait_for(lambda: results)

    analysis: FrameAnalysis = results[0]
    assert analysis.offside is not None
    assert analysis.explanation is not None
    assert analysis.explanation.band.value == "none"  # no models, nothing to call


def test_every_stage_reports_unavailable_without_models(runner):
    stages = []
    runner.stage_progress.connect(stages.append)
    runner.completed.connect(lambda _analysis: None)
    runner.analyse(1, blank_frame())
    assert wait_for(lambda: len(stages) >= 7)

    assert all(report.state is StageState.UNAVAILABLE for report in stages)


# -- staleness ------------------------------------------------------------


def test_a_stale_result_never_overwrites_a_newer_confirm(runner):
    """The operator confirmed frame 5, then immediately confirmed frame 9
    before frame 5 finished. Frame 5's result must not land afterwards and
    silently replace what the operator is now looking at."""
    completed = []
    runner.completed.connect(lambda analysis: completed.append(analysis.frame_index))

    runner.analyse(5, blank_frame())
    runner.analyse(9, blank_frame())

    assert wait_for(lambda: len(completed) >= 1, timeout=5.0)
    time.sleep(0.2)
    QApplication.processEvents()

    assert 5 not in completed
    assert 9 in completed


# -- failure ----------------------------------------------------------------


def test_a_pipeline_crash_reaches_the_operator_not_a_dead_thread(runner, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner.pipeline, "analyse", explode)

    failures = []
    completions = []
    runner.failed.connect(lambda frame_id, message: failures.append((frame_id, message)))
    runner.completed.connect(completions.append)

    runner.analyse(3, blank_frame())
    assert wait_for(lambda: failures)

    assert failures[0][0] == 3
    assert failures[0][1]
    assert not completions
