"""M2.7 — the per-stage checklist the operator watches while offside runs.

Driven headlessly with synthetic `StageReport`s standing in for what
`OffsideRunner` forwards from the real pipeline. What is worth protecting: the
panel stays hidden until there is something to show, rows fill in in pipeline
order rather than only on completion, a stage the panel doesn't recognise
does not crash it, and a failure clears the "running" guess it made rather
than leaving a row lying about still being in progress.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from apps.desktop.ui.widgets.offside_progress import (  # noqa: E402
    PIPELINE_STAGES,
    OffsideProgressPanel,
)
from offside.pipeline import StageReport, StageState  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    return OffsideProgressPanel()


def report(key: str, state: StageState, summary: str = "ok") -> StageReport:
    _key, phase, title, _short = next(s for s in PIPELINE_STAGES if s[0] == key)
    return StageReport(key=key, phase=phase, title=title, state=state, summary=summary)


def test_the_panel_is_hidden_until_a_run_starts(panel):
    assert not panel.isVisibleTo(panel.parent() or panel)
    assert panel.isVisible() is False


def test_starting_a_run_shows_the_panel_and_resets_every_row(panel):
    panel.start(90)
    assert panel.isVisible()
    assert "90" in panel._caption.text()
    for row in panel._rows.values():
        assert row._detail.text() in ("waiting", "running")


def test_the_first_stage_is_marked_running_immediately(panel):
    """A checklist that shows every row as merely "waiting" the instant it
    appears looks identical to a tool stuck before it started."""
    panel.start(1)
    first_key = PIPELINE_STAGES[0][0]
    assert panel._rows[first_key]._detail.text() == "running"


def test_a_finished_stage_marks_the_next_one_running(panel):
    panel.start(1)
    panel.report_stage(report("detection", StageState.OK))

    assert panel._rows["detection"]._detail.text() == "done"
    assert panel._rows["body_keypoints"]._detail.text() == "running"


def test_stages_fill_in_one_at_a_time_in_pipeline_order(panel):
    panel.start(1)
    keys = [key for key, *_rest in PIPELINE_STAGES]

    for key in keys:
        panel.report_stage(report(key, StageState.OK))
        done = [k for k, row in panel._rows.items() if row._detail.text() == "done"]
        assert set(done) == set(keys[: keys.index(key) + 1])


def test_a_degraded_stage_is_shown_as_degraded_not_folded_into_done(panel):
    panel.start(1)
    panel.report_stage(report("pitch_calibration", StageState.DEGRADED, "directional only"))
    assert panel._rows["pitch_calibration"]._detail.text() == "done, degraded"
    assert panel._rows["pitch_calibration"].toolTip() == "Pitch calibration\ndirectional only"


def test_an_unavailable_stage_is_visibly_different_from_a_healthy_one(panel):
    panel.start(1)
    panel.report_stage(report("team_assignment", StageState.UNAVAILABLE, "no players"))
    assert panel._rows["team_assignment"]._detail.text() == "unavailable"


def test_an_unrecognised_stage_key_does_not_crash_the_panel(panel):
    """Defensive against a future milestone adding a stage this panel has not
    been taught about yet — it should be ignored, not fatal."""
    panel.start(1)
    panel.report_stage(
        StageReport(key="m3_something", phase="M3", title="Future", state=StageState.OK, summary="")
    )  # must not raise


def test_finishing_updates_the_caption(panel):
    panel.start(1)
    panel.finish()
    assert "COMPLETE" in panel._caption.text()


def test_a_failure_shows_the_message_and_clears_stuck_running_rows(panel):
    panel.start(1)
    panel.report_stage(report("detection", StageState.OK))
    # body_keypoints is now "running" — the guess the panel made — and then
    # the pipeline crashes before it ever reports back.
    panel.fail("The offside pipeline could not run on this frame.")

    assert panel._error.isVisible()
    assert "could not run" in panel._error.text()
    assert panel._rows["body_keypoints"]._detail.text() == "waiting"
    # A stage that had already finished keeps showing what actually happened.
    assert panel._rows["detection"]._detail.text() == "done"


def test_a_second_run_clears_the_previous_ones_state(panel):
    panel.start(1)
    panel.report_stage(report("detection", StageState.UNAVAILABLE, "no detector"))

    panel.start(2)
    assert panel._rows["detection"]._detail.text() == "running"
    assert "2" in panel._caption.text()
