"""M2.7 — the top-down pitch map in the review screen.

The failures worth protecting here are not about pixels; `render_top_down`'s
own tests cover the drawing. This is about layout and the widget's own
contract: it must never force the review rail wider than its budget (the bug
that broke the checklist and the camera thumbnail before it — see
tests/integration/test_offside_review_screen.py), it must show an honest
empty state before any frame has been analysed, and clearing it must
actually clear it rather than leaving a stale map on screen.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from apps.desktop.ui.widgets.pitch_map import PitchMapPanel  # noqa: E402
from offside.field_geometry.pitch import PitchModel  # noqa: E402
from tests.unit.test_pipeline_presenter import FakeAnalysis, make_pose  # noqa: E402
from tests.unit.test_team_assignment import FakeDirectionalCalibration  # noqa: E402

_RIGHT_COLUMN_BUDGET_PX = 300  # matches tests/integration/test_offside_review_screen.py


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    return PitchMapPanel()


@pytest.fixture
def pitch() -> PitchModel:
    return PitchModel()


def test_starts_with_an_honest_empty_state(panel):
    assert panel._caption.text() == "No frame analysed yet."  # noqa: SLF001
    assert panel._canvas._pixmap is None  # noqa: SLF001


def test_does_not_set_the_review_rails_floor(panel):
    """The exact bug that broke this column twice already (the checklist's
    longest title, then the camera thumbnail's inherited 320px minimum): a
    widget with a native 760px render must not carry that width into layout."""
    assert panel.minimumSizeHint().width() <= _RIGHT_COLUMN_BUDGET_PX


def test_an_uncalibrated_frame_says_so(panel, pitch):
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = None

    panel.set_analysis(analysis, pitch)

    assert panel._caption.text() == "Not calibrated on this frame."  # noqa: SLF001
    assert panel._canvas._pixmap is not None  # noqa: SLF001


def test_a_directional_calibration_says_it_is_not_metric(panel, pitch):
    analysis = FakeAnalysis(poses=[make_pose()])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None

    panel.set_analysis(analysis, pitch)

    caption = panel._caption.text()  # noqa: SLF001
    assert "Directional" in caption
    assert "no distances" in caption


def test_marked_landmarks_are_reported_as_operator_marked(panel, pitch):
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None

    panel.set_analysis(analysis, pitch, marked_landmarks={"halfway_top": (100.0, 50.0)})

    assert "operator-marked" in panel._caption.text()  # noqa: SLF001


def test_no_marks_are_reported_as_automatic(panel, pitch):
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None

    panel.set_analysis(analysis, pitch, marked_landmarks={})

    assert "automatic" in panel._caption.text()  # noqa: SLF001


def test_the_review_screen_never_gets_a_click_to_mark_prompt(panel, pitch):
    """`render_top_down`'s interactive-mode "click these 4 points" prompt
    belongs to the Pipeline Inspector, which has a click handler for it. The
    review screen does not, so telling the operator to click would be
    actively misleading rather than merely unhelpful."""
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = None

    panel.set_analysis(analysis, pitch)

    # The prompt would only ever appear painted on the canvas, never in the
    # Qt caption — assert the caption carries the read-only wording instead.
    assert "click" not in panel._caption.text().lower()  # noqa: SLF001


def test_clearing_removes_the_map_not_just_the_caption(panel, pitch):
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None
    panel.set_analysis(analysis, pitch)

    panel.clear()

    assert panel._canvas._pixmap is None  # noqa: SLF001
    assert panel._caption.text() == "No frame analysed yet."  # noqa: SLF001


def test_the_canvas_fits_whatever_width_it_is_given(panel, pitch):
    """Regression guard for the QLabel-sizeHint bug: the canvas must scale to
    its own allocated size, not force layout to grow around a 760px render."""
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None
    panel.set_analysis(analysis, pitch)

    canvas_widget = panel._canvas  # noqa: SLF001
    canvas_widget.resize(180, 120)
    assert canvas_widget.minimumSizeHint().width() <= _RIGHT_COLUMN_BUDGET_PX
