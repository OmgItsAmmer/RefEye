"""The manual pitch-marking dialog — the shipped app's fallback when both
automatic paths (directional line detection, the auto-landmark detector)
come up short on a real frame.

Driven headlessly, clicking the frame widget directly rather than through
real mouse events (Qt's synthetic click plumbing is what's under test
elsewhere; here the thing worth protecting is the dialog's own state
machine — does a click actually record the *selected* landmark, does Done
stay disabled below the mathematical minimum, does re-opening on a frame
that already has marks show them).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from apps.desktop.ui.widgets.landmark_marking_dialog import LandmarkMarkingDialog  # noqa: E402
from offside.field_geometry.pitch import PitchModel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def pitch() -> PitchModel:
    return PitchModel()


@pytest.fixture
def dialog(app, image, pitch):
    widget = LandmarkMarkingDialog(image, pitch)
    yield widget
    widget.deleteLater()


def _click(dialog, x: float, y: float) -> None:
    dialog._on_frame_clicked(x, y)  # noqa: SLF001 — bypassing real Qt mouse events on purpose


def test_starts_with_nothing_marked(dialog):
    assert dialog.correspondences() == []
    assert not dialog._done_button.isEnabled()  # noqa: SLF001


def test_a_click_marks_the_currently_selected_landmark(dialog):
    landmark = dialog._selected_landmark()  # noqa: SLF001
    _click(dialog, 120.0, 340.0)

    correspondences = dialog.correspondences()
    assert len(correspondences) == 1
    assert correspondences[0].landmark == landmark
    assert correspondences[0].image_xy == (120.0, 340.0)


def test_the_pitch_coordinate_matches_the_pitch_models_own(dialog, pitch):
    landmark = dialog._selected_landmark()  # noqa: SLF001
    _click(dialog, 50.0, 60.0)
    assert dialog.correspondences()[0].pitch_xy == pitch.landmark(landmark)


def test_marking_advances_to_the_next_unmarked_landmark(dialog):
    first = dialog._selected_landmark()  # noqa: SLF001
    _click(dialog, 10.0, 10.0)
    second = dialog._selected_landmark()  # noqa: SLF001
    assert second != first, "selection did not move on after marking"


def test_re_marking_the_same_landmark_replaces_it_not_duplicates(dialog):
    landmark = dialog._selected_landmark()  # noqa: SLF001
    _click(dialog, 10.0, 10.0)
    dialog._picker.setCurrentIndex(  # noqa: SLF001
        dialog._picker.findData(landmark)  # noqa: SLF001
    )
    _click(dialog, 999.0, 999.0)

    matches = [c for c in dialog.correspondences() if c.landmark == landmark]
    assert len(matches) == 1
    assert matches[0].image_xy == (999.0, 999.0)


def test_done_stays_disabled_below_the_mathematical_minimum(dialog):
    for offset in range(3):
        _click(dialog, 10.0 + offset, 10.0)
    assert len(dialog.correspondences()) == 3
    assert not dialog._done_button.isEnabled()  # noqa: SLF001


def test_done_enables_at_four_marks(dialog):
    for offset in range(4):
        _click(dialog, 10.0 + offset, 10.0)
    assert len(dialog.correspondences()) == 4
    assert dialog._done_button.isEnabled()  # noqa: SLF001


def test_opening_with_existing_marks_shows_them_immediately(app, image, pitch):
    existing = {"corner_left_top": (5.0, 6.0), "corner_right_top": (600.0, 6.0)}
    dialog = LandmarkMarkingDialog(image, pitch, initial_marks=existing)
    try:
        result = {c.landmark: c.image_xy for c in dialog.correspondences()}
        assert result == existing
        assert dialog._marked_list.count() == 2  # noqa: SLF001
    finally:
        dialog.deleteLater()


def test_cancel_is_reachable_without_marking_anything(dialog):
    """The dialog must never trap the operator — closing without marking a
    single point is always a valid outcome, matching this codebase's
    standing rule that overrides and exits are never blocked."""
    assert dialog.reject() is None  # does not raise
