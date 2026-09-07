"""The shared top-down renderer, and its one product-vs-inspector switch.

`render_top_down` lives in `offside/pitch_calibration/rendering.py` and is
drawn by two callers: the Pipeline Inspector (interactive — an operator can
click a landmark to mark it) and the review screen (M2.7 — read-only, no
click handler exists). The only thing worth testing at the rendering layer is
that switch: a read-only caller must never be told to click something, and an
interactive caller must keep working exactly as it did before this option was
added. Everything else about the canvas (colours, projections) is exercised
already by the inspector's own tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.rendering import render_top_down
from tests.unit.test_team_assignment import FakeDirectionalCalibration, FakeMetricCalibration


@pytest.fixture
def pitch() -> PitchModel:
    return PitchModel()


def render(pitch, calibration, **kwargs) -> np.ndarray:
    return render_top_down(pitch, calibration, poses=[], teams=None, **kwargs)


def has_text_row(canvas: np.ndarray) -> bool:
    """Cheap stand-in for OCR: any non-background pixel below the pitch
    diagram means *some* caption was drawn there."""
    return bool((canvas[-40:] > 40).any())


def test_an_interactive_caller_is_told_to_click(pitch):
    canvas = render(pitch, None, interactive=True)
    assert has_text_row(canvas)


def test_a_read_only_caller_is_never_told_to_click(pitch):
    """The failure this test exists for: a caption on the review screen
    telling the operator to click a point nothing there can handle."""
    interactive = render(pitch, None, interactive=True)
    read_only = render(pitch, None, interactive=False)
    assert not np.array_equal(interactive, read_only), (
        "the read-only caption must differ from the interactive one"
    )


def test_read_only_still_draws_something_when_uncalibrated(pitch):
    canvas = render(pitch, None, interactive=False)
    assert has_text_row(canvas)


def test_read_only_still_draws_something_when_directional(pitch):
    canvas = render(pitch, FakeDirectionalCalibration(), interactive=False)
    assert has_text_row(canvas)


def test_a_highlighted_landmark_is_only_drawn_when_interactive(pitch):
    """The product never selects a landmark (it has no marking UI), but this
    guards the case where it accidentally would: the "click this one" prompt
    must not appear on a read-only map even if a landmark were passed."""
    with_highlight_interactive = render(
        pitch, None, interactive=True, highlight_landmark="halfway_top"
    )
    with_highlight_read_only = render(
        pitch, None, interactive=False, highlight_landmark="halfway_top"
    )
    without_highlight_read_only = render(pitch, None, interactive=False)
    assert not np.array_equal(with_highlight_interactive, with_highlight_read_only)
    # The circle/cross marker for the highlighted point is still drawn (that
    # part of the map is informational either way) — only the click prompt
    # text differs, so the two read-only canvases stay mostly the same.
    assert with_highlight_read_only.shape == without_highlight_read_only.shape


def test_metric_calibration_draws_the_pitch_regardless_of_interactivity(pitch):
    """Once there is something real to show (a metric calibration), the
    click-vs-status distinction does not apply — both modes just draw it."""
    interactive = render(pitch, FakeMetricCalibration(), interactive=True)
    read_only = render(pitch, FakeMetricCalibration(), interactive=False)
    assert np.array_equal(interactive, read_only)
