"""M2.7 — the offside verdict in the operator's review screen (redesigned).

Driven headlessly against real explanations from M2.6. The failures worth
catching here are not layout ones; they are the panel quietly saying something
the decision did not:

* a withheld call appearing as a call,
* a requirement row's colour disagreeing with its own percentage,
* the suggestion box appearing with nothing actionable in it (or staying
  hidden when there is something to say),
* an override that changes the text but not the line drawn on the frame,
* a stale verdict left on screen after the frame changes.

Each of those looks perfectly fine in a screenshot, which is exactly why they
are asserted here rather than eyeballed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from apps.desktop.ui.widgets.offside_review import (  # noqa: E402
    ConfidenceMeter,
    OffsideReviewPanel,
    band_caption,
)
from offside.decision_support.explainer import DecisionExplainer  # noqa: E402
from offside.decision_support.explanation import ConfidenceBand  # noqa: E402
from offside.offside_line.line import Verdict  # noqa: E402
from tests.unit.test_decision_support import (  # noqa: E402
    FakeCalibration,
    healthy,
    make_decision,
    make_teams,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    widget = OffsideReviewPanel()
    yield widget
    widget.deleteLater()


def explain_with(**overrides):
    inputs = healthy(**overrides)
    decision = inputs.pop("decision")
    return decision, DecisionExplainer().explain(decision, **inputs)


# -- the verdict word and confidence badge -----------------------------------


def test_an_empty_panel_says_so_rather_than_showing_a_stale_call(panel):
    panel.set_explanation(explain_with()[1])
    panel.clear()
    assert panel.explanation is None
    assert "No frame" in panel._headline.text()
    assert "NOT" not in panel._verdict_word.text()  # no leftover verdict text
    assert panel._suggestions.isVisibleTo(panel) is False


def test_a_published_verdict_reaches_the_word_and_the_headline(panel):
    _, explanation = explain_with()
    panel.set_explanation(explanation)

    assert "OFFSIDE" in panel._verdict_word.text().upper()
    assert panel._headline.text() == explanation.headline
    assert "0.30m" in panel._detail.text()


def test_a_withheld_verdict_is_never_shown_as_a_call(panel):
    """The failure this panel exists to prevent: the geometry read offside,
    the chain could not carry it, and the operator sees "offside"."""
    _, explanation = explain_with(calibration=FakeCalibration(confidence=0.12))
    panel.set_explanation(explanation)

    assert explanation.geometry_verdict is Verdict.OFFSIDE
    word = panel._verdict_word.text().upper()
    assert "OFFSIDE" not in word
    assert "NO CALL" in word
    assert "please judge" in panel._headline.text().lower()


def test_the_confidence_badge_shows_the_band_not_the_verdict(panel):
    """The verdict lives in the big word; the small badge now carries how
    sure the tool is, in the same colour vocabulary as everywhere else."""
    _, explanation = explain_with()
    panel.set_explanation(explanation)

    assert explanation.band is ConfidenceBand.HIGH
    assert panel._confidence_badge._label.text().upper() == "HIGH"


def test_the_confidence_value_is_a_percentage(panel):
    """Literal ask: 'percentage confidence we achieve', not a 0-1 decimal."""
    _, explanation = explain_with()
    panel.set_explanation(explanation)
    assert panel._confidence_value.text().endswith("%")
    assert "." not in panel._confidence_value.text()


# -- the requirements checklist ----------------------------------------------


def test_every_stage_gets_a_row_including_the_healthy_ones(panel):
    """A panel that lists only problems cannot be used to check there are
    none."""
    _, explanation = explain_with()
    panel.set_explanation(explanation)

    visible = [row for row in panel._rows if row.isVisibleTo(panel)]
    assert len(visible) == len(explanation.signals) == 5


def test_a_healthy_row_reads_good_with_a_percentage(panel):
    _, explanation = explain_with()
    panel.set_explanation(explanation)

    row = panel._rows[0]
    assert "GOOD" in row._score.text()
    assert "%" in row._score.text()


def test_a_weak_row_reads_weak_not_good(panel):
    _, explanation = explain_with(teams=make_teams(confidence=0.5))
    panel.set_explanation(explanation)

    team_index = next(
        i for i, s in enumerate(explanation.signals) if s.key == "teams"
    )
    row = panel._rows[team_index]
    assert "WEAK" in row._score.text()


def test_a_blocking_row_reads_poor_even_with_a_nonzero_score(panel):
    """`blocking` must dominate the score-based read — a stage that is
    missing outright is not merely "weak", regardless of what number it
    happens to carry."""
    _, explanation = explain_with(teams=make_teams(known=False))
    panel.set_explanation(explanation)

    team_index = next(
        i for i, s in enumerate(explanation.signals) if s.key == "teams"
    )
    row = panel._rows[team_index]
    assert "POOR" in row._score.text()


def test_a_stage_with_nothing_to_say_shows_n_a_not_zero(panel):
    """Zero reads as a stage that failed; this one simply had no opinion."""
    _, explanation = explain_with(identities=None)
    panel.set_explanation(explanation)

    scores = [row._score.text() for row in panel._rows[: len(explanation.signals)]]
    assert "N/A" in scores
    assert panel.explanation.confidence > 0.5


def test_rows_are_reused_rather_than_stacked_up_between_frames(panel):
    """Frame-by-frame stepping refreshes this panel constantly; rebuilding
    rows each time would leak widgets for the length of a review session."""
    for _ in range(5):
        panel.set_explanation(explain_with()[1])
    assert len(panel._rows) == 5


def test_the_row_tooltip_carries_the_full_reason(panel):
    """The percentage and the colour are the at-a-glance read; the full
    sentence must still be reachable for whoever wants it."""
    _, explanation = explain_with(teams=make_teams(confidence=0.62))
    panel.set_explanation(explanation)

    team_index = next(
        i for i, s in enumerate(explanation.signals) if s.key == "teams"
    )
    row = panel._rows[team_index]
    assert "kits were told apart" in row.toolTip()


# -- the suggestion box -------------------------------------------------------


def test_the_suggestion_box_is_hidden_when_nothing_is_weak(panel):
    _, explanation = explain_with()
    panel.set_explanation(explanation)

    assert explanation.band is ConfidenceBand.HIGH
    assert panel._suggestions.isVisibleTo(panel) is False


def test_the_suggestion_box_appears_when_something_is_weak(panel):
    _, explanation = explain_with(
        calibration=FakeCalibration(confidence=0.9, is_metric=False)
    )
    panel.set_explanation(explanation)

    assert panel._suggestions.isVisibleTo(panel) is True
    texts = " ".join(
        row.text() for row in panel._suggestions._rows if row.isVisibleTo(panel)
    )
    assert "mark four" in texts


def test_the_suggestion_box_is_cleared_on_an_operator_call(panel):
    """The operator's own call has nothing left to suggest — the tool's
    advice was for reaching a call the operator has now made by hand."""
    panel.set_explanation(explain_with(teams=make_teams(confidence=0.5))[1])
    panel.set_override(Verdict.OFFSIDE)
    assert panel._suggestions.isVisibleTo(panel) is False


# -- the override -----------------------------------------------------------


def test_an_override_replaces_the_verdict_the_tool_offered(panel):
    _, explanation = explain_with()
    panel.set_explanation(explanation)
    panel.set_override(Verdict.ONSIDE)

    assert panel.explanation.verdict is Verdict.ONSIDE
    assert panel.explanation.is_operator_call


def test_an_override_says_it_is_the_operators_call_not_a_score(panel):
    """Showing a confidence next to a human's decision would suggest the tool
    endorsed it."""
    panel.set_explanation(explain_with()[1])
    panel.set_override(Verdict.OFFSIDE)
    assert panel._confidence_badge._label.text().upper() == "YOUR CALL"
    assert panel._confidence_value.text() == ""


def test_clearing_an_override_hands_the_call_back_to_the_tool(panel):
    _, explanation = explain_with()
    panel.set_explanation(explanation)
    panel.set_override(Verdict.ONSIDE)
    panel.set_override(None)

    assert panel.explanation is explanation
    assert not panel.explanation.is_operator_call


def test_an_override_is_announced_so_the_frame_overlay_can_follow(panel):
    seen = []
    panel.override_changed.connect(seen.append)
    panel.set_override(Verdict.OFFSIDE)
    panel.set_override(None)
    assert seen == [Verdict.OFFSIDE, None]


def test_the_tools_own_reading_survives_an_override(panel):
    """Overruling the tool must not delete what it said — the operator may
    want to compare, and whoever reviews the decision later will certainly
    want to see what was overruled."""
    _, explanation = explain_with(teams=make_teams(confidence=0.62))
    panel.set_explanation(explanation)
    panel.set_override(Verdict.ONSIDE)

    assert "Offside position" in panel._detail.text()
    rows = [row._score.text() for row in panel._rows if row.isVisibleTo(panel)]
    assert len(rows) == len(explanation.signals)


# -- the frame overlay ------------------------------------------------------


def test_the_overlay_is_drawn_on_a_copy_of_the_frame(panel):
    """The review buffer's frame is shared with every other view; drawing
    into it would burn the line onto the frame everywhere it appears."""
    decision, explanation = explain_with()
    panel.set_explanation(explanation)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    painted = panel.render_overlay(frame, decision)

    assert painted is not frame
    assert frame.max() == 0, "the caller's frame was drawn into"
    assert painted.max() > 0, "nothing was drawn"


def test_the_overlay_follows_the_override(panel):
    """The panel and the picture must never disagree about what the call is."""
    decision, explanation = explain_with()
    panel.set_explanation(explanation)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    as_offside = panel.render_overlay(frame, decision)
    panel.set_override(Verdict.ONSIDE)
    as_onside = panel.render_overlay(frame, decision)

    assert not np.array_equal(as_offside, as_onside)


def test_no_decision_means_no_line(panel):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert panel.render_overlay(frame, None) is frame


# -- the meter --------------------------------------------------------------


def test_the_meter_marks_the_floor_a_withheld_call_fell_short_of(app):
    """The number alone does not tell an operator whether 0.38 was nearly
    enough, which is the first thing they ask when a call is withheld."""
    meter = ConfidenceMeter()
    meter.resize(100, 6)
    meter.set_value(0.38, ConfidenceBand.LOW, floor=0.4)
    assert meter._floor == pytest.approx(0.4)
    assert meter._value == pytest.approx(0.38)


def test_the_meter_clamps_rather_than_painting_outside_itself(app):
    meter = ConfidenceMeter()
    meter.set_value(4.2, ConfidenceBand.HIGH, floor=-1.0)
    assert meter._value == 1.0 and meter._floor == 0.0


# -- the one-line form ------------------------------------------------------


def test_the_caption_carries_the_band_for_places_too_narrow_for_the_panel():
    _, explanation = explain_with(decision=make_decision(confidence=0.5))
    caption = band_caption(explanation)
    assert explanation.headline in caption
    assert "check the frame" in caption


def test_the_caption_of_nothing_is_not_an_empty_string():
    assert band_caption(None) == "No offside decision for this frame"
