"""M2.7 wiring: the analyzer screen, the panel, and the frame beneath it.

The panel's own behaviour is covered in `tests/unit/test_offside_review_panel.py`.
What is left to go wrong is the join between the two halves of the screen —
the verdict shown in the rail and the line drawn on the video. If those ever
disagree, the operator is looking at one call and reading another, which is a
worse failure than showing no line at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from apps.desktop.ui.screens.analyzer_screen import AnalyzerScreen  # noqa: E402
from apps.desktop.viewmodels.main_viewmodel import MainViewModel  # noqa: E402
from core.config.loader import load_settings  # noqa: E402
from offside.decision_support.explainer import DecisionExplainer  # noqa: E402
from offside.field_geometry.pitch import PitchModel  # noqa: E402
from offside.offside_line.line import Verdict  # noqa: E402
from tests.unit.test_decision_support import healthy  # noqa: E402
from tests.unit.test_pipeline_presenter import FakeAnalysis  # noqa: E402
from tests.unit.test_team_assignment import FakeDirectionalCalibration  # noqa: E402


@pytest.fixture
def settings():
    return load_settings("config/default.yaml", local_path=None, apply_env=False)


@pytest.fixture
def screen(qtbot, settings):
    viewmodel = MainViewModel(settings)
    widget = AnalyzerScreen(
        video_service=viewmodel.video,
        recent_window_seconds=settings.buffer.recent_window_seconds,
        analyze_shortcut_label=settings.shortcuts.analyze,
        feature_cache=viewmodel.feature_cache,
    )
    qtbot.addWidget(widget)
    yield widget
    viewmodel.shutdown()


def make_call():
    inputs = healthy()
    decision = inputs.pop("decision")
    return decision, DecisionExplainer().explain(decision, **inputs)


def test_the_screen_starts_with_no_offside_call(screen):
    assert screen.offside.explanation is None


def test_a_decision_reaches_both_the_panel_and_the_frame(screen):
    decision, explanation = make_call()
    screen.set_offside_decision(decision, explanation)

    assert screen.offside.explanation is explanation
    assert screen.review._offside_decision is decision
    assert screen.review._offside_explanation is explanation


def test_an_override_reaches_the_frame_not_only_the_panel(screen):
    """The failure this test exists for: the rail says onside, the line on the
    frame is still drawn in the colour of the tool's offside call."""
    decision, explanation = make_call()
    screen.set_offside_decision(decision, explanation)

    screen.offside.set_override(Verdict.ONSIDE)

    drawn = screen.review._offside_explanation
    assert drawn.verdict is Verdict.ONSIDE
    assert drawn.is_operator_call


def test_clearing_removes_the_line_as_well_as_the_verdict(screen):
    """A line left over from the previous candidate is worse than no line: it
    reads as a call about the frame currently on screen."""
    decision, explanation = make_call()
    screen.set_offside_decision(decision, explanation)
    screen.clear_offside_decision()

    assert screen.offside.explanation is None
    assert screen.review._offside_decision is None


def test_the_m1_review_screen_still_works_without_any_offside_analysis(screen):
    """M2 is additive. A screen that was never given an offside decision must
    behave exactly as it did in M1."""
    assert screen.review._offside_decision is None
    assert screen.review._with_offside_line("frame-sentinel") == "frame-sentinel"


# -- small-window layout -----------------------------------------------------
#
# The right rail is a QScrollArea over three stacked panels (camera/preview,
# alternatives, offside). Found on real use: a fixed-width row anywhere in
# that stack (a checklist label that doesn't wrap, a hardcoded button row)
# does not just look cramped on a small window — it sets a hard floor on the
# whole column's minimum width that the horizontal scrollbar policy then
# hides silently, at *any* window size, not only a small one. These assert a
# budget rather than a pixel-exact layout so the check survives copy edits.

_RIGHT_COLUMN_BUDGET_PX = 300  # comfortably inside the app's own minimum window


def test_the_offside_checklist_fits_the_review_rail(screen):
    for row in screen.offside_progress._rows.values():  # noqa: SLF001
        assert row.minimumSizeHint().width() <= _RIGHT_COLUMN_BUDGET_PX, (
            f"{row.toolTip()!r} is too wide for the review rail"
        )


def test_the_camera_preview_does_not_set_the_columns_floor(screen):
    """`RecentClipPreview` subclasses the main video surface, which sets a
    320px minimum meant for the large left-hand video — inherited here by a
    corner thumbnail unless explicitly overridden. This is the one that
    actually broke the column: found only once the offside checklist's own
    fix made this the widest thing left in it."""
    assert screen._top_right_stack.minimumSizeHint().width() <= _RIGHT_COLUMN_BUDGET_PX  # noqa: SLF001


def test_the_override_buttons_wrap_to_two_rows_not_one(screen):
    """Four buttons in a single row is wider than the rail at any normal
    window size, not only a small one — this is the shape of bug a "small
    screen" report is actually describing. A 2x2 grid caps each row at two
    buttons regardless of how many choices exist."""
    screen.offside.show()
    screen.offside.resize(280, screen.offside.sizeHint().height())
    buttons = [*screen.offside._override_buttons.values(), screen.offside._clear_button]  # noqa: SLF001
    rows = {button.y() for button in buttons}
    assert len(rows) == 2, "expected the four override buttons on two rows, not one"


def test_the_pitch_map_does_not_set_the_columns_floor(screen):
    """Same class of bug as the checklist and the camera thumbnail: the map
    renders natively at 760px and must not carry that into layout."""
    assert screen.pitch_map.minimumSizeHint().width() <= _RIGHT_COLUMN_BUDGET_PX


# -- the pitch map (M2.7) -----------------------------------------------


def test_set_pitch_analysis_reaches_the_map(screen):
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None
    pitch = PitchModel()

    screen.set_pitch_analysis(analysis, pitch)

    assert screen.pitch_map._canvas._pixmap is not None  # noqa: SLF001


def test_starting_a_new_offside_check_clears_the_previous_map(screen):
    """A map left over from the previous candidate would show a calibration
    that has nothing to do with the frame now on screen — the same reasoning
    that already clears the offside verdict on a fresh start."""
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None
    screen.set_pitch_analysis(analysis, PitchModel())

    screen.show_offside_started(7)

    assert screen.pitch_map._canvas._pixmap is None  # noqa: SLF001


def test_clearing_the_offside_decision_clears_the_map_too(screen):
    analysis = FakeAnalysis(poses=[])
    analysis.calibration = FakeDirectionalCalibration()
    analysis.teams = None
    screen.set_pitch_analysis(analysis, PitchModel())

    screen.clear_offside_decision()

    assert screen.pitch_map._canvas._pixmap is None  # noqa: SLF001
