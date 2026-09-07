"""The inspector's widgets, driven headlessly.

Renders the real panels against a synthetic analysis and checks that what the
operator would see actually appears — a swatch per player, the flag on the
ones awaiting confirmation, and a correction that reaches the caller. Runs
offscreen, so it needs no display.

This deliberately does not screenshot-compare: the value here is that the
widgets build, populate and emit, which is the class of failure that would
otherwise only show up when someone opens the window.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from offside.team_assignment.teams import TEAM_A, TEAM_B, PlayerRole  # noqa: E402
from tests.unit.test_pipeline_presenter import (  # noqa: E402
    FakeAnalysis,
    make_assignment,
    make_player,
    make_pose,
)
from tools.pipeline_debugger import presenter  # noqa: E402
from tools.pipeline_debugger.panels import (  # noqa: E402
    ACTION_CONFIRM,
    ACTION_TEAM_B,
    FlowPanel,
    SwatchCell,
    TeamsPanel,
)
from tools.pipeline_debugger.pipeline import StageReport, StageState  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def analysis():
    players = [
        make_player(0, TEAM_A),
        make_player(1, TEAM_A, needs_confirmation=True),
        make_player(2, TEAM_B, patterned=True),
        make_player(3, None, role=PlayerRole.GOALKEEPER, needs_confirmation=True),
    ]
    return FakeAnalysis(
        teams=make_assignment(players),
        poses=[make_pose(200 + index * 60) for index in range(4)],
    )


def _teams_panel(analysis) -> TeamsPanel:
    panel = TeamsPanel()
    panel.update_from(
        analysis,
        presenter.team_columns(analysis),
        presenter.confirmation_summary(analysis),
        presenter.kit_summary(analysis),
    )
    return panel


def test_every_player_gets_a_swatch_on_screen(qt_app, analysis):
    panel = _teams_panel(analysis)

    cells = panel.findChildren(SwatchCell)
    assert len(cells) == len(analysis.teams.players)
    assert {cell.card.index for cell in cells} == {0, 1, 2, 3}


def test_players_awaiting_confirmation_are_marked(qt_app, analysis):
    panel = _teams_panel(analysis)

    flagged = [cell.card.index for cell in panel.findChildren(SwatchCell)
               if cell.card.needs_confirmation]
    assert sorted(flagged) == [1, 3]


def test_nothing_is_correctable_until_a_player_is_picked(qt_app, analysis):
    panel = _teams_panel(analysis)

    assert panel.selected_index is None
    # Acting on "no selection" would silently apply to the wrong player.
    panel._emit_action(ACTION_TEAM_B)  # must be a no-op, not a crash


def test_picking_a_player_and_correcting_them_reaches_the_caller(qt_app, analysis):
    panel = _teams_panel(analysis)
    received: list[tuple[str, list[int]]] = []
    panel.action_requested.connect(lambda action, indices: received.append((action, indices)))

    cell = next(c for c in panel.findChildren(SwatchCell) if c.card.index == 2)
    cell.clicked.emit(cell.card.index)
    panel._emit_action(ACTION_TEAM_B)

    assert received == [(ACTION_TEAM_B, [2])]


def test_a_column_can_confirm_all_of_its_flagged_players_at_once(qt_app, analysis):
    panel = _teams_panel(analysis)
    received: list[tuple[str, list[int]]] = []
    panel.action_requested.connect(lambda action, indices: received.append((action, indices)))

    grids = [
        child for child in panel._grids_holder.children() if hasattr(child, "confirm_all")
    ]
    team_a_grid = next(grid for grid in grids if grid.column.team_id == TEAM_A)
    team_a_grid.confirm_all.emit([card.index for card in team_a_grid.column.cards
                                  if card.needs_confirmation])

    assert received == [(ACTION_CONFIRM, [1])]


def test_the_pipeline_view_lists_unbuilt_stages_too(qt_app):
    analysis = FakeAnalysis(
        reports=[
            StageReport(key="a", phase="M2.3", title="Teams", state=StageState.OK,
                        summary="done"),
            StageReport(key="b", phase="M2.5", title="Offside line",
                        state=StageState.PENDING, summary="not built yet — the geometry"),
        ]
    )
    panel = FlowPanel()
    panel.update_from(presenter.flow_steps(analysis))

    rendered = " ".join(label.text() for label in panel.findChildren(QLabel))
    assert "M2.3" in rendered
    assert "M2.5" in rendered
    assert "NOT BUILT" in rendered
