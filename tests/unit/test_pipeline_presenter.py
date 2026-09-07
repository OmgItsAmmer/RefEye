"""What the inspector puts on screen.

The inspector exists to make the pipeline trustworthy, which makes the choice
and wording of what it displays part of the product rather than incidental UI
code. These tests cover that choice without needing a display: the widgets in
`panels.py` only render what these functions return.
"""

from __future__ import annotations

import numpy as np
import pytest

from offside.body_keypoints.keypoints import (
    SOURCE_ANKLE,
    GroundPoint,
    PlayerPose,
)
from offside.team_assignment.teams import (
    SOURCE_KIT_COLOUR,
    SOURCE_OPERATOR,
    TEAM_A,
    TEAM_B,
    JerseyColor,
    PlayerRole,
    PlayerTeam,
    TeamAssignment,
    TeamColorModel,
)
from tools.pipeline_debugger import presenter
from tools.pipeline_debugger.pipeline import StageReport, StageState


class FakeFrame:
    def __init__(self, image):
        self.image = image


class FakeAnalysis:
    """The parts of a FrameAnalysis the presenter reads."""

    def __init__(self, teams=None, poses=None, reports=None, image=None):
        self.teams = teams
        self.poses = poses or []
        self.reports = reports or []
        self.frame = FakeFrame(
            image if image is not None else np.full((720, 1280, 3), 40, dtype=np.uint8)
        )


def make_pose(x: float = 200.0, y: float = 400.0) -> PlayerPose:
    return PlayerPose(
        frame_id=1,
        bbox_xyxy=(x - 15, y - 90, x + 15, y),
        detection_confidence=0.9,
        ground_point=GroundPoint(xy=(x, y), confidence=0.9, source=SOURCE_ANKLE, reason="t"),
        source_model="test",
    )


def make_player(
    index: int,
    team_id: str | None,
    *,
    needs_confirmation: bool = False,
    role: PlayerRole = PlayerRole.OUTFIELD,
    patterned: bool = False,
    source: str = SOURCE_KIT_COLOUR,
) -> PlayerTeam:
    return PlayerTeam(
        index=index,
        team_id=team_id,
        role=role,
        confidence=0.8,
        source=source,
        reason="because the shirt measured that way",
        anchor_xy=(200.0 + index * 40, 350.0),
        track_id=f"t{index}",
        color=JerseyColor(
            vector=(50.0, 10.0, 10.0, 50.0, 10.0, 10.0),
            bgr=(40, 40, 200),
            secondary_bgr=(200, 60, 40) if patterned else (40, 40, 200),
            patterned=patterned,
            pixel_count=120,
            confidence=0.9,
            source="torso_keypoints",
            reason="120 shirt pixels",
            sample_count=4,
        ),
        vote_share=0.9,
        frames_pooled=4,
        needs_confirmation=needs_confirmation,
    )


def make_assignment(players, attacking=TEAM_A) -> TeamAssignment:
    model = TeamColorModel(
        centroids={TEAM_A: (50.0, 10.0, 10.0), TEAM_B: (80.0, -5.0, -5.0)},
        swatches={TEAM_A: (40, 40, 200), TEAM_B: (220, 220, 220)},
        spreads={TEAM_A: 8.0, TEAM_B: 9.0},
        separation=52.0,
        outlier_threshold=22.0,
        sample_count=len(players),
        confidence=0.8,
    )
    return TeamAssignment(
        players=players, color_model=model, attacking_team_id=attacking, confidence=0.7
    )


@pytest.fixture
def analysis() -> FakeAnalysis:
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


class TestTeamColumns:
    def test_one_column_per_kit_plus_one_for_everyone_else(self, analysis):
        columns = presenter.team_columns(analysis)

        assert [column.team_id for column in columns] == [TEAM_A, TEAM_B, None]
        assert [column.count for column in columns] == [2, 1, 1]

    def test_the_columns_say_which_side_is_attacking(self, analysis):
        columns = presenter.team_columns(analysis)

        assert "attacking" in columns[0].title
        assert "defending" in columns[1].title

    def test_a_column_says_how_many_of_its_players_need_a_look(self, analysis):
        columns = presenter.team_columns(analysis)

        assert "1 to confirm" in columns[0].subtitle
        assert "all settled" in columns[1].subtitle

    def test_each_card_carries_the_crop_it_was_measured_from(self, analysis):
        columns = presenter.team_columns(analysis)
        card = columns[0].cards[0]

        # The evidence beside the conclusion: a swatch with no crop next to it
        # is an assertion the operator cannot check.
        assert card.crop is not None
        assert card.crop.shape[0] > 0 and card.crop.shape[1] > 0

    def test_a_patterned_kit_shows_two_different_colours(self, analysis):
        columns = presenter.team_columns(analysis)

        solid = columns[0].cards[0]
        striped = columns[1].cards[0]
        assert not solid.has_two_colours
        assert striped.has_two_colours

    def test_cards_explain_themselves(self, analysis):
        card = presenter.team_columns(analysis)[0].cards[0]

        assert any("shirt measured" in line for line in card.lines)
        assert any("4 frames" in line for line in card.lines)

    def test_an_operator_correction_is_labelled_as_one(self):
        players = [make_player(0, TEAM_A, source=SOURCE_OPERATOR)]
        analysis = FakeAnalysis(teams=make_assignment(players), poses=[make_pose()])

        card = presenter.team_columns(analysis)[0].cards[0]

        assert card.is_operator_set
        assert card.label.endswith("*")

    def test_no_teams_yet_means_no_columns(self):
        assert presenter.team_columns(FakeAnalysis()) == []


class TestKitSummary:
    def test_the_measured_kits_are_stated_in_checkable_numbers(self, analysis):
        lines = presenter.kit_summary(analysis)

        assert any("52" in line for line in lines)
        assert any("apart" in line for line in lines)

    def test_no_model_says_so_plainly(self):
        analysis = FakeAnalysis(teams=TeamAssignment())

        assert "No kit colours" in presenter.kit_summary(analysis)[0]


class TestConfirmationSummary:
    def test_it_says_when_there_is_nothing_to_do(self):
        players = [make_player(0, TEAM_A), make_player(1, TEAM_B)]
        analysis = FakeAnalysis(teams=make_assignment(players))

        assert "nothing to confirm" in presenter.confirmation_summary(analysis)

    def test_it_counts_what_is_waiting(self, analysis):
        assert "2 of 4 players need a look" in presenter.confirmation_summary(analysis)


class TestFlowSteps:
    def test_every_stage_appears_including_unbuilt_ones(self):
        analysis = FakeAnalysis(
            reports=[
                StageReport(
                    key="a", phase="M2.3", title="Teams", state=StageState.OK,
                    summary="done", details=["a fact", "WARNING: a worry"],
                ),
                StageReport(
                    key="b", phase="M2.5", title="Offside line",
                    state=StageState.PENDING, summary="not built yet",
                ),
            ]
        )

        steps = presenter.flow_steps(analysis)

        assert [step.phase for step in steps] == ["M2.3", "M2.5"]
        assert steps[1].state == "pending"

    def test_warnings_are_separated_from_facts(self):
        analysis = FakeAnalysis(
            reports=[
                StageReport(
                    key="a", phase="M2.3", title="Teams", state=StageState.DEGRADED,
                    summary="partly", details=["a fact", "WARNING: a worry"],
                )
            ]
        )

        step = presenter.flow_steps(analysis)[0]

        assert step.facts == ["a fact"]
        assert step.warnings == ["a worry"]


class TestPlayerCrop:
    def test_a_crop_off_the_edge_of_the_frame_is_clipped_not_crashed(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        assert presenter.player_crop(image, (-50, -50, 20, 20)) is not None
        assert presenter.player_crop(image, (200, 200, 260, 260)) is None


class TestPitchMapAsInstruction:
    """The top-down map before calibration is not a placeholder — it is how
    the operator is told *which* point to click. Asking someone to click
    "centre_mark" in a video without showing them where that is on a pitch is
    not a question anybody can answer."""

    def test_a_click_on_the_map_maps_back_to_the_pitch_position(self):
        from offside.field_geometry.pitch import PitchModel
        from tools.pipeline_debugger import overlays

        pitch = PitchModel()
        scale, offset_x, offset_y = overlays.top_down_transform(pitch)
        canvas_point = (offset_x + 52.5 * scale, offset_y + 34.0 * scale)

        # The dot drawn and the dot clicked must be the same dot, or the map
        # quietly stops selecting what it displays.
        assert overlays.canvas_to_pitch(pitch, canvas_point) == pytest.approx(
            (52.5, 34.0)
        )

    def test_the_map_draws_the_landmark_being_asked_for(self):
        from offside.field_geometry.pitch import PitchModel
        from tools.pipeline_debugger import overlays

        pitch = PitchModel()
        plain = overlays.render_top_down(pitch, None, [])
        highlighted = overlays.render_top_down(
            pitch, None, [], highlight_landmark="centre_mark"
        )

        assert not np.array_equal(plain, highlighted)

    def test_marked_landmarks_are_shown_as_marked(self):
        from offside.field_geometry.pitch import PitchModel
        from tools.pipeline_debugger import overlays

        pitch = PitchModel()
        before = overlays.render_top_down(pitch, None, [])
        after = overlays.render_top_down(
            pitch, None, [], marked_landmarks={"centre_mark": (10.0, 10.0)}
        )

        assert not np.array_equal(before, after)


class TestLandmarkGuide:
    """The technical names are correct and useless to somebody who does not
    know football. Every point the operator can be asked for must come with a
    plain label and a description of where it physically is."""

    def test_every_landmark_the_app_offers_is_described(self):
        from offside.field_geometry.pitch import PitchModel

        undescribed = set(PitchModel().landmarks()) - {
            name for name, _, _ in presenter.LANDMARK_GUIDE
        }
        assert not undescribed, f"no plain-words description for: {undescribed}"

    def test_no_choice_is_lost_even_if_the_guide_falls_behind(self):
        class ExtraPitch:
            def landmarks(self):
                return {"centre_mark": (0, 0), "some_new_point": (1, 1)}

        choices = presenter.landmark_choices(ExtraPitch())

        # A landmark that exists but cannot be chosen would be a silent gap.
        assert {name for _, name in choices} == {"centre_mark", "some_new_point"}

    def test_the_easiest_points_to_find_are_offered_first(self):
        from offside.field_geometry.pitch import PitchModel

        choices = presenter.landmark_choices(PitchModel())

        assert choices[0][1].startswith("corner_")
        assert "Corner flag" in choices[0][0]

    def test_descriptions_avoid_football_jargon(self):
        for name, _, description in presenter.LANDMARK_GUIDE:
            lowered = description.lower()
            for jargon in ("penalty area", "goal area", "six-yard", "18-yard"):
                assert jargon not in lowered, f"{name} describes itself with '{jargon}'"
