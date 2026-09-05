"""Team assignment policy: kits, keepers, sides, and operator corrections.

Synthetic scenes rather than footage — flat green "pitch", flat coloured
"shirts" — because what is under test here is the *decision-making*: what the
stage concludes when two kits are similar, when the keeper stands alone, when
the ball is nowhere near anyone, and when the operator disagrees with it. The
real-footage question (can a shirt actually be measured off a 90px-tall
broadcast player) is a different test, and lives in
tests/integration/test_team_assignment_footage.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from offside.body_keypoints.keypoints import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SOURCE_ANKLE,
    SOURCE_BBOX_BOTTOM,
    GroundPoint,
    Keypoint,
    PlayerPose,
)
from offside.team_assignment import (
    TEAM_A,
    TEAM_B,
    JerseyColorExtractor,
    PlayerRole,
    TeamAssigner,
)

GRASS = (50, 130, 60)
RED_KIT = (40, 40, 200)
BLUE_KIT = (200, 60, 40)
KEEPER_KIT = (180, 60, 180)  # purple: outside the grass hue window
NEAR_RED_KIT = (45, 48, 190)  # a red that a human would struggle to tell apart

FRAME_SIZE = (720, 1280)
PLAYER_W, PLAYER_H = 30, 90


class FakeMetricCalibration:
    """A metric calibration with a trivial 10px-per-metre mapping."""

    is_metric = True
    can_draw_offside_line = True

    def to_pitch(self, points):
        return np.array([[x / 10.0, y / 10.0] for x, y in points], dtype=np.float64)


class FakeDirectionalCalibration:
    """Directional only: a goal-line direction, but no metres."""

    is_metric = False
    can_draw_offside_line = True

    def offside_line_through(self, point):
        return point, (0.0, 1.0)


def make_scene() -> np.ndarray:
    image = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    image[:, :] = GRASS
    return image


def add_player(
    image: np.ndarray,
    x: float,
    y: float,
    kit: tuple[int, int, int],
    *,
    with_keypoints: bool = True,
    track_id: str | None = None,
) -> PlayerPose:
    """Paint a player at (x, y) — x, y is the point their feet stand on."""
    x1, y1 = x - PLAYER_W / 2, y - PLAYER_H
    x2, y2 = x + PLAYER_W / 2, y

    shoulder_y = y1 + PLAYER_H * 0.25
    hip_y = y1 + PLAYER_H * 0.55
    left_x, right_x = x1 + PLAYER_W * 0.15, x2 - PLAYER_W * 0.15

    cv2.rectangle(
        image,
        (int(left_x), int(shoulder_y)),
        (int(right_x), int(hip_y)),
        kit,
        -1,
    )

    keypoints = {}
    if with_keypoints:
        keypoints = {
            LEFT_SHOULDER: Keypoint(LEFT_SHOULDER, (left_x, shoulder_y), 0.9),
            RIGHT_SHOULDER: Keypoint(RIGHT_SHOULDER, (right_x, shoulder_y), 0.9),
            LEFT_HIP: Keypoint(LEFT_HIP, (left_x, hip_y), 0.9),
            RIGHT_HIP: Keypoint(RIGHT_HIP, (right_x, hip_y), 0.9),
            LEFT_ANKLE: Keypoint(LEFT_ANKLE, (x, y), 0.9),
        }

    return PlayerPose(
        frame_id=1,
        bbox_xyxy=(x1, y1, x2, y2),
        detection_confidence=0.9,
        ground_point=GroundPoint(
            xy=(x, y),
            confidence=0.9 if with_keypoints else 0.3,
            source=SOURCE_ANKLE if with_keypoints else SOURCE_BBOX_BOTTOM,
            reason="test",
        ),
        source_model="test",
        keypoints=keypoints,
        track_id=track_id,
    )


def two_team_scene(
    *,
    kit_a: tuple[int, int, int] = RED_KIT,
    kit_b: tuple[int, int, int] = BLUE_KIT,
    keeper: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, list[PlayerPose]]:
    image = make_scene()
    poses = []
    for index in range(5):
        poses.append(add_player(image, 400 + index * 60, 400 + index * 20, kit_a))
    for index in range(5):
        poses.append(add_player(image, 700 + index * 60, 420 + index * 20, kit_b))
    if keeper is not None:
        # Far to the left: alone at one end of the goal-to-goal axis.
        poses.append(add_player(image, 120, 500, keeper))
    return image, poses


@pytest.fixture
def assigner() -> TeamAssigner:
    return TeamAssigner()


class TestKitMeasurement:
    def test_torso_colour_comes_from_the_shirt_not_the_grass(self):
        image = make_scene()
        pose = add_player(image, 400, 400, RED_KIT)

        colour = JerseyColorExtractor().extract(image, pose)

        assert colour.is_usable
        # The kit is red; the grass around it is green. Averaging the whole box
        # would land somewhere between the two.
        blue, green, red = colour.bgr
        assert red > green + 40

    def test_a_player_who_is_all_grass_is_reported_unmeasured_not_green(self):
        image = make_scene()
        pose = add_player(image, 400, 400, RED_KIT, with_keypoints=False)
        # Repaint over the shirt: the player is now indistinguishable from
        # the pitch, which is what heavy occlusion looks like.
        cv2.rectangle(image, (380, 300), (420, 400), GRASS, -1)

        colour = JerseyColorExtractor().extract(image, pose)

        assert not colour.is_usable
        assert colour.confidence == 0.0
        assert "grass" in colour.reason

    def test_missing_keypoints_fall_back_to_the_box_with_lower_confidence(self):
        image = make_scene()
        posed = add_player(image, 400, 400, RED_KIT)
        unposed = add_player(image, 600, 400, RED_KIT, with_keypoints=False)

        extractor = JerseyColorExtractor()
        measured = extractor.extract(image, posed)
        guessed = extractor.extract(image, unposed)

        assert measured.confidence > guessed.confidence
        assert guessed.is_usable  # still usable — losing the player is worse


class TestClustering:
    def test_two_kits_split_into_two_groups(self, assigner):
        image, poses = two_team_scene()

        assignment = assigner.assign(image, poses)

        counts = assignment.counts()
        assert counts[TEAM_A] == 5
        assert counts[TEAM_B] == 5
        assert counts["unassigned"] == 0
        assert all(p.role is PlayerRole.OUTFIELD for p in assignment.players)

    def test_every_player_appears_even_when_their_kit_cannot_be_measured(self, assigner):
        image, poses = two_team_scene()
        hidden = add_player(image, 200, 300, RED_KIT, with_keypoints=False)
        cv2.rectangle(image, (180, 210), (220, 300), GRASS, -1)
        poses.append(hidden)

        assignment = assigner.assign(image, poses)

        # Same rule as M2.2: dropping a player would quietly change the
        # defender ranking in M2.5.
        assert len(assignment.players) == len(poses)
        assert assignment.by_index(len(poses) - 1).role is PlayerRole.UNKNOWN

    def test_similar_kits_are_reported_as_uncertain_rather_than_confident(self, assigner):
        image, poses = two_team_scene(kit_a=RED_KIT, kit_b=NEAR_RED_KIT)

        assignment = assigner.assign(image, poses)

        assert assignment.confidence < 0.5
        assert any("close in colour" in reason for reason in assignment.reasons)

    def test_the_same_frame_gives_the_same_answer_twice(self):
        image, poses = two_team_scene()

        first = TeamAssigner().assign(image, poses)
        second = TeamAssigner().assign(image, poses)

        assert [p.team_id for p in first.players] == [p.team_id for p in second.players]

    def test_team_labels_do_not_flip_between_frames(self, assigner):
        image, poses = two_team_scene()

        first = assigner.assign(image, poses)
        # Same players, listed in the opposite order — a fresh clustering could
        # easily hand the two kits each other's names.
        second = assigner.assign(image, list(reversed(poses)))

        first_kit = first.color_model.centroids[TEAM_A]
        second_kit = second.color_model.centroids[TEAM_A]
        assert np.allclose(first_kit, second_kit, atol=5.0)

    def test_one_odd_kit_does_not_take_a_whole_team_slot(self, assigner):
        """The failure this was actually found by, on the reference clip.

        Kits there are white and maroon — closer to each other than either is
        to the goalkeeper's red. Plain two-means gave the lone red player one
        of the two slots and lumped white and maroon together into the other,
        and no distance-based outlier test could see it: a group of one sits
        exactly on its own centre, so its residual is zero.
        """
        image = make_scene()
        white, maroon, lone_red = (230, 235, 233), (105, 80, 80), (50, 60, 195)
        poses = [add_player(image, 300 + i * 60, 400, white) for i in range(6)]
        poses += [add_player(image, 700 + i * 60, 430, maroon) for i in range(6)]
        poses.append(add_player(image, 1100, 500, lone_red))

        assignment = assigner.assign(image, poses)

        counts = assignment.counts()
        assert counts[TEAM_A] == 6
        assert counts[TEAM_B] == 6
        assert assignment.by_index(len(poses) - 1).team_id is None

    def test_too_few_players_is_a_refusal_not_a_guess(self, assigner):
        image = make_scene()
        poses = [add_player(image, 400, 400, RED_KIT), add_player(image, 500, 400, BLUE_KIT)]

        assignment = assigner.assign(image, poses)

        assert assignment.color_model is None
        assert assignment.confidence == 0.0
        assert all(p.team_id is None for p in assignment.players)
        assert assignment.warnings


class TestGoalkeeper:
    def test_an_odd_kit_alone_at_one_end_is_the_goalkeeper(self, assigner):
        image, poses = two_team_scene(keeper=KEEPER_KIT)

        assignment = assigner.assign(image, poses, calibration=FakeMetricCalibration())

        keepers = assignment.goalkeepers()
        assert len(keepers) == 1
        assert keepers[0].index == len(poses) - 1
        assert keepers[0].team_id is None
        assert "beyond every other player" in keepers[0].reason

    def test_the_keeper_never_drags_a_team_colour_toward_itself(self, assigner):
        image, without = two_team_scene()
        baseline = assigner.assign(image, without, calibration=FakeMetricCalibration())

        image, with_keeper = two_team_scene(keeper=KEEPER_KIT)
        assigner.reset()
        result = assigner.assign(image, with_keeper, calibration=FakeMetricCalibration())

        # This is the reason the fit runs twice — see clustering.py.
        for team in (TEAM_A, TEAM_B):
            assert np.allclose(
                baseline.color_model.centroids[team],
                result.color_model.centroids[team],
                atol=2.0,
            )

    def test_without_calibration_the_keeper_is_flagged_not_guessed(self, assigner):
        image, poses = two_team_scene(keeper=KEEPER_KIT)

        assignment = assigner.assign(image, poses, calibration=None)

        assert assignment.goalkeepers() == []
        assert any("goalkeeper" in warning for warning in assignment.warnings)

    def test_directional_calibration_is_enough_to_rank_players(self, assigner):
        image, poses = two_team_scene(keeper=KEEPER_KIT)

        assignment = assigner.assign(
            image, poses, calibration=FakeDirectionalCalibration()
        )

        assert len(assignment.goalkeepers()) == 1


class TestAttackingSide:
    def test_the_side_with_the_ball_is_the_attacking_side(self, assigner):
        image, poses = two_team_scene()
        ball = (poses[0].ground_point.xy[0] + 10, poses[0].ground_point.xy[1])

        assignment = assigner.assign(image, poses, ball_xy=ball)

        assert assignment.attacking_team_id == assignment.by_index(0).team_id
        assert assignment.defending_team_id != assignment.attacking_team_id

    def test_no_ball_means_no_sides_and_a_capped_confidence(self, assigner):
        image, poses = two_team_scene()

        assignment = assigner.assign(image, poses, ball_xy=None)

        assert assignment.attacking_team_id is None
        assert not assignment.sides_are_known
        assert assignment.confidence <= 0.35
        assert any("attacking" in warning for warning in assignment.warnings)

    def test_a_ball_far_from_everyone_does_not_pick_a_side(self, assigner):
        image, poses = two_team_scene()

        assignment = assigner.assign(image, poses, ball_xy=(50.0, 50.0))

        assert assignment.attacking_team_id is None

    def test_the_defending_side_includes_its_goalkeeper(self, assigner):
        image, poses = two_team_scene(keeper=KEEPER_KIT)
        ball = (poses[0].ground_point.xy[0] + 10, poses[0].ground_point.xy[1])

        assignment = assigner.assign(
            image, poses, calibration=FakeMetricCalibration(), ball_xy=ball
        )
        keeper = assignment.goalkeepers()[0]
        assigner.overrides.pin_player(
            keeper.anchor_xy,
            team_id=assignment.defending_team_id,
            role=PlayerRole.GOALKEEPER,
        )
        corrected = assigner.assign(
            image, poses, calibration=FakeMetricCalibration(), ball_xy=ball
        )

        # Law 11 counts the second-last *opponent*, which is usually the
        # keeper plus one defender — so the keeper must stay in this list.
        assert any(p.role is PlayerRole.GOALKEEPER for p in corrected.opponents())


class TestOperatorOverrides:
    def test_pinning_a_player_beats_the_measured_colour(self, assigner):
        image, poses = two_team_scene()
        before = assigner.assign(image, poses)
        target = before.by_index(0)
        other_team = TEAM_B if target.team_id == TEAM_A else TEAM_A

        assigner.overrides.pin_player(target.anchor_xy, team_id=other_team)
        after = assigner.assign(image, poses)

        pinned = after.by_index(0)
        assert pinned.team_id == other_team
        assert pinned.is_operator_set
        assert pinned.confidence == 1.0

    def test_swapping_labels_moves_everyone_and_the_attacking_side(self, assigner):
        image, poses = two_team_scene()
        ball = (poses[0].ground_point.xy[0] + 10, poses[0].ground_point.xy[1])
        before = assigner.assign(image, poses, ball_xy=ball)

        assigner.overrides.swap_teams()
        after = assigner.assign(image, poses, ball_xy=ball)

        assert after.by_index(0).team_id != before.by_index(0).team_id
        assert after.attacking_team_id != before.attacking_team_id
        # ...and the attacking side still contains the player with the ball.
        assert after.by_index(0).team_id == after.attacking_team_id

    def test_the_operator_can_name_the_attacking_side_with_no_ball_on_screen(
        self, assigner
    ):
        image, poses = two_team_scene()
        assigner.overrides.set_attacking_team(TEAM_A)

        assignment = assigner.assign(image, poses, ball_xy=None)

        assert assignment.attacking_team_id == TEAM_A
        assert not any("which side is attacking" in w for w in assignment.warnings)

    def test_excluding_a_player_removes_them_from_both_teams(self, assigner):
        image, poses = two_team_scene()
        before = assigner.assign(image, poses)

        assigner.overrides.pin_player(
            before.by_index(0).anchor_xy, role=PlayerRole.UNKNOWN
        )
        after = assigner.assign(image, poses)

        assert after.by_index(0).team_id is None
        assert after.by_index(0).role is PlayerRole.UNKNOWN

    def test_a_pin_follows_a_player_who_moves_a_little(self, assigner):
        image, poses = two_team_scene()
        before = assigner.assign(image, poses)
        assigner.overrides.pin_player(before.by_index(0).anchor_xy, team_id=TEAM_B)

        moved = make_scene()
        shifted = [
            add_player(
                moved,
                pose.ground_point.xy[0] + 8,
                pose.ground_point.xy[1] + 4,
                RED_KIT if index < 5 else BLUE_KIT,
            )
            for index, pose in enumerate(poses)
        ]
        after = assigner.assign(moved, shifted)

        assert after.by_index(0).is_operator_set

    def test_clearing_corrections_returns_to_what_was_measured(self, assigner):
        image, poses = two_team_scene()
        before = assigner.assign(image, poses)

        assigner.overrides.pin_player(before.by_index(0).anchor_xy, team_id=TEAM_B)
        assigner.overrides.clear()
        after = assigner.assign(image, poses)

        assert after.by_index(0).team_id == before.by_index(0).team_id
        assert not after.by_index(0).is_operator_set
