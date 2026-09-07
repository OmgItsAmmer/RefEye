"""The offside line, the verdict, and the refusal to over-claim.

Synthetic geometry with exact answers, because the point of these tests is
that the *decision* is right — including the decision not to decide. Whether
real footage supplies good enough inputs is a different question, tested
against the clip.

Coordinates: a fake metric calibration where 10 image pixels = 1 metre, so a
player at image x=500 stands at 50m up the pitch. That keeps every expected
number in the tests something a reader can check by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from offside.body_keypoints.keypoints import (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    SOURCE_ANKLE,
    SOURCE_BBOX_BOTTOM,
    GroundPoint,
    Keypoint,
    PlayerPose,
)
from offside.offside_line import (
    DepthAxis,
    OffsideLineCalculator,
    Verdict,
    infer_attack_direction,
)
from offside.second_last_defender import rank_opponents
from offside.team_assignment.teams import (
    SOURCE_KIT_COLOUR,
    TEAM_A,
    TEAM_B,
    PlayerRole,
    PlayerTeam,
    TeamAssignment,
)

IMAGE_SIZE = (1280, 720)


class MetricCalibration:
    """10 pixels to the metre, no rotation — checkable by hand."""

    is_metric = True
    can_draw_offside_line = True

    def to_pitch(self, points):
        return np.array([[x / 10.0, y / 10.0] for x, y in points], dtype=np.float64)

    def to_image(self, points):
        return np.array([[x * 10.0, y * 10.0] for x, y in points], dtype=np.float64)

    def offside_line_through(self, point):
        return point, (0.0, 1.0)


class DirectionalCalibration:
    is_metric = False
    can_draw_offside_line = True

    def offside_line_through(self, point):
        return point, (0.0, 1.0)


def make_pose(x: float, y: float = 400.0, *, confidence: float = 0.9) -> PlayerPose:
    """A player standing at image (x, y), with both ankles measured there."""
    return PlayerPose(
        frame_id=1,
        bbox_xyxy=(x - 15, y - 90, x + 15, y),
        detection_confidence=0.9,
        ground_point=GroundPoint(
            xy=(x, y), confidence=confidence, source=SOURCE_ANKLE, reason="test"
        ),
        source_model="test",
        keypoints={
            LEFT_ANKLE: Keypoint(LEFT_ANKLE, (x, y), confidence),
            RIGHT_ANKLE: Keypoint(RIGHT_ANKLE, (x, y), confidence),
        },
    )


def unposed(x: float, y: float = 400.0) -> PlayerPose:
    return PlayerPose(
        frame_id=1,
        bbox_xyxy=(x - 15, y - 90, x + 15, y),
        detection_confidence=0.9,
        ground_point=GroundPoint(
            xy=(x, y), confidence=0.27, source=SOURCE_BBOX_BOTTOM, reason="guess"
        ),
        source_model="test",
    )


def player(index: int, team_id: str | None, *, role=PlayerRole.OUTFIELD, confidence=0.9):
    return PlayerTeam(
        index=index,
        team_id=team_id,
        role=role,
        confidence=confidence,
        source=SOURCE_KIT_COLOUR,
        reason="test",
        anchor_xy=(0.0, 0.0),
        track_id=f"t{index}",
    )


def scene(attackers, defenders, *, keeper: float | None = None, attacking=TEAM_A):
    """Build poses and a team assignment from image x positions."""
    poses, players = [], []
    for x in attackers:
        players.append(player(len(poses), attacking))
        poses.append(make_pose(x))
    other = TEAM_B if attacking == TEAM_A else TEAM_A
    for x in defenders:
        players.append(player(len(poses), other))
        poses.append(make_pose(x))
    if keeper is not None:
        players.append(player(len(poses), other, role=PlayerRole.GOALKEEPER))
        poses.append(make_pose(keeper))
    return poses, TeamAssignment(players=players, attacking_team_id=attacking)


@pytest.fixture
def calculator() -> OffsideLineCalculator:
    return OffsideLineCalculator()


class TestTheLawItself:
    def test_an_attacker_beyond_the_second_last_opponent_is_offside(self, calculator):
        # Keeper at 75m, defenders at 70m and 60m: the second-last opponent
        # is the defender at 70m. The attacker stands at 72m.
        poses, teams = scene([720.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=1.0,
        )

        assert decision.verdict is Verdict.OFFSIDE
        assert decision.attacker.margin == pytest.approx(2.0, abs=0.01)
        assert decision.second_last_defender.depth == pytest.approx(70.0, abs=0.01)

    def test_an_attacker_behind_the_line_is_onside(self, calculator):
        poses, teams = scene([650.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        assert decision.verdict is Verdict.ONSIDE
        assert decision.attacker.margin == pytest.approx(-5.0, abs=0.01)

    def test_the_line_is_the_second_last_opponent_not_the_last(self, calculator):
        """The commonest misreading of Law 11. With a keeper at 75m and
        defenders at 70m and 60m, the line is at 70m, not 75m or 60m."""
        poses, teams = scene([720.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=1.0,
        )

        assert decision.second_last_defender.depth == pytest.approx(70.0, abs=0.01)
        assert decision.verdict is Verdict.OFFSIDE

    def test_the_goalkeeper_is_ranked_like_anyone_else(self, calculator):
        """A keeper who has come out is not the last opponent, and the Law
        counts opponents, not outfielders."""
        # Keeper has come out to 68m; defenders at 70m and 65m. The deepest
        # opponent is now the defender at 70m, and the *second*-last is the
        # keeper — ranked like anybody else rather than skipped.
        poses, teams = scene([720.0], [700.0, 650.0], keeper=680.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=1.0,
        )

        assert decision.second_last_defender.depth == pytest.approx(68.0, abs=0.01)
        assert decision.second_last_defender.index == 3

    def test_beyond_the_defender_but_behind_the_ball_is_onside(self, calculator):
        """A geometry-only tool gets this wrong constantly: the Law needs the
        attacker nearer the goal line than the ball as well."""
        poses, teams = scene([720.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(900.0, 400.0), attack_sign=1.0,
        )

        assert decision.verdict is Verdict.ONSIDE
        assert "not beyond the ball" in decision.attacker.reason

    def test_arms_are_never_measured(self, calculator):
        """Law 11: the arms are not an offside surface. A reaching attacker's
        wrist is often their furthest-forward point, so measuring it would
        call a legal attacker offside."""
        from offside.body_keypoints.keypoints import RIGHT_WRIST

        poses, teams = scene([695.0], [600.0, 700.0], keeper=750.0)
        # An arm stretched 4m past the line — must be ignored entirely.
        poses[0].keypoints[RIGHT_WRIST] = Keypoint(RIGHT_WRIST, (740.0, 380.0), 0.99)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        assert decision.attacker.point[0] == pytest.approx(695.0, abs=0.01)
        assert decision.verdict is not Verdict.OFFSIDE


class TestRefusingToOverClaim:
    def test_a_margin_inside_the_error_is_too_close_to_call(self, calculator):
        """The product, not a failure mode: a 5cm margin from foot points
        measured to the nearest half metre is not a verdict."""
        poses, teams = scene([700.5], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        assert decision.verdict is Verdict.TOO_CLOSE
        assert "inside" in decision.attacker.reason
        assert "Too close to call" in decision.headline()

    def test_a_guessed_foot_position_widens_the_error(self, calculator):
        """A box-bottom guess must not produce the same confident call a
        measured ankle does."""
        measured_poses, teams = scene([740.0], [600.0, 700.0], keeper=750.0)
        measured = calculator.decide(
            measured_poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        guessed_poses = list(measured_poses)
        guessed_poses[0] = unposed(740.0)
        guessed = calculator.decide(
            guessed_poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        assert guessed.attacker.uncertainty > measured.attacker.uncertainty
        assert guessed.confidence < measured.confidence

    def test_no_calibration_means_no_call(self, calculator):
        poses, teams = scene([650.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(poses, teams, None, IMAGE_SIZE)

        assert decision.verdict is Verdict.INCONCLUSIVE
        assert any("not calibrated" in w for w in decision.warnings)

    def test_unknown_sides_mean_no_call(self, calculator):
        poses, teams = scene([650.0], [600.0, 700.0], keeper=750.0)
        teams.attacking_team_id = None

        decision = calculator.decide(poses, teams, MetricCalibration(), IMAGE_SIZE)

        assert decision.verdict is Verdict.INCONCLUSIVE
        assert any("which side is attacking" in w for w in decision.warnings)

    def test_one_opponent_is_not_enough_for_a_second_last(self, calculator):
        poses, teams = scene([650.0], [600.0])

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        assert decision.verdict is Verdict.INCONCLUSIVE
        assert any("at least two" in w for w in decision.warnings)

    def test_a_defender_who_could_not_be_placed_is_reported(self, calculator):
        """The unmeasurable defender is exactly the one who might have been
        second-last, so their absence can never be silent."""

        class PartialCalibration(MetricCalibration):
            def to_pitch(self, points):
                return np.array(
                    [
                        [np.nan, np.nan] if x > 690 else [x / 10.0, y / 10.0]
                        for x, y in points
                    ],
                    dtype=np.float64,
                )

        poses, teams = scene([650.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, PartialCalibration(), IMAGE_SIZE,
            ball_xy=(500.0, 400.0), attack_sign=1.0,
        )

        assert any("could not be placed" in w for w in decision.warnings)


class TestWhichEndIsBeingDefended:
    def test_the_goalkeeper_settles_the_direction(self):
        axis = DepthAxis.from_calibration(MetricCalibration(), IMAGE_SIZE)

        direction = infer_attack_direction(axis, [50.0, 55.0], [60.0, 62.0], [75.0])

        assert direction.sign == 1.0
        assert direction.source == "goalkeeper"
        assert direction.confidence > 0.8

    def test_a_keeper_who_has_come_out_is_not_evidence(self):
        """A keeper level with their own defensive line says nothing about
        which end is being defended, and reading the direction off them there
        would invert the verdict."""
        axis = DepthAxis.from_calibration(MetricCalibration(), IMAGE_SIZE)

        direction = infer_attack_direction(axis, [72.0], [70.0, 65.0], [68.0])

        assert direction.source == "team_shape"

    def test_team_shape_is_the_fallback(self):
        axis = DepthAxis.from_calibration(MetricCalibration(), IMAGE_SIZE)

        direction = infer_attack_direction(axis, [20.0, 25.0], [60.0, 65.0], [])

        assert direction.sign == 1.0
        assert direction.source == "team_shape"
        assert 0.0 < direction.confidence <= 0.75

    def test_teams_mixed_together_means_the_direction_is_unknown(self):
        axis = DepthAxis.from_calibration(MetricCalibration(), IMAGE_SIZE)

        direction = infer_attack_direction(axis, [50.0, 51.0], [50.5, 50.2], [])

        assert not direction.known
        assert "mixed together" in direction.reason

    def test_the_operator_outranks_both(self):
        axis = DepthAxis.from_calibration(MetricCalibration(), IMAGE_SIZE)

        direction = infer_attack_direction(
            axis, [20.0], [60.0], [75.0], operator_sign=-1.0
        )

        assert direction.sign == -1.0
        assert direction.confidence == 1.0

    def test_reading_the_pitch_backwards_reverses_the_verdict(self, calculator):
        """Proof that the direction matters: the identical picture read from
        the other end gives the opposite answer, which is why it is inferred
        with its own confidence rather than assumed."""
        poses, teams = scene([720.0], [600.0, 700.0], keeper=750.0)

        forwards = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=1.0,
        )
        backwards = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=-1.0,
        )

        assert forwards.verdict is Verdict.OFFSIDE
        assert backwards.verdict is not Verdict.OFFSIDE


class TestWithoutMetres:
    def test_a_verdict_is_still_possible_at_directional_level(self, calculator):
        """Most of offside is 'is this attacker beyond that defender', which
        needs an ordering, not a map."""
        poses, teams = scene([700.0], [600.0, 650.0], keeper=560.0)

        decision = calculator.decide(
            poses,
            teams,
            DirectionalCalibration(),
            IMAGE_SIZE,
            ball_xy=(600.0, 400.0),
            attack_sign=-1.0,
        )

        assert decision.verdict is Verdict.OFFSIDE
        assert decision.axis_unit == "px"

    def test_the_error_scales_with_the_player_on_screen(self, calculator):
        """Without metres the only scale is the player's own height, which
        shrinks with distance exactly as the measurement error does."""
        poses, teams = scene([700.0], [600.0, 650.0], keeper=560.0)
        decision = calculator.decide(
            poses, teams, DirectionalCalibration(), IMAGE_SIZE, attack_sign=-1.0
        )

        assert decision.attacker.uncertainty == pytest.approx(0.35 * 90, abs=0.01)


class TestReporting:
    def test_the_headline_reads_as_english(self, calculator):
        poses, teams = scene([720.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=1.0,
        )

        assert decision.headline().startswith("Offside position")
        assert "2.00m" in decision.headline()

    def test_every_attacker_beyond_the_line_is_listed(self, calculator):
        """Which of them is involved in play is football judgement and out of
        scope, so all of them are reported rather than one being chosen."""
        poses, teams = scene([720.0, 760.0, 550.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(540.0, 400.0), attack_sign=1.0,
        )

        beyond = [a for a in decision.attackers if a.verdict is Verdict.OFFSIDE]
        assert len(beyond) == 2
        assert decision.attacker.index == 1  # the most advanced one leads
        assert any("judgement for the operator" in r for r in decision.reasons)

    def test_the_line_comes_back_ready_to_draw(self, calculator):
        poses, teams = scene([720.0], [600.0, 700.0], keeper=750.0)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(600.0, 400.0), attack_sign=1.0,
        )

        (point, direction) = decision.line
        assert point == pytest.approx((700.0, 400.0), abs=0.01)
        assert direction == pytest.approx((0.0, 1.0), abs=0.01)


class TestRankingDirectly:
    def test_ranking_orders_from_the_defended_goal_outwards(self):
        from offside.offside_line.axis import AttackDirection

        axis = DepthAxis.from_calibration(MetricCalibration(), IMAGE_SIZE)
        direction = AttackDirection(1.0, 0.9, "test", "test")
        poses, teams = scene([], [600.0, 700.0, 650.0], keeper=750.0)

        ranking = rank_opponents(teams.opponents(), poses, axis, direction)

        assert [round(p.depth) for p in ranking.ranked] == [75, 70, 65, 60]
        assert ranking.second_last.depth == pytest.approx(70.0)


class TestTheConfidenceFloor:
    """A verdict the inputs cannot support is withheld, not published."""

    # Both sides strung out across the whole pitch: far enough apart on
    # average to establish *a* direction, nowhere near consistent enough for
    # that direction to be worth much. Geometrically this reads "offside by
    # 28m"; on evidence it reads "I do not really know which way they are
    # playing", which is what happened on the reference clip.
    SPREAD_OUT = ([200.0, 900.0], [580.0, 620.0, 1000.0])

    def test_a_barely_established_direction_withholds_the_verdict(self, calculator):
        poses, teams = scene(*self.SPREAD_OUT)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE, ball_xy=(150.0, 400.0)
        )

        assert decision.direction.known, "this scene is meant to exercise the floor"
        assert decision.verdict is Verdict.INCONCLUSIVE
        assert any("too uncertain to report" in w for w in decision.warnings)

    def test_the_refusal_names_what_to_go_and_fix(self, calculator):
        poses, teams = scene(*self.SPREAD_OUT)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE, ball_xy=(150.0, 400.0)
        )

        # The useful half of a low confidence score is which input caused it.
        assert any("which end is being defended" in w for w in decision.warnings)

    def test_a_stated_direction_lifts_the_same_frame_to_a_verdict(self, calculator):
        """Proof the floor is about evidence rather than geometry: the same
        picture, with the one uncertain input supplied, becomes a call."""
        poses, teams = scene(*self.SPREAD_OUT)

        decision = calculator.decide(
            poses, teams, MetricCalibration(), IMAGE_SIZE,
            ball_xy=(150.0, 400.0), attack_sign=1.0,
        )

        assert decision.verdict is Verdict.OFFSIDE
