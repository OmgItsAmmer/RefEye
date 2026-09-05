"""Body-point policy: which point on a player offside is measured from.

Pure logic, no model — everything here is about the decisions the pipeline
makes when a skeleton is partial, wrong, or missing, which on broadcast
footage is most of the time.
"""

from __future__ import annotations

import pytest

from offside.body_keypoints.ground_point import (
    estimate_ground_point,
    leading_offside_point,
)
from offside.body_keypoints.keypoints import (
    ARM_KEYPOINTS,
    COCO_KEYPOINT_NAMES,
    LEFT_ANKLE,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    OFFSIDE_SURFACE_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_KNEE,
    SOURCE_ANKLE,
    SOURCE_BBOX_BOTTOM,
    SOURCE_KNEE_PROJECTED,
    GroundPoint,
    Keypoint,
    PlayerPose,
)

BOX = (100.0, 100.0, 140.0, 200.0)  # a 40x100 player

POLICY = {
    "min_keypoint_confidence": 0.5,
    "knee_projection_penalty": 0.55,
    "bbox_fallback_penalty": 0.3,
}


def kp(name: str, x: float, y: float, conf: float = 0.9) -> Keypoint:
    return Keypoint(name=name, xy=(x, y), confidence=conf)


def pose(keypoints: dict[str, Keypoint], ground: GroundPoint | None = None) -> PlayerPose:
    return PlayerPose(
        frame_id=1,
        bbox_xyxy=BOX,
        detection_confidence=0.9,
        ground_point=ground
        or GroundPoint(xy=(120.0, 200.0), confidence=0.27, source=SOURCE_BBOX_BOTTOM, reason="test"),
        source_model="test",
        keypoints=keypoints,
    )


class TestKeypointVocabulary:
    def test_coco_order_is_the_full_seventeen(self):
        assert len(COCO_KEYPOINT_NAMES) == 17
        assert COCO_KEYPOINT_NAMES[0] == "nose"
        assert COCO_KEYPOINT_NAMES[-2:] == (LEFT_ANKLE, RIGHT_ANKLE)

    def test_arms_are_excluded_from_offside_surfaces(self):
        """A reaching player's wrist is often their furthest-forward point;
        Law 11 does not allow measuring from it."""
        assert not (ARM_KEYPOINTS & OFFSIDE_SURFACE_KEYPOINTS)
        assert LEFT_WRIST not in OFFSIDE_SURFACE_KEYPOINTS
        assert LEFT_SHOULDER in OFFSIDE_SURFACE_KEYPOINTS
        assert LEFT_ANKLE in OFFSIDE_SURFACE_KEYPOINTS


class TestGroundPoint:
    def test_uses_the_lower_ankle_as_the_planted_foot(self):
        """The raised foot of a running player is off the pitch plane, so
        projecting it would place them a stride further forward."""
        ground = estimate_ground_point(
            {
                LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 180.0),   # in the air
                RIGHT_ANKLE: kp(RIGHT_ANKLE, 130.0, 199.0),  # planted
            },
            BOX,
            0.9,
            **POLICY,
        )
        assert ground.source == SOURCE_ANKLE
        assert ground.xy == (130.0, 199.0)
        assert ground.is_measured

    def test_ignores_an_ankle_below_the_confidence_bar(self):
        ground = estimate_ground_point(
            {
                LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 199.0, conf=0.2),
                LEFT_KNEE: kp(LEFT_KNEE, 118.0, 160.0, conf=0.8),
            },
            BOX,
            0.9,
            **POLICY,
        )
        assert ground.source == SOURCE_KNEE_PROJECTED

    def test_single_visible_ankle_says_so_in_its_reason(self):
        ground = estimate_ground_point(
            {LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 199.0)}, BOX, 0.9, **POLICY
        )
        assert ground.source == SOURCE_ANKLE
        assert "other foot was not visible" in ground.reason

    def test_knee_fallback_takes_x_from_the_knee_and_y_from_the_box(self):
        """The box centre is skewed by an outstretched arm; the knee is not."""
        ground = estimate_ground_point(
            {
                LEFT_KNEE: kp(LEFT_KNEE, 112.0, 155.0, conf=0.8),
                RIGHT_KNEE: kp(RIGHT_KNEE, 128.0, 162.0, conf=0.8),
            },
            BOX,
            0.9,
            **POLICY,
        )
        assert ground.source == SOURCE_KNEE_PROJECTED
        assert ground.xy == (128.0, 200.0)  # lower knee's x, box bottom's y
        assert ground.confidence == pytest.approx(0.8 * 0.55)

    def test_falls_back_to_the_box_bottom_when_nothing_is_visible(self):
        ground = estimate_ground_point({}, BOX, 0.8, **POLICY)
        assert ground.source == SOURCE_BBOX_BOTTOM
        assert ground.xy == (120.0, 200.0)
        assert ground.confidence == pytest.approx(0.8 * 0.3)
        assert not ground.is_measured
        assert "can be off by a stride" in ground.reason

    def test_a_guess_never_outranks_a_measurement(self):
        """Confidence has to be comparable across rungs, or M2.6 cannot use
        it to decide whether a call is safe to make."""
        measured = estimate_ground_point(
            {LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 199.0, conf=0.55)}, BOX, 1.0, **POLICY
        )
        knee = estimate_ground_point(
            {LEFT_KNEE: kp(LEFT_KNEE, 110.0, 160.0, conf=0.95)}, BOX, 1.0, **POLICY
        )
        box = estimate_ground_point({}, BOX, 1.0, **POLICY)

        assert measured.confidence > knee.confidence > box.confidence


class TestLeadingOffsidePoint:
    def test_picks_the_most_advanced_legal_body_part(self):
        attacking = (1.0, 0.0)  # goal is to the right
        point = leading_offside_point(
            pose(
                {
                    LEFT_SHOULDER: kp(LEFT_SHOULDER, 135.0, 120.0),
                    LEFT_KNEE: kp(LEFT_KNEE, 120.0, 160.0),
                    LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 199.0),
                }
            ),
            attacking,
            min_keypoint_confidence=0.5,
        )
        assert point.keypoint_name == LEFT_SHOULDER
        assert not point.is_fallback

    def test_never_measures_from_an_arm(self):
        """The wrist is furthest forward here — and must still lose."""
        point = leading_offside_point(
            pose(
                {
                    LEFT_WRIST: kp(LEFT_WRIST, 190.0, 130.0),   # reaching out
                    LEFT_SHOULDER: kp(LEFT_SHOULDER, 138.0, 120.0),
                }
            ),
            (1.0, 0.0),
            min_keypoint_confidence=0.5,
        )
        assert point.keypoint_name == LEFT_SHOULDER

    def test_direction_decides_which_end_of_the_player_counts(self):
        keypoints = {
            LEFT_SHOULDER: kp(LEFT_SHOULDER, 138.0, 120.0),
            LEFT_ANKLE: kp(LEFT_ANKLE, 102.0, 199.0),
        }
        rightwards = leading_offside_point(
            pose(keypoints), (1.0, 0.0), min_keypoint_confidence=0.5
        )
        leftwards = leading_offside_point(
            pose(keypoints), (-1.0, 0.0), min_keypoint_confidence=0.5
        )
        assert rightwards.keypoint_name == LEFT_SHOULDER
        assert leftwards.keypoint_name == LEFT_ANKLE

    def test_falls_back_to_the_ground_point_and_admits_it(self):
        ground = GroundPoint(
            xy=(120.0, 200.0), confidence=0.24, source=SOURCE_BBOX_BOTTOM, reason="test"
        )
        point = leading_offside_point(
            pose({}, ground=ground), (1.0, 0.0), min_keypoint_confidence=0.5
        )
        assert point.is_fallback
        assert point.xy == ground.xy
        assert "no body keypoint" in point.reason

    def test_low_confidence_keypoints_do_not_steer_the_line(self):
        point = leading_offside_point(
            pose(
                {
                    LEFT_SHOULDER: kp(LEFT_SHOULDER, 190.0, 120.0, conf=0.1),
                    LEFT_ANKLE: kp(LEFT_ANKLE, 130.0, 199.0, conf=0.9),
                }
            ),
            (1.0, 0.0),
            min_keypoint_confidence=0.5,
        )
        assert point.keypoint_name == LEFT_ANKLE

    def test_a_zero_direction_is_a_programming_error(self):
        with pytest.raises(ValueError):
            leading_offside_point(
                pose({LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 199.0)}),
                (0.0, 0.0),
                min_keypoint_confidence=0.5,
            )


class TestPlayerPose:
    def test_reports_whether_it_actually_has_a_skeleton(self):
        assert pose({LEFT_ANKLE: kp(LEFT_ANKLE, 110.0, 199.0)}).has_pose
        assert not pose({}).has_pose

    def test_offside_surface_points_filter_arms_and_weak_points(self):
        player = pose(
            {
                LEFT_WRIST: kp(LEFT_WRIST, 190.0, 130.0),
                LEFT_SHOULDER: kp(LEFT_SHOULDER, 138.0, 120.0),
                LEFT_KNEE: kp(LEFT_KNEE, 120.0, 160.0, conf=0.1),
            }
        )
        names = {p.name for p in player.offside_surface_points(0.5)}
        assert names == {LEFT_SHOULDER}
