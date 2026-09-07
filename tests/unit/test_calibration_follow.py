"""Carrying a calibration across a moving camera.

The bug being fixed here is invisible by construction: four marked points
always agree with each other, so a calibration can go completely wrong as the
camera pans while its reprojection error stays at zero. These tests move a
synthetic scene by a *known* transform and check the marks moved with it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from offside.pitch_calibration.homography import PointCorrespondence
from offside.pitch_calibration.tracking import CalibrationFollower

SIZE = (720, 1280)


def textured_scene(seed: int = 7) -> np.ndarray:
    """A frame with enough detail to track — the sterile flat-green images used
    elsewhere have no corners at all, and would test nothing."""
    rng = np.random.default_rng(seed)
    image = np.full((*SIZE, 3), 60, dtype=np.uint8)
    image[:, :] = (50, 130, 60)
    for _ in range(400):
        x, y = int(rng.integers(0, SIZE[1])), int(rng.integers(0, SIZE[0]))
        colour = tuple(int(c) for c in rng.integers(80, 255, size=3))
        cv2.rectangle(image, (x, y), (x + 9, y + 9), colour, -1)
    for offset in range(0, SIZE[1], 160):
        cv2.line(image, (offset, 0), (offset + 200, SIZE[0]), (240, 240, 240), 2)
    return image


def shifted(image: np.ndarray, dx: float, dy: float, scale: float = 1.0) -> np.ndarray:
    """The same scene as the camera would see it after a pan (and zoom)."""
    matrix = np.array(
        [[scale, 0.0, dx], [0.0, scale, dy]], dtype=np.float64
    )
    return cv2.warpAffine(image, matrix, (SIZE[1], SIZE[0]), borderMode=cv2.BORDER_REFLECT)


def marks(points) -> list[PointCorrespondence]:
    return [
        PointCorrespondence(image_xy=point, pitch_xy=(index * 10.0, 0.0), landmark=f"m{index}")
        for index, point in enumerate(points)
    ]


@pytest.fixture
def follower() -> CalibrationFollower:
    return CalibrationFollower()


class TestFollowingTheCamera:
    def test_marks_move_with_a_panning_camera(self, follower):
        scene = textured_scene()
        original = [(400.0, 300.0), (700.0, 320.0), (500.0, 500.0), (900.0, 520.0)]
        follower.anchor(marks(original), scene, frame_id=0)

        result = follower.update(shifted(scene, 40, 15), frame_id=1)

        assert result is not None and not result.lost
        for before, after in zip(original, result.correspondences):
            assert after.image_xy[0] == pytest.approx(before[0] + 40, abs=2.0)
            assert after.image_xy[1] == pytest.approx(before[1] + 15, abs=2.0)

    def test_the_pitch_side_of_a_mark_never_changes(self, follower):
        """Only the *image* position moves. The point on the pitch a mark
        refers to is a fact about the pitch, not about the camera."""
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=0)

        result = follower.update(shifted(scene, 30, 0), frame_id=1)

        assert result.correspondences[0].pitch_xy == (0.0, 0.0)
        assert result.correspondences[0].landmark == "m0"

    def test_marks_follow_a_zoom_as_well_as_a_pan(self, follower):
        scene = textured_scene()
        point = (600.0, 400.0)
        follower.anchor(marks([point]), scene, frame_id=0)

        # Zoom about the origin: a point at 600,400 should land near 630,420.
        result = follower.update(shifted(scene, 0, 0, scale=1.05), frame_id=1)

        assert result.correspondences[0].image_xy[0] == pytest.approx(630, abs=6.0)
        assert result.correspondences[0].image_xy[1] == pytest.approx(420, abs=6.0)

    def test_movement_accumulates_over_several_frames(self, follower):
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=0)

        for step in range(1, 6):
            result = follower.update(shifted(scene, 20 * step, 0), frame_id=step)

        assert result.correspondences[0].image_xy[0] == pytest.approx(500, abs=6.0)
        assert result.frames_since_anchor == 5

    def test_a_still_camera_moves_nothing(self, follower):
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=0)

        result = follower.update(scene.copy(), frame_id=1)

        assert result.moved_px < 1.0
        assert result.correspondences[0].image_xy == pytest.approx((400.0, 300.0), abs=1.0)


class TestHonesty:
    def test_trust_falls_the_further_the_marks_are_carried(self, follower):
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=0)

        first = follower.update(shifted(scene, 10, 0), frame_id=1)
        for step in range(2, 40):
            last = follower.update(shifted(scene, 10 * step, 0), frame_id=step)

        # Drift is cumulative and cannot be seen in the marks themselves, so
        # confidence has to fall with distance from the human's own frame.
        assert last.confidence < first.confidence

    def test_carried_marks_explain_themselves(self, follower):
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=0)

        result = follower.update(shifted(scene, 60, 0), frame_id=1)

        assert any("carried" in reason for reason in result.reasons)
        assert any("camera moved" in reason for reason in result.reasons)

    def test_losing_the_camera_drops_the_marks_rather_than_guessing(self, follower):
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=0)

        # A completely unrelated frame: registration cannot succeed, and
        # transforming the marks by a matrix nobody believes would be worse
        # than admitting the pitch is gone.
        result = follower.update(np.zeros((*SIZE, 3), dtype=np.uint8), frame_id=1)

        assert result.lost
        assert result.correspondences == []
        assert any("lost track" in warning for warning in result.warnings)
        assert not follower.has_anchor

    def test_nothing_anchored_means_nothing_reported(self, follower):
        assert follower.update(textured_scene(), frame_id=1) is None

    def test_the_anchor_frame_itself_is_reported_as_untouched(self, follower):
        scene = textured_scene()
        follower.anchor(marks([(400.0, 300.0)]), scene, frame_id=12)

        result = follower.update(scene, frame_id=12)

        assert result.confidence == 1.0
        assert result.frames_since_anchor == 0
        assert "marked on this frame" in result.reasons[0]


class TestPlayersAreExcluded:
    def test_a_moving_player_does_not_drag_the_camera_estimate(self, follower):
        """A feature on a running player reports the player's motion, not the
        camera's — so players are masked out before features are chosen."""
        scene = textured_scene()
        box = (500.0, 250.0, 620.0, 460.0)

        # A big, high-contrast "player" that moves the *other* way to the pan.
        def frame_with_player(camera_dx: float, player_x: int) -> np.ndarray:
            image = shifted(scene, camera_dx, 0)
            cv2.rectangle(image, (player_x, 250), (player_x + 120, 460), (10, 10, 250), -1)
            for stripe in range(player_x, player_x + 120, 12):
                cv2.line(image, (stripe, 250), (stripe, 460), (250, 250, 10), 3)
            return image

        follower.anchor(marks([(300.0, 600.0)]), frame_with_player(0, 500), 0, [box])
        result = follower.update(frame_with_player(30, 380), frame_id=1, exclude_boxes=[
            (380.0, 250.0, 500.0, 460.0)
        ])

        assert result.correspondences[0].image_xy[0] == pytest.approx(330, abs=4.0)
