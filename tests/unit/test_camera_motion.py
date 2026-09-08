"""The shared camera-motion primitive both trackers compose.

Behavioural contract is the same one `offside.pitch_calibration.tracking`
already proved out for `CalibrationFollower` — moving a synthetic scene by a
*known* transform and checking the estimate recovers it — kept here as its
own test now that the logic is a standalone, reusable module rather than
private to one caller.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.tracking.camera_motion import (
    CameraMotionEstimator,
    to_gray,
    transform_box,
    transform_point,
)

SIZE = (720, 1280)


def textured_scene(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((*SIZE, 3), 60, dtype=np.uint8)
    image[:, :] = (50, 130, 60)
    for _ in range(400):
        x, y = int(rng.integers(0, SIZE[1])), int(rng.integers(0, SIZE[0]))
        colour = tuple(int(c) for c in rng.integers(80, 255, size=3))
        cv2.rectangle(image, (x, y), (x + 9, y + 9), colour, -1)
    return image


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(image, matrix, (SIZE[1], SIZE[0]), borderMode=cv2.BORDER_REFLECT)


@pytest.fixture
def estimator() -> CameraMotionEstimator:
    return CameraMotionEstimator()


class TestEstimatingMotion:
    def test_recovers_a_known_pan(self, estimator):
        scene = textured_scene()
        gray_a = to_gray(scene)
        features = estimator.detect(gray_a)

        gray_b = to_gray(shifted(scene, 35, 12))
        result = estimator.estimate(gray_a, features, gray_b)

        assert result is not None
        point = transform_point(result.matrix, (400.0, 300.0))
        assert point[0] == pytest.approx(435.0, abs=2.0)
        assert point[1] == pytest.approx(312.0, abs=2.0)
        assert result.inlier_ratio > 0.5

    def test_no_features_yet_is_none_not_a_crash(self, estimator):
        scene = textured_scene()
        assert estimator.estimate(to_gray(scene), None, to_gray(scene)) is None

    def test_an_unrelated_frame_is_not_registered(self, estimator):
        scene = textured_scene()
        features = estimator.detect(to_gray(scene))

        unrelated = np.zeros((*SIZE, 3), dtype=np.uint8)
        result = estimator.estimate(to_gray(scene), features, to_gray(unrelated))

        assert result is None

    def test_excluded_boxes_have_no_features_placed_in_them(self, estimator):
        scene = textured_scene()
        box = (400.0, 300.0, 700.0, 600.0)

        features = estimator.detect(to_gray(scene), exclude_boxes=[box])

        assert features is not None
        x1, y1, x2, y2 = box
        inside = [
            f for f in features.reshape(-1, 2) if x1 < f[0] < x2 and y1 < f[1] < y2
        ]
        assert inside == []


class TestTransformingBoxes:
    def test_a_translated_box_keeps_its_size(self):
        matrix = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
        box = (100.0, 100.0, 140.0, 200.0)

        warped = transform_box(matrix, box)

        assert warped == pytest.approx((120.0, 105.0, 160.0, 205.0))

    def test_no_matrix_needed_case_is_the_caller_s_job_not_this_function_s(self):
        """`transform_point` is not asked to special-case identity — callers
        (both trackers) skip calling it at all when there is nothing to
        compensate for, which is what the `motion_matrix is None` branches in
        `_Track.predict`/`observe` exist to do."""
        identity = np.eye(3)
        assert transform_point(identity, (12.0, 34.0)) == pytest.approx((12.0, 34.0))
