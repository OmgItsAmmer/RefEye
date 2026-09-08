"""Frame-to-frame camera motion: masked feature tracking + RANSAC homography.

Extracted from `offside.pitch_calibration.tracking.CalibrationFollower`, which
originally had this logic inline for one purpose (carrying operator-marked
pitch landmarks across a pan). `offside.player_identity.tracker.IdentityTracker`
needs exactly the same primitive for a different purpose (compensating a
track's predicted box for camera motion before association, so a pan does not
read as every player moving). Two copies of "how did the static scene move
between these two frames" would drift apart the first time one was tuned and
the other was not, so it is written once here and both callers compose it.

## Why this approach

A broadcast main camera pans, tilts and zooms from a fixed position — it
rotates, it does not translate. Under pure rotation and zoom, every static
point in the scene (pitch, stands, hoardings, roof) moves between frames by
one shared homography, regardless of how far away it is. So the motion can be
recovered from features anywhere in the frame, as long as anything that moves
on its own (a player) is masked out first — a feature on a running player
reports that player's motion, not the camera's.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]


@dataclass
class MotionEstimate:
    """One frame-to-frame registration."""

    matrix: np.ndarray
    inlier_ratio: float
    #: Tracked feature positions in the new frame, ready to seed the next
    #: call — re-detecting from scratch every frame is unnecessary and loses
    #: features that would have tracked fine.
    features: np.ndarray


class CameraMotionEstimator:
    """Stateless per call: the caller owns the previous frame's grey image and
    feature set, and passes both in — that is what lets `CalibrationFollower`
    and `IdentityTracker` each keep their own notion of "since when" without
    sharing mutable state neither of them should be reaching into.
    """

    def __init__(
        self,
        *,
        max_features: int = 600,
        quality_level: float = 0.01,
        min_distance_px: int = 12,
        ransac_threshold_px: float = 3.0,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.5,
        flow_window_px: int = 31,
        flow_pyramid_levels: int = 4,
        exclude_box_margin: float = 0.25,
    ):
        self._max_features = max_features
        self._quality_level = quality_level
        self._min_distance_px = min_distance_px
        self._ransac_threshold_px = ransac_threshold_px
        self._min_inliers = min_inliers
        self._min_inlier_ratio = min_inlier_ratio
        self._exclude_box_margin = exclude_box_margin
        # A fast pan moves the scene tens of pixels between frames; a window
        # too small loses it, which is exactly when registration matters most.
        self._flow_params = {
            "winSize": (flow_window_px, flow_window_px),
            "maxLevel": flow_pyramid_levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    def detect(
        self, gray: np.ndarray, exclude_boxes: list[Box] | None = None
    ) -> np.ndarray | None:
        """Corner features to track, with any moving object masked out."""
        mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
        for box in exclude_boxes or []:
            x1, y1, x2, y2 = box
            pad_x = (x2 - x1) * self._exclude_box_margin
            pad_y = (y2 - y1) * self._exclude_box_margin
            cv2.rectangle(
                mask,
                (int(x1 - pad_x), int(y1 - pad_y)),
                (int(x2 + pad_x), int(y2 + pad_y)),
                0,
                -1,
            )

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_features,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance_px,
            mask=mask,
        )

    def estimate(
        self,
        previous_gray: np.ndarray,
        previous_features: np.ndarray | None,
        gray: np.ndarray,
    ) -> MotionEstimate | None:
        """The homography that carries `previous_gray` onto `gray`, or None
        when registration cannot be trusted — never a matrix nobody believes.
        """
        if previous_features is None or len(previous_features) < self._min_inliers:
            return None

        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, previous_features, None, **self._flow_params
        )
        if moved is None or status is None:
            return None

        good = status.ravel() == 1
        before, after = previous_features[good], moved[good]
        if len(before) < self._min_inliers:
            return None

        matrix, mask = cv2.findHomography(
            before, after, cv2.RANSAC, self._ransac_threshold_px
        )
        if matrix is None or mask is None:
            return None

        inliers = int(mask.sum())
        ratio = inliers / max(1, len(before))
        if inliers < self._min_inliers or ratio < self._min_inlier_ratio:
            return None

        return MotionEstimate(
            matrix=matrix,
            inlier_ratio=ratio,
            features=after[mask.ravel() == 1].reshape(-1, 1, 2),
        )


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    vector = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(vector[2]) < 1e-9:
        return point
    return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))


def transform_box(matrix: np.ndarray, box: Box) -> Box:
    """A box's four corners, individually warped and re-enclosed.

    Individually, not just the two opposite corners: a homography with any
    rotation or shear turns a rectangle into a general quadrilateral, and
    warping only the corners a box happens to store would silently drop the
    other two.
    """
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    warped = [transform_point(matrix, corner) for corner in corners]
    xs = [p[0] for p in warped]
    ys = [p[1] for p in warped]
    return (min(xs), min(ys), max(xs), max(ys))
