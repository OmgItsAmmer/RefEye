"""Carrying one calibration across a shot as the camera moves.

## The bug this exists to fix

Marked landmarks are image points. Mark four corners on one frame and the
calibration is right — for that frame. A broadcast camera then pans, tilts and
zooms constantly, and those four marks stay pinned to the same *pixels* while
the pitch moves underneath them. Every measurement built on them is quietly
wrong from that moment on.

Worse, nothing catches it. The four marked points still agree with each other
perfectly, so the reprojection error stays at zero while the calibration
describes a camera that stopped existing several seconds ago. That is the
exact failure this project keeps refusing to ship: confident and wrong.

## Why a single homography per frame is the right correction

A broadcast main camera pans, tilts and zooms from a fixed position. It
rotates; it does not travel. Under pure rotation and zoom, **every** static
point in the scene — pitch, stands, hoardings, roof — moves between frames by
one shared homography, regardless of how far away it is. So the camera's
movement can be recovered from features anywhere in the frame, and applied to
the marks to carry them along with the pitch.

The exception is anything that moves by itself, which is why players are
masked out before features are chosen: a feature on a running player reports
that player's motion, not the camera's.

## Where the honesty lives

Drift is cumulative and cannot be detected from the marks themselves, so the
confidence reported here is not measured from them. It comes from the quality
of each frame-to-frame registration (how many features agreed) and from how
long the calibration has been carried since a human last placed it. Both are
reported in plain words, and when registration fails outright the follower
says it has lost the pitch rather than continuing to transform points by a
matrix it no longer believes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from offside.pitch_calibration.homography import PointCorrespondence
from vision.tracking.camera_motion import CameraMotionEstimator, to_gray, transform_point

Box = tuple[float, float, float, float]


@dataclass
class FollowResult:
    """Where the operator's marks have moved to, and how much to trust them."""

    correspondences: list[PointCorrespondence]
    #: Multiplier on the calibration's own confidence, falling as the marks
    #: are carried further from the frame a human actually placed them on.
    confidence: float
    frames_since_anchor: int
    #: How far the marks moved on this frame, in pixels — the visible sign
    #: that the camera is moving at all.
    moved_px: float
    inlier_ratio: float
    lost: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CalibrationFollower:
    """Follows the camera so a calibration outlives the frame it was made on."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_features: int = 600,
        quality_level: float = 0.01,
        min_distance_px: int = 12,
        redetect_below: int = 120,
        ransac_threshold_px: float = 3.0,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.5,
        flow_window_px: int = 31,
        flow_pyramid_levels: int = 4,
        confidence_decay_per_frame: float = 0.004,
        min_confidence: float = 0.3,
        exclude_box_margin: float = 0.25,
    ):
        self._enabled = enabled
        self._redetect_below = redetect_below
        self._motion = CameraMotionEstimator(
            max_features=max_features,
            quality_level=quality_level,
            min_distance_px=min_distance_px,
            ransac_threshold_px=ransac_threshold_px,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            flow_window_px=flow_window_px,
            flow_pyramid_levels=flow_pyramid_levels,
            exclude_box_margin=exclude_box_margin,
        )
        self._decay = confidence_decay_per_frame
        self._min_confidence = min_confidence

        self._correspondences: list[PointCorrespondence] = []
        self._previous_gray: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._anchor_frame: int | None = None
        self._frames_since_anchor = 0
        self._confidence = 1.0

    @classmethod
    def from_config(cls, config):
        return cls(
            enabled=config.enabled,
            max_features=config.max_features,
            quality_level=config.quality_level,
            min_distance_px=config.min_distance_px,
            redetect_below=config.redetect_below,
            ransac_threshold_px=config.ransac_threshold_px,
            min_inliers=config.min_inliers,
            min_inlier_ratio=config.min_inlier_ratio,
            flow_window_px=config.flow_window_px,
            flow_pyramid_levels=config.flow_pyramid_levels,
            confidence_decay_per_frame=config.confidence_decay_per_frame,
            min_confidence=config.min_confidence,
            exclude_box_margin=config.exclude_box_margin,
        )

    @property
    def has_anchor(self) -> bool:
        return bool(self._correspondences)

    @property
    def correspondences(self) -> list[PointCorrespondence]:
        return list(self._correspondences)

    @property
    def frames_since_anchor(self) -> int:
        return self._frames_since_anchor

    def reset(self) -> None:
        """Forget the marks and the camera — a cut, a new clip, a re-mark."""
        self._correspondences = []
        self._previous_gray = None
        self._features = None
        self._anchor_frame = None
        self._frames_since_anchor = 0
        self._confidence = 1.0

    def anchor(
        self,
        correspondences: list[PointCorrespondence],
        image: np.ndarray,
        frame_id: int,
        exclude_boxes: list[Box] | None = None,
    ) -> None:
        """Pin the operator's marks to this frame and start following it."""
        self._correspondences = list(correspondences)
        self._anchor_frame = frame_id
        self._frames_since_anchor = 0
        self._confidence = 1.0
        self._previous_gray = to_gray(image)
        self._features = self._motion.detect(self._previous_gray, exclude_boxes)

    def update(
        self,
        image: np.ndarray,
        frame_id: int,
        exclude_boxes: list[Box] | None = None,
    ) -> FollowResult | None:
        """Move the marks with the camera. None when nothing is anchored."""
        if not self._enabled or not self._correspondences:
            return None
        if frame_id == self._anchor_frame:
            return FollowResult(
                correspondences=self.correspondences,
                confidence=1.0,
                frames_since_anchor=0,
                moved_px=0.0,
                inlier_ratio=1.0,
                reasons=["the pitch was marked on this frame"],
            )

        gray = to_gray(image)
        if self._previous_gray is None or self._features is None:
            self._previous_gray = gray
            self._features = self._motion.detect(gray, exclude_boxes)
            return self._carry_on(0.0, 1.0, [])

        estimate = self._motion.estimate(self._previous_gray, self._features, gray)
        self._previous_gray = gray

        if estimate is None:
            # Do not keep transforming points by a matrix we do not believe.
            self.reset()
            return FollowResult(
                correspondences=[],
                confidence=0.0,
                frames_since_anchor=0,
                moved_px=0.0,
                inlier_ratio=0.0,
                lost=True,
                warnings=[
                    "lost track of the camera — the pitch marks belonged to an "
                    "earlier view and cannot be carried any further; mark the "
                    "pitch again on this frame"
                ],
            )

        moved = self._apply(estimate.matrix)
        self._frames_since_anchor += 1
        # Deliberately a ratchet: confidence can fall but never recover. Error
        # injected by one poor registration is carried in the marks from then
        # on, so a run of clean frames afterwards is not evidence that it went
        # away. Only a human re-marking the pitch resets this.
        self._confidence = max(
            0.0, min(self._confidence, estimate.inlier_ratio) - self._decay
        )

        if len(estimate.features) < self._redetect_below:
            self._features = self._motion.detect(gray, exclude_boxes)
        else:
            self._features = estimate.features

        return self._carry_on(moved, estimate.inlier_ratio, [])

    # -- internals ----------------------------------------------------------

    def _carry_on(
        self, moved: float, inlier_ratio: float, warnings: list[str]
    ) -> FollowResult:
        reasons = []
        if self._frames_since_anchor:
            reasons.append(
                f"the pitch was marked {self._frames_since_anchor} frame(s) ago and "
                "has been carried along with the camera since"
            )
        if moved > 1.0:
            reasons.append(f"the camera moved {moved:.0f}px on this frame")

        result = FollowResult(
            correspondences=self.correspondences,
            confidence=self._confidence,
            frames_since_anchor=self._frames_since_anchor,
            moved_px=moved,
            inlier_ratio=inlier_ratio,
            reasons=reasons,
            warnings=list(warnings),
        )
        if self._confidence < self._min_confidence:
            result.warnings.append(
                "the marks have been carried a long way from the frame they were "
                "placed on — re-mark the pitch before trusting distances"
            )
        return result

    def _apply(self, motion: np.ndarray) -> float:
        moved = 0.0
        updated: list[PointCorrespondence] = []
        for correspondence in self._correspondences:
            point = transform_point(motion, correspondence.image_xy)
            moved = max(
                moved,
                float(
                    np.hypot(
                        point[0] - correspondence.image_xy[0],
                        point[1] - correspondence.image_xy[1],
                    )
                ),
            )
            updated.append(
                PointCorrespondence(
                    image_xy=point,
                    pitch_xy=correspondence.pitch_xy,
                    landmark=correspondence.landmark,
                )
            )
        self._correspondences = updated
        return moved
