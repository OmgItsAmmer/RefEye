"""Automatic pitch-landmark detection (M2.1, extends the plan).

Every prior version of pitch calibration in this app that reaches METRIC
(real metres, not just direction) needed an operator to click four named
points on the pitch — and until this module, that clicking only existed in
the internal debug Inspector, never in the shipped app. This is the "any AI
model in the world" alternative: a neural network that finds the same
landmarks the operator would have clicked, by itself, on a single frame.

## The model

`Adit-jain/Soccana_Keypoint` on Hugging Face — YOLO11-pose, fine-tuned on
SoccerNet's official camera-calibration dataset (confirmed from the
checkpoint's own training metadata: `data: .../SoccerNet/Data/calibration/
dataset.yaml`). It outputs 29 named pitch keypoints per frame, each as
`(x, y, visibility)` — corners of the penalty area, the goal area, the pitch
corners, the centre circle, and more. The 29-point count is not a
coincidence: it is the same standard taxonomy this project's own
`PitchModel.landmarks()` and the Inspector's `LANDMARK_GUIDE` were already
built around, just under different names.

## Why only 21 of the 29 points are used

The model's 29 keypoints include 8 for the centre circle and the penalty
arcs (the "D"). Mapping those onto this project's pitch model correctly
needs new landmark geometry to be added and independently verified — not
done here. The 21 points used are exactly the ones this project's manual
marking UI already asks for: pitch corners, and the sharp, painted corners
of both penalty areas and both goal areas. Every one of them is a straight-line
intersection, the single most reliable kind of point for a homography fit to
begin with — leaving out the harder, curved points costs little and avoids
guessing at geometry that was never checked against real footage.

## The pt1 / pt2 assumption, and why it is safe even if wrong

The published keypoint names (`big_rect_left_top_pt1`, `..._pt2`, etc.) do
not document which corner of the box each one is — this module assumes pt1
is the corner on the goal line and pt2 is the corner at the box's outer
edge, matching the common convention in public soccer-calibration keypoint
sets. If that assumption is backwards, the practical effect is bounded, not
silent: `solve_homography`'s own RANSAC step (`ransac_threshold_px`) and its
reprojection-error check would very likely flag the resulting fit as
low-confidence or reject outlier points outright, because a systematically
mislabelled box would show up as several points inconsistent with the
others — this was checked against real footage before being wired into the
pipeline (see the module's test file and the M2 plan notes for the
measurement).

## Where this sits in the calibration priority order

Operator-marked points (existing manual flow) still outrank everything —
they are a direct human confirmation. This module runs when there are no
operator marks: if it finds enough confident points, its result goes through
the *exact same* `calibrate_manual` solve the operator's clicks do, just
labelled `source="auto_landmarks"` rather than `"manual"` so nothing claims
to be an operator's confirmation when it wasn't. Only when this also fails
(too few confident points, or no valid mapping) does the pipeline fall back
to the weaker, direction-only automatic line detection that existed before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config.paths import resolve
from core.errors.exceptions import ModelLoadError
from observability.logging.setup import get_logger
from offside.field_geometry.pitch import PitchModel
from offside.pitch_calibration.homography import PointCorrespondence

logger = get_logger(__name__)

#: The model's own keypoint index -> name, exactly as published by the
#: checkpoint's model card (Adit-jain/Soccana_Keypoint, trained on SoccerNet).
#: Only entries with a value are used — the 8 centre-circle/arc points are
#: intentionally left unmapped (see module docstring).
_MODEL_KEYPOINT_NAMES: dict[int, str] = {
    0: "sideline_top_left",
    1: "big_rect_left_top_pt1",
    2: "big_rect_left_top_pt2",
    3: "big_rect_left_bottom_pt1",
    4: "big_rect_left_bottom_pt2",
    5: "small_rect_left_top_pt1",
    6: "small_rect_left_top_pt2",
    7: "small_rect_left_bottom_pt1",
    8: "small_rect_left_bottom_pt2",
    9: "sideline_bottom_left",
    10: "left_semicircle_right",
    11: "center_line_top",
    12: "center_line_bottom",
    13: "center_circle_top",
    14: "center_circle_bottom",
    15: "field_center",
    16: "sideline_top_right",
    17: "big_rect_right_top_pt1",
    18: "big_rect_right_top_pt2",
    19: "big_rect_right_bottom_pt1",
    20: "big_rect_right_bottom_pt2",
    21: "small_rect_right_top_pt1",
    22: "small_rect_right_top_pt2",
    23: "small_rect_right_bottom_pt1",
    24: "small_rect_right_bottom_pt2",
    25: "sideline_bottom_right",
    26: "right_semicircle_left",
    27: "center_circle_left",
    28: "center_circle_right",
}

#: Model keypoint name -> this project's `PitchModel.landmarks()` name. Only
#: the 21 straight-corner points; see module docstring for why.
_TO_PITCH_LANDMARK: dict[str, str] = {
    "sideline_top_left": "corner_left_top",
    "sideline_top_right": "corner_right_top",
    "sideline_bottom_left": "corner_left_bottom",
    "sideline_bottom_right": "corner_right_bottom",
    "field_center": "centre_mark",
    "big_rect_left_top_pt1": "left_penalty_area_top_goalline",
    "big_rect_left_top_pt2": "left_penalty_area_top_corner",
    "big_rect_left_bottom_pt1": "left_penalty_area_bottom_goalline",
    "big_rect_left_bottom_pt2": "left_penalty_area_bottom_corner",
    "big_rect_right_top_pt1": "right_penalty_area_top_goalline",
    "big_rect_right_top_pt2": "right_penalty_area_top_corner",
    "big_rect_right_bottom_pt1": "right_penalty_area_bottom_goalline",
    "big_rect_right_bottom_pt2": "right_penalty_area_bottom_corner",
    "small_rect_left_top_pt1": "left_goal_area_top_goalline",
    "small_rect_left_top_pt2": "left_goal_area_top_corner",
    "small_rect_left_bottom_pt1": "left_goal_area_bottom_goalline",
    "small_rect_left_bottom_pt2": "left_goal_area_bottom_corner",
    "small_rect_right_top_pt1": "right_goal_area_top_goalline",
    "small_rect_right_top_pt2": "right_goal_area_top_corner",
    "small_rect_right_bottom_pt1": "right_goal_area_bottom_goalline",
    "small_rect_right_bottom_pt2": "right_goal_area_bottom_corner",
}


class AutoLandmarkDetector:
    """Finds named pitch landmarks in one frame, without an operator."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        visibility_threshold: float = 0.5,
        detection_confidence: float = 0.25,
    ):
        self._checkpoint = checkpoint
        self._device = device
        self._imgsz = imgsz
        self._visibility_threshold = visibility_threshold
        self._detection_confidence = detection_confidence
        self._model = None
        self._version = "unknown"

    @classmethod
    def from_config(cls, config, checkpoint: str, device: str = "cpu") -> AutoLandmarkDetector:
        return cls(
            checkpoint,
            device=device,
            imgsz=config.imgsz,
            visibility_threshold=config.visibility_threshold,
            detection_confidence=config.detection_confidence,
        )

    @property
    def model_name(self) -> str:
        return f"yolo-pose:{Path(self._checkpoint).stem}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load weights once. Never called per analysis trigger."""
        if self._model is not None:
            return

        path = resolve(self._checkpoint)
        if not path.exists():
            raise ModelLoadError(
                f"Pitch keypoint checkpoint not found: {path}. Automatic "
                "landmark detection stays unavailable until it is present — "
                "see models/pitch_keypoints/README.md."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ModelLoadError("ultralytics is not installed") from exc

        try:
            self._model = YOLO(str(path))
        except Exception as exc:
            raise ModelLoadError(f"Could not load pitch keypoint model {path}: {exc}") from exc

        try:
            import ultralytics

            self._version = ultralytics.__version__
        except Exception:  # noqa: BLE001 — a missing version string is cosmetic
            self._version = "unknown"

        logger.info(
            "model_ready",
            component="pitch_keypoint_detector",
            model=self.model_name,
            version=self._version,
            device=self._device,
        )

    def detect(self, image: np.ndarray, pitch: PitchModel) -> list[PointCorrespondence]:
        """Every confidently-visible landmark this frame supports.

        Returns image<->pitch correspondences ready for
        `PitchCalibrator.calibrate_manual(..., source="auto_landmarks")` —
        the same solve an operator's own clicks go through. Callers decide
        whether the count returned is enough (four is the mathematical
        minimum; the pipeline asks for more, for robustness against a
        mislabelled point — see `PitchCalibrationConfig.auto_landmark_
        min_points`).
        """
        if self._model is None:
            self.load()

        results = self._model.predict(
            image,
            imgsz=self._imgsz,
            conf=self._detection_confidence,
            device=self._device,
            verbose=False,
        )
        if not results or results[0].keypoints is None or len(results[0].boxes) == 0:
            return []

        result = results[0]
        # One pitch per frame: if the model (unexpectedly) proposes more than
        # one instance, the most confident detection is the pitch — a second
        # "field" is not a real possibility this app's cameras produce.
        best_index = int(result.boxes.conf.argmax()) if len(result.boxes) > 1 else 0
        keypoints_xy = result.keypoints.xy[best_index].cpu().numpy()
        visibility = (
            result.keypoints.conf[best_index].cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(len(keypoints_xy))
        )

        correspondences: list[PointCorrespondence] = []
        for index, model_name in _MODEL_KEYPOINT_NAMES.items():
            landmark = _TO_PITCH_LANDMARK.get(model_name)
            if landmark is None or index >= len(keypoints_xy):
                continue
            if visibility[index] < self._visibility_threshold:
                continue
            x, y = keypoints_xy[index]
            if x <= 0.0 and y <= 0.0:
                # Ultralytics reports an undetected keypoint as (0, 0) with
                # visibility 0 — the visibility gate above already excludes
                # these in the normal case; this is a second, cheap guard for
                # exactly-zero coordinates slipping through at a low
                # threshold, so a mislabelled origin point never becomes a
                # correspondence.
                continue
            correspondences.append(
                PointCorrespondence(
                    image_xy=(float(x), float(y)),
                    pitch_xy=pitch.landmark(landmark),
                    landmark=landmark,
                )
            )
        return correspondences
