"""Pose estimation adapter — body keypoints for players already detected.

Implements the `PoseEstimator` protocol (core/interfaces/vision.py). All
Ultralytics-specific types stay inside this file; callers see only
`PlayerPose` objects in full-frame coordinates.

## Why crops, not whole frames (measured, not assumed)

This is a **top-down** pose estimator: the M1 detector says who is a player,
and each of those boxes is cropped, padded, upscaled and posed individually.
That is not a stylistic choice. Measured on the client reference clip
(`data/videos/client_m2_test_video.mp4`, 1280x720 broadcast footage where
players stand roughly 90-100px tall):

    whole-frame pose, imgsz 1280 :  1-2 of ~17 players found
    per-player crops, imgsz 256  : 11-13 of ~17 players found,
                                   most with a confident ankle

A wide broadcast shot simply does not give a pose model enough pixels per
player to work with; cropping restores them. The cost is one small inference
per player instead of one large one per frame, which batches well on GPU.

## Why one PlayerPose per detection, always

The detector and tracker stay authoritative for identity — a pose never
creates, removes or renames a player. Players whose pose fails still come
back, carrying a bounding-box fallback ground point and a warning, because
silently dropping them would quietly corrupt the second-last-defender
ranking in M2.5: the defender who is hardest to pose (occluded, distant) is
exactly the one whose absence would change the offside line.

## The neighbour guard

Padding a crop to catch a stretched leg also drags in whoever is standing
next to the player. The pose model is asked for several people per crop and
the one whose box best overlaps the *original* detection wins; if nothing
overlaps well enough, the pose is rejected rather than attached to the wrong
player. A silently swapped skeleton would be far more damaging than a missing
one.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.config.paths import resolve
from core.domain.models import Detection, FramePacket
from core.errors.exceptions import ModelLoadError
from observability.logging.setup import get_logger
from offside.body_keypoints.ground_point import estimate_ground_point
from offside.body_keypoints.keypoints import (
    COCO_KEYPOINT_NAMES,
    GroundPoint,
    Keypoint,
    PlayerPose,
)
from vision.detection.classes import PERSON_CLASSES

logger = get_logger(__name__)


class YoloPoseEstimator:
    """Ultralytics YOLO-pose wrapped behind the PoseEstimator protocol."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        crop_imgsz: int = 256,
        keypoint_confidence_threshold: float = 0.5,
        person_confidence_threshold: float = 0.25,
        match_iou: float = 0.3,
        crop_padding_x: float = 0.25,
        crop_padding_y: float = 0.15,
        min_box_height_px: float = 24.0,
        knee_projection_penalty: float = 0.55,
        bbox_fallback_penalty: float = 0.3,
        max_players_per_frame: int = 30,
        batch_size: int = 32,
    ):
        self._checkpoint = checkpoint
        self._device = device
        self._crop_imgsz = crop_imgsz
        self._keypoint_confidence = keypoint_confidence_threshold
        self._person_confidence = person_confidence_threshold
        self._match_iou = match_iou
        self._crop_padding_x = crop_padding_x
        self._crop_padding_y = crop_padding_y
        self._min_box_height_px = min_box_height_px
        self._knee_projection_penalty = knee_projection_penalty
        self._bbox_fallback_penalty = bbox_fallback_penalty
        self._max_players_per_frame = max_players_per_frame
        self._batch_size = batch_size

        self._model = None
        self._version = "unknown"

    # -- PoseEstimator protocol --------------------------------------------

    @property
    def model_name(self) -> str:
        return f"yolo-pose:{Path(self._checkpoint).stem}"

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def keypoint_confidence_threshold(self) -> float:
        """Exposed so downstream phases score against the same bar this
        estimator used, instead of inventing a second threshold."""
        return self._keypoint_confidence

    def load(self) -> None:
        """Load weights once. Never called per analysis trigger (section 37)."""
        if self._model is not None:
            return

        path = resolve(self._checkpoint)
        if not path.exists():
            raise ModelLoadError(
                f"Pose checkpoint not found: {path}. Offside body-point "
                "estimation stays unavailable until it is present — see "
                "models/pose/README.md for how to obtain it."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ModelLoadError("ultralytics is not installed") from exc

        try:
            self._model = YOLO(str(path))
        except Exception as exc:
            raise ModelLoadError(f"Could not load pose model {path}: {exc}") from exc

        try:
            import ultralytics

            self._version = ultralytics.__version__
        except Exception:  # noqa: BLE001 — a missing version string is cosmetic
            self._version = "unknown"

        logger.info(
            "model_ready",
            component="pose_estimator",
            model=self.model_name,
            version=self._version,
            device=self._device,
        )

    def warmup(self) -> None:
        """One throwaway crop so the operator's first trigger isn't slow."""
        if self._model is None:
            self.load()

        blank = np.zeros((self._crop_imgsz, self._crop_imgsz // 2, 3), dtype=np.uint8)
        started = time.perf_counter()
        self._predict([blank])
        logger.info(
            "model_warmed_up",
            component="pose_estimator",
            model=self.model_name,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def estimate(
        self,
        frame: FramePacket,
        detections: list[Detection],
    ) -> list[PlayerPose]:
        """One `PlayerPose` per person detection, in frame coordinates."""
        return self.estimate_batch([frame], [detections])[0]

    def estimate_batch(
        self,
        frames: list[FramePacket],
        detections_per_frame: list[list[Detection]],
    ) -> list[list[PlayerPose]]:
        """Pose every player across several frames in as few passes as possible.

        The triggered offside path needs a handful of frames around the
        contact moment, each with a dozen-plus players; per-crop calls would
        spend most of the time on launch overhead.
        """
        if self._model is None:
            self.load()

        # Flatten to one job per player so a single batch can span frames —
        # a dozen crops rarely fills the GPU on its own.
        jobs: list[_CropJob] = []
        results: list[list[PlayerPose]] = [[] for _ in frames]

        for frame_index, (frame, detections) in enumerate(
            zip(frames, detections_per_frame)
        ):
            people = [d for d in detections if d.class_name in PERSON_CLASSES]
            if len(people) > self._max_players_per_frame:
                # Keep the most confident; a frame with 60 "people" is picking
                # up the crowd, and posing all of them helps nobody.
                people = sorted(people, key=lambda d: d.confidence, reverse=True)[
                    : self._max_players_per_frame
                ]

            for detection in people:
                job = _CropJob(
                    frame_index=frame_index,
                    frame_id=frame.frame_id,
                    detection=detection,
                )
                if frame.image is None:
                    job.warning = "this frame carried no image to pose against"
                else:
                    job.attach_crop(
                        frame.image,
                        padding_x=self._crop_padding_x,
                        padding_y=self._crop_padding_y,
                        min_box_height_px=self._min_box_height_px,
                    )
                jobs.append(job)

        posed = 0
        for chunk in _chunks([j for j in jobs if j.crop is not None], self._batch_size):
            predictions = self._predict([job.crop for job in chunk])
            for job, prediction in zip(chunk, predictions):
                job.keypoints = self._keypoints_for(job, prediction)
                if job.keypoints:
                    posed += 1

        for job in jobs:
            results[job.frame_index].append(self._to_player_pose(job))

        logger.debug(
            "pose_estimated",
            component="pose_estimator",
            players=len(jobs),
            posed=posed,
            frames=len(frames),
        )
        return results

    # -- internals ----------------------------------------------------------

    def _predict(self, crops: list[np.ndarray]):
        # max_det > 1 deliberately: a padded crop often contains a neighbour,
        # and the right answer is to pick the best-overlapping person rather
        # than trust that the most confident one is the player we asked about.
        return self._model.predict(
            crops,
            verbose=False,
            conf=self._person_confidence,
            imgsz=self._crop_imgsz,
            device=self._device,
            max_det=4,
        )

    def _keypoints_for(self, job: _CropJob, prediction) -> dict[str, Keypoint]:
        boxes = getattr(prediction, "boxes", None)
        keypoints = getattr(prediction, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return {}

        xyxy = boxes.xyxy.cpu().numpy()
        target = job.detection_box_in_crop()

        best_index, best_iou = -1, 0.0
        for index, box in enumerate(xyxy):
            overlap = _iou(tuple(box), target)
            if overlap > best_iou:
                best_index, best_iou = index, overlap

        if best_index < 0 or best_iou < self._match_iou:
            job.warning = (
                "the pose model locked onto a different player inside this "
                "crop, so its skeleton was rejected"
            )
            return {}

        xy = keypoints.xy[best_index].cpu().numpy()
        confidences = keypoints.conf
        conf = (
            confidences[best_index].cpu().numpy()
            if confidences is not None
            else np.ones(len(xy), dtype=np.float32)
        )

        offset_x, offset_y = job.crop_origin
        return {
            name: Keypoint(
                name=name,
                xy=(float(point[0]) + offset_x, float(point[1]) + offset_y),
                confidence=float(score),
            )
            for name, point, score in zip(COCO_KEYPOINT_NAMES, xy, conf)
        }

    def _to_player_pose(self, job: _CropJob) -> PlayerPose:
        detection = job.detection
        ground = estimate_ground_point(
            job.keypoints,
            detection.bbox_xyxy,
            detection.confidence,
            min_keypoint_confidence=self._keypoint_confidence,
            knee_projection_penalty=self._knee_projection_penalty,
            bbox_fallback_penalty=self._bbox_fallback_penalty,
        )

        warnings: list[str] = []
        if job.warning:
            warnings.append(job.warning)
        if not job.keypoints and not job.warning:
            warnings.append("no pose could be estimated for this player")
        if not ground.is_measured:
            warnings.append(ground.reason)

        return PlayerPose(
            frame_id=job.frame_id,
            bbox_xyxy=detection.bbox_xyxy,
            detection_confidence=detection.confidence,
            ground_point=ground,
            source_model=self.model_name,
            keypoints=job.keypoints,
            warnings=warnings,
        )


class _CropJob:
    """One player box awaiting (or carrying) its skeleton."""

    __slots__ = (
        "crop",
        "crop_origin",
        "detection",
        "frame_id",
        "frame_index",
        "keypoints",
        "warning",
    )

    def __init__(self, frame_index: int, frame_id: int, detection: Detection):
        self.frame_index = frame_index
        self.frame_id = frame_id
        self.detection = detection
        self.crop: np.ndarray | None = None
        self.crop_origin: tuple[float, float] = (0.0, 0.0)
        self.keypoints: dict[str, Keypoint] = {}
        self.warning: str | None = None

    def attach_crop(
        self,
        image: np.ndarray,
        *,
        padding_x: float,
        padding_y: float,
        min_box_height_px: float,
    ) -> None:
        x1, y1, x2, y2 = self.detection.bbox_xyxy
        height = y2 - y1

        if height < min_box_height_px:
            # Below this a player is a smudge; a skeleton drawn on it would
            # look like data while being noise.
            self.warning = (
                f"player is too small in frame ({height:.0f}px tall) for "
                "reliable body points"
            )
            return

        # Wider padding horizontally than vertically: what escapes a player
        # box is usually a stretched leg or trailing arm, not extra headroom.
        width = x2 - x1
        pad_x, pad_y = width * padding_x, height * padding_y

        frame_height, frame_width = image.shape[:2]
        cx1 = max(0, round(x1 - pad_x))
        cy1 = max(0, round(y1 - pad_y))
        cx2 = min(frame_width, round(x2 + pad_x))
        cy2 = min(frame_height, round(y2 + pad_y))

        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            self.warning = "player box fell outside the frame"
            return

        self.crop = image[cy1:cy2, cx1:cx2]
        self.crop_origin = (float(cx1), float(cy1))

    def detection_box_in_crop(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.detection.bbox_xyxy
        offset_x, offset_y = self.crop_origin
        return (x1 - offset_x, y1 - offset_y, x2 - offset_x, y2 - offset_y)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


__all__ = ["GroundPoint", "PlayerPose", "YoloPoseEstimator"]
