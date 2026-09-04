"""YOLO object detector adapter.

Implements the `ObjectDetector` protocol (architecture.md section 14). All
Ultralytics-specific types stay inside this file; callers see only
`Detection` objects with domain class names.

A note on ball detection, per architecture.md section 14: a generic COCO
`sports ball` class is NOT assumed accurate enough for broadcast football.
The ball is often tiny, motion-blurred, occluded, or confused with pitch
markings. This adapter therefore takes a separate, lower confidence threshold
for the ball than for people, and downstream code is built to tolerate frames
with no ball at all (architecture.md section 25). Replacing this with a
fine-tuned football detector is a config change, not a code change.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.domain.models import Detection, FramePacket
from core.config.paths import resolve
from core.errors.exceptions import ModelLoadError
from observability.logging.setup import get_logger
from vision.detection.classes import BALL, COCO_TO_DOMAIN

logger = get_logger(__name__)


class YoloObjectDetector:
    """Ultralytics YOLO wrapped behind the ObjectDetector protocol."""

    def __init__(
        self,
        checkpoint: str,
        confidence_threshold: float = 0.35,
        ball_confidence_threshold: float | None = None,
        device: str = "cpu",
        imgsz: int = 640,
        max_detections: int = 60,
    ):
        self._checkpoint = checkpoint
        self._confidence = confidence_threshold
        # The ball is the hardest and most important object; hold it to a
        # looser bar than people and let the trajectory logic filter noise.
        self._ball_confidence = (
            ball_confidence_threshold
            if ball_confidence_threshold is not None
            else confidence_threshold * 0.5
        )
        self._device = device
        self._imgsz = imgsz
        self._max_detections = max_detections

        self._model = None
        self._names: dict[int, str] = {}
        self._version = "unknown"

    # -- ObjectDetector protocol -------------------------------------------

    @property
    def model_name(self) -> str:
        return f"yolo:{Path(self._checkpoint).stem}"

    @property
    def model_version(self) -> str:
        return self._version

    def load(self) -> None:
        """Load weights once. Never called per analysis trigger (section 37)."""
        if self._model is not None:
            return

        path = resolve(self._checkpoint)
        if not path.exists():
            raise ModelLoadError(f"Detector checkpoint not found: {path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ModelLoadError("ultralytics is not installed") from exc

        try:
            self._model = YOLO(str(path))
            self._names = dict(self._model.names)
        except Exception as exc:  # noqa: BLE001 — surface as a domain error
            raise ModelLoadError(f"Could not load detector {path}: {exc}") from exc

        try:
            import ultralytics

            self._version = ultralytics.__version__
        except Exception:  # noqa: BLE001
            self._version = "unknown"

        logger.info(
            "model_ready",
            component="detector",
            model=self.model_name,
            version=self._version,
            device=self._device,
        )

    def warmup(self) -> None:
        """Run one throwaway inference so the first real trigger isn't slow."""
        if self._model is None:
            self.load()
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        started = time.perf_counter()
        self._predict(blank)
        logger.info(
            "model_warmed_up",
            component="detector",
            model=self.model_name,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def detect(self, frame: FramePacket) -> list[Detection]:
        if frame.image is None:
            return []
        if self._model is None:
            self.load()

        results = self._predict(frame.image)
        return self._to_detections(results, frame.frame_id)

    def detect_batch(self, frames: list[FramePacket]) -> list[list[Detection]]:
        """Detect over several frames in one call.

        Batching matters for triggered analysis, where a whole clip is
        processed at once — per-frame calls waste most of the GPU time on
        launch overhead.
        """
        usable = [f for f in frames if f.image is not None]
        if not usable:
            return [[] for _ in frames]
        if self._model is None:
            self.load()

        images = [f.image for f in usable]
        results = self._predict(images)

        by_frame: dict[int, list[Detection]] = {}
        for frame, result in zip(usable, results):
            by_frame[frame.frame_id] = self._to_detections([result], frame.frame_id)

        return [by_frame.get(f.frame_id, []) for f in frames]

    # -- internals ----------------------------------------------------------

    def _predict(self, source):
        # The lowest of the per-class thresholds is applied at inference time;
        # per-class filtering happens in _to_detections.
        return self._model.predict(
            source,
            verbose=False,
            conf=min(self._confidence, self._ball_confidence),
            imgsz=self._imgsz,
            max_det=self._max_detections,
            device=self._device,
        )

    def _to_detections(self, results, frame_id: int) -> list[Detection]:
        detections: list[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, classes):
                native = self._names.get(int(cls_id))
                domain_class = COCO_TO_DOMAIN.get(native)
                if domain_class is None:
                    continue

                threshold = self._ball_confidence if domain_class == BALL else self._confidence
                if conf < threshold:
                    continue

                detections.append(
                    Detection(
                        frame_id=frame_id,
                        class_name=domain_class,
                        confidence=float(conf),
                        bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        source_model=self.model_name,
                    )
                )

        return detections
