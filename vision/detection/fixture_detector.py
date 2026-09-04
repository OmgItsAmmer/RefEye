"""Colour-blob detector for the synthetic development fixture.

**Development only. Never point this at real footage.**

The generated fixture (`tools/video_sampling/make_sample_clip.py`) draws
players and the ball as flat colour blocks. A COCO-trained detector correctly
sees nothing in them, which would leave the whole downstream pipeline
untestable in an environment with no licensed broadcast material.

This adapter closes that gap: it finds those exact colours and emits the same
`Detection` objects a real detector would, so tracking, feature extraction,
action spotting, refinement, de-duplication and ranking all run against
deterministic input. It is a test double sitting at the adapter seam the
architecture already defines — the reason that seam exists (section 4.3).

On real video it will detect essentially nothing, by design. Selected with:

    ai:
      detector:
        provider: "fixture"
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.models import Detection, FramePacket
from vision.detection.classes import BALL, GOALKEEPER, PLAYER

MODEL_NAME = "fixture-colour-blobs"
MODEL_VERSION = "1.0"

# HSV ranges matching the fixture generator's palette.
_HSV_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], str]] = {
    # name: (lower, upper, domain class)
    "ball": ((0, 0, 225), (180, 40, 255), BALL),
    "team_a": ((0, 150, 90), (12, 255, 255), PLAYER),
    "team_b": ((100, 150, 90), (130, 255, 255), PLAYER),
    "keeper": ((22, 150, 150), (38, 255, 255), GOALKEEPER),
}


class FixtureColorDetector:
    """Detects the synthetic fixture's colour blocks."""

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        min_person_area: int = 120,
        min_ball_area: int = 12,
        max_ball_area: int = 2000,
        **_ignored,
    ):
        self._confidence = confidence_threshold
        self._min_person_area = min_person_area
        self._min_ball_area = min_ball_area
        self._max_ball_area = max_ball_area

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    def load(self) -> None:
        """No weights to load."""

    def warmup(self) -> None:
        """Nothing to warm."""

    def detect(self, frame: FramePacket) -> list[Detection]:
        if frame.image is None:
            return []

        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for lower, upper, domain_class in _HSV_RANGES.values():
            mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
            )

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if domain_class == BALL:
                    if not (self._min_ball_area <= area <= self._max_ball_area):
                        continue
                elif area < self._min_person_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                detections.append(
                    Detection(
                        frame_id=frame.frame_id,
                        class_name=domain_class,
                        confidence=0.9,
                        bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                        source_model=MODEL_NAME,
                    )
                )

        return detections

    def detect_batch(self, frames: list[FramePacket]) -> list[list[Detection]]:
        return [self.detect(frame) for frame in frames]
