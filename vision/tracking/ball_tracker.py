"""Ball trajectory tracking with short-gap tolerance.

The ball needs different logic from players (architecture.md section 15). It
is small, fast, frequently occluded by legs and bodies, and often missing
from detector output entirely. Two rules from the architecture drive this:

  * Section 25 — tolerate short gaps. A missing detection is normal, not a
    failure. Bridge briefly using the last known velocity, and lower
    confidence while coasting.
  * Section 26 — never compute velocity across a camera cut. On a cut the
    trajectory resets; a ball "teleporting" across a scene change would
    otherwise read as an enormous acceleration and fake a contact event.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.domain.models import Detection

Point = tuple[float, float]


@dataclass(frozen=True)
class BallState:
    frame_id: int
    timestamp_ms: int
    center: Point | None
    confidence: float | None
    velocity: Point | None
    speed: float | None
    is_interpolated: bool
    segment: int


class BallTracker:
    """Single-object trajectory with gap bridging and per-segment continuity."""

    def __init__(
        self,
        max_gap_frames: int = 6,
        max_jump_px: float = 260.0,
        interpolated_confidence_decay: float = 0.6,
    ):
        self._max_gap_frames = max_gap_frames
        self._max_jump_px = max_jump_px
        self._decay = interpolated_confidence_decay

        self._last: BallState | None = None
        self._gap = 0
        self._segment = 0

    @property
    def segment(self) -> int:
        return self._segment

    def start_new_segment(self) -> None:
        """Camera cut: break trajectory continuity, do not carry velocity over."""
        self._segment += 1
        self._last = None
        self._gap = 0

    def reset(self) -> None:
        self.start_new_segment()

    def update(
        self,
        frame_id: int,
        timestamp_ms: int,
        detections: list[Detection],
    ) -> BallState:
        observed = self._pick_ball(detections)

        if observed is not None:
            state = self._accept(frame_id, timestamp_ms, observed)
        else:
            state = self._coast(frame_id, timestamp_ms)

        self._last = state
        return state

    # -- internals ----------------------------------------------------------

    def _pick_ball(self, detections: list[Detection]) -> Detection | None:
        """Choose the most plausible ball detection for this frame.

        With no prior, take the most confident. With a prior, prefer the
        candidate closest to where the ball should be — the highest-confidence
        blob is often a distant pitch marking or a player's head.
        """
        balls = [d for d in detections if d.class_name == "ball"]
        if not balls:
            return None

        if self._last is None or self._last.center is None:
            return max(balls, key=lambda d: d.confidence)

        predicted = self._predicted_center()

        def distance(detection: Detection) -> float:
            cx = (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2.0
            cy = (detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) / 2.0
            return math.hypot(cx - predicted[0], cy - predicted[1])

        best = min(balls, key=distance)

        # Every candidate this frame is implausibly far from where the ball
        # should be. Falling back to "just take the most confident blob" here
        # is how a sock, an ad-board logo, or a crowd member gets locked onto
        # as the ball — the ball detector deliberately runs at half the normal
        # confidence threshold (it must, to catch a small fast-moving ball),
        # so "most confident" is not a meaningful filter on its own. Coasting
        # (this frame's job, once _pick_ball returns None) is the tracker's
        # designed answer to a missed detection (architecture.md section 25)
        # and is far more likely to be right than any of these candidates.
        if distance(best) > self._max_jump_px:
            return None
        return best

    def _predicted_center(self) -> Point:
        assert self._last is not None and self._last.center is not None
        if self._last.velocity is None:
            return self._last.center
        return (
            self._last.center[0] + self._last.velocity[0],
            self._last.center[1] + self._last.velocity[1],
        )

    def _accept(self, frame_id: int, timestamp_ms: int, detection: Detection) -> BallState:
        x1, y1, x2, y2 = detection.bbox_xyxy
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        velocity: Point | None = None
        if self._last is not None and self._last.center is not None:
            frames_elapsed = max(1, frame_id - self._last.frame_id)
            velocity = (
                (center[0] - self._last.center[0]) / frames_elapsed,
                (center[1] - self._last.center[1]) / frames_elapsed,
            )

        self._gap = 0
        return BallState(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            center=center,
            confidence=detection.confidence,
            velocity=velocity,
            speed=math.hypot(*velocity) if velocity else None,
            is_interpolated=False,
            segment=self._segment,
        )

    def _coast(self, frame_id: int, timestamp_ms: int) -> BallState:
        """No detection this frame: extrapolate briefly, then give up."""
        self._gap += 1

        if (
            self._last is None
            or self._last.center is None
            or self._gap > self._max_gap_frames
        ):
            return BallState(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                center=None,
                confidence=None,
                velocity=None,
                speed=None,
                is_interpolated=False,
                segment=self._segment,
            )

        predicted = self._predicted_center()
        confidence = (self._last.confidence or 0.0) * self._decay

        return BallState(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            center=predicted,
            confidence=confidence,
            velocity=self._last.velocity,
            speed=self._last.speed,
            is_interpolated=True,
            segment=self._segment,
        )
