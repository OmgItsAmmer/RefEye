"""ByteTrack-class multi-object tracker for players.

Implements the `MultiObjectTracker` protocol (architecture.md section 15).

The core ByteTrack idea is kept: associate high-confidence detections first,
then give low-confidence detections a second chance against the tracks that
are still unmatched. That recovers players through partial occlusion, which
is exactly what happens in a crowded box.

Motion is a constant-velocity prediction rather than a full Kalman filter —
enough for association between adjacent broadcast frames, and it keeps the
dependency surface small. Ball tracking is deliberately NOT handled here; the
ball moves too fast and disappears too often for player-style association
(see `ball_tracker.py`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from core.domain.models import Detection, FramePacket, TrackObservation

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_of(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass
class _Track:
    track_id: str
    object_type: str
    bbox: Box
    confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 1
    misses: int = 0
    last_frame_id: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return center_of(self.bbox)

    def predict(self) -> Box:
        """Constant-velocity extrapolation of where this track should be next."""
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def update(self, bbox: Box, confidence: float, frame_id: int) -> None:
        old_cx, old_cy = self.center
        self.bbox = bbox
        new_cx, new_cy = self.center

        # Smooth the velocity estimate; raw frame-to-frame deltas are noisy
        # when the detector jitters a box by a few pixels.
        vx, vy = self.velocity
        self.velocity = (0.6 * vx + 0.4 * (new_cx - old_cx), 0.6 * vy + 0.4 * (new_cy - old_cy))

        self.confidence = confidence
        self.hits += 1
        self.misses = 0
        self.last_frame_id = frame_id
        self.history.append((new_cx, new_cy))
        if len(self.history) > 60:
            self.history.pop(0)

    def mark_missed(self) -> None:
        self.misses += 1
        # Coast on the last known velocity while unmatched.
        self.bbox = self.predict()


class ByteTracker:
    """Two-stage IoU association tracker.

    Parameters mirror ByteTrack's: a high threshold that starts new tracks,
    a low threshold that only ever extends existing ones, and a buffer that
    keeps a lost track alive for a few frames before discarding it.
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.2,
        match_iou: float = 0.25,
        max_misses: int = 15,
        min_hits: int = 2,
        track_classes: frozenset[str] | None = None,
    ):
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._match_iou = match_iou
        self._max_misses = max_misses
        self._min_hits = min_hits
        self._track_classes = track_classes

        self._tracks: list[_Track] = []
        self._id_counter = itertools.count(1)
        self._segment = 0

    # -- MultiObjectTracker protocol ---------------------------------------

    def update(
        self,
        frame: FramePacket,
        detections: list[Detection],
    ) -> list[TrackObservation]:
        candidates = [
            d
            for d in detections
            if self._track_classes is None or d.class_name in self._track_classes
        ]

        high = [d for d in candidates if d.confidence >= self._high_threshold]
        low = [
            d
            for d in candidates
            if self._low_threshold <= d.confidence < self._high_threshold
        ]

        for track in self._tracks:
            track.age += 1

        unmatched_tracks = list(self._tracks)

        # Stage 1: confident detections may both extend and create tracks.
        unmatched_tracks, unmatched_high = self._associate(
            unmatched_tracks, high, frame.frame_id
        )
        # Stage 2: weak detections rescue tracks that would otherwise be lost.
        unmatched_tracks, _ = self._associate(unmatched_tracks, low, frame.frame_id)

        for track in unmatched_tracks:
            track.mark_missed()

        for detection in unmatched_high:
            self._tracks.append(
                _Track(
                    track_id=f"t{next(self._id_counter)}",
                    object_type=detection.class_name,
                    bbox=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    last_frame_id=frame.frame_id,
                    history=[center_of(detection.bbox_xyxy)],
                )
            )

        self._tracks = [t for t in self._tracks if t.misses <= self._max_misses]

        return [
            TrackObservation(
                track_id=t.track_id,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                object_type=t.object_type,
                bbox_xyxy=t.bbox,
                center_xy=t.center,
                confidence=t.confidence,
            )
            for t in self._tracks
            # Suppress one-frame flickers; a track must be seen twice to exist.
            if t.hits >= self._min_hits and t.misses == 0
        ]

    def reset(self) -> None:
        """Drop all tracks. Called on a camera cut or stream discontinuity."""
        self._tracks.clear()

    def start_new_segment(self) -> None:
        """Camera cut: identities cannot survive it (architecture.md section 26)."""
        self._segment += 1
        self.reset()

    @property
    def segment(self) -> int:
        return self._segment

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    # -- internals ----------------------------------------------------------

    def _associate(
        self,
        tracks: list[_Track],
        detections: list[Detection],
        frame_id: int,
    ) -> tuple[list[_Track], list[Detection]]:
        """Greedy IoU matching against each track's predicted position."""
        if not tracks or not detections:
            return tracks, detections

        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            predicted = track.predict()
            for j, detection in enumerate(detections):
                if detection.class_name != track.object_type:
                    continue
                cost[i, j] = iou(predicted, detection.bbox_xyxy)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Greedy is sufficient here: at broadcast frame rates, player boxes
        # rarely contend for the same detection strongly enough for the
        # Hungarian assignment to differ.
        while True:
            i, j = np.unravel_index(np.argmax(cost), cost.shape)
            if cost[i, j] < self._match_iou:
                break
            tracks[i].update(detections[j].bbox_xyxy, detections[j].confidence, frame_id)
            matched_tracks.add(int(i))
            matched_dets.add(int(j))
            cost[i, :] = 0.0
            cost[:, j] = 0.0

        return (
            [t for i, t in enumerate(tracks) if i not in matched_tracks],
            [d for j, d in enumerate(detections) if j not in matched_dets],
        )
