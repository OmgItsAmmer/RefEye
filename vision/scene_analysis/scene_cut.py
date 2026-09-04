"""Camera-cut detection.

Broadcast football is heavily cut: wide play, close-ups, replays, crowd,
graphics. Detecting the cut matters because motion must never be computed
across one (architecture.md section 26) — the ball appears to jump the whole
frame, which would otherwise look like a violent contact event.

Method: colour-histogram correlation between consecutive analysis frames.
Cheap enough to run on every background frame, and robust to the pan/zoom
that would fool a raw pixel-difference approach.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.models import SceneState


class SceneCutDetector:
    def __init__(
        self,
        correlation_threshold: float = 0.60,
        min_frames_between_cuts: int = 4,
        closeup_person_height_ratio: float = 0.55,
    ):
        self._threshold = correlation_threshold
        self._min_gap = min_frames_between_cuts
        self._closeup_ratio = closeup_person_height_ratio

        self._previous_hist: np.ndarray | None = None
        self._frames_since_cut = 0
        self._last_correlation = 1.0

    @property
    def last_correlation(self) -> float:
        return self._last_correlation

    def update(self, image: np.ndarray) -> bool:
        """Returns True when this frame begins a new shot."""
        hist = self._histogram(image)
        self._frames_since_cut += 1

        if self._previous_hist is None:
            self._previous_hist = hist
            return False

        correlation = float(
            cv2.compareHist(self._previous_hist, hist, cv2.HISTCMP_CORREL)
        )
        self._last_correlation = correlation
        self._previous_hist = hist

        # Debounce: a hard cut followed by a fast pan can otherwise register
        # as several cuts in a row and repeatedly reset the trackers.
        if correlation < self._threshold and self._frames_since_cut >= self._min_gap:
            self._frames_since_cut = 0
            return True

        return False

    def classify(self, image: np.ndarray, person_boxes: list) -> SceneState:
        """Coarse shot classification.

        Only `is_closeup` is inferred here (a person filling much of the
        frame height). Replay and graphic detection need broadcaster-specific
        cues, so they are reported as unknown rather than guessed — the
        architecture requires replay behaviour to stay configurable until
        confirmed with the client (section 27).
        """
        height = image.shape[0]
        tallest = 0.0
        for box in person_boxes:
            tallest = max(tallest, (box[3] - box[1]) / max(height, 1))

        is_closeup = tallest >= self._closeup_ratio

        return SceneState(
            is_gameplay=not is_closeup,
            is_replay=False,
            is_closeup=is_closeup,
            has_major_graphic=False,
            confidence=0.5 if is_closeup else 0.7,
        )

    def reset(self) -> None:
        self._previous_hist = None
        self._frames_since_cut = 0

    @staticmethod
    def _histogram(image: np.ndarray) -> np.ndarray:
        # Downscale first: the histogram is scale-invariant, and this keeps
        # the per-frame cost negligible in the continuous background loop.
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
