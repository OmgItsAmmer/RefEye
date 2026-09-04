"""Continuous lightweight CV pipeline.

Architecture.md sections 4.4, 4.5 and 13: keep preparing cheap, useful data in
the background so that pressing the hotkey does not restart all processing
from zero, but do NOT run expensive analysis continuously when focused
analysis after a trigger produces the same useful result.

Concretely this runs detection on every Nth frame (configurable stride),
tracks players and the ball, watches for camera cuts, and writes bounded
per-frame features into the `FeatureCache`.

It runs on the ingest thread's callback but submits detection through the
inference scheduler at LOW priority, so a triggered analysis always wins the
GPU (section 34). When the scheduler refuses the job, the frame is skipped —
background work is droppable by design.
"""

from __future__ import annotations

import threading
import time

from ai.inference_runtime.scheduler import (
    InferencePriority,
    InferenceScheduler,
    SchedulerBusy,
)
from core.domain.models import FrameFeatures, FramePacket
from observability.logging.setup import get_logger
from vision.features.feature_cache import CachedFrame, FeatureCache
from vision.scene_analysis.scene_cut import SceneCutDetector

logger = get_logger(__name__)


class LiveCVPipeline:
    def __init__(
        self,
        detector,
        tracker,
        ball_tracker,
        feature_cache: FeatureCache,
        scheduler: InferenceScheduler,
        frame_stride: int = 3,
    ):
        self._detector = detector
        self._tracker = tracker
        self._ball_tracker = ball_tracker
        self._cache = feature_cache
        self._scheduler = scheduler
        self._stride = max(1, frame_stride)

        self._scene_cuts = SceneCutDetector()
        self._lock = threading.Lock()
        self._processed = 0
        self._skipped = 0
        self._cut_count = 0
        self._enabled = detector is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "processed": self._processed,
                "skipped": self._skipped,
                "camera_cuts": self._cut_count,
                "cached_frames": len(self._cache),
                "active_tracks": getattr(self._tracker, "active_track_count", 0),
            }

    def on_frame(self, frame: FramePacket) -> None:
        """Called for every decoded frame. Must return quickly."""
        if not self._enabled or frame.image is None:
            return

        if frame.frame_id % self._stride != 0:
            return

        try:
            self._scheduler.submit(
                lambda: self._process(frame),
                priority=InferencePriority.LOW,
                label="live_cv",
            )
        except SchedulerBusy:
            # Expected and healthy: a triggered analysis is using the GPU.
            with self._lock:
                self._skipped += 1

    def process_now(
        self,
        frames: list[FramePacket],
        batch_size: int = 16,
        time_budget_seconds: float = 8.0,
    ) -> None:
        """Fill feature-cache gaps for a triggered analysis window.

        Runs on the analysis thread, so it must not turn a several-hundred-
        frame window on a slow (e.g. CPU-only) machine into a multi-minute
        stall. Two things keep that bounded:

          * batched detection — one forward pass over `batch_size` frames
            instead of one pass per frame, which is what actually made this
            slow (per-call overhead dominates on small single-frame batches).
          * a hard time budget — if the machine is too slow to finish the
            whole gap in time, stop and let action spotting run on whatever
            got cached. A shorter, real analysis beats a long stall, and
            missing feature data degrades gracefully (architecture.md
            section 25) rather than failing the request.
        """
        missing = [
            f for f in frames if f.image is not None and self._cache.get(f.frame_id) is None
        ]
        if not missing:
            return

        started = time.monotonic()
        processed = 0

        for start in range(0, len(missing), batch_size):
            if time.monotonic() - started > time_budget_seconds:
                logger.info(
                    "live_cv_gap_fill_budget_exceeded",
                    filled=processed,
                    remaining=len(missing) - processed,
                )
                break

            batch = missing[start : start + batch_size]
            self._process_batch(batch)
            processed += len(batch)

    # -- internals ----------------------------------------------------------

    def _process(self, frame: FramePacket) -> None:
        """Single-frame path used by the continuous background loop."""
        if frame.image is None:
            return
        try:
            detections = self._detector.detect(frame)
        except Exception as exc:  # noqa: BLE001 — one bad frame must not stop the loop
            logger.warning("detection_failed", frame_id=frame.frame_id, error=str(exc))
            return
        self._finish_frame(frame, detections)

    def _process_batch(self, frames: list[FramePacket]) -> None:
        """Batched path used when filling a gap for a triggered analysis."""
        try:
            batch_detections = self._detector.detect_batch(frames)
        except Exception as exc:  # noqa: BLE001 — one bad batch must not stop the request
            logger.warning(
                "batch_detection_failed", frame_count=len(frames), error=str(exc)
            )
            return

        for frame, detections in zip(frames, batch_detections):
            self._finish_frame(frame, detections)

    def _finish_frame(self, frame: FramePacket, detections) -> None:
        scene_cut = self._scenecut(frame)

        tracks = self._tracker.update(frame, detections)
        ball = self._ball_tracker.update(frame.frame_id, frame.timestamp_ms, detections)

        nearest_track_id = self._nearest_person_track(tracks, ball.center)

        features = FrameFeatures(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            ball_center=ball.center,
            # An interpolated position is not an observation; recording its
            # confidence as None keeps "we saw the ball" honest downstream.
            ball_confidence=None if ball.is_interpolated else ball.confidence,
            nearest_player_track_id=nearest_track_id,
            ball_velocity=ball.velocity,
            ball_speed=ball.speed,
            scene_cut=scene_cut,
        )

        self._cache.put(
            CachedFrame(
                features=features,
                tracks=tracks,
                scene_cut=scene_cut,
                segment=self._ball_tracker.segment,
            )
        )

        with self._lock:
            self._processed += 1

    def _scenecut(self, frame: FramePacket) -> bool:
        cut = self._scene_cuts.update(frame.image)
        if not cut:
            return False

        # Identity and motion cannot survive a cut (architecture.md section 26).
        self._tracker.start_new_segment()
        self._ball_tracker.start_new_segment()

        with self._lock:
            self._cut_count += 1

        logger.debug("camera_cut_detected", frame_id=frame.frame_id)
        return True

    @staticmethod
    def _nearest_person_track(tracks, ball_center) -> str | None:
        if ball_center is None or not tracks:
            return None

        best_id, best_distance = None, float("inf")
        for observation in tracks:
            if observation.object_type == "ball":
                continue
            dx = observation.center_xy[0] - ball_center[0]
            dy = observation.center_xy[1] - ball_center[1]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_id = observation.track_id
        return best_id
