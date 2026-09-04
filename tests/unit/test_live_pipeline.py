"""LiveCVPipeline.process_now: batched gap-filling with a time budget.

Regression coverage for a real CPU-bottleneck bug: process_now used to call
the detector once per frame, sequentially, on the analysis thread. For a
20-second trigger window on a slow machine that could take 40-90+ seconds,
during which the operator sees nothing but "Analysing...". The fix batches
detection calls and caps total time spent, so a slow machine still returns
a result — with less feature data, not none.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ai.inference_runtime.scheduler import InferenceScheduler
from core.domain.models import FramePacket
from vision.features.feature_cache import FeatureCache
from vision.live_pipeline import LiveCVPipeline
from vision.tracking.ball_tracker import BallTracker
from vision.tracking.iou_tracker import ByteTracker


def make_frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id * 40,
        capture_timestamp_ms=frame_id * 40,
        width=64,
        height=36,
        source_id="test",
        image=np.zeros((36, 64, 3), dtype=np.uint8),
    )


class _CountingBatchDetector:
    """Records how it was called — one batch call, not N single calls."""

    def __init__(self, delay_per_batch: float = 0.0):
        self.batch_calls = 0
        self.single_calls = 0
        self.batch_sizes: list[int] = []
        self._delay = delay_per_batch

    def detect(self, frame):
        self.single_calls += 1
        return []

    def detect_batch(self, frames):
        self.batch_calls += 1
        self.batch_sizes.append(len(frames))
        if self._delay:
            time.sleep(self._delay)
        return [[] for _ in frames]


@pytest.fixture
def pipeline():
    detector = _CountingBatchDetector()
    cache = FeatureCache(max_frames=1000)
    scheduler = InferenceScheduler(max_queue_size=8)
    pipe = LiveCVPipeline(
        detector=detector,
        tracker=ByteTracker(min_hits=1),
        ball_tracker=BallTracker(),
        feature_cache=cache,
        scheduler=scheduler,
        frame_stride=1,
    )
    return pipe, detector, cache


class TestProcessNowBatching:
    def test_uses_batched_detection_not_per_frame_calls(self, pipeline):
        pipe, detector, cache = pipeline
        frames = [make_frame(i) for i in range(50)]

        pipe.process_now(frames, batch_size=16)

        assert detector.single_calls == 0, "process_now must never call detect() per frame"
        assert detector.batch_calls > 0
        # 50 frames at batch_size=16 -> 4 batch calls (16+16+16+2), not 50.
        assert detector.batch_calls == 4

    def test_all_frames_end_up_cached(self, pipeline):
        pipe, detector, cache = pipeline
        frames = [make_frame(i) for i in range(30)]

        pipe.process_now(frames, batch_size=10)

        for frame in frames:
            assert cache.get(frame.frame_id) is not None

    def test_skips_frames_already_in_cache(self, pipeline):
        """No point re-running detection for frames the background loop
        already covered — process_now only fills gaps."""
        pipe, detector, cache = pipeline
        frames = [make_frame(i) for i in range(10)]

        pipe.process_now(frames[:5])
        assert detector.batch_calls == 1

        pipe.process_now(frames)  # first 5 already cached
        assert detector.batch_sizes[-1] == 5

    def test_empty_gap_makes_no_detector_call(self, pipeline):
        pipe, detector, cache = pipeline
        frames = [make_frame(i) for i in range(5)]

        pipe.process_now(frames)
        detector.batch_calls = 0  # reset

        pipe.process_now(frames)  # everything already cached
        assert detector.batch_calls == 0

    def test_respects_time_budget_on_a_slow_detector(self):
        """A slow (e.g. CPU) machine must get a partial result, not a stall."""
        detector = _CountingBatchDetector(delay_per_batch=0.15)
        cache = FeatureCache(max_frames=1000)
        scheduler = InferenceScheduler(max_queue_size=8)
        pipe = LiveCVPipeline(
            detector=detector,
            tracker=ByteTracker(min_hits=1),
            ball_tracker=BallTracker(),
            feature_cache=cache,
            scheduler=scheduler,
            frame_stride=1,
        )

        frames = [make_frame(i) for i in range(200)]  # would take ~2s unbudgeted

        started = time.monotonic()
        pipe.process_now(frames, batch_size=10, time_budget_seconds=0.4)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0, f"process_now ignored its time budget (took {elapsed:.2f}s)"
        assert len(cache) < len(frames), "a budget-limited run should cache fewer than all frames"
        assert len(cache) > 0, "at least some progress should be made before the budget expires"
