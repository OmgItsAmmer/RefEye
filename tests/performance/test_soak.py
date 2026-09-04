"""Long-session behaviour: bounded memory and repeated triggers.

Architecture.md section 36 ("full-match runtime must not cause continuous
memory growth") and section 50's performance/stability list.

These are slower than the rest of the suite. Run them with:

    pytest tests/performance -m soak
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest

from ai.inference_runtime.scheduler import InferenceScheduler
from ai.model_registry.registry import ModelRegistry
from analysis.pipelines.analysis_pipeline import AnalysisPipeline
from analysis.request_manager.manager import AnalysisRequestManager
from analysis.request_manager.states import RequestState
from core.config.loader import load_settings
from video.frame_access.video_service import VideoService
from vision.features.feature_cache import FeatureCache
from vision.live_pipeline import LiveCVPipeline

pytestmark = pytest.mark.soak

CLIP = Path("tests/fixtures/sample_match.mp4")


def build_stack(loop: bool = True):
    if not CLIP.exists():
        pytest.skip("fixture clip missing — run make fixture")

    settings = load_settings("config/default.yaml", local_path=None, apply_env=False)
    settings.runtime.device = "cpu"
    settings.video.local_file.loop = loop
    settings.ai.detector.provider = "fixture"

    cache = FeatureCache(max_frames=settings.ai.feature_cache.max_frames)
    scheduler = InferenceScheduler(max_queue_size=32)
    scheduler.start()

    registry = ModelRegistry(settings, cache)
    registry.load_all()

    live = LiveCVPipeline(
        detector=registry.get_detector(),
        tracker=registry.get_tracker(),
        ball_tracker=registry.get_ball_tracker(),
        feature_cache=cache,
        scheduler=scheduler,
        frame_stride=settings.ai.detector.live_frame_stride,
    )
    video = VideoService(settings, on_frame=live.on_frame)
    manager = AnalysisRequestManager(
        max_queue_size=4,
        default_window_ms=settings.buffer.recent_window_seconds * 1000,
    )
    manager.set_pipeline(
        AnalysisPipeline(settings, video, registry, cache, scheduler, live)
    )
    manager.start()
    return settings, video, manager, scheduler, cache, live


def test_buffers_and_caches_stay_bounded_over_a_long_run():
    """The core non-functional promise: memory plateaus, it does not climb."""
    settings, video, manager, scheduler, cache, _live = build_stack(loop=True)
    video.start()

    try:
        # Let the buffers reach their configured ceiling first.
        deadline = time.time() + 90
        while time.time() < deadline and video.stats().buffer_bytes < 1:
            time.sleep(0.5)
        time.sleep(20)
        gc.collect()

        first = video.stats()
        first_cache = len(cache)

        time.sleep(25)
        gc.collect()
        second = video.stats()

        limit = settings.buffer.max_decoded_megabytes * 1024 * 1024

        assert second.decoded_frames > first.decoded_frames, "playback stalled"
        assert second.buffer_bytes <= limit
        assert len(cache) <= settings.ai.feature_cache.max_frames
        # Plateaued: still decoding, but holding roughly the same memory.
        assert second.buffer_bytes <= first.buffer_bytes * 1.05
        assert len(cache) <= max(first_cache, 1) * 1.05 + 5
    finally:
        video.stop()
        manager.stop()
        scheduler.stop()


def test_repeated_triggers_do_not_corrupt_state():
    """Architecture.md section 17: repeated shortcuts must stay safe."""
    settings, video, manager, scheduler, cache, _live = build_stack(loop=True)
    video.start()

    try:
        deadline = time.time() + 60
        while time.time() < deadline and len(cache) < 60:
            time.sleep(0.5)

        tracked = []
        for _ in range(6):
            frame = video.latest_frame()
            if frame is not None:
                tracked.append(
                    manager.submit("soak", triggered_at_ms=frame.timestamp_ms)
                )
            time.sleep(0.4)

        deadline = time.time() + 90
        while time.time() < deadline and any(
            t.state
            not in (RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED)
            for t in tracked
        ):
            time.sleep(0.3)

        # Every request reached a terminal state; none left dangling.
        assert all(
            t.state
            in (RequestState.COMPLETED, RequestState.FAILED, RequestState.CANCELLED)
            for t in tracked
        )
        # Request ids stay unique under rapid triggering.
        assert len({t.request_id for t in tracked}) == len(tracked)
        # At least one actually completed rather than all being shed.
        assert any(t.state == RequestState.COMPLETED for t in tracked)
        # Video kept running throughout.
        assert video.stats().decoded_frames > 0
    finally:
        video.stop()
        manager.stop()
        scheduler.stop()


def test_stream_end_leaves_the_app_healthy():
    """Reaching the end of a file is not a crash."""
    settings, video, manager, scheduler, cache, _live = build_stack(loop=False)
    video.start()

    try:
        deadline = time.time() + 120
        while time.time() < deadline and video.state.value == "running":
            time.sleep(0.5)

        # Buffered frames remain available for analysis after playback ends.
        assert video.stats().decoded_frames > 0
        assert len(cache) > 0
    finally:
        video.stop()
        manager.stop()
        scheduler.stop()
