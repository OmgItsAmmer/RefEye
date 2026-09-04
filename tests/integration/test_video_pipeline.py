"""End-to-end ingest: file -> demux -> decode -> normalized timeline -> buffers.

No Qt here — VideoService is deliberately UI-framework-free, so the whole
pipeline is testable headlessly.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from core.config.loader import load_settings
from core.errors.exceptions import AnalysisWindowUnavailableError
from video.frame_access.video_service import StreamState, VideoService

FIXTURE = Path("tests/fixtures/sample_match.mp4")


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory) -> Path:
    """Use the committed sample clip, or synthesize one if it is absent.

    The real fixture is gitignored (video files stay out of git), so CI and a
    fresh clone must still be able to run this test.
    """
    if FIXTURE.exists():
        return FIXTURE

    av = pytest.importorskip("av")
    path = tmp_path_factory.mktemp("clips") / "synthetic.mp4"

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=25)
    stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"

    for i in range(75):  # 3 seconds
        img = np.zeros((180, 320, 3), np.uint8)
        img[:, :] = (40, 110, 55)
        x = int(20 + 280 * (i / 75))
        y = int(90 + 40 * math.sin(i / 6))
        img[max(0, y - 4) : y + 4, max(0, x - 4) : x + 4] = (245, 245, 245)
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return path


def make_settings(clip: Path, **buffer_overrides):
    settings = load_settings("config/default.yaml", local_path=None, apply_env=False)
    settings.video.local_file.path = str(clip)
    settings.video.local_file.loop = False
    for key, value in buffer_overrides.items():
        setattr(settings.buffer, key, value)
    return settings


def wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_pipeline_decodes_frames_into_the_buffer(clip_path):
    settings = make_settings(clip_path)
    frames_seen = []
    service = VideoService(settings, on_frame=frames_seen.append)

    service.start()
    try:
        assert wait_for(lambda: len(frames_seen) >= 10), "no frames reached the subscriber"

        stats = service.stats()
        assert stats.decoded_frames >= 10
        assert stats.decode_errors == 0
        assert stats.buffer_frames > 0
    finally:
        service.stop()


def test_decoded_frames_are_downscaled_to_analysis_resolution(clip_path):
    settings = make_settings(clip_path)
    settings.video.analysis_resolution.width = 320
    settings.video.analysis_resolution.height = 180

    frames = []
    service = VideoService(settings, on_frame=frames.append)
    service.start()
    try:
        assert wait_for(lambda: len(frames) >= 3)
        frame = frames[0]
        assert frame.image.shape == (180, 320, 3)
        assert frame.image.dtype == np.uint8
        assert frame.width == 320 and frame.height == 180
    finally:
        service.stop()


def test_frame_ids_and_timestamps_are_monotonic(clip_path):
    settings = make_settings(clip_path)
    frames = []
    service = VideoService(settings, on_frame=frames.append)

    service.start()
    try:
        assert wait_for(lambda: len(frames) >= 20)
    finally:
        service.stop()

    ids = [f.frame_id for f in frames]
    timestamps = [f.timestamp_ms for f in frames]

    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids), "frame ids must be unique"
    assert timestamps == sorted(timestamps)


def test_buffer_stays_bounded_over_playback(clip_path):
    """The core non-functional requirement: memory must not grow without limit."""
    settings = make_settings(clip_path, max_decoded_frames=15, decoded_buffer_seconds=1)
    service = VideoService(settings)

    service.start()
    try:
        assert wait_for(lambda: service.stats().decoded_frames >= 50)
        stats = service.stats()
        assert stats.buffer_frames <= 15
        assert stats.buffer_duration_ms <= 1000
    finally:
        service.stop()


def test_memory_ceiling_binds_before_the_frame_count(clip_path):
    """RAM, not frame count, is the limit that protects the workstation."""
    settings = make_settings(
        clip_path,
        max_decoded_frames=100_000,
        decoded_buffer_seconds=3600,
        max_decoded_megabytes=2,
    )
    settings.video.analysis_resolution.width = 320
    settings.video.analysis_resolution.height = 180

    service = VideoService(settings)
    service.start()
    try:
        assert wait_for(lambda: service.stats().decoded_frames >= 60)
        stats = service.stats()
        assert stats.buffer_bytes <= 2 * 1024 * 1024
        # Far below the frame/duration caps: the byte cap is what bound it.
        assert stats.buffer_frames < 1000
    finally:
        service.stop()


def test_recent_clip_is_extracted_from_the_buffer(clip_path):
    settings = make_settings(clip_path)
    service = VideoService(settings)

    service.start()
    try:
        assert wait_for(lambda: service.stats().buffer_frames >= 25)

        latest = service.latest_frame()
        clip = service.get_recent_clip(
            end_timestamp_ms=latest.timestamp_ms,
            duration_ms=500,
            request_id="test-req",
        )

        assert clip.request_id == "test-req"
        assert clip.frames
        assert clip.start_frame_id <= clip.end_frame_id
        assert clip.end_timestamp_ms - clip.start_timestamp_ms <= 500
    finally:
        service.stop()


def test_recent_clip_fails_loudly_when_window_is_empty(clip_path):
    settings = make_settings(clip_path)
    service = VideoService(settings)

    with pytest.raises(AnalysisWindowUnavailableError):
        service.get_recent_clip(end_timestamp_ms=0, duration_ms=1000)


def test_frames_are_retrievable_by_id(clip_path):
    settings = make_settings(clip_path)
    service = VideoService(settings)

    service.start()
    try:
        assert wait_for(lambda: service.stats().buffer_frames >= 10)
        latest = service.latest_frame()
        assert service.get_frame(latest.frame_id) is not None
        assert service.get_frame(999_999) is None
    finally:
        service.stop()


def test_missing_source_reports_error_without_crashing():
    """Graceful degradation: a bad source must leave the app alive."""
    settings = make_settings(Path("does/not/exist.mp4"))
    states = []
    service = VideoService(settings, on_state_change=lambda s, m: states.append((s, m)))

    service.start()
    try:
        assert wait_for(lambda: service.state == StreamState.ERROR, timeout=5.0)
        assert any(s == StreamState.ERROR for s, _ in states)
        # A human-readable reason must reach the UI, not just the log.
        error_message = next(m for s, m in states if s == StreamState.ERROR)
        assert error_message and "exist" in error_message
    finally:
        service.stop()


def test_a_failing_frame_subscriber_does_not_stop_ingest(clip_path):
    settings = make_settings(clip_path)
    calls = {"n": 0}

    def bad_subscriber(_frame):
        calls["n"] += 1
        raise RuntimeError("subscriber blew up")

    service = VideoService(settings, on_frame=bad_subscriber)
    service.start()
    try:
        assert wait_for(lambda: service.stats().decoded_frames >= 20)
        assert calls["n"] >= 5
        assert service.state == StreamState.RUNNING
    finally:
        service.stop()


def test_unsupported_input_type_is_rejected_clearly(clip_path):
    settings = make_settings(clip_path)
    settings.video.input_type = "rtsp"
    states = []
    service = VideoService(settings, on_state_change=lambda s, m: states.append((s, m)))

    service.start()
    try:
        assert wait_for(lambda: service.state == StreamState.ERROR, timeout=5.0)
        message = next(m for s, m in states if s == StreamState.ERROR)
        assert "rtsp" in message
    finally:
        service.stop()
