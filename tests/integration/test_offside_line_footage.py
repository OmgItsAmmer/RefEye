"""The offside verdict on real broadcast footage, end to end.

The unit tests prove the Law is applied correctly to inputs that are known to
be right. This asks the different question: given what the earlier phases
actually produce on a real clip — noisy foot points, a partly-calibrated
pitch, teams assigned from kit colour — does the verdict come out *honest*?

That mostly means checking it refuses. On this clip the pitch is not manually
marked, so a metric verdict is impossible; the correct behaviour is to say so
rather than to produce a number, and these tests fail if it ever stops
saying so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config.loader import load_settings
from core.domain.models import FramePacket
from offside.offside_line import Verdict
from tools.pipeline_debugger.pipeline import OffsidePipeline

CLIP = Path("data/videos/client_m2_test_video.mp4")
DETECTOR_CHECKPOINT = Path("models/detector/yolo11n.pt")
POSE_CHECKPOINT = Path("models/pose/yolo11n-pose.pt")
RUN_START = 90
RUN_LENGTH = 12


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def analyses(settings):
    if not CLIP.exists():
        pytest.skip("client reference clip is not in git")
    if not (DETECTOR_CHECKPOINT.exists() and POSE_CHECKPOINT.exists()):
        pytest.skip("model checkpoints not present — see models/pose/README.md")

    import cv2

    from ai.model_registry.registry import ModelRegistry
    from vision.features.feature_cache import FeatureCache

    registry = ModelRegistry(
        settings, FeatureCache(max_frames=settings.ai.feature_cache.max_frames)
    )
    registry.load_all()
    if registry.get_detector() is None or registry.get_pose_estimator() is None:
        pytest.skip("models unavailable")

    pipeline = OffsidePipeline(settings, registry)
    capture = cv2.VideoCapture(str(CLIP))
    capture.set(cv2.CAP_PROP_POS_FRAMES, RUN_START)

    results = []
    try:
        for offset in range(RUN_LENGTH):
            ok, image = capture.read()
            if not ok:
                break
            frame_id = RUN_START + offset
            height, width = image.shape[:2]
            frame = FramePacket(
                frame_id=frame_id,
                pts=frame_id,
                timestamp_ms=frame_id * 33,
                capture_timestamp_ms=frame_id * 33,
                width=width,
                height=height,
                source_id="test",
                image=image,
            )
            results.append(pipeline.analyse(frame, frame_id))
    finally:
        capture.release()

    if len(results) < 5:
        pytest.skip("could not decode a run of frames")
    return results


def test_the_stage_runs_on_every_frame(analyses):
    """A stage that silently produced nothing would look identical to one
    that was never built."""
    assert all(analysis.offside is not None for analysis in analyses)
    for analysis in analyses:
        report = {r.phase: r for r in analysis.reports}["M2.5"]
        assert "not built yet" not in report.summary


def test_an_uncalibrated_pitch_never_produces_a_verdict(analyses):
    """The clip is not marked by hand, so no metric call is possible. Saying
    so is the correct behaviour, and it must not quietly change."""
    for analysis in analyses:
        if analysis.calibration is not None and analysis.calibration.can_draw_offside_line:
            continue
        decision = analysis.offside
        assert decision.verdict is Verdict.INCONCLUSIVE
        assert decision.warnings, "refused without saying why"


def test_every_refusal_explains_itself(analyses):
    for analysis in analyses:
        decision = analysis.offside
        if decision.verdict is Verdict.INCONCLUSIVE:
            assert decision.warnings
            assert decision.headline().startswith("No call")


def test_a_verdict_never_exceeds_its_own_confidence(analyses):
    for analysis in analyses:
        decision = analysis.offside
        assert 0.0 <= decision.confidence <= 1.0
        if decision.verdict is Verdict.INCONCLUSIVE:
            assert decision.confidence == 0.0


def test_the_same_frame_gives_the_same_verdict_twice(settings, analyses):
    """A verdict that changes between two runs of the same clip is not
    something an operator can be asked to trust."""
    frame = analyses[-1].frame
    from ai.model_registry.registry import ModelRegistry
    from vision.features.feature_cache import FeatureCache

    registry = ModelRegistry(
        settings, FeatureCache(max_frames=settings.ai.feature_cache.max_frames)
    )
    registry.load_all()

    first = OffsidePipeline(settings, registry).analyse(frame, frame.frame_id)
    second = OffsidePipeline(settings, registry).analyse(frame, frame.frame_id)

    assert first.offside.verdict is second.offside.verdict
    assert first.offside.headline() == second.offside.headline()
