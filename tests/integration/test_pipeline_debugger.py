"""The pipeline debugger runs, and reports every stage honestly.

The debugger is a development tool, but it is the thing a human looks at to
judge whether M2 works — so the properties that matter are that it does not
quietly omit a stage, and that it never dresses up a fallback as a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.config.loader import load_settings
from core.domain.models import FramePacket
from offside.offside_line import Verdict
from tools.pipeline_debugger.pipeline import OffsidePipeline, StageState

CLIP = Path("data/videos/client_m2_test_video.mp4")


class _StubDetector:
    model_name = "stub-detector"

    def detect(self, frame):
        return []


class _EmptyRegistry:
    """No models at all — the debugger must still render a full report."""

    def get_detector(self):
        return None

    def get_pose_estimator(self):
        return None


class _DetectorOnlyRegistry:
    def get_detector(self):
        return _StubDetector()

    def get_pose_estimator(self):
        return None


def blank_frame() -> FramePacket:
    return FramePacket(
        frame_id=0,
        pts=0,
        timestamp_ms=0,
        capture_timestamp_ms=0,
        width=1280,
        height=720,
        source_id="test",
        image=np.full((720, 1280, 3), 40, dtype=np.uint8),
    )


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_every_plan_stage_appears_even_with_no_models(settings):
    """A missing phase must look missing. If a stage silently vanished from
    the panel, the gap would read as 'done'."""
    analysis = OffsidePipeline(settings, _EmptyRegistry()).analyse(blank_frame(), 0)
    phases = {report.phase for report in analysis.reports}

    assert {"M2.1", "M2.2", "M2.3", "M2.4", "M2.5", "M2.6"} <= phases


def test_no_stage_of_the_pipeline_is_still_pending(settings):
    """Every phase of M2 is built as of M2.6. This is the test that will fail
    first if a stage is ever removed from the panel rather than implemented —
    a missing stage must read as missing, never as done."""
    analysis = OffsidePipeline(settings, _EmptyRegistry()).analyse(blank_frame(), 0)

    pending = [r.phase for r in analysis.reports if r.state is StageState.PENDING]
    assert not pending
    assert all("not built yet" not in r.summary for r in analysis.reports)


def test_decision_support_refuses_rather_than_reporting_nothing(settings):
    """M2.6 is built, so on a frame with nobody on it it must say why there is
    no call — not stay silent, which would be indistinguishable from a stage
    that never ran."""
    analysis = OffsidePipeline(settings, _EmptyRegistry()).analyse(blank_frame(), 0)
    report = {r.phase: r for r in analysis.reports}["M2.6"]

    assert analysis.explanation is not None
    assert analysis.explanation.verdict is Verdict.INCONCLUSIVE
    assert report.summary.startswith("No call")
    assert analysis.explanation.weakest is not None, "a refusal must name a cause"


def test_player_identity_reports_no_players_rather_than_pending(settings):
    """M2.4 is built, so on a frame with nobody on it the stage must say that
    — not "not built", which would hide a failing stage behind a missing one."""
    analysis = OffsidePipeline(settings, _EmptyRegistry()).analyse(blank_frame(), 0)
    report = {r.phase: r for r in analysis.reports}["M2.4"]

    assert report.state is not StageState.PENDING
    assert "not built yet" not in report.summary
    assert analysis.identities is not None


def test_team_assignment_reports_no_players_rather_than_pending(settings):
    """M2.3 is built, so on a frame it cannot work with it must say *why* it
    produced nothing — a built stage reporting 'not built' would hide the
    difference between a missing phase and a failing one."""
    analysis = OffsidePipeline(settings, _EmptyRegistry()).analyse(blank_frame(), 0)
    report = {r.phase: r for r in analysis.reports}["M2.3"]

    assert report.state is StageState.UNAVAILABLE
    assert "not built yet" not in report.summary
    assert analysis.teams is not None


def test_missing_models_report_unavailable_rather_than_crashing(settings):
    analysis = OffsidePipeline(settings, _EmptyRegistry()).analyse(blank_frame(), 0)
    by_phase = {report.phase: report for report in analysis.reports}

    assert by_phase["M2.2"].state is StageState.UNAVAILABLE
    assert analysis.poses == []


def test_a_frame_with_no_pitch_reports_no_calibration(settings):
    analysis = OffsidePipeline(settings, _DetectorOnlyRegistry()).analyse(blank_frame(), 0)
    calibration = analysis.calibration

    assert calibration is not None
    assert not calibration.can_draw_offside_line
    by_phase = {report.phase: report for report in analysis.reports}
    assert by_phase["M2.1"].state is StageState.UNAVAILABLE


def test_marking_four_landmarks_reaches_metric_calibration(settings):
    """The operator path in the debug UI: pick a landmark, click it, repeat.
    Four marks is what turns 'directional' into 'metric'."""
    pipeline = OffsidePipeline(settings, _DetectorOnlyRegistry())
    marks = {
        "corner_left_top": (280.0, 210.0),
        "corner_right_top": (1180.0, 250.0),
        "corner_right_bottom": (1500.0, 690.0),
        "corner_left_bottom": (-140.0, 620.0),
    }
    for landmark, point in marks.items():
        pipeline.mark_landmark(point, landmark)

    analysis = pipeline.analyse(blank_frame(), 0)
    assert analysis.calibration.is_metric
    assert analysis.calibration.source == "manual"

    # And the top-down projection now works.
    projected = analysis.calibration.to_pitch([(700.0, 450.0)])
    assert projected is not None and np.all(np.isfinite(projected))


def test_marking_the_same_landmark_twice_moves_it(settings):
    pipeline = OffsidePipeline(settings, _DetectorOnlyRegistry())
    pipeline.mark_landmark((10.0, 10.0), "centre_mark")
    pipeline.mark_landmark((20.0, 20.0), "centre_mark")

    assert len(pipeline.manual_correspondences) == 1
    assert pipeline.manual_correspondences[0].image_xy == (20.0, 20.0)


@pytest.mark.skipif(not CLIP.exists(), reason="client reference clip is not in git")
def test_runs_on_the_real_clip_with_real_models(settings):
    import cv2

    from ai.model_registry.registry import ModelRegistry
    from vision.features.feature_cache import FeatureCache

    registry = ModelRegistry(
        settings, FeatureCache(max_frames=settings.ai.feature_cache.max_frames)
    )
    state = registry.load_all()
    if registry.get_detector() is None:
        pytest.skip(f"models unavailable: {state.message}")

    capture = cv2.VideoCapture(str(CLIP))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, image = capture.read()
    capture.release()
    assert ok

    height, width = image.shape[:2]
    frame = FramePacket(
        frame_id=200,
        pts=200,
        timestamp_ms=6666,
        capture_timestamp_ms=6666,
        width=width,
        height=height,
        source_id="test",
        image=image,
    )

    analysis = OffsidePipeline(settings, registry).analyse(frame, 200)

    assert analysis.players, "no players found on real footage"
    assert analysis.line_mask is not None
    by_phase = {report.phase: report for report in analysis.reports}
    assert by_phase["M2.2"].state in (StageState.OK, StageState.DEGRADED)
    # Metric calibration no longer requires an operator to mark the pitch —
    # the auto-landmark detector (offside/pitch_calibration/auto_landmarks.py)
    # can reach it by itself on a frame with enough visible pitch markings.
    # It must still say honestly that it did, not claim to be the operator.
    if analysis.calibration.is_metric:
        assert analysis.calibration.source == "auto_landmarks"

    # The torso-yield count (added after a real frame showed 19/19 feet
    # measured but M2.3 found usable shirt colour on only 4 of them — the
    # gap this line exists to surface one stage earlier).
    assert analysis.torso_confident_count is not None
    assert 0 <= analysis.torso_confident_count <= len(analysis.poses)
    assert any(
        "confident torso" in detail for detail in by_phase["M2.2"].details
    )


@pytest.mark.skipif(not CLIP.exists(), reason="client reference clip is not in git")
def test_marks_follow_the_camera_through_the_pipeline(settings):
    """End to end: mark the pitch once, then let the camera pan.

    Without the follower the marks stay pinned to the same pixels while the
    pitch moves away underneath them, and nothing reports it — four marks
    always agree with each other, so the reprojection error stays at zero
    while the calibration goes badly wrong.
    """
    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 90)
    frames = []
    for offset in range(25):
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        frames.append(
            FramePacket(
                frame_id=90 + offset,
                pts=90 + offset,
                timestamp_ms=(90 + offset) * 33,
                capture_timestamp_ms=(90 + offset) * 33,
                width=width,
                height=height,
                source_id="test",
                image=image,
            )
        )
    capture.release()
    if len(frames) < 10:
        pytest.skip("could not decode a run of frames")

    pipeline = OffsidePipeline(settings, _DetectorOnlyRegistry())
    for landmark, point in (
        ("corner_left_top", (280.0, 210.0)),
        ("corner_right_top", (1180.0, 250.0)),
        ("corner_right_bottom", (1500.0, 690.0)),
        ("corner_left_bottom", (-140.0, 620.0)),
    ):
        pipeline.mark_landmark(point, landmark)

    first = pipeline.analyse(frames[0], frames[0].frame_id)
    assert first.calibration.is_metric

    start = {c.landmark: c.image_xy for c in pipeline.manual_correspondences}
    for frame in frames[1:]:
        analysis = pipeline.analyse(frame, frame.frame_id)

    moved = {c.landmark: c.image_xy for c in pipeline.manual_correspondences}
    drift = max(
        abs(moved[name][0] - start[name][0]) for name in start if name in moved
    )

    assert analysis.calibration.is_metric, "the calibration was dropped mid-shot"
    assert drift > 20, f"the marks did not follow the camera (moved {drift:.0f}px)"
    # Carried marks must cost confidence, or the operator has no signal that
    # the calibration is ageing.
    assert analysis.calibration.confidence < first.calibration.confidence
    assert any("carried" in reason for reason in analysis.calibration.reasons)
