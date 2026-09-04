"""Evaluate the pipeline against ground-truth labels.

Architecture.md sections 21 and 50: model behaviour is judged on labelled
footage, through the same harness for every model, not by isolated manual
scripts.

**Scope warning.** These run on the *synthetic* fixture, whose contacts are
scripted and whose figures are colour blocks. Passing here proves the
evaluation machinery works and that the pipeline finds contacts it should
find. It says nothing about accuracy on real broadcast footage — that needs
client clips and is the first thing to do once they are available
(architecture.md section 21).
"""

from __future__ import annotations

import json
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

CLIP = Path("tests/fixtures/sample_match.mp4")
LABELS = Path("tests/fixtures/sample_match.labels.json")

#: How close a candidate must land to count as matching a labelled event.
#: Configurable rather than asserted as a product promise (architecture §3).
TOLERANCE_FRAMES = 12


@pytest.fixture(scope="module")
def analysis_result():
    if not CLIP.exists() or not LABELS.exists():
        pytest.skip(
            "fixture clip/labels missing — run "
            "`python -m tools.video_sampling.make_sample_clip`"
        )

    settings = load_settings("config/default.yaml", local_path=None, apply_env=False)
    settings.runtime.device = "cpu"
    settings.video.local_file.loop = False
    settings.ai.detector.provider = "fixture"
    settings.ai.action_spotter.provider = "kinematic"

    cache = FeatureCache(max_frames=1200)
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
    video.start()

    try:
        deadline = time.time() + 60
        while time.time() < deadline and len(cache) < 80:
            time.sleep(0.5)

        latest_frame = video.latest_frame()
        assert latest_frame is not None, "no video was decoded"

        tracked = manager.submit("evaluation", triggered_at_ms=latest_frame.timestamp_ms)

        deadline = time.time() + 120
        while time.time() < deadline and tracked.state not in (
            RequestState.COMPLETED,
            RequestState.FAILED,
        ):
            time.sleep(0.2)

        assert tracked.state == RequestState.COMPLETED, tracked.error_message
        yield tracked.result, json.loads(LABELS.read_text(encoding="utf-8"))
    finally:
        video.stop()
        manager.stop()
        scheduler.stop()


def test_analysis_returns_multiple_candidates(analysis_result):
    """Never collapse to one result (architecture.md section 58, rule 7)."""
    result, _ = analysis_result
    assert len(result.candidates) >= 2


def test_candidates_are_distinct_frames(analysis_result):
    """De-duplication must not leave the operator paging identical images."""
    result, _ = analysis_result
    frames = [c.refined_frame_id for c in result.candidates]
    assert len(frames) == len(set(frames))


def test_candidates_are_ranked_by_rank_score(analysis_result):
    result, _ = analysis_result
    scores = [c.evidence["rank_score"] for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_every_candidate_records_its_source_model(analysis_result):
    """Architecture.md section 58, rule 13: results carry model provenance."""
    result, _ = analysis_result
    for candidate in result.candidates:
        assert candidate.evidence.get("source_model")
    assert result.diagnostics["models"]["action_spotter"]["name"]


def test_recall_against_labelled_events(analysis_result):
    """At least one labelled contact is recovered within tolerance.

    Deliberately a weak floor: the fixture's scripted events are the only
    ground truth available here, and a stricter threshold would encode an
    accuracy claim this data cannot support.
    """
    result, labels = analysis_result

    labelled = [event["frame"] for event in labels["events"]]
    found = [c.refined_frame_id for c in result.candidates]

    matched = [
        frame
        for frame in labelled
        if any(abs(frame - candidate) <= TOLERANCE_FRAMES for candidate in found)
    ]

    assert matched, (
        f"no labelled event matched within {TOLERANCE_FRAMES} frames.\n"
        f"  labelled: {labelled}\n"
        f"  found:    {sorted(found)}"
    )


def test_reports_timing_for_each_stage(analysis_result):
    """Runtime is a first-class evaluation metric (architecture.md section 21)."""
    result, _ = analysis_result
    for key in ("spotting_ms", "refinement_ms", "total_ms"):
        assert key in result.diagnostics
    assert result.diagnostics["total_ms"] < 30_000
