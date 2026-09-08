"""M1 detection — a ball missed at the primary resolution gets one retry.

Found on a real confirm: the operator could see a labelled "BALL" box on
screen while the verdict said "no ball was detected". Traced to two separate
detector passes disagreeing — M2's own high-resolution pass (what the
verdict is built from) and the live preview's lower-resolution pass (what
produced the visible box) — with nothing reconciling the two. This is the
fix: when the high-resolution pass finds no plausible ball, retry once,
ball-only, at the same resolution the live preview uses, so a ball visibly
on screen elsewhere in the app is not silently contradicted without ever
being looked for the same way. Never used to add players — the primary pass
stays the only source of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config.loader import load_settings
from core.domain.models import Detection, FramePacket
from offside.pipeline import OffsidePipeline
from vision.detection.classes import BALL, PLAYER

LIVE_PREVIEW_IMGSZ = 640  # core/config default: ai.detector.imgsz


def make_frame(frame_id: int = 1) -> FramePacket:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id,
        capture_timestamp_ms=frame_id,
        width=1280,
        height=720,
        source_id="test",
        image=image,
    )


class _ResolutionSensitiveDetector:
    """Finds the ball only at one specific resolution — mirroring the real
    case this fallback exists for: a save/rebound with the ball tight
    against the keeper's hands, missed by the primary pass, caught by the
    cheaper one. `_imgsz` starts at the live preview's own resolution,
    matching how the same detector instance sits between M2 calls in the
    real app — M2 only ever bumps it up for the duration of its own call.
    """

    model_name = "stub-detector"

    def __init__(self, ball_visible_at: int | None):
        self._imgsz = LIVE_PREVIEW_IMGSZ
        self._ball_visible_at = ball_visible_at
        self.calls: list[int] = []

    def detect(self, frame: FramePacket) -> list[Detection]:
        self.calls.append(self._imgsz)
        players = [
            Detection(
                frame_id=frame.frame_id,
                class_name=PLAYER,
                confidence=0.9,
                bbox_xyxy=(400.0 + step * 40, 400.0, 430.0 + step * 40, 490.0),
                source_model=self.model_name,
            )
            for step in range(4)
        ]
        if self._imgsz != self._ball_visible_at:
            return players
        # Plausible relative to the ~90px-tall players above.
        ball_box = (700.0, 440.0, 715.0, 455.0)
        return players + [
            Detection(
                frame_id=frame.frame_id,
                class_name=BALL,
                confidence=0.4,
                bbox_xyxy=ball_box,
                source_model=self.model_name,
            )
        ]


class _Registry:
    def __init__(self, detector):
        self._detector = detector

    def get_detector(self):
        return self._detector

    def get_pose_estimator(self):
        return None


@pytest.fixture
def settings():
    loaded = load_settings("config/default.yaml", local_path=None, apply_env=False)
    # No real checkpoint needed for this test — auto-landmarks is a separate,
    # real model this test has no reason to load.
    loaded.offside.pitch_calibration.auto_landmarks.enabled = False
    return loaded


def test_a_ball_missed_at_the_primary_resolution_is_retried_at_the_live_previews(
    settings,
):
    detector = _ResolutionSensitiveDetector(ball_visible_at=LIVE_PREVIEW_IMGSZ)
    pipeline = OffsidePipeline(settings, _Registry(detector))

    analysis = pipeline.analyse(make_frame(), 1)

    assert analysis.ball is not None
    # The primary (higher-resolution) call first, then the fallback.
    assert detector.calls == [pipeline._detector_imgsz, LIVE_PREVIEW_IMGSZ]
    report = {r.key: r for r in analysis.reports}["detection"]
    assert "fallback pass" in " ".join(report.details)
    assert "ball found" in report.summary


def test_a_ball_missed_at_every_resolution_is_still_reported_honestly(settings):
    detector = _ResolutionSensitiveDetector(ball_visible_at=None)
    pipeline = OffsidePipeline(settings, _Registry(detector))

    analysis = pipeline.analyse(make_frame(), 1)

    assert analysis.ball is None
    report = {r.key: r for r in analysis.reports}["detection"]
    assert "no ball this frame" in report.summary
    assert "a missing ball is normal" in " ".join(report.details)
    assert "fallback" not in " ".join(report.details)


def test_no_retry_when_the_primary_pass_already_found_the_ball(settings):
    """The whole point is a *fallback* — a ball the primary pass already
    caught must not trigger a second, redundant detector call."""

    class _AlwaysBallDetector(_ResolutionSensitiveDetector):
        def detect(self, frame):
            self.calls.append(self._imgsz)
            return [
                Detection(
                    frame_id=frame.frame_id,
                    class_name=PLAYER,
                    confidence=0.9,
                    bbox_xyxy=(400.0, 400.0, 430.0, 490.0),
                    source_model=self.model_name,
                ),
                Detection(
                    frame_id=frame.frame_id,
                    class_name=BALL,
                    confidence=0.4,
                    bbox_xyxy=(700.0, 440.0, 715.0, 455.0),
                    source_model=self.model_name,
                ),
            ]

    always_ball = _AlwaysBallDetector(ball_visible_at=None)
    pipeline = OffsidePipeline(settings, _Registry(always_ball))

    analysis = pipeline.analyse(make_frame(), 1)

    assert analysis.ball is not None
    assert always_ball.calls == [pipeline._detector_imgsz]  # one call, not two


def test_the_fallback_pass_never_contributes_players(settings):
    """The primary pass stays the only source of players — a lower-
    resolution fallback must not silently add a second, less reliable
    player count on top of it."""
    detector = _ResolutionSensitiveDetector(ball_visible_at=LIVE_PREVIEW_IMGSZ)
    pipeline = OffsidePipeline(settings, _Registry(detector))

    analysis = pipeline.analyse(make_frame(), 1)

    assert len(analysis.players) == 4  # exactly the primary pass's own count


def test_a_fallback_ball_is_tagged_by_source_not_blended_in_silently(settings):
    detector = _ResolutionSensitiveDetector(ball_visible_at=LIVE_PREVIEW_IMGSZ)
    pipeline = OffsidePipeline(settings, _Registry(detector))

    analysis = pipeline.analyse(make_frame(), 1)

    assert "fallback" in analysis.ball.source_model
