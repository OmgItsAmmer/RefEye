"""OffsidePipeline.warm_up — priming identity tracking before a confirm.

The bug this guards against was found on a real confirm: M2.4 only reports
`CONFIRMED` after several *consecutive* matched frames, but the app calls
`OffsidePipeline.analyse()` exactly once, on the single frame the operator
confirmed — so every identity was judged after one frame and could never be
anything but `TENTATIVE`, no matter how clean the kit colours were (see
`offside/pipeline.py::OffsidePipeline.warm_up`). These tests reproduce that
failure directly and confirm `warm_up()` fixes it, using a stub detector and
pose estimator so no real model checkpoint is needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.config.loader import load_settings
from core.domain.models import Detection, FramePacket
from offside.body_keypoints.keypoints import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SOURCE_ANKLE,
    GroundPoint,
    Keypoint,
    PlayerPose,
)
from offside.pipeline import OffsidePipeline
from offside.player_identity import IdentityState
from vision.detection.classes import PLAYER

FRAME_SIZE = (720, 1280)
PLAYER_W, PLAYER_H = 30, 90
GRASS = (50, 130, 60)
RED_KIT = (40, 40, 200)
BLUE_KIT = (200, 60, 40)

#: Fixed pitch positions for six players — three per kit. A confirmed frame
#: draws them each with a tiny (1px) jitter, mimicking ordinary frame-to-frame
#: motion, so IoU association is exercised rather than trivially perfect.
ROSTER = [
    (400, 400, RED_KIT),
    (460, 420, RED_KIT),
    (520, 440, RED_KIT),
    (700, 420, BLUE_KIT),
    (760, 440, BLUE_KIT),
    (820, 460, BLUE_KIT),
]


def _box(x: float, y: float) -> tuple[float, float, float, float]:
    return (x - PLAYER_W / 2, y - PLAYER_H, x + PLAYER_W / 2, y)


def _make_image(frame_id: int) -> np.ndarray:
    image = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    image[:, :] = GRASS
    jitter = (frame_id % 3) - 1
    for x, y, kit in ROSTER:
        x1, y1, x2, y2 = _box(x + jitter, y)
        shoulder_y, hip_y = y1 + PLAYER_H * 0.25, y1 + PLAYER_H * 0.55
        left_x, right_x = x1 + PLAYER_W * 0.15, x2 - PLAYER_W * 0.15
        cv2.rectangle(
            image, (int(left_x), int(shoulder_y)), (int(right_x), int(hip_y)), kit, -1
        )
    return image


def _make_frame(frame_id: int) -> FramePacket:
    image = _make_image(frame_id)
    height, width = image.shape[:2]
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id,
        capture_timestamp_ms=frame_id,
        width=width,
        height=height,
        source_id="test",
        image=image,
    )


class _StubDetector:
    model_name = "stub-detector"

    def detect(self, frame: FramePacket) -> list[Detection]:
        jitter = (frame.frame_id % 3) - 1
        return [
            Detection(
                frame_id=frame.frame_id,
                class_name=PLAYER,
                confidence=0.9,
                bbox_xyxy=_box(x + jitter, y),
                source_model=self.model_name,
            )
            for x, y, _ in ROSTER
        ]


class _StubPoseEstimator:
    model_name = "stub-pose"

    def estimate(self, frame: FramePacket, detections: list[Detection]) -> list[PlayerPose]:
        poses = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            shoulder_y = y1 + (y2 - y1) * 0.25
            hip_y = y1 + (y2 - y1) * 0.55
            left_x, right_x = x1 + (x2 - x1) * 0.15, x2 - (x2 - x1) * 0.15
            ankle_xy = ((x1 + x2) / 2.0, y2)
            keypoints = {
                LEFT_SHOULDER: Keypoint(LEFT_SHOULDER, (left_x, shoulder_y), 0.9),
                RIGHT_SHOULDER: Keypoint(RIGHT_SHOULDER, (right_x, shoulder_y), 0.9),
                LEFT_HIP: Keypoint(LEFT_HIP, (left_x, hip_y), 0.9),
                RIGHT_HIP: Keypoint(RIGHT_HIP, (right_x, hip_y), 0.9),
                LEFT_ANKLE: Keypoint(LEFT_ANKLE, ankle_xy, 0.9),
            }
            poses.append(
                PlayerPose(
                    frame_id=frame.frame_id,
                    bbox_xyxy=detection.bbox_xyxy,
                    detection_confidence=detection.confidence,
                    ground_point=GroundPoint(
                        xy=ankle_xy, confidence=0.9, source=SOURCE_ANKLE, reason="test"
                    ),
                    source_model="stub-pose",
                    keypoints=keypoints,
                )
            )
        return poses


class _StubRegistry:
    resolved_device = "cpu"

    def get_detector(self):
        return _StubDetector()

    def get_pose_estimator(self):
        return _StubPoseEstimator()


@pytest.fixture
def pipeline() -> OffsidePipeline:
    settings = load_settings("config/default.yaml", local_path=None, apply_env=False)
    # No real checkpoint needed for this test — only the stub detector/pose
    # above stand in for models, and auto-landmarks is a separate, real model
    # this test has no reason to load.
    settings.offside.pitch_calibration.auto_landmarks.enabled = False
    return OffsidePipeline(settings, _StubRegistry())


def test_without_warm_up_every_identity_is_tentative(pipeline):
    """Reproduces the bug directly: one call, one frame each, forever new."""
    frame = _make_frame(1)
    analysis = pipeline.analyse(frame, 1)

    assert analysis.identities is not None
    assert len(analysis.identities.identities) == len(ROSTER)
    assert all(
        identity.state is IdentityState.TENTATIVE
        for identity in analysis.identities.identities
    )
    assert analysis.identities.confidence == 0.0


def test_warm_up_lets_identities_reach_confirmed(pipeline):
    """The fix: feed the frames leading up to the confirm through the same
    stages first, so tracks already have real continuity by the time the
    confirmed frame is judged."""
    warm_up_frames = [_make_frame(frame_id) for frame_id in range(1, 7)]
    pipeline.warm_up(warm_up_frames)

    confirmed = _make_frame(7)
    analysis = pipeline.analyse(confirmed, 7)

    assert analysis.identities is not None
    confirmed_identities = [
        identity
        for identity in analysis.identities.identities
        if identity.state is IdentityState.CONFIRMED
    ]
    # Every player was visible, unoccluded and unambiguous throughout — all
    # six should have earned CONFIRMED, not just cleared the floor.
    assert len(confirmed_identities) == len(ROSTER)
    assert analysis.identities.confidence == 1.0


def test_warm_up_writes_no_pipeline_run_log(pipeline, tmp_path, monkeypatch):
    """Warm-up frames are priming, not a result — they must not each produce
    a JSON file under logs/CLI, or one confirm would spam the log directory
    with N-1 files nobody asked to see."""
    pipeline._run_log_config.directory = str(tmp_path)  # noqa: SLF001 — test setup
    warm_up_frames = [_make_frame(frame_id) for frame_id in range(1, 4)]

    pipeline.warm_up(warm_up_frames)

    assert list(tmp_path.iterdir()) == []
