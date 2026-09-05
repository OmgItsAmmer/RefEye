"""Pose estimator against the real model and real broadcast footage.

Skips cleanly when either binary asset is absent (both are gitignored — see
models/pose/README.md). When present this exercises the part that unit tests
cannot: whether crops of ~90px-tall broadcast players actually yield usable
body points, and whether coordinates come back in full-frame space rather
than crop space — an off-by-a-crop-origin bug would look plausible in every
log line while silently placing every player in the wrong spot.

The reference clip is a development fixture, not a design target (M2_Plan
section 4): assertions here stay behavioural (alignment, coordinate space,
honest confidence) and deliberately avoid pinning accuracy numbers to this
one video.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.domain.models import Detection, FramePacket
from offside.body_keypoints.keypoints import (
    SOURCE_ANKLE,
    SOURCE_BBOX_BOTTOM,
)
from vision.detection.classes import BALL, PLAYER

POSE_CHECKPOINT = Path("models/pose/yolo11n-pose.pt")
DETECTOR_CHECKPOINT = Path("models/detector/yolo11n.pt")
CLIP = Path("data/videos/client_m2_test_video.mp4")

#: Frames spread across the clip's open play (it ends on a tight celebration
#: shot, which is a different problem than a wide defensive line).
SAMPLE_FRAMES = (120, 200, 300)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


@pytest.fixture(scope="module")
def clip_frames() -> list[FramePacket]:
    if not CLIP.exists():
        pytest.skip(f"{CLIP} not present — the client reference clip is not in git")

    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    frames: list[FramePacket] = []
    try:
        for frame_id in SAMPLE_FRAMES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, image = capture.read()
            if not ok:
                continue
            height, width = image.shape[:2]
            frames.append(
                FramePacket(
                    frame_id=frame_id,
                    pts=frame_id,
                    timestamp_ms=int(frame_id * 1000 / 30),
                    capture_timestamp_ms=int(frame_id * 1000 / 30),
                    width=width,
                    height=height,
                    source_id="client_m2_test_video",
                    image=image,
                )
            )
    finally:
        capture.release()

    if not frames:
        pytest.skip("could not decode any sample frame from the reference clip")
    return frames


@pytest.fixture(scope="module")
def detections(clip_frames) -> list[list[Detection]]:
    """Real player boxes — the pose estimator's input contract is the M1
    detector's output, so the test feeds it exactly that."""
    if not DETECTOR_CHECKPOINT.exists():
        pytest.skip(f"{DETECTOR_CHECKPOINT} not present — see models/pose/README.md")

    from vision.detection.yolo_detector import YoloObjectDetector

    detector = YoloObjectDetector(
        checkpoint=str(DETECTOR_CHECKPOINT),
        confidence_threshold=0.35,
        ball_confidence_threshold=0.15,
        device=_device(),
        # Wide broadcast framing: players are small, and 640 finds almost
        # none of them. This mirrors what the offside path needs, not the
        # live preview's cheaper setting.
        imgsz=1280,
    )
    detector.load()
    return detector.detect_batch(clip_frames)


@pytest.fixture(scope="module")
def estimator():
    if not POSE_CHECKPOINT.exists():
        pytest.skip(f"{POSE_CHECKPOINT} not present — see models/pose/README.md")

    from offside.body_keypoints.estimator import YoloPoseEstimator

    instance = YoloPoseEstimator(checkpoint=str(POSE_CHECKPOINT), device=_device())
    instance.load()
    instance.warmup()
    return instance


@pytest.fixture(scope="module")
def poses(estimator, clip_frames, detections):
    return estimator.estimate_batch(clip_frames, detections)


def _people(frame_detections: list[Detection]) -> list[Detection]:
    return [d for d in frame_detections if d.class_name != BALL]


def test_the_detector_actually_finds_players_in_this_footage(detections):
    """Guards the fixture itself: if detection collapses, every assertion
    below would pass vacuously on an empty list."""
    counts = [len(_people(d)) for d in detections]
    assert min(counts) >= 6, f"expected a defensive line's worth of players, got {counts}"


def test_one_pose_per_detected_player(poses, detections):
    """Identity stays with the detector/tracker: a pose never adds or drops
    a player, or M2.5's second-last-defender ranking silently shifts."""
    for frame_poses, frame_detections in zip(poses, detections):
        assert len(frame_poses) == len(_people(frame_detections))


def test_ball_detections_are_never_posed(poses, detections):
    posed_boxes = {p.bbox_xyxy for frame in poses for p in frame}
    ball_boxes = {
        d.bbox_xyxy for frame in detections for d in frame if d.class_name == BALL
    }
    assert not (posed_boxes & ball_boxes)


def test_keypoints_come_back_in_full_frame_coordinates(poses, clip_frames):
    """The crop-origin offset is the easiest thing to get wrong here, and it
    fails invisibly — every keypoint would land near the top-left corner."""
    frame = clip_frames[0]
    for player in poses[0]:
        for keypoint in player.keypoints.values():
            x, y = keypoint.xy
            assert -50 <= x <= frame.width + 50
            assert -50 <= y <= frame.height + 50

    inside_own_box = 0
    total = 0
    for player in poses[0]:
        if not player.has_pose:
            continue
        x1, y1, x2, y2 = player.bbox_xyxy
        for keypoint in player.confident_keypoints(0.5).values():
            total += 1
            margin = player.box_height * 0.5
            if x1 - margin <= keypoint.x <= x2 + margin and y1 - margin <= keypoint.y <= y2 + margin:
                inside_own_box += 1

    assert total > 0, "no confident keypoints at all on real footage"
    # Crop-space coordinates would put nearly everything outside its own box.
    assert inside_own_box / total > 0.9


def test_most_players_get_a_real_measured_foot_position(poses):
    """The reason this phase exists: a box bottom is a guess, an ankle is a
    measurement. A regression that quietly falls back everywhere would still
    "work" — every downstream number would just get worse."""
    all_poses = [p for frame in poses for p in frame]
    measured = [p for p in all_poses if p.ground_point.source == SOURCE_ANKLE]
    assert len(measured) / len(all_poses) > 0.4, (
        "too few players got an ankle-based ground point; pose estimation is "
        "degrading on this footage"
    )


def test_players_without_a_pose_still_appear_and_say_why(poses):
    unposed = [p for frame in poses for p in frame if not p.has_pose]
    for player in unposed:
        assert player.warnings, "a failed pose must explain itself"
        assert player.ground_point.source == SOURCE_BBOX_BOTTOM
        assert not player.ground_point.is_measured


def test_ground_point_sits_at_the_bottom_of_the_player(poses):
    """Whatever rung of the ladder produced it, the ground point must be
    near the player's feet — M2.1 projects it as a point on the pitch."""
    for player in (p for frame in poses for p in frame):
        _, y1, _, y2 = player.bbox_xyxy
        assert player.ground_point.xy[1] > y1 + (y2 - y1) * 0.5


def test_measured_ground_points_outrank_guessed_ones(poses):
    """Confidence has to be comparable across players, or M2.6 cannot use it
    to decide whether an offside call is safe to make."""
    all_poses = [p for frame in poses for p in frame]
    measured = [p.ground_point.confidence for p in all_poses if p.ground_point.is_measured]
    guessed = [p.ground_point.confidence for p in all_poses if not p.ground_point.is_measured]

    if measured and guessed:
        assert min(measured) > max(guessed)


def test_every_pose_carries_provenance(poses, estimator):
    for player in (p for frame in poses for p in frame):
        assert player.source_model == estimator.model_name
        assert player.frame_id in SAMPLE_FRAMES


def test_single_frame_estimate_matches_the_batch_path(estimator, clip_frames, detections, poses):
    single = estimator.estimate(clip_frames[0], detections[0])
    assert len(single) == len(poses[0])
    assert [p.bbox_xyxy for p in single] == [p.bbox_xyxy for p in poses[0]]


def test_a_frame_with_no_image_degrades_instead_of_crashing(estimator, clip_frames):
    blind = FramePacket(
        frame_id=999,
        pts=999,
        timestamp_ms=0,
        capture_timestamp_ms=0,
        width=1280,
        height=720,
        source_id="test",
        image=None,
    )
    detection = Detection(
        frame_id=999,
        class_name=PLAYER,
        confidence=0.9,
        bbox_xyxy=(100.0, 100.0, 140.0, 200.0),
        source_model="test",
    )

    result = estimator.estimate(blind, [detection])
    assert len(result) == 1
    assert not result[0].has_pose
    assert result[0].warnings
