"""Team assignment against the real models and real broadcast footage.

Skips cleanly when either checkpoint is absent (both are gitignored — see
models/pose/README.md).

What only real footage can answer: whether a ~90px-tall broadcast player has
enough clean shirt pixels to measure a kit colour from at all, once grass and
skin are removed. Everything about *what the stage concludes* — similar kits,
keeper handling, operator corrections — is decided in
tests/unit/test_team_assignment.py, where the inputs can be controlled.

The reference clip is a development fixture, not a design target (M2_Plan
section 4): the assertions here stay behavioural — every player accounted
for, kits measured, teams reported with honest confidence — and deliberately
avoid pinning numbers to the kit colours of this one match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.domain.models import Detection, FramePacket
from offside.team_assignment import TEAM_IDS, PlayerRole, TeamAssigner
from vision.detection.classes import BALL

POSE_CHECKPOINT = Path("models/pose/yolo11n-pose.pt")
DETECTOR_CHECKPOINT = Path("models/detector/yolo11n.pt")
CLIP = Path("data/videos/client_m2_test_video.mp4")

SAMPLE_FRAMES = (120, 200, 300)

#: A run of consecutive frames, for the per-player pooling and voting.
RUN_START = 90
RUN_LENGTH = 20


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
    if not DETECTOR_CHECKPOINT.exists():
        pytest.skip(f"{DETECTOR_CHECKPOINT} not present — see models/pose/README.md")

    from vision.detection.yolo_detector import YoloObjectDetector

    detector = YoloObjectDetector(
        checkpoint=str(DETECTOR_CHECKPOINT),
        confidence_threshold=0.35,
        ball_confidence_threshold=0.15,
        device=_device(),
        imgsz=1280,
    )
    detector.load()
    return detector.detect_batch(clip_frames)


@pytest.fixture(scope="module")
def poses(clip_frames, detections):
    if not POSE_CHECKPOINT.exists():
        pytest.skip(f"{POSE_CHECKPOINT} not present — see models/pose/README.md")

    from offside.body_keypoints.estimator import YoloPoseEstimator

    estimator = YoloPoseEstimator(checkpoint=str(POSE_CHECKPOINT), device=_device())
    estimator.load()
    return estimator.estimate_batch(clip_frames, detections)


@pytest.fixture(scope="module")
def assignments(clip_frames, detections, poses):
    assigner = TeamAssigner()
    results = []
    for frame, frame_detections, frame_poses in zip(clip_frames, detections, poses):
        ball = [d for d in frame_detections if d.class_name == BALL]
        ball_xy = None
        if ball:
            x1, y1, x2, y2 = max(ball, key=lambda d: d.confidence).bbox_xyxy
            ball_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        results.append(
            assigner.assign(frame.image, frame_poses, ball_xy=ball_xy)
        )
    return results


def test_kits_can_actually_be_measured_on_broadcast_players(assignments):
    """The fixture guard: if shirt sampling collapsed to nothing, every
    assertion below would pass vacuously on empty assignments."""
    for assignment in assignments:
        measured = [p for p in assignment.players if p.color is not None]
        assert measured, "no player's kit colour could be measured at all"
        assert len(measured) >= len(assignment.players) * 0.5


def test_two_teams_are_found_and_both_are_populated(assignments):
    for assignment in assignments:
        assert assignment.color_model is not None
        counts = assignment.counts()
        assert counts[TEAM_IDS[0]] >= 2
        assert counts[TEAM_IDS[1]] >= 2


def test_every_detected_player_is_accounted_for(assignments, poses):
    """Same contract as M2.2: a player who cannot be classified is reported
    as unknown, never dropped — the hardest player to measure is exactly the
    one whose absence would move the offside line."""
    for assignment, frame_poses in zip(assignments, poses):
        assert len(assignment.players) == len(frame_poses)
        assert {p.index for p in assignment.players} == set(range(len(frame_poses)))


def test_confidence_is_reported_honestly_rather_than_asserted(assignments):
    for assignment in assignments:
        assert 0.0 <= assignment.confidence <= 1.0
        # Without a metric pitch and with no goalkeeper confirmation, this
        # footage should never produce a "certain" claim.
        if not assignment.sides_are_known:
            assert assignment.confidence <= 0.35
            assert assignment.warnings


def test_no_goalkeeper_is_claimed_without_a_calibrated_pitch(assignments):
    """Depth on the pitch is what identifies a keeper; with no calibration
    passed in, the honest answer is "cannot tell", not a guess."""
    for assignment in assignments:
        assert all(p.role is not PlayerRole.GOALKEEPER for p in assignment.players)


def test_the_same_footage_gives_the_same_teams_twice(clip_frames, poses):
    """A verdict that changes between two runs of the same clip is not
    something an operator can be asked to trust."""
    first = TeamAssigner().assign(clip_frames[0].image, poses[0])
    second = TeamAssigner().assign(clip_frames[0].image, poses[0])
    assert [p.team_id for p in first.players] == [p.team_id for p in second.players]


@pytest.fixture(scope="module")
def consecutive_frames() -> list[FramePacket]:
    """A run of consecutive frames — the only way to exercise the part of this
    stage that decides a player's team from *many* frames rather than one."""
    if not CLIP.exists():
        pytest.skip(f"{CLIP} not present — the client reference clip is not in git")

    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    frames: list[FramePacket] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, RUN_START)
        for offset in range(RUN_LENGTH):
            ok, image = capture.read()
            if not ok:
                break
            frame_id = RUN_START + offset
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

    if len(frames) < 5:
        pytest.skip("could not decode a run of frames from the reference clip")
    return frames


@pytest.fixture(scope="module")
def run_assignments(consecutive_frames):
    if not (POSE_CHECKPOINT.exists() and DETECTOR_CHECKPOINT.exists()):
        pytest.skip("model checkpoints not present — see models/pose/README.md")

    from offside.body_keypoints.estimator import YoloPoseEstimator
    from vision.detection.yolo_detector import YoloObjectDetector

    detector = YoloObjectDetector(
        checkpoint=str(DETECTOR_CHECKPOINT),
        confidence_threshold=0.35,
        ball_confidence_threshold=0.15,
        device=_device(),
        imgsz=1280,
    )
    detector.load()
    estimator = YoloPoseEstimator(checkpoint=str(POSE_CHECKPOINT), device=_device())
    estimator.load()

    detections = detector.detect_batch(consecutive_frames)
    poses = estimator.estimate_batch(consecutive_frames, detections)

    assigner = TeamAssigner()
    return [
        assigner.assign(frame.image, frame_poses, frame_id=frame.frame_id)
        for frame, frame_poses in zip(consecutive_frames, poses)
    ]


def test_a_players_team_does_not_flicker_between_frames(run_assignments):
    """The reason team decisions are pooled per player rather than made per
    frame. A player flickering between sides is both wrong and destroys the
    operator's trust in everything else drawn on the frame."""
    flips = comparisons = 0
    previous: dict[str, str] = {}
    for assignment in run_assignments:
        current = {
            player.track_id: player.team_id
            for player in assignment.players
            if player.track_id and player.team_id
        }
        for track_id, team in current.items():
            if track_id in previous:
                comparisons += 1
                flips += previous[track_id] != team
        previous = current

    assert comparisons > 50, "not enough tracked players to judge stability"
    assert flips / comparisons < 0.05, f"{flips}/{comparisons} team changes between frames"


def test_solid_kits_are_not_mistaken_for_striped_ones(run_assignments):
    """Both kits in this clip are solid. A pattern test based on brightness
    called half the players striped here — it was finding creases in shirts."""
    patterned = total = 0
    for assignment in run_assignments:
        for player in assignment.players:
            if player.color is not None:
                total += 1
                patterned += player.color.patterned

    assert total > 0
    assert patterned / total < 0.2, f"{patterned}/{total} solid kits read as patterned"


def test_evidence_accumulates_into_settled_players(run_assignments):
    """Nothing is claimed as settled on first sight; confidence has to be
    earned over frames, and then most players should reach it."""
    assert run_assignments[0].settled() == []
    assert len(run_assignments[-1].settled()) >= 5


def test_an_operator_correction_survives_into_the_result(clip_frames, poses):
    assigner = TeamAssigner()
    before = assigner.assign(clip_frames[0].image, poses[0])
    target = next(p for p in before.players if p.team_id is not None)
    other = TEAM_IDS[1] if target.team_id == TEAM_IDS[0] else TEAM_IDS[0]

    assigner.overrides.pin_player(target.anchor_xy, team_id=other)
    after = assigner.assign(clip_frames[0].image, poses[0])

    corrected = after.by_index(target.index)
    assert corrected.team_id == other
    assert corrected.is_operator_set
